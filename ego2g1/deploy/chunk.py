"""The active action chunk, and the RTC prefix bookkeeping.

The index arithmetic here is the easiest thing in the whole client to get subtly
wrong, so it is stated once, explicitly:

  A chunk generated against an anchor taken at time t0 has, at slot k, the action
  for time  t0 + (k+1)/fps.  (Training gathers poses at [0..H]/fps with row 0 as
  the anchor, and hand commands at [1..H]/fps — so slot k is one tick AHEAD of
  the anchor, not level with it.)

  At a re-plan we have consumed `j` slots, so "now" is t0 + j/fps, and that is
  the new anchor. A new chunk's slot i is for time
        t_new + (i+1)/fps  =  t0 + (j + i + 1)/fps  =  old slot (j + i).

  Hence: new slot i  <->  old slot j+i, and the RTC prefix is simply
  `actions[j:]`, left-aligned to row 0. That is exactly LeRobot's
  `original_queue[last_index:]`.

The eef dims of that leftover are deltas from the OLD anchor, so they must be
re-anchored to the new one before they mean anything (se3.reanchor_chunk). The
hand dims are absolute and pass through.
"""

import threading

import numpy as np

from ego2g1.common import layout, se3


class ChunkQueue:
    """Thread-safe: the control thread consumes, the inference thread replaces."""

    def __init__(self, horizon: int, fps: int):
        self.horizon = int(horizon)
        self.fps = int(fps)
        self._lock = threading.Lock()
        self._actions: np.ndarray | None = None   # (H, 30) raw robot action space
        self._anchor: dict | None = None          # {hand: (4,4)} pelvis frame
        self._index = 0                           # slots consumed from this chunk

    # --- state --------------------------------------------------------------

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._actions is not None

    @property
    def index(self) -> int:
        with self._lock:
            return self._index

    def remaining(self) -> int:
        with self._lock:
            return 0 if self._actions is None else self.horizon - self._index

    def anchor(self) -> dict | None:
        with self._lock:
            return None if self._anchor is None else dict(self._anchor)

    # --- consumption --------------------------------------------------------

    def pop(self):
        """Next action (30,), or None if exhausted. Advances the index."""
        with self._lock:
            if self._actions is None or self._index >= self.horizon:
                return None
            a = self._actions[self._index].copy()
            self._index += 1
            return a

    def peek_slot(self) -> int:
        with self._lock:
            return self._index

    # --- replacement --------------------------------------------------------

    def replace(self, actions, anchor: dict, d: int) -> None:
        """Install a new chunk generated against `anchor`, skipping `d` slots.

        Slots [0, d) of the new chunk correspond to the ticks that elapsed WHILE
        we were inferring — the robot already executed the old chunk's actions for
        those instants, and under RTC the new chunk was guided to agree with them.
        Re-executing them would rewind the robot, so we start at slot d.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.horizon, layout.DIM):
            raise ValueError(f"expected ({self.horizon}, {layout.DIM}), got {actions.shape}")
        with self._lock:
            self._actions = actions
            self._anchor = dict(anchor)
            self._index = int(np.clip(d, 0, self.horizon))

    def clear(self) -> None:
        with self._lock:
            self._actions = None
            self._anchor = None
            self._index = 0

    # --- RTC ----------------------------------------------------------------

    def rtc_prefix(self, anchor_new: dict, start: int):
        """The guidance target for the next chunk: what the robot is ABOUT TO DO,
        re-anchored to the new anchor.

        `start` is a WALL-CLOCK-aligned slot index supplied by the caller, not our
        consumption index — and the difference matters. The control thread runs
        AHEAD of the wall clock (it pops actions to keep `lookahead_s` of joint
        trajectory queued), so `self._index` points a few slots into the future.
        Slicing there would hand the server a target ~lookahead ahead of reality
        and RTC would pull every new chunk forward in time — a nudge at each seam,
        which is precisely the discontinuity it exists to remove.

        The right alignment is by TIME: row 0 must be the action for the instant
        the new chunk's slot 0 will execute. That set INCLUDES slots we have
        already popped and IK'd but not yet executed — they are sitting in the
        trajectory buffer and they WILL run during the inference, so the new chunk
        must agree with them. The prefix is "what the robot will do", not "what we
        have not looked at yet".

        Returns `(prefix (H, 30) float32, n_real)`, left-aligned to the new chunk's
        slot 0 and zero-padded, or `(None, 0)` if there is no chunk yet (first
        inference of an episode -> plain sampling).

        `n_real` MUST travel with the prefix. The padding is not a benign filler:
        a zero vec9 decodes to the ZERO matrix (det 0), not a pose. If the server
        weights those rows — which it will, whenever the leftover is shorter than
        the guidance band — it guides the chunk toward a degenerate target. The
        LeRobot reference clamps its horizon to the leftover length for exactly
        this reason.
        """
        with self._lock:
            if self._actions is None or self._anchor is None:
                return None, 0
            s = int(np.clip(start, 0, self.horizon))
            leftover = self._actions[s:].copy()
            anchor_old = dict(self._anchor)

        if len(leftover) == 0:
            return None, 0

        rehomed = se3.reanchor_chunk(leftover, anchor_old, anchor_new)

        out = np.zeros((self.horizon, layout.DIM), dtype=np.float32)
        out[: len(rehomed)] = rehomed
        return out, len(rehomed)


def targets_from(action: np.ndarray, anchor: dict) -> dict:
    """One action row -> {hand: (4,4)} absolute pelvis-frame flange targets."""
    return {h: se3.compose(anchor[h], action[layout.EEF[h]]) for h in layout.HANDS}


def hands_from(action: np.ndarray) -> dict:
    """One action row -> {hand: (6,)} absolute motor commands, clipped to [0,1]."""
    return {h: np.clip(np.asarray(action[layout.HAND[h]], dtype=np.float32), 0.0, 1.0)
            for h in layout.HANDS}
