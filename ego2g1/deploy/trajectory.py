"""The joint-trajectory buffer: the seam between the 30 Hz planner and the
500 Hz emitter.

The control thread appends timestamped joint knots (one per IK solve). The
emitter thread reads it at 500 Hz and linearly interpolates. That decoupling is
the only reason a ~400 ms inference stall does not stall the robot: the buffer
still has knots from the previous chunk, so the emitter keeps emitting.

Past the last knot we HOLD it. We never extrapolate — extrapolating a joint
trajectory off the end of a plan is how you get a runaway, and "stopped" is the
only safe thing to do when we have run out of plan.
"""

import bisect
import threading

import numpy as np


class TrajectoryBuffer:
    """Thread-safe, timestamped joint knots with linear interpolation.

    Times are `time.monotonic()` seconds. Not wall-clock: the emitter and the
    planner must agree on a clock that cannot jump.
    """

    def __init__(self, dof: int, *, keep_past: float = 0.5):
        self._dof = int(dof)
        self._keep_past = float(keep_past)
        self._t: list[float] = []
        self._q: list[np.ndarray] = []
        self._lock = threading.Lock()

    def seed(self, t: float, q) -> None:
        """Start the buffer at the measured configuration. Until this is called
        the emitter has nothing to send and must not publish."""
        q = self._check(q)
        with self._lock:
            self._t = [float(t)]
            self._q = [q]

    def push(self, t: float, q) -> None:
        """Append a knot. Out-of-order knots are dropped rather than reordered:
        a knot in the past means the planner fell behind, and silently splicing
        it into the timeline would rewind the robot."""
        q = self._check(q)
        t = float(t)
        with self._lock:
            if not self._t:
                self._t, self._q = [t], [q]
                return
            if t <= self._t[-1]:
                return  # stale; the emitter has already passed this instant
            self._t.append(t)
            self._q.append(q)

    def replace_after(self, t: float, knots) -> None:
        """Drop everything strictly after `t` and splice in a new plan.

        Used when a new chunk lands: the tail of the old plan is no longer what
        we intend to do. Knots at or before `t` are kept so the interpolation the
        emitter is *currently inside* stays continuous.
        """
        with self._lock:
            cut = bisect.bisect_right(self._t, float(t))
            self._t = self._t[:cut]
            self._q = self._q[:cut]
            for kt, kq in knots:
                kt = float(kt)
                if self._t and kt <= self._t[-1]:
                    continue
                self._t.append(kt)
                self._q.append(self._check(kq))

    def reseed(self, t: float, q) -> None:
        """Drop the whole plan and restart from `q` at time `t`.

        Use this at a splice, with `q = eval(now)` — the value the emitter is
        actually sending right now.

        Why not just `replace_after(now, [])`: that KEEPS the last knot at or
        before `now`, and after a hold (the emitter ran off the end of the plan and
        froze) that knot can be hundreds of milliseconds in the past. The next
        pushed knot then lands at `now + 1/fps`, so the segment spans
        [t_stale, now + 1/fps] but is evaluated at `now` — alpha ~0.92 after a
        400 ms hold. The emitter jumps almost the entire way to the new knot in ONE
        emit period: a 0.15 rad step at 500 Hz is 75 rad/s, exactly the torque
        spike dds.py warns about. Re-anchoring at `now` makes the segment one tick
        wide again, so the clamp's step limit is actually a RATE limit.
        """
        q = self._check(q)
        with self._lock:
            self._t = [float(t)]
            self._q = [q]

    def eval(self, t: float):
        """Interpolated joints at time `t`, or None if the buffer is unseeded.

        Before the first knot -> the first knot. After the last -> the last (hold).
        """
        t = float(t)
        with self._lock:
            if not self._t:
                return None
            if t <= self._t[0]:
                return self._q[0].copy()
            if t >= self._t[-1]:
                return self._q[-1].copy()
            i = bisect.bisect_right(self._t, t)
            t0, t1 = self._t[i - 1], self._t[i]
            q0, q1 = self._q[i - 1], self._q[i]
            span = t1 - t0
            a = 0.0 if span <= 0 else (t - t0) / span
            return q0 + a * (q1 - q0)

    def runway(self, t: float) -> float:
        """Seconds of plan remaining after `t`. Zero means the emitter is holding
        the last knot — i.e. we have run out of plan and the robot is frozen."""
        with self._lock:
            if not self._t:
                return 0.0
            return max(0.0, self._t[-1] - float(t))

    def prune(self, t: float) -> None:
        """Drop knots older than `t - keep_past`. Keeps one knot before the cut so
        interpolation at `t` still brackets."""
        with self._lock:
            cutoff = float(t) - self._keep_past
            i = bisect.bisect_right(self._t, cutoff)
            if i > 1:
                del self._t[: i - 1]
                del self._q[: i - 1]

    def last(self):
        with self._lock:
            return (self._t[-1], self._q[-1].copy()) if self._t else (None, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._t)

    def _check(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (self._dof,):
            raise ValueError(f"expected ({self._dof},) joints, got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"non-finite joints: {q}")
        return q
