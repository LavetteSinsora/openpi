"""The recorded episode, dressed up as a policy server.

`DatasetClient` has PolicyClient's interface and returns the dataset's OWN action
labels instead of predictions. Everything downstream of `client.infer` — the
measured-FK anchor, the delta composition, mink IK, the safety clamp, the 500 Hz
emitter — runs exactly as it does with a real checkpoint. So this replays the
action labels through the real control path with the model out of the loop.

What that tests, and what `check.replay` (rung 6) does not: rung 6 streams the
dataset's stored `arm_qpos` straight to the motors, so it never touches an action
label. It proves the plumbing (joint order, sign, units, CRC, rates, hands). This
proves the TRANSFORMS: if `inv(pose[t0]) @ pose[t0+k]` or the anchor convention is
wrong, rung 6 still looks perfect and this does not.

The anchor is the robot's measured FK, never the stored pose — the deltas are
composed onto where the arm ACTUALLY is, so IK error feeds forward exactly as it
will at deployment. That is the same "measured" anchor mode the dashboard replay
renders, which is why the robot should trace the dashboard video.

Which tick a chunk is anchored at: after a re-plan we have popped `j` slots of the
active chunk, so "now" is t0 + j/fps and that is the new anchor tick (the
arithmetic in chunk.py, in reverse). The loop owns `j`; we read it through the
callback `attach_consumed` installs. Counting queries instead — assuming a chunk
is always consumed whole — silently slides the dataset out of step with the robot
every time the loop re-plans a few slots early.
"""

import logging
import pathlib

import numpy as np

from ego2g1.common import layout
from ego2g1.eval_replay import dataset_io as dio
from ego2g1.eval_replay.rollout import gt_actions

logger = logging.getLogger(__name__)


class DatasetClient:
    """Drop-in for `deploy.client.PolicyClient`, backed by a recorded episode."""

    def __init__(self, dataset_root, *, episode: int | None = None,
                 source_episode: str | None = None, horizon: int = 50,
                 fps: int | None = None):
        root = pathlib.Path(dataset_root)
        if episode is None:
            if source_episode is None:
                raise ValueError("give episode= or source_episode=")
            found = dio.source_to_episodes(root, source_episode)
            if len(found) > 1:
                logger.info("source %s -> LeRobot episodes %s; using %d "
                            "(one sub-episode = one timeline)",
                            source_episode, found, found[0])
            episode = found[0]

        self.ep = dio.load_episode(root, episode)

        # The PolicyClient surface the loop reads.
        self.action_horizon = int(horizon)
        self.action_dim = layout.DIM
        self.fps = int(fps or round(self.ep.fps))
        self.hands = layout.HANDS
        self.rtc = {"sampler": "dataset"}
        self.rtc_training = False
        self.metadata = {"ego2g1": {"source": "dataset"}}

        self.task = self.ep.task
        self.tick = 0            # dataset tick the ACTIVE chunk is anchored at
        self.queries = 0
        self.exhausted = False
        self._consumed = lambda: 0

    def attach_consumed(self, fn) -> None:
        """Install a callable returning how many slots of the active chunk the
        loop has popped. Must be attached before the loop starts."""
        self._consumed = fn

    @property
    def n_frames(self) -> int:
        return self.ep.n_frames

    def next_tick(self) -> int:
        """The tick the NEXT chunk will be anchored at. Pure — the camera reads
        this before `infer` mutates anything."""
        if self.queries == 0:
            return 0
        return min(self.tick + int(self._consumed()), self.ep.n_frames - 1)

    def infer(self, image, state, prompt, *, prev_chunk=None, d: int = 0) -> dict:
        """`image`, `state` and `prompt` are ignored — that is the point. They are
        what a policy would condition on; the labels are indexed by time instead.

        `prev_chunk` (the RTC prefix) is likewise ignored: the labels are already
        self-consistent across chunks, so there is nothing to guide. Run the loop
        in blocking mode; async + RTC has nothing to add here and would only make
        the tick bookkeeping harder to reason about.
        """
        self.tick = self.next_tick()
        actions = gt_actions(self.ep, [self.tick], self.action_horizon)[0]
        self.queries += 1
        self.exhausted = self.tick + self.action_horizon >= self.ep.n_frames - 1

        logger.info("chunk %d: dataset tick %d/%d%s", self.queries, self.tick,
                    self.ep.n_frames - 1, "  (last)" if self.exhausted else "")
        return {
            "actions": actions.astype(np.float32),
            "client_latency_s": 0.0,
            "policy_timing": {},
            "rtc": {"sampler": "dataset"},
        }


class DatasetCamera:
    """The camera the loop insists on having. Serves the recorded frame at the
    tick the client is about to be queried for, so a Rerun/preview stream shows
    what the policy WOULD have seen. Nothing consumes the pixels.

    `video=False` (the default) skips the AV1 decode entirely and returns a black
    frame: the action source ignores images, so paying ~300 frames of decode to
    feed a stub is only worth it when a human is looking.
    """

    def __init__(self, client: DatasetClient, *, video: bool = False,
                 size: int = 224):
        self.client = client
        self.video = video
        self._frames = None
        self._blank = np.zeros((size, size, 3), dtype=np.uint8)

    def connect(self) -> None:
        if not self.video:
            return
        ep = self.client.ep
        self._frames = dio.read_video_frames(ep.video_path, ep.n_frames, ep.fps)
        logger.info("decoded %d frames from %s", len(self._frames), ep.video_path.name)

    def read(self) -> np.ndarray:
        if self._frames is None:
            return self._blank
        return self._frames[min(self.client.next_tick(), len(self._frames) - 1)]

    def close(self) -> None:
        self._frames = None
