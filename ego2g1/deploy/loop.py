"""The control loop: five threads, and the timing that ties them together.

  T1 arm emitter   500 Hz   traj_arm.eval(now) -> rt/lowcmd
  T2 hand emitter  200 Hz   traj_hand.eval(now) -> rt/brainco/*/cmd
  T3 state RX      DDS      handled by unitree_sdk2py's own callback thread
  T4 control       30 Hz    build obs, IK, top up the trajectory, trigger inference
  T5 inference     ad hoc   the blocking websocket call

T5 exists so that T4 never blocks. If T4 blocked on a ~400 ms inference it would
stop producing IK knots, the trajectory would drain, and T1 would run off the end
and freeze the robot mid-motion. With T5 separate, T4 keeps solving IK on the
CURRENT chunk's remaining slots while the GPU works on the next one. That overlap
is precisely what RTC exists to reconcile.

Timing (the part to get right):

  A chunk with anchor at t0 has, at slot k, the action for time t0 + (k+1)/fps.
  We trigger inference at slot `trigger`, promising the server a delay of `d`
  ticks. Inference runs; meanwhile we execute slots trigger..trigger+d-1. The new
  chunk is then installed at index d, whose action is for t0' + (d+1)/fps — i.e.
  exactly one tick after the splice instant. Continuous by construction.

  If the chunk lands EARLY we hold it until the promised instant (rather than
  splicing early and desynchronising from the mask the server used). If it lands
  LATE we skip forward to where we actually are — never rewind — and count a
  budget violation.
"""

import dataclasses
import logging
import threading
import time

import numpy as np

from ego2g1.common import layout
from ego2g1.deploy import chunk as _chunk
from ego2g1.deploy import safety as _safety
from ego2g1.deploy.trajectory import TrajectoryBuffer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LoopConfig:
    task: str
    fps: int = 30
    arm_hz: float = 500.0
    hand_hz: float = 200.0
    # Seconds of joint trajectory we keep queued ahead of the emitter. Enough to
    # ride out a late control tick; short enough that a re-plan is still reactive.
    lookahead_s: float = 0.10
    # Re-plan when this many slots remain unexecuted (must exceed d, or the chunk
    # lands after we have run out of plan).
    replan_margin: int = 8
    blocking: bool = False


class DeployLoop:
    def __init__(self, cfg: LoopConfig, *, dds, camera, kinematics, client, budget,
                 limits: _safety.SafetyLimits = _safety.SafetyLimits()):
        self.cfg = cfg
        self.dds = dds
        self.cam = camera
        self.kin = kinematics
        self.client = client
        self.budget = budget

        self.H = client.action_horizon
        self.fps = client.fps
        self.dt = 1.0 / self.fps

        self.queue = _chunk.ChunkQueue(self.H, self.fps)
        self.traj_arm = TrajectoryBuffer(layout.ARM_DOF)
        self.traj_hand = TrajectoryBuffer(layout.HAND_DIM * len(layout.HANDS))

        self.clamp = _safety.Clamp(limits)
        self.watchdog = _safety.Watchdog(limits, on_trip=self.dds.damp)
        self.limits = limits

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self._anchor_time = 0.0             # monotonic time of the active chunk's anchor
        self._pending = None                # result from T5, awaiting its splice instant
        self._inferring = False
        self._infer_lock = threading.Lock()

        # The starvation watchdog must not be armed until a plan has EXISTED. Before
        # the first chunk the trajectory is legitimately a single held knot, so
        # runway is 0 by construction — and the first inference on a cold policy
        # server includes a JIT compile that can take tens of seconds. Arming from
        # t=0 damps the robot before it has ever moved.
        self._armed = False

        # If the server has RTC switched off, sending it a prefix is pointless: it
        # returns an unguided chunk that we would still splice at slot d, i.e. an
        # unconstrained seam. Better to know, and to say so.
        self._server_rtc = bool(getattr(client, "rtc", {}).get("enabled", True))
        if not self._server_rtc and not cfg.blocking:
            logger.warning(
                "server has RTC disabled: chunks will be unguided but still spliced "
                "at slot d, so seams rely on the joint clamp alone. Consider --blocking."
            )

        self.stats = {"ticks": 0, "chunks": 0, "late": 0, "starved": 0}

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        arm_q = self.dds.arm_q()
        now = time.monotonic()
        self.traj_arm.seed(now, arm_q)
        self.traj_hand.seed(now, np.zeros(layout.HAND_DIM * len(layout.HANDS)))
        self.clamp.reset(arm_q)
        self.kin.ground(arm_q)

        for fn, name in ((self._emit_arm, "emit-arm"),
                         (self._emit_hand, "emit-hand"),
                         (self._control, "control")):
            t = threading.Thread(target=self._guard(fn), name=name, daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("loop started: %d Hz control, %.0f Hz arm, %.0f Hz hand",
                    self.fps, self.cfg.arm_hz, self.cfg.hand_hz)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)

    def estop(self, reason: str = "manual") -> None:
        self.watchdog.trip(reason)

    def _guard(self, fn):
        """Any uncaught exception in ANY thread must damp the robot, not silently
        kill the thread and leave the arm stiff at its last setpoint."""
        def run():
            try:
                fn()
            except Exception as e:
                logger.exception("thread %s died", threading.current_thread().name)
                self.watchdog.trip(f"{threading.current_thread().name} died: {e}")
        return run

    # --- T1: arm emitter ----------------------------------------------------

    def _emit_arm(self) -> None:
        period = 1.0 / self.cfg.arm_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            now = time.monotonic()

            if not self.watchdog.tripped:
                self.watchdog.check_state_age(self.dds.lowstate_age())
                q = self.traj_arm.eval(now)
                if q is not None and not self.watchdog.tripped:
                    self.dds.send_arm(q)         # waist pinned to 0 inside
                self.traj_arm.prune(now)

            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    # --- T2: hand emitter ---------------------------------------------------

    def _emit_hand(self) -> None:
        period = 1.0 / self.cfg.hand_hz
        n = layout.HAND_DIM
        while not self._stop.is_set():
            t0 = time.perf_counter()
            now = time.monotonic()

            if not self.watchdog.tripped:
                v = self.traj_hand.eval(now)
                if v is not None:
                    self.dds.send_hands({h: v[i * n:(i + 1) * n]
                                         for i, h in enumerate(layout.HANDS)})
                self.traj_hand.prune(now)

            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    # --- T4: control --------------------------------------------------------

    def _control(self) -> None:
        # Poll faster than the action rate: the loop is event-driven (top up the
        # trajectory, splice when due), not strictly one-action-per-tick.
        period = self.dt / 3.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self.watchdog.tripped:
                time.sleep(period)
                continue

            now = time.monotonic()
            self._maybe_splice(now)
            self._top_up(now)
            self._maybe_infer(now)

            # Starvation only means something once we have HAD a plan. Arming it
            # before the first chunk lands would fire during the (possibly very
            # long, JIT-compiling) first inference.
            if self._armed:
                self.watchdog.check_starvation(self.traj_arm.runway(now), now)
            if self.cam.age() > self.limits.max_camera_age:
                self.watchdog.trip(f"camera stale for {self.cam.age():.2f}s — "
                                   "policy would run on a frozen frame")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _top_up(self, now: float) -> None:
        """Keep `lookahead_s` of joint trajectory queued, one IK solve per action.

        One solve per tick is ~1 ms of a 33 ms budget (measured). Batch-solving the
        whole chunk would cost 50-250 ms on top of the inference stall and would be
        wasted on any chunk we abandon early.
        """
        while self.traj_arm.runway(now) < self.cfg.lookahead_s:
            anchor = self.queue.anchor()
            if anchor is None:
                return
            slot = self.queue.peek_slot()
            action = self.queue.pop()
            if action is None:
                return  # chunk exhausted; the emitter holds the last knot

            t_k = self._anchor_time + (slot + 1) * self.dt
            if t_k <= now:
                continue  # we fell behind; drop this slot rather than rewind

            if not _safety.sanity_check_action(action):
                self.watchdog.trip(f"non-finite or absurd action at slot {slot}")
                return

            targets = _chunk.targets_from(action, anchor)
            q = self.kin.solve(targets)
            self.watchdog.check_tracking(self.kin.tracking_error(targets))
            q = self.clamp(q, self.dt)
            self.traj_arm.push(t_k, q)

            hands = _chunk.hands_from(action)
            self.traj_hand.push(t_k, np.concatenate([hands[h] for h in layout.HANDS]))
            self._last_hand = hands
            self.stats["ticks"] += 1

    # --- inference ----------------------------------------------------------

    def _should_infer(self, now: float) -> bool:
        if self._inferring or self._pending is not None:
            return False
        if not self.queue.ready:
            return True                       # first chunk of the episode
        if self.cfg.blocking:
            return self.queue.remaining() <= 0
        trigger = self.H - self.budget.d - self.cfg.replan_margin
        return self.queue.peek_slot() >= max(1, trigger)

    def _maybe_infer(self, now: float) -> None:
        if not self._should_infer(now):
            return

        arm_q = self.dds.arm_q()
        anchor_new = self.kin.flange_poses(arm_q)

        # The state's hand block must be the command being emitted RIGHT NOW, not
        # self._last_hand — which _top_up sets to the furthest-future slot it has
        # popped (up to lookahead_s ahead). During a closing grasp that is ~3 ticks
        # early, a train/serve mismatch in 12 of 30 state dims exactly when the hand
        # is moving fastest. eval(now) is the ground truth the emitter uses.
        n = layout.HAND_DIM
        hv = self.traj_hand.eval(now)
        if hv is not None:
            hand_now = {h: hv[i * n:(i + 1) * n] for i, h in enumerate(layout.HANDS)}
        else:
            hand_now = self._last_hand
        state = self.kin.state(arm_q, hand_now)

        image = self.cam.read()
        if image is None:
            self.watchdog.trip("no camera frame")
            return

        # The prefix must be aligned to the WALL CLOCK, not to queue.index (which
        # runs `lookahead_s` ahead — see ChunkQueue.rtc_prefix). Row 0 has to be the
        # action for the instant the new chunk's slot 0 will execute:
        #     new slot 0 time  = now + 1/fps
        #     old slot m  time = anchor_time + (m+1)/fps
        #     =>  m = (now - anchor_time) * fps
        prefix, n_prefix = None, 0
        use_rtc = not self.cfg.blocking and self._server_rtc and self.queue.ready
        if use_rtc:
            m = int(round((now - self._anchor_time) * self.fps))
            prefix, n_prefix = self.queue.rtc_prefix(anchor_new, m)
        d = 0 if prefix is None else min(self.budget.d, max(0, n_prefix - 1))
        t_request = now

        if self.cfg.blocking:
            self._run_infer(image, state, prefix, d, n_prefix, anchor_new, t_request)
            return

        with self._infer_lock:
            self._inferring = True
        threading.Thread(
            target=self._guard(
                lambda: self._run_infer(image, state, prefix, d, n_prefix, anchor_new, t_request)
            ),
            name="inference", daemon=True,
        ).start()

    def _run_infer(self, image, state, prefix, d, n_prefix, anchor_new, t_request) -> None:
        in_flight = self.queue.ready
        try:
            out = self.client.infer(image, state, self.cfg.task, prev_chunk=prefix, d=d,
                                    n_prefix=n_prefix)
            latency = out["client_latency_s"]
            self.budget.observe(latency)
            with self._infer_lock:
                self._pending = {
                    "actions": np.asarray(out["actions"], dtype=np.float32),
                    "anchor": anchor_new,
                    "d": int(d),
                    "t_request": t_request,
                    "latency": latency,
                    "sampler": out.get("rtc", {}).get("sampler", "?"),
                    # Was a chunk actually EXECUTING while this inference ran? That
                    # is the difference between splicing into a moving trajectory
                    # and starting one from rest, and it changes the arithmetic. Key
                    # it off the queue, NOT the prefix: with the server's RTC off we
                    # send no prefix yet a chunk is very much in flight.
                    "in_flight": in_flight,
                }
        finally:
            with self._infer_lock:
                self._inferring = False

    def _maybe_splice(self, now: float) -> None:
        with self._infer_lock:
            p = self._pending
            if p is None:
                return
            # Hold an early chunk until the instant we PROMISED the server. Splicing
            # early would desynchronise us from the guidance mask it used.
            splice_at = p["t_request"] + p["d"] * self.dt
            if now < splice_at:
                return
            self._pending = None

        if not p["in_flight"]:
            # Nothing was executing while we inferred — episode start, blocking
            # mode, or recovery from an exhausted chunk. The arm has been HOLDING,
            # so the anchor we took is still where it is, and no slot of this chunk
            # has "already happened". Run it from the top, timed from now.
            #
            # (Skipping `elapsed` slots here — which the in-flight arithmetic would
            # do — would silently discard the first ~8 actions of every chunk and
            # make the motion jerk at every boundary.)
            start = 0
            self._anchor_time = now
        else:
            elapsed = int(round((now - p["t_request"]) * self.fps))
            start = p["d"]
            if elapsed > p["d"]:
                # We executed PAST the frozen prefix: the new chunk's committed slots
                # no longer describe what the robot did. Skip to where we actually
                # are — never rewind — and record it.
                self.budget.note_violation()
                self.stats["late"] += 1
                start = min(elapsed, self.H - 1)
                logger.warning("chunk late: promised d=%d, elapsed=%d ticks", p["d"], elapsed)
            self._anchor_time = p["t_request"]

        self.queue.replace(p["actions"], p["anchor"], start)

        # Re-anchor BOTH trajectories at `now`, seeded with what the emitter is
        # sending this instant. replace_after(now, []) would instead keep the last
        # knot <= now, which after a hold can be far in the past — the next pushed
        # knot then spans that stale point to now+dt and the emitter jumps almost
        # all of it in one period (F13). Reseeding makes the next segment exactly
        # one tick wide, so the clamp's per-knot step limit is a genuine rate limit.
        q_now = self.traj_arm.eval(now)
        h_now = self.traj_hand.eval(now)
        if q_now is not None:
            self.traj_arm.reseed(now, q_now)
            self.clamp.reset(q_now)
        if h_now is not None:
            self.traj_hand.reseed(now, h_now)

        # Close the loop: re-ground the IK at the MEASURED joints (within a chunk we
        # warm-start from the previous solution instead, for continuity).
        self.kin.ground(self.dds.arm_q())

        self._armed = True   # a plan now exists; starvation is meaningful from here
        self.stats["chunks"] += 1
        logger.debug("chunk %d spliced: sampler=%s d=%d latency=%.0f ms start=%d",
                     self.stats["chunks"], p["sampler"], p["d"], p["latency"] * 1000, start)
