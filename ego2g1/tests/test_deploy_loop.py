"""DeployLoop against fake hardware: the threading and the RTC splice.

This is the only place the concurrency gets exercised before it reaches a robot.
The fakes are deliberately faithful about the two things that matter: the policy
takes real wall-clock time to answer (so inference genuinely overlaps execution),
and the "robot" integrates the commands it receives (so a discontinuity in the
emitted stream is visible as one).
"""

import pathlib
import sys
import threading
import time

import numpy as np
import pytest

from ego2g1.common import layout, se3
from ego2g1.deploy import loop as _loop
from ego2g1.deploy import safety as _safety
from ego2g1.deploy.client import DelayBudget

REPO = pathlib.Path(__file__).resolve().parents[4]
pytest.importorskip("mujoco")
pytest.importorskip("mink")


class FakeDDS:
    """Records every commanded joint vector, and reports them back as 'measured'
    (a robot that tracks perfectly)."""

    def __init__(self, arm0):
        self._arm = np.asarray(arm0, dtype=np.float64).copy()
        self.sent: list[tuple[float, np.ndarray]] = []
        self.hands_sent: list[np.ndarray] = []
        self.damped = False
        self._lock = threading.Lock()

    def arm_q(self):
        with self._lock:
            return self._arm.copy()

    def lowstate_age(self):
        return 0.001

    def send_arm(self, q, *, waist=None):
        if self.damped:
            return
        with self._lock:
            self._arm = np.asarray(q, dtype=np.float64).copy()
            self.sent.append((time.monotonic(), self._arm.copy()))

    def send_hands(self, cmds):
        if self.damped:
            return
        self.hands_sent.append(np.concatenate([cmds[h] for h in layout.HANDS]))

    def damp(self):
        self.damped = True


class FakeCamera:
    def read(self):
        return np.zeros((224, 224, 3), np.uint8)

    def age(self):
        return 0.0

    def close(self):
        pass


class FakePolicy:
    """A policy that takes real time to answer and returns a chunk continuing
    smoothly from the anchor.

    Unlike a trivial fake, this one RESPECTS the RTC prefix: the first `d` slots of
    the returned chunk are copied from `prev_chunk` (what a correct guided sampler
    converges to on the pinned rows). That makes seam continuity depend on the
    prefix being right — the audit's blind spot was a fake that ignored it, so
    every seam looked smooth regardless of the RTC wiring. It also lets us assert
    that `d` and `n_prefix` arrive with a sane relationship.
    """

    action_horizon = 50
    action_dim = 30
    fps = 30
    rtc_training = False
    rtc = {"enabled": True}
    hands = ("left", "right")

    def __init__(self, latency_s=0.25, drift=0.002, respect_prefix=True):
        self.latency_s = latency_s
        self.drift = drift
        self.respect_prefix = respect_prefix
        self.calls = []
        self._lock = threading.Lock()

    def infer(self, image, state, prompt, *, prev_chunk=None, d=0, n_prefix=None):
        time.sleep(self.latency_s)
        with self._lock:
            self.calls.append({"d": d, "n_prefix": n_prefix,
                               "has_prefix": prev_chunk is not None,
                               "state": np.asarray(state).copy()})
            if prev_chunk is not None:
                # invariants a correct client must uphold
                assert n_prefix is not None and n_prefix >= 1
                assert 0 <= d < n_prefix, (d, n_prefix)

        a = np.zeros((self.action_horizon, layout.DIM), np.float32)
        for h in layout.HANDS:
            T = np.tile(np.eye(4), (self.action_horizon, 1, 1))
            T[:, 0, 3] = self.drift * np.arange(1, self.action_horizon + 1)
            a[:, layout.EEF[h]] = se3.se3_to_vec9(T)
            a[:, layout.HAND[h]] = 0.3

        if self.respect_prefix and prev_chunk is not None and d > 0:
            # a correct guided sampler holds slots [0,d) at the prefix
            a[:d] = np.asarray(prev_chunk)[:d]

        return {"actions": a, "client_latency_s": self.latency_s,
                "rtc": {"sampler": "guided" if prev_chunk is not None else "plain",
                        "d": d, "n_prefix": n_prefix}}


@pytest.fixture
def kin():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from ego2g1.deploy.kinematics import Kinematics
    return Kinematics(REPO)


def _run(kin, *, latency=0.25, seconds=3.0, blocking=False, initial_d=8,
         first_latency=None, max_starvation=1.0, overlap=10):
    import pandas as pd
    f = sorted((REPO / "lerobot_datasets/ego2g1/put_bottle_in_box").glob("data/*/*.parquet"))[0]
    arm0 = np.stack(pd.read_parquet(f)["arm_qpos"].to_numpy())[0]

    dds = FakeDDS(arm0)
    policy = _VarLatencyPolicy(latency, first_latency) if first_latency is not None \
        else FakePolicy(latency_s=latency)
    budget = DelayBudget(policy.fps, initial=initial_d)
    lp = _loop.DeployLoop(
        _loop.LoopConfig(task="test", fps=policy.fps, blocking=blocking,
                         arm_hz=200.0, hand_hz=100.0),
        dds=dds, camera=FakeCamera(), kinematics=kin, client=policy, budget=budget,
        limits=_safety.SafetyLimits(max_joint_step=0.15, max_tracking_error_m=0.25,
                                    max_starvation=max_starvation),
    )
    lp.start()
    time.sleep(seconds)
    lp.stop()
    return lp, dds, policy, budget


class _VarLatencyPolicy(FakePolicy):
    """First call takes `first_latency`, the rest `steady`. Models a cold server
    whose first inference JIT-compiles for far longer than steady state."""

    def __init__(self, steady, first_latency):
        super().__init__(latency_s=steady)
        self._steady = steady
        self._first = first_latency
        self._n = 0

    def infer(self, *a, **kw):
        self.latency_s = self._first if self._n == 0 else self._steady
        self._n += 1
        return super().infer(*a, **kw)


def test_emitter_never_starves_while_inference_runs(kin):
    """THE reason T5 is a separate thread. A 250 ms inference is 7+ control ticks;
    if it blocked the planner, the trajectory would drain and the arm would freeze
    mid-motion. The commanded stream must stay dense throughout."""
    lp, dds, policy, _ = _run(kin, latency=0.25, seconds=3.0)

    assert not lp.watchdog.tripped, lp.watchdog.reason
    assert len(policy.calls) >= 2, "no re-planning happened"

    t = np.array([s[0] for s in dds.sent])
    gaps = np.diff(t)
    # emitter runs at 200 Hz here; no gap should approach the inference latency
    assert gaps.max() < 0.05, f"emitter stalled for {gaps.max()*1000:.0f} ms"


@pytest.mark.parametrize("latency,initial_d", [
    (0.10, 4),    # fast
    (0.25, 8),    # the old fixed point
    (0.35, 12),   # the SHIPPED default d — where F2 lived
    (0.45, 16),   # slow; d approaches the budget cap
])
def test_commanded_joints_stay_continuous_across_chunk_seams(kin, latency, initial_d):
    """A chunk splice must not step the joints, ACROSS the latency/d space — not
    just at the one fixed point (0.25 s, d=8) the original fixture used, which was
    the single region where none of F1/F2/F13 fired. FakePolicy now respects the
    RTC prefix, so a mis-aligned or unweighted prefix shows up here as a seam."""
    lp, dds, policy, _ = _run(kin, latency=latency, initial_d=initial_d, seconds=3.5)
    assert not lp.watchdog.tripped, lp.watchdog.reason
    assert len(policy.calls) >= 2, "no re-planning happened"

    q = np.stack([s[1] for s in dds.sent])
    step = np.abs(np.diff(q, axis=0)).max(axis=1)
    # 0.15 rad / 30 Hz tick, emitted at 200 Hz -> <= ~0.023 rad per emit
    assert step.max() < 0.03, f"joint discontinuity {step.max():.4f} rad (latency={latency})"


def test_cold_start_first_inference_does_not_estop(kin):
    """F1 regression. A slow first inference (a cold server JIT-compiles) must NOT
    trip the starvation watchdog — there is legitimately no plan yet. The old code
    armed starvation from t=0 and damped the robot before it ever moved."""
    lp, dds, policy, _ = _run(kin, latency=0.15, first_latency=1.3, seconds=3.0,
                              max_starvation=1.0)
    assert not lp.watchdog.tripped, lp.watchdog.reason
    assert not dds.damped
    assert len(policy.calls) >= 2, "loop never recovered after the slow first call"


def test_delay_budget_is_capped(kin):
    """F3 regression. A sustained slow inference must not drive d past the chunk —
    d>=H installs an empty chunk and the loop stops planning. It must saturate."""
    lp, dds, policy, budget = _run(kin, latency=0.6, initial_d=8, seconds=4.0)
    assert budget.d <= budget.max_d
    assert budget.d < policy.action_horizon
    # and the loop kept planning despite the slow calls
    assert len(policy.calls) >= 2


def test_splice_after_a_hold_does_not_catastrophically_lurch(kin):
    """F13, live smoke test. The tight, deterministic regression lives in
    test_deploy.py::test_reseed_after_a_hold_bounds_the_emit_velocity — a
    wall-clock velocity bound here is inherently jittery under full-suite load
    (the emitter thread is itself starved), so this only asserts the absence of a
    CATASTROPHIC spike: pre-fix the audit measured ~70 rad/s in one 2 ms emit."""
    lp, dds, policy, _ = _run(kin, latency=1.1, first_latency=0.2, initial_d=6,
                              seconds=3.5, max_starvation=5.0)
    t = np.array([s[0] for s in dds.sent])
    q = np.stack([s[1] for s in dds.sent])
    dq = np.abs(np.diff(q, axis=0)).max(axis=1)
    dt = np.diff(t)
    vel = dq[dt > 0] / dt[dt > 0]
    assert vel.max() < 25.0, f"catastrophic lurch of {vel.max():.1f} rad/s after a hold"


def test_rtc_prefix_is_sent_on_every_replan_but_not_the_first(kin):
    """First inference of an episode has no leftover -> plain sampling. Every
    subsequent one must carry the re-anchored prefix, or there is no RTC."""
    lp, dds, policy, _ = _run(kin, latency=0.2, seconds=3.0)
    assert len(policy.calls) >= 2
    assert policy.calls[0]["has_prefix"] is False
    assert policy.calls[0]["d"] == 0
    for c in policy.calls[1:]:
        assert c["has_prefix"] is True
        assert c["d"] > 0


def test_state_sent_to_the_policy_is_wellformed(kin):
    lp, dds, policy, _ = _run(kin, latency=0.2, seconds=2.0)
    s = policy.calls[0]["state"]
    assert s.shape == (layout.DIM,)
    assert np.all(np.isfinite(s))
    for h in layout.HANDS:
        hb = s[layout.HAND[h]]
        assert np.all((hb >= 0) & (hb <= 1)), "hand block must be a [0,1] command"


def test_blocking_mode_runs_without_rtc(kin):
    """--blocking is the bring-up path: no overlap, no prefix, no guidance."""
    lp, dds, policy, _ = _run(kin, latency=0.15, seconds=3.0, blocking=True)
    assert not lp.watchdog.tripped, lp.watchdog.reason
    assert len(policy.calls) >= 1
    assert all(c["has_prefix"] is False for c in policy.calls)
    assert all(c["d"] == 0 for c in policy.calls)


def test_delay_budget_adapts_to_measured_latency(kin):
    lp, dds, policy, budget = _run(kin, latency=0.30, seconds=4.0, initial_d=4)
    st = budget.stats()
    if st["n"] >= 5:
        # 0.30 s at 30 Hz is ~9 ticks; with headroom the budget should exceed the
        # naive initial guess rather than stay stuck at it.
        assert budget.d >= 9, st


def test_watchdog_damps_on_stale_state(kin):
    """If we cannot see the robot we must not command it — and the stop must be a
    damping command, not merely silence (the firmware holds the last setpoint)."""
    import pandas as pd
    f = sorted((REPO / "lerobot_datasets/ego2g1/put_bottle_in_box").glob("data/*/*.parquet"))[0]
    arm0 = np.stack(pd.read_parquet(f)["arm_qpos"].to_numpy())[0]

    dds = FakeDDS(arm0)
    dds.lowstate_age = lambda: 10.0          # the robot went quiet
    policy = FakePolicy(latency_s=0.05)
    lp = _loop.DeployLoop(
        _loop.LoopConfig(task="test", fps=30, arm_hz=200.0, hand_hz=100.0),
        dds=dds, camera=FakeCamera(), kinematics=kin, client=policy,
        budget=DelayBudget(30, initial=6),
    )
    lp.start()
    time.sleep(0.5)
    lp.stop()

    assert lp.watchdog.tripped
    assert dds.damped, "watchdog must DAMP, not just stop publishing"


def test_rtc_prefix_is_aligned_to_the_wall_clock_not_the_planner(kin):
    """Regression: the planner pops ahead of realtime to keep `lookahead_s` of
    trajectory queued, so queue.index sits ~3 slots in the future. Slicing the RTC
    prefix there hands the server a guidance target ~100 ms ahead of what the robot
    is actually about to do, and RTC then drags every chunk forward in time — a
    nudge at every seam, which is the exact discontinuity it exists to remove.

    The prefix start must be derived from monotonic time, not from consumption.
    """
    import pathlib as _p

    import pandas as pd
    f = sorted((REPO / "lerobot_datasets/ego2g1/put_bottle_in_box").glob("data/*/*.parquet"))[0]
    arm0 = np.stack(pd.read_parquet(f)["arm_qpos"].to_numpy())[0]

    dds, policy = FakeDDS(arm0), FakePolicy(latency_s=0.25)
    lp = _loop.DeployLoop(
        _loop.LoopConfig(task="t", fps=30, arm_hz=200.0, hand_hz=100.0),
        dds=dds, camera=FakeCamera(), kinematics=kin, client=policy,
        budget=DelayBudget(30, initial=8),
        limits=_safety.SafetyLimits(max_tracking_error_m=0.25),
    )

    skews = []
    inner = lp.queue.rtc_prefix

    def spy(anchor_new, start):
        now = time.monotonic()
        wall = (now - lp._anchor_time) * lp.fps
        skews.append(start - wall)
        return inner(anchor_new, start)

    lp.queue.rtc_prefix = spy
    lp.start()
    time.sleep(4.0)
    lp.stop()

    assert skews, "no re-planning happened"
    worst = max(abs(s) for s in skews)
    # sub-slot agreement; the old (index-based) code sat ~3.7 slots out
    assert worst < 1.0, f"prefix misaligned by {worst:.2f} slots ({worst/30*1000:.0f} ms)"
