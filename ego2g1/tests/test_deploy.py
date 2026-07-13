"""Deploy-side logic that must be right before it touches hardware.

Nothing here imports DDS (unitree_sdk2py lives on the robot PC, not here), so
these run anywhere. The kinematics tests need mujoco + the outer repo's
data_extraction; they skip if either is absent.
"""

import pathlib
import sys

import numpy as np
import pytest

from ego2g1.common import layout, se3
from ego2g1.deploy import chunk as _chunk
from ego2g1.deploy import safety as _safety
from ego2g1.deploy.trajectory import TrajectoryBuffer

# tests/ -> ego2g1/ -> openpi/ -> third_party/ -> the outer repo
REPO = pathlib.Path(__file__).resolve().parents[4]
DATASET = REPO / "lerobot_datasets/ego2g1/put_bottle_in_box"


# --- trajectory buffer -------------------------------------------------------


def test_traj_interpolates_linearly():
    t = TrajectoryBuffer(2)
    t.seed(10.0, [0.0, 0.0])
    t.push(11.0, [1.0, -2.0])
    assert t.eval(10.5) == pytest.approx([0.5, -1.0])
    assert t.eval(10.25) == pytest.approx([0.25, -0.5])


def test_traj_holds_past_the_end_never_extrapolates():
    """Running off the end of a plan must FREEZE, not keep going. Extrapolating a
    joint trajectory is how you get a runaway."""
    t = TrajectoryBuffer(2)
    t.seed(0.0, [0.0, 0.0])
    t.push(1.0, [1.0, 1.0])
    assert t.eval(5.0) == pytest.approx([1.0, 1.0])
    assert t.eval(-5.0) == pytest.approx([0.0, 0.0])
    assert t.runway(5.0) == 0.0


def test_traj_drops_stale_knots_rather_than_rewinding():
    t = TrajectoryBuffer(1)
    t.seed(0.0, [0.0])
    t.push(1.0, [1.0])
    t.push(0.5, [99.0])          # late knot, in the past -> must be ignored
    assert t.eval(1.0) == pytest.approx([1.0])
    assert len(t) == 2


def test_reseed_after_a_hold_bounds_the_emit_velocity():
    """F13, deterministic (no threads). After the emitter holds past the end of a
    plan, a splice must not deliver the whole clamp step in one emit period.
    `reseed(now, eval(now))` makes the next segment one control tick wide, so the
    clamp's per-knot step becomes a genuine rate limit; `replace_after` keeps the
    stale pre-hold knot and the emitter jumps almost all the way in one period.
    """
    fps, emit_hz, clamp_step = 30, 500, 0.15

    def splice_jump(use_reseed):
        tb = TrajectoryBuffer(1)
        tb.seed(0.0, [0.0])
        tb.push(0.05, [0.5])          # a short plan ending at t=0.05
        now = 0.4                     # emitter held q=0.5 for 350 ms
        q_held = float(tb.eval(now)[0])          # what the last emit sent
        if use_reseed:
            tb.reseed(now, [q_held])
        else:
            tb.replace_after(now, [])            # keeps the STALE knot at t=0.05
        tb.push(now + 1 / fps, [q_held + clamp_step])  # next knot, one tick ahead
        q_first = float(tb.eval(now)[0])         # first emit after the splice
        # the discontinuity is between the last held emit and the first post-splice
        # one, in a single emit period
        return abs(q_first - q_held) * emit_hz   # rad/s

    rate_limit = clamp_step * fps                # 4.5 rad/s
    # reseed: the first post-splice emit is still at q_held (alpha 0), no jump
    assert splice_jump(use_reseed=True) < 0.5 * rate_limit
    # replace_after: the stale segment puts the first emit ~0.9 of the way to the
    # new knot in ONE period — a ~60x-rate-limit spike (the audit measured 70 rad/s)
    assert splice_jump(use_reseed=False) > 10 * rate_limit


def test_traj_replace_after_keeps_the_current_segment():
    """Splicing a new plan must not break the interpolation the emitter is inside."""
    t = TrajectoryBuffer(1)
    t.seed(0.0, [0.0])
    t.push(1.0, [1.0])
    t.push(2.0, [2.0])
    t.replace_after(1.0, [(1.5, [10.0])])
    assert t.eval(0.5) == pytest.approx([0.5])    # old segment intact
    assert t.eval(1.5) == pytest.approx([10.0])   # new plan in force
    assert t.eval(9.0) == pytest.approx([10.0])   # and it holds


def test_traj_rejects_nonfinite():
    t = TrajectoryBuffer(2)
    with pytest.raises(ValueError):
        t.seed(0.0, [np.nan, 0.0])


# --- chunk queue / RTC bookkeeping -------------------------------------------


def _fake_chunk(h=50, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, layout.DIM), np.float32)
    for hand in layout.HANDS:
        T = np.tile(np.eye(4), (h, 1, 1))
        T[:, :3, 3] = rng.normal(0, 0.02, size=(h, 3))
        a[:, layout.EEF[hand]] = se3.se3_to_vec9(T)
        a[:, layout.HAND[hand]] = rng.uniform(0, 1, size=(h, layout.HAND_DIM))
    return a


def _identity_anchor():
    return {h: np.eye(4) for h in layout.HANDS}


def test_queue_pops_in_order_and_exhausts():
    q = _chunk.ChunkQueue(50, 30)
    a = _fake_chunk()
    q.replace(a, _identity_anchor(), d=0)
    for i in range(50):
        assert q.pop() == pytest.approx(a[i])
    assert q.pop() is None
    assert q.remaining() == 0


def test_queue_replace_skips_the_committed_prefix():
    """The new chunk's first d slots are the ticks that elapsed during inference:
    the robot already executed them. Re-executing would rewind it."""
    q = _chunk.ChunkQueue(50, 30)
    a = _fake_chunk()
    q.replace(a, _identity_anchor(), d=7)
    assert q.index == 7
    assert q.pop() == pytest.approx(a[7])


def test_rtc_prefix_is_the_leftover_left_aligned():
    """new slot i <-> old slot j+i, so the prefix is actions[j:] with row 0 first."""
    q = _chunk.ChunkQueue(50, 30)
    a = _fake_chunk()
    anchor = _identity_anchor()
    q.replace(a, anchor, d=0)
    j = 12
    for _ in range(j):
        q.pop()

    prefix, n_real = q.rtc_prefix(anchor, j)  # same anchor -> re-anchor is identity
    assert n_real == 50 - j
    assert prefix.shape == (50, layout.DIM)
    # row i of the prefix is old slot j+i
    assert prefix[: 50 - j] == pytest.approx(a[j:], abs=1e-5)
    # and the tail is zero padding, which prefix_weights never reads
    assert prefix[50 - j:] == pytest.approx(0.0)


def test_rtc_prefix_preserves_absolute_targets_under_a_moved_anchor():
    """THE invariant. The prefix must point at the same physical poses the old
    chunk did, expressed from the new anchor. If this drifts, RTC guides the new
    chunk toward poses nobody intended and the robot lurches at the seam."""
    q = _chunk.ChunkQueue(50, 30)
    a = _fake_chunk()
    rng = np.random.default_rng(3)

    old = {}
    new = {}
    for h in layout.HANDS:
        T = np.eye(4); T[:3, 3] = rng.normal(size=3); old[h] = T
        U = np.eye(4); U[:3, 3] = rng.normal(size=3); new[h] = U

    q.replace(a, old, d=0)
    j = 9
    for _ in range(j):
        q.pop()
    prefix, _ = q.rtc_prefix(new, j)

    for h in layout.HANDS:
        want = se3.compose(old[h], a[j:, layout.EEF[h]])             # what the old chunk meant
        got = se3.compose(new[h], prefix[: 50 - j, layout.EEF[h]])   # what the prefix means now
        assert got == pytest.approx(want, abs=1e-5)


def test_rtc_prefix_none_before_the_first_chunk():
    q = _chunk.ChunkQueue(50, 30)
    assert q.rtc_prefix(_identity_anchor(), 0) == (None, 0)


def test_rtc_prefix_start_is_independent_of_consumption_index():
    """The prefix is aligned by WALL CLOCK, not by how many actions we happen to
    have popped. The planner pops ahead of realtime to keep the trajectory topped
    up, so slicing at queue.index would send a target ~lookahead into the future
    and RTC would drag every chunk forward in time."""
    q = _chunk.ChunkQueue(50, 30)
    a = _fake_chunk()
    anchor = _identity_anchor()
    q.replace(a, anchor, d=0)
    for _ in range(20):          # planner has run ahead to index 20
        q.pop()
    # ...but the wall clock says slot 17 is what executes next
    prefix, n_real = q.rtc_prefix(anchor, 17)
    assert n_real == 50 - 17
    assert prefix[0] == pytest.approx(a[17], abs=1e-5)
    assert prefix[3] == pytest.approx(a[20], abs=1e-5)


def test_targets_and_hands_split():
    a = _fake_chunk()[0]
    anchor = _identity_anchor()
    tgt = _chunk.targets_from(a, anchor)
    assert set(tgt) == set(layout.HANDS)
    assert tgt["left"].shape == (4, 4)
    hands = _chunk.hands_from(a)
    assert hands["right"].shape == (layout.HAND_DIM,)
    assert (hands["right"] >= 0).all() and (hands["right"] <= 1).all()


# --- safety ------------------------------------------------------------------


def test_clamp_limits_the_joint_step():
    c = _safety.Clamp(_safety.SafetyLimits(max_joint_step=0.1, max_joint_vel=1e9))
    c.reset(np.zeros(14))
    out = c(np.full(14, 1.0), dt=1 / 30)
    assert np.abs(out).max() == pytest.approx(0.1)
    assert c.clamped_ticks == 1


def test_clamp_passes_small_steps_through():
    c = _safety.Clamp(_safety.SafetyLimits(max_joint_step=0.1, max_joint_vel=1e9))
    c.reset(np.zeros(14))
    q = np.full(14, 0.05)
    assert c(q, dt=1 / 30) == pytest.approx(q)
    assert c.clamped_ticks == 0


def test_watchdog_trips_only_after_repeated_strikes():
    tripped = []
    w = _safety.Watchdog(_safety.SafetyLimits(trip_after=3), on_trip=lambda: tripped.append(1))
    for _ in range(2):
        w.check_state_age(10.0)
    assert not w.tripped          # a single spike is noise
    w.check_state_age(10.0)
    assert w.tripped and tripped  # sustained is a fault


def test_watchdog_strikes_reset_on_recovery():
    w = _safety.Watchdog(_safety.SafetyLimits(trip_after=3), on_trip=lambda: None)
    w.check_state_age(10.0)
    w.check_state_age(10.0)
    w.check_state_age(0.001)      # recovered
    w.check_state_age(10.0)
    w.check_state_age(10.0)
    assert not w.tripped


def test_sanity_check_rejects_garbage_actions():
    good = _fake_chunk()[0]
    assert _safety.sanity_check_action(good)

    nan = good.copy(); nan[3] = np.nan
    assert not _safety.sanity_check_action(nan)

    huge = good.copy(); huge[layout.EEF["left"]][:3] = [9.0, 0, 0]  # 9 m in one tick
    huge[0] = 9.0
    assert not _safety.sanity_check_action(huge)

    assert not _safety.sanity_check_action(np.zeros(7))


# --- kinematics (needs mujoco + data_extraction) ------------------------------

mujoco = pytest.importorskip("mujoco")
pytestmark_ds = pytest.mark.skipif(not DATASET.exists(), reason="dataset not present")


@pytest.fixture(scope="module")
def kin():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    pytest.importorskip("mink")
    from ego2g1.deploy.kinematics import Kinematics
    return Kinematics(REPO)


def test_base_is_the_fixed_pelvis(kin):
    """Pelvis == the base link, a pure z translation. If this ever picks up a
    rotation, the fixed-base assumption has broken."""
    assert kin.base[:3, :3] == pytest.approx(np.eye(3), abs=1e-9)
    assert kin.base[:3, 3] == pytest.approx([0, 0, 0.793], abs=1e-6)


@pytestmark_ds
def test_fk_reproduces_the_dataset_state(kin):
    """The single most load-bearing check in the client: joint order, waist==0,
    flange site, pelvis frame, and vec9 encoding, all at once."""
    import pandas as pd
    f = sorted(DATASET.glob("data/*/*.parquet"))[0]
    df = pd.read_parquet(f)
    arm = np.stack(df["arm_qpos"].to_numpy())
    state = np.stack(df["state"].to_numpy())

    for t in range(0, len(arm), 40):
        poses = kin.flange_poses(arm[t])
        for h in layout.HANDS:
            got = se3.se3_to_vec9(poses[h])
            assert got == pytest.approx(state[t, layout.EEF[h]], abs=1e-5)


def test_state_hand_block_is_the_command_not_encoders(kin):
    """Training's state hand-block is the retargeted COMMAND — no encoders exist in
    the data. kinematics.state() must take what we sent, and clip it to [0,1]."""
    arm = np.zeros(layout.ARM_DOF)
    cmds = {"left": np.full(6, 0.25), "right": np.full(6, 2.0)}  # right is out of range
    s = kin.state(arm, cmds)
    assert s.shape == (layout.DIM,)
    assert s[layout.HAND["left"]] == pytest.approx(0.25)
    assert s[layout.HAND["right"]] == pytest.approx(1.0)  # clipped, not passed through


@pytestmark_ds
def test_dataset_actions_reconstruct_the_recorded_joints(kin):
    """Rung 7 (`check.py replay-actions`) with the robot replaced by a kinematic
    stand-in: the same DatasetClient, the same measured-FK anchoring, the same
    delta composition and IK the real loop runs — just no DDS.

    This is the gate on the ACTION LABELS. `test_fk_reproduces_the_dataset_state`
    proves the state encoding; nothing proves the actions unless the composed
    chunks reconstruct the motion they were derived from. A sign flip or a
    transposed anchor in `inv(pose[t0]) @ pose[t0+k]` would sail past every other
    test in this file and only show up as a lunging robot.
    """
    from ego2g1.deploy import chunk as _chunk
    from ego2g1.deploy import dataset_client as _dsc

    client = _dsc.DatasetClient(DATASET, episode=0)
    ep = client.ep
    H = client.action_horizon

    # The robot starts where the ramp leaves it: the episode's first posture.
    q_meas = ep.arm_qpos[0].astype(np.float64)
    kin.ground(q_meas)

    consumed = {"j": 0}
    client.attach_consumed(lambda: consumed["j"])
    zero_hands = {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS}

    q_out, t = [q_meas.copy()], 0
    while t < ep.n_frames - 1:
        # chunk boundary: anchor on MEASURED FK, then fetch (loop._maybe_infer)
        anchor = kin.flange_poses(q_meas)
        actions = client.infer(None, kin.state(q_meas, zero_hands), ep.task)["actions"]
        kin.ground(q_meas)

        k = 0
        while k < H and t < ep.n_frames - 1:
            q_meas = kin.solve(_chunk.targets_from(actions[k], anchor))  # loop._top_up
            q_out.append(q_meas.copy())
            k, t = k + 1, t + 1
        consumed["j"] = k

    q_out = np.stack(q_out)[: ep.n_frames]
    err = np.degrees(np.abs(q_out - ep.arm_qpos[: len(q_out)]))
    # Measured 0.19 deg mean / 1.66 deg max. The ceiling is loose enough to absorb
    # QP jitter and tight enough that any real frame error blows through it.
    assert err.mean() < 0.5, f"mean {err.mean():.3f} deg"
    assert err.max() < 5.0, f"max {err.max():.3f} deg"


def test_vendored_g1_sim_matches_source():
    """The vendored G1 sim (deploy/_g1_sim/) must be byte-identical to the training
    repo's data_extraction. This is the guarantee that makes `ego2g1/` shippable as
    one folder without the deployment IK silently drifting from the IK that made
    the labels. Re-vendor with `python -m ego2g1.deploy.vendor_g1_sim`.

    Integrity (vendored file == its recorded hash) runs everywhere. Drift (source
    == recorded hash) runs only where data_extraction is present — i.e. NOT on the
    robot PC, which by design has only the copy.
    """
    import hashlib
    import json

    vendor = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "_g1_sim"
    manifest = json.loads((vendor / "MANIFEST.json").read_text())["files"]
    assert manifest, "empty manifest — vendoring never ran"

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for rel, want in manifest.items():
        assert sha(vendor / rel) == want, f"vendored {rel} corrupt vs MANIFEST"

    de = REPO / "data_extraction"
    if not de.exists():
        pytest.skip("data_extraction absent (robot PC) — integrity checked, drift cannot be")
    drifted = [rel for rel, want in manifest.items() if sha(de / rel) != want]
    assert not drifted, ("data_extraction changed since vendoring; re-run "
                         f"`python -m ego2g1.deploy.vendor_g1_sim`. Stale: {drifted}")


@pytestmark_ds
def test_vendored_kinematics_matches_external(kin):
    """The vendored copy must produce the SAME kinematics as the external repo, not
    merely the same bytes — the belt to the manifest's braces. FK a spread of
    configurations through both and require agreement to float precision."""
    from ego2g1.deploy.kinematics import Kinematics

    vendored = Kinematics()          # default -> deploy/_g1_sim
    rng = np.random.default_rng(0)
    for _ in range(8):
        q = rng.uniform(-1.0, 1.0, size=layout.ARM_DOF)
        a, b = kin.flange_poses(q), vendored.flange_poses(q)   # kin = Kinematics(REPO)
        for h in layout.HANDS:
            assert a[h] == pytest.approx(b[h], abs=1e-9)


def test_deploy_does_not_import_jax():
    """The robot PC has no JAX. If this fails, ego2g1.deploy just became unshippable."""
    import subprocess
    code = (
        "import sys; import ego2g1.deploy.chunk, ego2g1.deploy.trajectory, "
        "ego2g1.deploy.safety, ego2g1.common.se3; "
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m); "
        "print('clean')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(pathlib.Path(__file__).resolve().parents[2]))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "clean" in r.stdout


# --- RTC mask (F2/F8) and delay budget (F3) ---------------------------------

from ego2g1.serve import rtc as _rtc  # noqa: E402
from ego2g1.deploy.client import DelayBudget  # noqa: E402


@pytest.mark.parametrize("d", [0, 1, 3, 8, 12, 18, 30])
def test_prefix_weight_is_nonzero_at_the_splice_slot(d):
    """F2. The client splices at slot d, so the mask MUST carry weight there. With
    the old absolute horizon and shipped d>=horizon it was exactly 0 — RTC guided
    only the slots that get discarded."""
    w = _rtc.prefix_weights(d, overlap=10, horizon=50, n_real=50)
    assert w.shape == (50,)
    if d < 50:
        assert w[d] > 0.0, f"splice slot d={d} is unconstrained"
    assert (w[:d] == 1.0).all()  # committed prefix fully pinned


def test_prefix_weight_never_touches_padding():
    """F8. When the real prefix is shorter than the band, the mask must stop at
    n_real — a zero vec9 is the zero matrix, not a pose."""
    w = _rtc.prefix_weights(d=3, overlap=10, horizon=50, n_real=5)
    assert (w[5:] == 0.0).all(), "weight leaked onto zero padding"
    assert w[3] > 0.0             # but the splice slot inside n_real still carries


def test_prefix_weight_caps_d_at_n_real():
    """F14-adjacent. d beyond the real prefix cannot pin padding as committed."""
    w = _rtc.prefix_weights(d=20, overlap=10, horizon=50, n_real=6)
    assert (w[6:] == 0.0).all()
    assert (w[:6] == 1.0).all()   # everything real is pinned, nothing past it


def test_delay_budget_saturates_instead_of_exceeding_the_chunk():
    """F3. Ten 2 s inferences must not push d past the cap (old code -> d=69)."""
    b = DelayBudget(30, initial=8, max_d=20)
    for _ in range(10):
        b.observe(2.0)
    assert b.d <= 20
    assert b.saturated > 0        # and it recorded that it clipped


def test_serve_record_reads_metadata_before_wrapping():
    """F6. PolicyRecorder is a bare BasePolicy with no .metadata; reading it off the
    wrapper is an AttributeError. Assert the source captures meta before the wrap."""
    src = pathlib.Path(__file__).resolve().parents[1] / "serve" / "__main__.py"
    text = src.read_text()
    assert "metadata=meta" in text
    assert text.index("meta = policy.metadata") < text.index("PolicyRecorder(policy")
