"""The bring-up ladder. Walk it in order; each rung gates the next.

    python -m ego2g1.deploy.check listen     # 1. DDS only, no commands   [robot]
    python -m ego2g1.deploy.check fk         # 2. FK vs dataset state     [offline]
    python -m ego2g1.deploy.check ik         # 3. IK vs dataset joints    [offline]
    python -m ego2g1.deploy.check camera     # 4. one frame, saved to disk[robot]
    python -m ego2g1.deploy.check hand-sweep # 5. one finger at a time    [robot]
    python -m ego2g1.deploy.check replay     # 6. recorded JOINTS, no model   [robot]
    python -m ego2g1.deploy.check replay-actions # 7. recorded ACTIONS, no model [robot]
    python -m ego2g1.deploy.check latency    # 8. round trip to the server [no robot]

Rungs 2 and 3 need no hardware and no checkpoint, and between them they validate
joint order, the waist==0 assumption, the flange frame, the pelvis frame, the
vec9 encoding, and the IK — i.e. most of what can silently be wrong.

Rungs 6 and 7 both drive the real arm from a recording with the policy out of the
loop, and they are not the same test. 6 streams the episode's stored joints
straight to the motors: it never touches an action label, and it proves the
plumbing (joint order, sign, units, CRC, rates, hands, e-stop). 7 feeds the
episode's ACTION LABELS through the real control loop — measured-FK anchor, delta
composition, mink IK, safety clamp — and so proves the transforms. A frame or
anchor bug leaves 6 looking perfect and shows up only in 7, which is why 6 runs
first: it makes 7 interpretable.
"""

import dataclasses
import logging
import pathlib
import sys
import time

import numpy as np
import tyro

from ego2g1.common import layout, se3


def _ramp_seconds(q_now, q_start, ramp_s: float, max_speed: float) -> float:
    """See deploy/ramp.py — one implementation, shared with `deploy
    --start-from-episode` and `eval_real`'s snap-back, so the three cannot drift
    apart on the one thing they all get to be wrong about: how fast the arm moves
    when nothing is watching it."""
    from ego2g1.deploy.ramp import ramp_seconds

    return ramp_seconds(q_now, q_start, ramp_s, max_speed)


def _dataset_episode(root: str, episode: int = 0):
    import pandas as pd
    files = sorted(pathlib.Path(root).glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}/data/")
    df = pd.read_parquet(files[min(episode, len(files) - 1)])
    return {
        "name": files[min(episode, len(files) - 1)].name,
        "arm": np.stack(df["arm_qpos"].to_numpy()),
        "state": np.stack(df["state"].to_numpy()),
        "pose": {h: np.stack(df[f"pose.{h}"].to_numpy()) for h in layout.HANDS},
        "hand": {h: np.stack(df[f"hand.{h}"].to_numpy()) for h in layout.HANDS},
    }


# --- 1. listen ---------------------------------------------------------------

def listen(iface: str | None = None, domain: int = 0, seconds: float = 5.0,
           hands: bool = True) -> None:
    """Subscribe only. No publishers, nothing commanded. Proves the DDS domain,
    the topic names, and that the Brainco bridge is actually running."""
    from ego2g1.deploy import dds as _dds

    d = _dds.G1DDS(network_interface=iface, domain=domain, enable_hands=hands)
    # Subscribe without arming the publisher path.
    d.connect()
    print(f"lowstate OK (age {d.lowstate_age()*1000:.0f} ms)\n")

    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        q = d.arm_q()
        print(f"  arm q  L {np.round(q[:7], 3)}  R {np.round(q[7:], 3)}")
        if hands:
            for h in layout.HANDS:
                age = d.hand_state_age(h)
                if age == float("inf"):
                    print(f"  hand {h:5s} NO STATE — is the Brainco bridge running?")
                else:
                    print(f"  hand {h:5s} {np.round(d.hand_q(h), 3)}  (age {age*1000:.0f} ms)")
        time.sleep(0.5)
    print("\nlisten OK — no commands were sent.")


# --- 2. fk -------------------------------------------------------------------

def fk(dataset: str, repo: str | None = None, episode: int = 0, tol: float = 1e-5) -> None:
    """FK the dataset's stored joints and compare to its stored state.

    Validates joint order, waist==0, the flange site, the pelvis frame, and the
    vec9 encoding in one shot. No hardware, no checkpoint.
    """
    from ego2g1.deploy.kinematics import Kinematics

    ep = _dataset_episode(dataset, episode)
    K = Kinematics(repo)
    print(f"{ep['name']}: {len(ep['arm'])} frames\n")

    worst = 0.0
    for h in layout.HANDS:
        errs = []
        for t in range(0, len(ep["arm"]), 5):
            got = se3.se3_to_vec9(K.flange_poses(ep["arm"][t])[h])
            errs.append(np.abs(got - ep["state"][t, layout.EEF[h]]))
        e = np.stack(errs)
        worst = max(worst, e.max())
        print(f"  {h:5s}  trans max {e[:, :3].max():.3e} m   rot6d max {e[:, 3:].max():.3e}")

    print(f"\nworst {worst:.3e}")
    if worst < tol:
        print("PASS — FK reproduces the dataset state.")
    else:
        sys.exit("FAIL — joint order, frame, or flange is wrong. Do NOT go to hardware.")


# --- 3. ik -------------------------------------------------------------------

def ik(dataset: str, repo: str | None = None, episode: int = 0, n: int = 150) -> None:
    """Track the dataset's stored poses with our IK; compare to its stored joints
    and time the solve. This is also where you learn whether one solve fits in a
    30 Hz tick."""
    from ego2g1.deploy.kinematics import Kinematics

    ep = _dataset_episode(dataset, episode)
    K = Kinematics(repo)
    K.ground(ep["arm"][0])

    n = min(n, len(ep["arm"]))
    q_err, t_err, dur = [], [], []
    for t in range(n):
        targets = {h: se3.vec9_to_se3(ep["pose"][h][t]) for h in layout.HANDS}
        t0 = time.perf_counter()
        q = K.solve(targets)
        dur.append((time.perf_counter() - t0) * 1000)
        q_err.append(np.abs(q - ep["arm"][t]))
        e = K.tracking_error(targets)
        t_err.append(max(e.values()))

    q_err, t_err, dur = np.stack(q_err), np.array(t_err), np.array(dur)
    budget = 1000.0 / 30
    print(f"{ep['name']}: {n} ticks, warm-started\n")
    print(f"  joint err    mean {q_err.mean():.4f} rad   max {q_err.max():.4f} rad")
    print(f"  flange err   mean {t_err.mean()*1000:.2f} mm   max {t_err.max()*1000:.2f} mm")
    print(f"  solve time   mean {dur.mean():.2f} ms   p95 {np.percentile(dur,95):.2f} ms")
    print(f"\n  30 Hz budget {budget:.1f} ms -> IK uses {dur.mean()/budget*100:.1f}%")
    if t_err.max() > 0.02:
        sys.exit("FAIL — IK cannot track the training poses. Frames are wrong.")
    print("PASS")


# --- 4. camera ---------------------------------------------------------------

def camera(host: str = "192.168.123.164", eye: str = "left",
           out: str = "check_camera.png") -> None:
    """Grab one frame and write it out. Then LOOK AT IT next to a training frame.

    This is the highest-risk open item in the whole deployment: the model was
    trained on Pico-headset egocentric video, and a systematically different
    viewpoint fails quietly and looks like a bad policy.
    """
    import cv2

    from ego2g1.deploy.camera import HeadCamera

    cam = HeadCamera(host=host, eye=eye)
    cam.connect()
    img = cam.read()
    cam.close()
    print(f"frame: {img.shape} {img.dtype}  range [{img.min()}, {img.max()}]")
    cv2.imwrite(out, img[..., ::-1])   # back to BGR for imwrite
    print(f"wrote {out} — compare it against a training video frame before trusting a rollout.")


# --- 5. hand sweep -----------------------------------------------------------

def hand_sweep(iface: str | None = None, domain: int = 0, hand: str = "right",
               motor: int = 2, lo: float = 0.0, hi: float = 0.6,
               seconds: float = 4.0) -> None:
    """Drive ONE Brainco motor slowly between two commands, printing the encoder.

    Commands are [0, 1] (0=open, 1=closed) — that much is settled. What this rung
    resolves is the ORDER: whether our [thumb_flex, thumb_rot, index, middle, ring,
    pinky] maps 1:1 onto Brainco's [Thumb, ThumbAux, Index, Middle, Ring, Pinky]. If
    commanding `motor` moves a different finger, the mapping is wrong — fix it
    before any policy runs.
    """
    from ego2g1.deploy import dds as _dds

    name = layout.HAND_MOTOR_ORDER[motor]
    print(f"sweeping {hand} motor {motor} ({name}) between {lo} and {hi}")
    print("WATCH THE HAND. Which finger actually moves?\n")

    d = _dds.G1DDS(network_interface=iface, domain=domain, enable_hands=True)
    d.connect()

    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            phase = 0.5 - 0.5 * np.cos(2 * np.pi * (time.monotonic() - t0) / seconds * 2)
            cmd = {h: np.zeros(layout.HAND_DIM, np.float32) for h in layout.HANDS}
            cmd[hand][motor] = lo + (hi - lo) * phase
            d.send_hands(cmd)
            print(f"  cmd {cmd[hand][motor]:.3f}  enc {np.round(d.hand_q(hand), 3)}", end="\r")
            time.sleep(1 / 200)
    finally:
        d.send_hands({h: np.zeros(layout.HAND_DIM, np.float32) for h in layout.HANDS})
        print("\n\nreturned to open.")


# --- 6. open-loop replay -----------------------------------------------------

def replay(dataset: str, repo: str = "../..", episode: int = 0,
           iface: str | None = None, domain: int = 0, fps: int = 30,
           hands: bool = True, ramp_s: float = 3.0, max_step: float = 0.15,
           max_ramp_speed: float = 0.5) -> None:
    """Drive the REAL arm from a recorded episode's joints. No model, no IK, no
    policy server.

    This is the rung that catches ~most deployment bugs: joint order, sign, units,
    CRC, publish rate, the hand mapping, and the e-stop — all with a trajectory we
    KNOW is good, so anything that looks wrong is the plumbing.
    """
    from ego2g1.deploy import dds as _dds
    from ego2g1.deploy import safety as _safety
    from ego2g1.deploy.trajectory import TrajectoryBuffer

    ep = _dataset_episode(dataset, episode)
    arm = ep["arm"]
    hand_cmds = np.concatenate([ep["hand"][h] for h in layout.HANDS], axis=1)
    print(f"{ep['name']}: {len(arm)} frames @ {fps} Hz = {len(arm)/fps:.1f} s")

    d = _dds.G1DDS(network_interface=iface, domain=domain, enable_hands=hands)
    d.connect()
    q0 = d.arm_q()
    print(f"measured arm: {np.round(q0, 3)}")
    print(f"episode start: {np.round(arm[0], 3)}")
    print(f"max |delta| to reach it: {np.abs(arm[0] - q0).max():.3f} rad")
    ramp_s = _ramp_seconds(q0, arm[0], ramp_s, max_ramp_speed)
    if input(f"\nramp to the episode start over {ramp_s:.1f}s and replay? [y/N] "
             ).strip().lower() != "y":
        return

    traj = TrajectoryBuffer(layout.ARM_DOF)
    htraj = TrajectoryBuffer(hand_cmds.shape[1])
    # The clamp guards against a corrupt or non-finite frame IN the recording
    # reaching the wire at full swing. It bounds each per-knot step; since knots are
    # 1/fps apart that is a rate limit. The episode is known-good so it should never
    # actually clamp — a non-zero count here means the recording has a jump.
    clamp = _safety.Clamp(_safety.SafetyLimits(max_joint_step=max_step))
    clamp.reset(arm[0])

    now = time.monotonic()
    traj.seed(now, q0)
    htraj.seed(now, hand_cmds[0])
    # ramp in (interpolated, not clamped — that is why the ramp DURATION is
    # stretched above), then the episode, clamped knot by knot.
    traj.push(now + ramp_s, arm[0])
    for k in range(1, len(arm)):
        traj.push(now + ramp_s + k / fps, clamp(arm[k], 1.0 / fps))
        htraj.push(now + ramp_s + k / fps, hand_cmds[k])
    if clamp.clamped_ticks:
        print(f"  WARNING: clamp fired {clamp.clamped_ticks}x — the recording has "
              f"joint jumps > {max_step} rad/tick (max seen {clamp.max_seen:.3f})")

    end = now + ramp_s + len(arm) / fps
    n = layout.HAND_DIM
    try:
        while time.monotonic() < end:
            t = time.monotonic()
            q = traj.eval(t)
            d.send_arm(q)
            if hands:
                v = htraj.eval(t)
                d.send_hands({h: v[i * n:(i + 1) * n] for i, h in enumerate(layout.HANDS)})
            time.sleep(1 / 500)
        print("replay complete.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("damping.")
        d.damp()


# --- 7. open-loop replay of the ACTION LABELS --------------------------------

def replay_actions(dataset: str, repo: str | None = None, episode: int | None = 0,
                   source_episode: str | None = None,
                   iface: str | None = None, domain: int = 0, hands: bool = True,
                   ramp_s: float = 3.0, max_step: float = 0.15,
                   max_ramp_speed: float = 0.5,
                   ik_iters: int = 5, collision_min_dist: float = 0.005,
                   video: bool = False, out: str = "replay_actions.npz") -> None:
    """Drive the REAL arm from the dataset's ACTION LABELS, through the real
    control loop, with the policy replaced by the recording.

    Rung 6 replays stored joints and so never touches an action label: it proves
    the plumbing. This runs the labels through the loop that deployment runs —
    measured-FK anchor, delta composition, mink IK, clamp, 500 Hz emitter — so it
    proves the TRANSFORMS. Run 6 first: if 6 is good and this is not, the bug is
    in the labels or the chunk math, not the wiring.

    The arm is ramped to the episode's first recorded posture before the loop
    starts, so the robot begins where the dashboard's MuJoCo replay begins.
    """
    from ego2g1.deploy import client as _client
    from ego2g1.deploy import dataset_client as _dsc
    from ego2g1.deploy import dds as _dds
    from ego2g1.deploy import kinematics as _kin
    from ego2g1.deploy import loop as _loop
    from ego2g1.deploy import safety as _safety
    from ego2g1.deploy.trajectory import TrajectoryBuffer

    client = _dsc.DatasetClient(dataset, episode=episode, source_episode=source_episode)
    ep = client.ep
    print(f"episode {ep.episode_index} ({ep.source_episode}): {ep.n_frames} frames "
          f"@ {client.fps} Hz = {ep.n_frames / client.fps:.1f} s")
    print(f"task: {ep.task!r}\n")

    cam = _dsc.DatasetCamera(client, video=video)
    cam.connect()
    kin = _kin.Kinematics(repo, collision_min_dist=collision_min_dist,
                          ik_iters=ik_iters, fps=client.fps)

    d = _dds.G1DDS(network_interface=iface, domain=domain, enable_hands=hands)
    d.connect()
    q0 = d.arm_q()
    start = ep.arm_qpos[0].astype(np.float64)
    hand0 = np.concatenate([ep.hand_left[0], ep.hand_right[0]])
    print(f"measured arm:  {np.round(q0, 3)}")
    print(f"episode start: {np.round(start, 3)}")
    print(f"max |delta| to reach it: {np.abs(start - q0).max():.3f} rad")
    ramp_s = _ramp_seconds(q0, start, ramp_s, max_ramp_speed)
    if input(f"\nramp to the episode start over {ramp_s:.1f}s, then replay the "
             "action labels? [y/N] ").strip().lower() != "y":
        return

    # --- ramp in. The loop seeds itself from the MEASURED joints, so it must find
    # the arm already at the episode's starting posture: the first chunk's deltas
    # are composed onto wherever the arm actually is, not onto the recording.
    n = layout.HAND_DIM
    ramp, hramp = TrajectoryBuffer(layout.ARM_DOF), TrajectoryBuffer(len(hand0))
    t_ramp = time.monotonic()
    ramp.seed(t_ramp, q0)
    hramp.seed(t_ramp, hand0)
    ramp.push(t_ramp + ramp_s, start)
    hramp.push(t_ramp + ramp_s, hand0)
    while time.monotonic() < t_ramp + ramp_s:
        t = time.monotonic()
        d.send_arm(ramp.eval(t))
        if hands:
            v = hramp.eval(t)
            d.send_hands({h: v[i * n:(i + 1) * n] for i, h in enumerate(layout.HANDS)})
        time.sleep(1 / 500)
    print(f"at episode start (residual {np.abs(d.arm_q() - start).max():.3f} rad)\n")

    loop = _loop.DeployLoop(
        _loop.LoopConfig(task=ep.task, fps=client.fps, blocking=True),
        dds=d, camera=cam, kinematics=kin, client=client,
        budget=_client.DelayBudget(client.fps, initial=0),
        limits=_safety.SafetyLimits(max_joint_step=max_step),
    )
    # The loop owns the only honest count of how much of a chunk was executed.
    client.attach_consumed(loop.queue.peek_slot)

    log = []
    loop.start()
    try:
        while not loop.watchdog.tripped:
            now = time.monotonic()
            q_cmd = loop.traj_arm.eval(now)
            if q_cmd is not None:
                log.append((now, q_cmd.copy(), d.arm_q()))
            if client.exhausted and loop.traj_arm.runway(now) <= 0.0:
                print("\nepisode complete.")
                break
            time.sleep(1 / client.fps)
        else:
            print(f"\nWATCHDOG TRIPPED: {loop.watchdog.reason}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        loop.stop()
        d.damp()
        print("damped.")

    if log:
        t, cmd, meas = (np.array([r[i] for r in log]) for i in range(3))
        err = np.abs(cmd - meas)
        print(f"\ntracking: mean {err.mean():.4f} rad   max {err.max():.4f} rad")
        print(f"clamped ticks: {loop.clamp.clamped_ticks}  (max step seen "
              f"{loop.clamp.max_seen:.3f} rad)")
        np.savez(out, t=t - t[0], q_cmd=cmd, q_meas=meas,
                 episode_index=ep.episode_index, fps=client.fps)
        print(f"wrote {out}")


# --- 8. policy-server latency ------------------------------------------------

def latency(host: str = "127.0.0.1", port: int = 8000, n: int = 20,
            frame_hw: tuple[int, int] = (480, 640),
            image_resize: tuple[int, int] | None = (224, 224)) -> None:
    """Time the round trip to the policy server. No robot, no camera.

    Run it TWICE: once on the server box (127.0.0.1, no tunnel) and once on the
    robot PC. The server-local number is pure inference; the difference between
    them is what the network costs. That difference is the entire question behind
    a split deployment (serve on a remote cluster, deploy on the Mac).

    What the numbers mean: the loop promises the server a delay of `d` ticks and
    splices the new chunk at slot d. DelayBudget caps d at max_d (20 ticks = 667 ms
    at 30 Hz). Past that the budget saturates: the loop still plans, but chunks land
    after the prefix they were guided against, so the RTC seam guarantee is gone and
    continuity rests on the joint clamp alone. So p95 is the number that matters,
    not the mean, and 667 ms is a cliff rather than a gradient.

    The first call includes an XLA compile (minutes on a cold server) and is
    reported separately — never let a policy's first-ever request happen with the
    robot in the loop.
    """
    from ego2g1.deploy import client as _client

    c = _client.PolicyClient(host, port, resize=image_resize)
    frame = np.random.randint(0, 255, (*frame_hw, 3), dtype=np.uint8)
    state = np.zeros(c.action_dim, dtype=np.float32)

    print(f"\nserver {host}:{port} | horizon {c.action_horizon} dim {c.action_dim} "
          f"fps {c.fps}")
    sent = c._prepare_image(frame)
    print(f"frame {frame.shape} ({frame.nbytes / 1e6:.2f} MB) -> wire {sent.shape} "
          f"({sent.nbytes / 1e3:.0f} KB)\n")

    t0 = time.monotonic()
    out = c.infer(frame, state, "latency check")
    print(f"first call (includes XLA compile): {time.monotonic() - t0:.1f} s"
          f"   actions {np.asarray(out['actions']).shape}\n")

    lat = []
    for i in range(n):
        out = c.infer(frame, state, "latency check")
        lat.append(out["client_latency_s"])
        print(f"  {i + 1:2d}/{n}  {lat[-1] * 1000:6.0f} ms", end="\r")
    lat = np.array(lat)

    budget_s = 20 / c.fps          # DelayBudget.max_d ticks
    p95 = float(np.quantile(lat, 0.95))
    print(f"\n\nmean {lat.mean() * 1000:.0f} ms   p95 {p95 * 1000:.0f} ms   "
          f"max {lat.max() * 1000:.0f} ms")
    print(f"d at p95: {int(np.ceil(p95 * 1.15 * c.fps))} ticks "
          f"(budget caps at 20 = {budget_s * 1000:.0f} ms)")
    if p95 > budget_s:
        print("\n  OVER BUDGET — the delay budget will saturate. Chunks splice at a "
              "slot the\n  robot has already passed: no RTC continuity guarantee, "
              "seams rest on the\n  clamp. Run --blocking, or move the server closer.")
    else:
        print(f"\n  OK — {(budget_s - p95) * 1000:.0f} ms of headroom.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.extras.subcommand_cli_from_dict({
        "listen": listen,
        "fk": fk,
        "ik": ik,
        "camera": camera,
        "hand-sweep": hand_sweep,
        "replay": replay,
        "replay-actions": replay_actions,
        "latency": latency,
    })
