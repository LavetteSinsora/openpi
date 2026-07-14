"""Teacher-forced policy eval on the REAL robot, with recorded observations.

    python -m ego2g1.deploy.eval_real \
        --dataset ../../lerobot_datasets/ego2g1/put_bottle_in_box \
        --episode 0 --host 127.0.0.1 --port 8000

The real-hardware twin of `ego2g1.eval_replay`, which runs this same loop in
MuJoCo. Same semantics: every K ticks, re-anchor the robot to the recording's
ground-truth posture, feed the policy the RECORDED image at that tick, execute the
first K of the 50 actions it predicts, repeat. In-distribution by construction;
measures local per-waypoint accuracy, not compounding drift.

Why this rung exists, and why it is the one to run BEFORE a live rollout:

  A bad live rollout has two candidate explanations that no other test separates —
  the policy is bad, or the G1's head camera does not show what the Pico headset
  showed in training. That viewpoint question is the biggest open risk in this whole
  deployment (see deploy/camera.py) and it fails QUIETLY: a systematically shifted
  FOV just looks like a mediocre checkpoint. Feed the policy the recording's own
  frames and the ambiguity disappears. Whatever goes wrong here is the policy, the
  transforms, or the robot — the camera is out of it.

  Correspondingly: this rung says NOTHING about whether the head camera is usable.
  It is the control, not the experiment.

The reset is a RAMP, not a teleport. eval_replay snaps its MuJoCo robot to ground
truth instantly; a real arm cannot, and by the time the policy has drifted for K
ticks the snap-back can be a large motion. So it goes through `ramp.ramp_to` at the
same 0.5 rad/s the bring-up rungs use, and then SETTLES before the next query — the
anchor is the measured FK, and reading it mid-coast anchors the chunk on a pose the
robot is not in. The pause costs nothing: a teacher-forced timeline is already
discontinuous at every snap. That is what the "visible snap" is.

Blocking by nature — no RTC, no async, no delay budget. Those exist to make chunk
SEAMS continuous, and this rung deliberately breaks the timeline at every seam.
"""

import dataclasses
import logging
import pathlib
import time

import numpy as np
import tyro

from ego2g1.common import layout
from ego2g1.deploy import chunk as _chunk
from ego2g1.deploy import ramp as _ramp
from ego2g1.deploy import safety as _safety

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    dataset: str
    episode: int = 0

    # --- policy server ---
    host: str = "127.0.0.1"
    port: int = 8000
    image_resize: tuple[int, int] | None = (224, 224)

    # How many of the 50 predicted actions to execute before re-anchoring to ground
    # truth. Lower = more teacher forcing, less drift, more snaps. 25 matches
    # eval_replay's default so the two are comparable.
    k: int = 25

    start_tick: int = 0
    max_segments: int = 0                  # 0 => to the end of the episode

    # --- robot ---
    network_interface: str | None = None
    domain: int = 0
    hands: bool = True

    # --- kinematics ---
    repo: str | None = None
    ik_iters: int = 5
    collision_min_dist: float = 0.005

    # --- safety ---
    max_joint_step: float = 0.15
    ramp_s: float = 3.0
    max_ramp_speed: float = 0.5
    settle_s: float = 0.5

    out: str = "eval_real.npz"


def main(args: Args) -> None:
    from ego2g1.deploy import client as _client
    from ego2g1.deploy import dds as _dds
    from ego2g1.deploy import kinematics as _kin
    from ego2g1.deploy.trajectory import TrajectoryBuffer
    from ego2g1.eval_replay import dataset_io as dio

    ep = dio.load_episode(pathlib.Path(args.dataset), args.episode)
    print(f"\nepisode {ep.episode_index} ({ep.source_episode}): {ep.n_frames} frames "
          f"@ {ep.fps:.0f} Hz")
    print(f"task: {ep.task!r}")

    client = _client.PolicyClient(args.host, args.port, resize=args.image_resize)
    if args.k > client.action_horizon:
        raise ValueError(f"--k {args.k} exceeds the policy's horizon "
                         f"{client.action_horizon}")
    fps = client.fps

    # The frames the policy trained on — decoded up front, not per segment: a
    # mid-episode decode stall would show up as the arm holding position while the
    # robot is stiff, which is exactly the thing we do not want to debug on hardware.
    print(f"decoding {ep.n_frames} frames from {ep.video_path.name} ...")
    frames = dio.read_video_frames(ep.video_path, ep.n_frames, ep.fps)

    kin = _kin.Kinematics(args.repo, collision_min_dist=args.collision_min_dist,
                          ik_iters=args.ik_iters, fps=fps)
    dds = _dds.G1DDS(network_interface=args.network_interface, domain=args.domain,
                     enable_hands=args.hands)
    dds.connect()

    n = layout.HAND_DIM
    dt = 1.0 / fps
    clamp = _safety.Clamp(_safety.SafetyLimits(max_joint_step=args.max_joint_step))

    def gt_at(t: int):
        q = ep.arm_qpos[t].astype(np.float64)
        hand = {"left": ep.hand_left[t].astype(np.float64),
                "right": ep.hand_right[t].astype(np.float64)}
        return q, hand

    q_gt0, _ = gt_at(args.start_tick)
    q_now = dds.arm_q()
    print(f"\nmeasured arm:  {np.round(q_now, 3)}")
    print(f"episode tick {args.start_tick}: {np.round(q_gt0, 3)}")
    print(f"max |delta|: {np.abs(q_gt0 - q_now).max():.3f} rad")
    print(f"\nteacher-forced: snap to ground truth every {args.k} ticks "
          f"({args.k / fps:.2f} s of policy motion per segment)")
    if input("\nramp to the episode tick and start? [y/N] ").strip().lower() != "y":
        return

    log = []
    tick = args.start_tick
    seg = 0
    try:
        while tick < ep.n_frames - 1:
            if args.max_segments and seg >= args.max_segments:
                break

            # --- teacher forcing: back to ground truth, then STOP moving ---------
            q_gt, hand_gt = gt_at(tick)
            print(f"\nsegment {seg}: snap to tick {tick}/{ep.n_frames - 1}")
            residual = _ramp.ramp_to(
                dds, q_gt, np.concatenate([hand_gt[h] for h in layout.HANDS]),
                ramp_s=args.ramp_s, max_speed=args.max_ramp_speed,
                hands=args.hands, settle_s=args.settle_s,
            )
            if residual > 0.15:
                raise RuntimeError(
                    f"arm did not reach the ground-truth posture (residual "
                    f"{residual:.3f} rad). Anchoring a chunk on a pose the robot is "
                    f"not in would send it somewhere else entirely.")

            # --- query: MEASURED anchor + state, RECORDED image ------------------
            arm_q = dds.arm_q()
            kin.ground(arm_q)
            anchor = kin.flange_poses(arm_q)
            # The hand block of the state is what we are COMMANDING right now, which
            # after the snap is the ground-truth hand. Same convention the live loop
            # uses (the emitter's current value), and in-distribution here by
            # construction.
            state = kin.state(arm_q, hand_gt)

            out = client.infer(frames[tick], state, ep.task)
            actions = np.asarray(out["actions"], dtype=np.float32)
            logger.info("segment %d: tick %d, %.0f ms, sampler=%s", seg, tick,
                        out["client_latency_s"] * 1000,
                        out.get("rtc", {}).get("sampler", "?"))

            # --- execute the first K, through the REAL transforms ---------------
            traj = TrajectoryBuffer(layout.ARM_DOF)
            htraj = TrajectoryBuffer(n * len(layout.HANDS))
            t0 = time.monotonic()
            traj.seed(t0, arm_q)
            htraj.seed(t0, np.concatenate([hand_gt[h] for h in layout.HANDS]))
            clamp.reset(arm_q)

            for k in range(args.k):
                if tick + k >= ep.n_frames - 1:
                    break
                action = actions[k]
                if not _safety.sanity_check_action(action):
                    raise RuntimeError(f"non-finite or absurd action at slot {k}")
                targets = _chunk.targets_from(action, anchor)
                q = clamp(kin.solve(targets), dt)
                traj.push(t0 + (k + 1) * dt, q)
                hands = _chunk.hands_from(action)
                htraj.push(t0 + (k + 1) * dt,
                           np.concatenate([hands[h] for h in layout.HANDS]))

            end = t0 + (args.k + 1) * dt
            while time.monotonic() < end:
                t = time.monotonic()
                q = traj.eval(t)
                if q is not None:
                    dds.send_arm(q)
                if args.hands:
                    v = htraj.eval(t)
                    if v is not None:
                        dds.send_hands({h: v[i * n:(i + 1) * n]
                                        for i, h in enumerate(layout.HANDS)})
                time.sleep(1 / 500)

            # Where the policy actually took the arm, vs where the recording says it
            # should be K ticks on. This is the number the rung exists to produce.
            q_end = dds.arm_q()
            t_end = min(tick + args.k, ep.n_frames - 1)
            q_gt_end = ep.arm_qpos[t_end].astype(np.float64)
            err = float(np.abs(q_end - q_gt_end).max())
            print(f"  after {args.k} ticks: max |q - q_gt| = {err:.3f} rad")
            log.append({"segment": seg, "tick": tick, "q_end": q_end,
                        "q_gt_end": q_gt_end, "err": err})

            tick += args.k
            seg += 1

        print("\neval complete.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("damping.")
        dds.damp()

    if log:
        errs = np.array([r["err"] for r in log])
        print(f"\nsegments: {len(log)}")
        print(f"drift after {args.k} ticks: mean {errs.mean():.3f} rad   "
              f"max {errs.max():.3f} rad")
        print(f"clamped ticks: {clamp.clamped_ticks} (max step seen "
              f"{clamp.max_seen:.3f} rad)")
        np.savez(args.out,
                 tick=np.array([r["tick"] for r in log]),
                 q_end=np.stack([r["q_end"] for r in log]),
                 q_gt_end=np.stack([r["q_gt_end"] for r in log]),
                 err=errs, k=args.k, fps=fps, episode_index=ep.episode_index)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
