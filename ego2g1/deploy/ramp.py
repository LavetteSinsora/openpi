"""Move the arm to a posture, slowly, outside the control loop.

Used in three places that all need the same thing and must not drift apart: the
bring-up rungs (6, 7) ramping to an episode's first posture, `deploy
--start-from-episode` doing the same before a rollout, and `eval_real` snapping
back to ground truth between teacher-forced segments.

The ramp is a plain interpolation and does NOT pass through the safety clamp — the
clamp guards knots leaving the IK, and there is no IK here. So the rate limit has
to be enforced by stretching the DURATION: a fixed 3 s is a promise about how long
we take, not about how fast we move, and the same 3 s that is gentle from a nearby
pose is a lunge from across the workspace.

`settle_s` matters more than it looks. Whatever runs next reads the MEASURED joints
to anchor on (FK -> flange poses -> the frame the policy's deltas compose onto). Read
them while the arm is still coasting into position and the anchor describes a pose
the robot is not in, so the composed targets land somewhere else entirely. Let it
stop first.
"""

import time

import numpy as np

from ego2g1.common import layout


def ramp_seconds(q_now, q_target, ramp_s: float, max_speed: float,
                 *, verbose: bool = True) -> float:
    """How long the ramp must take to stay under `max_speed` (rad/s)."""
    delta = float(np.abs(np.asarray(q_target) - np.asarray(q_now)).max())
    needed = delta / max_speed if max_speed > 0 else ramp_s
    if needed > ramp_s:
        if verbose:
            print(f"  ramp stretched {ramp_s:.1f}s -> {needed:.1f}s "
                  f"({delta:.2f} rad at {max_speed:.2f} rad/s)")
        return needed
    return ramp_s


def ramp_into(traj, htraj, now: float, q_target, hand_target=None, *,
              ramp_s: float = 3.0, max_speed: float = 0.5,
              verbose: bool = True) -> float:
    """Push the ramp as knots into trajectories something ELSE is emitting from.

    Returns the monotonic time the ramp lands.

    The ramp starts from `traj.eval(now)` — the value the emitter is sending THIS
    INSTANT — and never from the measured joints. Seeding a new segment at the
    measured pose looks more "honest" and is in fact a step discontinuity: the arm
    lags its command by the servo tracking error (0.1 rad is normal under load), so
    the command would jump backwards by that error in one emitter period. Command
    continuity is a property of the command stream, not of the robot.
    """
    q_from = traj.eval(now)
    ramp_s = ramp_seconds(q_from, q_target, ramp_s, max_speed, verbose=verbose)

    traj.reseed(now, q_from)
    traj.push(now + ramp_s, np.asarray(q_target, dtype=np.float64))

    if htraj is not None and hand_target is not None:
        h_from = htraj.eval(now)
        htraj.reseed(now, h_from)
        htraj.push(now + ramp_s, np.asarray(hand_target, dtype=np.float64))

    return now + ramp_s


def ramp_to(dds, q_target, hand_target=None, *, ramp_s: float = 3.0,
            max_speed: float = 0.5, hands: bool = True, settle_s: float = 0.3,
            hz: float = 500.0, verbose: bool = True) -> float:
    """Standalone ramp: own the emitter, move the arm, settle, return.

    For callers with no control loop running (the bring-up rungs, `deploy
    --start-from-episode`). Where an emitter is already running, use `ramp_into`
    instead and let it keep emitting — starting a second one would fight it.

    Returns the residual (rad) between the target and where the arm actually ended
    up. A large residual means the arm did not track the ramp — it is blocked, or
    the PD cannot hold it against gravity in that pose — and nothing downstream that
    anchors on the measured pose will be meaningful.
    """
    from ego2g1.deploy.trajectory import TrajectoryBuffer

    q_target = np.asarray(q_target, dtype=np.float64)
    n = layout.HAND_DIM
    now = time.monotonic()

    traj = TrajectoryBuffer(layout.ARM_DOF)
    traj.seed(now, dds.arm_q())        # nothing is emitting yet, so measured IS the command
    htraj = None
    if hands and hand_target is not None:
        htraj = TrajectoryBuffer(len(np.asarray(hand_target)))
        htraj.seed(now, np.asarray(hand_target, dtype=np.float64))

    landed = ramp_into(traj, htraj, now, q_target, hand_target,
                       ramp_s=ramp_s, max_speed=max_speed, verbose=verbose)

    period = 1.0 / hz
    end = landed + settle_s            # hold the target through the settle
    while time.monotonic() < end:
        t = time.monotonic()
        q = traj.eval(t)
        if q is not None:
            dds.send_arm(q)
        if htraj is not None:
            v = htraj.eval(t)
            if v is not None:
                dds.send_hands({h: v[i * n:(i + 1) * n]
                                for i, h in enumerate(layout.HANDS)})
        time.sleep(period)

    residual = float(np.abs(dds.arm_q() - q_target).max())
    if verbose:
        print(f"  at target (residual {residual:.3f} rad)")
    return residual
