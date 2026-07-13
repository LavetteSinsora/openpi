"""FK for the anchor + state, IK for the targets. Both reuse data_extraction's
tested sim, because the deployment kinematics MUST be the same kinematics that
generated the training labels.

Three facts inherited from data_extraction/sim/g1.py, none of them optional:

  * The IK is 14 DOF (7 per arm). Waist and legs are frozen at EXACTLY 0 rad for
    the whole dataset, over a fixed pelvis. So FK here zeros the full qpos and
    writes only the 14 arm joints — the real robot must be commanded to match
    (waist -> 0), or every pose we produce is expressed against a torso that
    isn't where the labels think it is.
  * The flange is the `*_ee_site`, which sits at the wrist_yaw_link origin with
    ZERO offset. Not a palm frame, not the hand mount.
  * Poses are in the PELVIS frame — the base link. Since waist == 0, pelvis->flange
    depends only on that arm's 7 joints, so we never need a world frame, an IMU,
    or the legs. mink solves in the MJCF world though, so pelvis-frame targets are
    pre-multiplied by the (constant) base pose on the way in.
"""

import pathlib
import sys

import numpy as np

from ego2g1.common import layout, se3


def _import_sim(data_extraction_path: str | pathlib.Path | None = None):
    """Return the G1 sim module — vendored copy by default, external on request.

    Default (path is None): the copy vendored into `deploy/_g1_sim/`, so `ego2g1/`
    is a single self-contained folder with no dependency on the outer repo. The
    copy is guaranteed to match the training sim by
    `test_deploy.py::test_vendored_g1_sim_matches_source` (re-vendor with
    `python -m ego2g1.deploy.vendor_g1_sim`).

    Explicit path: import data_extraction from that repo instead — for developing
    against an edited training sim before re-vendoring, and for the drift test
    itself, which loads BOTH and compares.
    """
    if data_extraction_path is None:
        from ego2g1.deploy._g1_sim.sim import g1
        return g1
    root = pathlib.Path(data_extraction_path).expanduser().resolve()
    # accept either the repo root or the data_extraction dir itself
    root = root.parent if root.name == "data_extraction" else root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from data_extraction.sim import g1
    return g1


class Kinematics:
    """Anchor/state FK and target IK against the training G1 model."""

    def __init__(self, data_extraction_path=None, *, collision_min_dist: float = 0.005,
                 ik_iters: int = 5, fps: int = 30):
        import mujoco

        self._g1 = _import_sim(data_extraction_path)
        self._mujoco = mujoco
        self.ik_iters = int(ik_iters)
        self.dt = 1.0 / float(fps)

        # One model, two MjData: the IK writes its solution into its own backend,
        # and FK must never see that. Sharing one backend "works" (mink solves
        # against its own Configuration) but couples two things that should not be.
        model = mujoco.MjModel.from_xml_path(self._g1.MODEL_XML)
        self._fk = self._g1.G1Backend(model=model)
        self._ik_backend = self._g1.G1Backend(model=model)
        self.ik = self._g1.DualArmIK(self._ik_backend, collision_min_dist=collision_min_dist)

        self.arm_adr = np.concatenate(
            [self._fk.arm_qpos_adr["left"], self._fk.arm_qpos_adr["right"]]
        )
        if len(self.arm_adr) != layout.ARM_DOF:
            raise RuntimeError(f"expected {layout.ARM_DOF} arm DOF, model has {len(self.arm_adr)}")

        # Fixed base: constant, so compute once.
        self.base = self._fk.base_pose()
        self.base_inv = se3.se3_inv(self.base)

    # --- FK -----------------------------------------------------------------

    def _write_arm(self, backend, arm_q14):
        arm_q14 = np.asarray(arm_q14, dtype=np.float64)
        if arm_q14.shape != (layout.ARM_DOF,):
            raise ValueError(f"expected ({layout.ARM_DOF},) arm joints, got {arm_q14.shape}")
        backend.data.qpos[:] = 0.0  # waist AND legs at exactly 0, as in training
        backend.data.qpos[self.arm_adr] = arm_q14
        self._mujoco.mj_forward(backend.model, backend.data)

    def flange_poses(self, arm_q14) -> dict:
        """Measured arm joints -> {hand: (4,4)} flange pose in the PELVIS frame.

        This is the anchor. It is the measured pose, never a commanded or stored
        one — the whole action chunk is expressed relative to it.
        """
        self._write_arm(self._fk, arm_q14)
        return {h: self.base_inv @ self._fk.flange_pose(h) for h in layout.HANDS}

    def state(self, arm_q14, hand_cmds: dict) -> np.ndarray:
        """Build the (30,) observation state the model expects.

        hand_cmds is the LAST COMMAND we sent, per hand — not encoder feedback.
        Training's state hand-block is the retargeted command (no robot ever
        executed those commands, so no encoders exist in the data). Encoders lag,
        and during a grasp they stall against the object and never reach the
        command; feeding them would put the model out of distribution exactly at
        the moment that matters.
        """
        poses = self.flange_poses(arm_q14)
        out = np.zeros(layout.DIM, dtype=np.float32)
        for h in layout.HANDS:
            out[layout.EEF[h]] = se3.se3_to_vec9(poses[h])
            out[layout.HAND[h]] = np.clip(np.asarray(hand_cmds[h], dtype=np.float32), 0.0, 1.0)
        return out

    # --- IK -----------------------------------------------------------------

    def ground(self, arm_q14) -> None:
        """Re-seed the IK at the MEASURED configuration.

        Called once per new chunk: that is what closes the loop. Within a chunk we
        warm-start from the previous SOLUTION instead, which is what keeps the
        joint trajectory continuous across ticks.
        """
        self._write_arm(self._ik_backend, arm_q14)
        self.ik.config.update(self._ik_backend.data.qpos.copy())

    def solve(self, targets: dict) -> np.ndarray:
        """{hand: (4,4)} pelvis-frame flange targets -> (14,) arm joints.

        Warm-started from wherever the IK last left off. mink's site targets are
        world-frame, hence the base pre-multiply.
        """
        q = self.ik.solve_tick(
            self.base @ np.asarray(targets["left"], dtype=np.float64),
            self.base @ np.asarray(targets["right"], dtype=np.float64),
            iters=self.ik_iters,
            dt=self.dt,
        )
        return q[self.arm_adr].copy()

    def tracking_error(self, targets: dict) -> dict:
        """Post-solve flange error per hand, in metres. The honest signal for
        'the IK could not reach that' — an unreachable target is silently
        approximated by the QP, not reported."""
        return {
            h: float(np.linalg.norm(
                self._ik_backend.flange_pose(h)[:3, 3]
                - (self.base @ np.asarray(targets[h], dtype=np.float64))[:3, 3]
            ))
            for h in layout.HANDS
        }
