"""MuJoCo kinematic backend + dual-arm differential IK for wrist_replay.

The backend is the swappable "robot" layer: today it writes IK solutions
straight into qpos (kinematic playback), later the same interface is
implemented by a dynamics sim, unitree_mujoco's DDS bridge, or the real
unitree_sdk2 client.

IK is mink (QP differential IK on MuJoCo): one FrameTask per flange site +
a low-cost posture task, with every non-arm DoF frozen exactly by zeroing
its velocity before integration (legs/waist never move).
"""

import os

import mujoco
import numpy as np

from ..common import frames

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_XML = os.path.join(os.path.dirname(HERE), "assets", "unitree_g1",
                         "scene_fixed_base.xml")

ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint",
             "left_wrist_yaw_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"],
}
EE_SITES = {"left": "left_ee_site", "right": "right_ee_site"}

# Ready pose (rad): all arm joints at zero. For the menagerie G1 this IS the
# natural "ready" configuration the user wants - the elbow link has a built-in
# 90-degree dogleg, so qpos 0 gives upper arm vertical along the torso,
# forearm horizontal pointing forward, hand in front of the body at
# chest/waist height (flange ~[0.20, +-0.15, 0.89], shoulder at z=1.085).
# A touch of shoulder roll keeps the arms clear of the torso shell.
NOMINAL_ARM_QPOS = {
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
}

# Bodies whose collision geoms take part in self-collision avoidance /
# clearance monitoring. Shoulder links are deliberately absent from the arm
# sets: they sit adjacent to the torso shell (permanently near-contact) and
# would keep the avoidance constraint saturated for no benefit - the failure
# mode that matters is the distal arm (elbow onward, where the hand mounts)
# entering the trunk.
_ARM_COLLISION_BODIES = {
    "left": ("left_elbow_link", "left_wrist_roll_link",
             "left_wrist_pitch_link", "left_wrist_yaw_link"),
    "right": ("right_elbow_link", "right_wrist_roll_link",
              "right_wrist_pitch_link", "right_wrist_yaw_link"),
}
_TRUNK_COLLISION_BODIES = (
    "pelvis", "torso_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
)


def collision_geom_groups(model):
    """-> {'left': [geom ids], 'right': [...], 'trunk': [...]} of collidable
    (contype/conaffinity != 0) geoms grouped for self-collision checks."""
    by_body = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            by_body.setdefault(model.body(model.geom_bodyid[g]).name, []).append(g)
    groups = {"trunk": [g for b in _TRUNK_COLLISION_BODIES
                        for g in by_body.get(b, [])]}
    for side in ("left", "right"):
        groups[side] = [g for b in _ARM_COLLISION_BODIES[side]
                        for g in by_body.get(b, [])]
    return groups


def self_clearance(backend, groups, side, distmax=0.2):
    """Min signed distance (m, negative = penetration) from `side`'s distal
    arm geoms to the trunk and to the other arm, at the backend's current
    qpos. This is the deployment-faithfulness monitor: the real robot can
    never realize a state where this is negative."""
    other = "right" if side == "left" else "left"
    lo = distmax
    for ga in groups[side]:
        for gb in groups["trunk"] + groups[other]:
            d = mujoco.mj_geomDistance(backend.model, backend.data,
                                       ga, gb, distmax, None)
            lo = min(lo, d)
    return float(lo)


class G1Backend:
    """Kinematic G1: set joint positions, read flange poses, render."""

    def __init__(self, xml_path=MODEL_XML, model=None):
        self.model = model if model is not None \
            else mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.arm_qpos_adr = {}
        self.arm_dof_adr = {}
        for side, names in ARM_JOINTS.items():
            self.arm_qpos_adr[side] = np.array(
                [self.model.joint(n).qposadr[0] for n in names])
            self.arm_dof_adr[side] = np.array(
                [self.model.joint(n).dofadr[0] for n in names])
        self.all_arm_dofs = np.concatenate(
            [self.arm_dof_adr["left"], self.arm_dof_adr["right"]])
        self._renderer = None
        self.reset_nominal()

    def reset_nominal(self):
        self.data.qpos[:] = 0.0
        for name, val in NOMINAL_ARM_QPOS.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = val
        mujoco.mj_forward(self.model, self.data)

    def set_qpos(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def flange_pose(self, side):
        """IK-achieved flange pose as 4x4 (this is what "measured" anchors
        read - the sim stand-in for encoder FK on the real robot)."""
        sid = self.model.site(EE_SITES[side]).id
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[sid]
        return T

    def base_pose(self):
        """Pelvis body's 4x4 world pose (the robot base frame states/actions
        are expressed in; constant while the base is fixed)."""
        bid = self.model.body("pelvis").id
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.data.xpos[bid]
        return T

    def world_to_base(self, T):
        """Re-express a world-frame 4x4 pose in the pelvis (base) frame."""
        return frames.se3_inv(self.base_pose()) @ T

    def shoulder_anchor(self, side):
        """World position of the shoulder-pitch joint axis (fixed while the
        base/waist are frozen) - the center of the arm's reach sphere."""
        jid = self.model.joint(f"{side}_shoulder_pitch_joint").id
        return self.data.xanchor[jid].copy()

    def max_reach(self, side):
        """Radius of the arm's reach sphere: max shoulder-anchor -> flange
        distance over the elbow's range (other joints at 0; qpos 0 has the
        elbow partly bent, so a single sample underestimates)."""
        saved = self.data.qpos.copy()
        elbow_adr = self.model.joint(f"{side}_elbow_joint").qposadr[0]
        lo, hi = self.model.joint(f"{side}_elbow_joint").range
        r = 0.0
        for e in np.linspace(lo, hi, 40):
            self.data.qpos[:] = 0.0
            self.data.qpos[elbow_adr] = e
            mujoco.mj_forward(self.model, self.data)
            r = max(r, float(np.linalg.norm(
                self.flange_pose(side)[:3, 3] - self.shoulder_anchor(side))))
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return r

    def render(self, width=480, height=360, cam_kwargs=None):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._cam)
            self._cam.azimuth = 150
            self._cam.elevation = -12
            self._cam.distance = 1.25
            self._cam.lookat[:] = [0.2, 0.0, 1.0]
        if cam_kwargs:
            for k, v in cam_kwargs.items():
                setattr(self._cam, k, v)
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()


class DualArmIK:
    """mink-based differential IK; arms only, everything else frozen."""

    def __init__(self, backend, position_cost=1.0, orientation_cost=0.3,
                 posture_cost=1e-3, solver=None, collision_min_dist=0.005):
        import mink
        import qpsolvers
        self._mink = mink
        self.backend = backend
        self.solver = solver or ("daqp" if "daqp" in qpsolvers.available_solvers
                                 else qpsolvers.available_solvers[0])
        self.config = mink.Configuration(backend.model)
        self.tasks = {}
        for side in ("left", "right"):
            self.tasks[side] = mink.FrameTask(
                frame_name=EE_SITES[side], frame_type="site",
                position_cost=position_cost,
                orientation_cost=orientation_cost, lm_damping=1e-2)
        self.posture = mink.PostureTask(backend.model, cost=posture_cost)
        self.config.update(backend.data.qpos.copy())
        self.posture.set_target_from_configuration(self.config)
        self._task_list = [self.tasks["left"], self.tasks["right"], self.posture]
        self._mask = np.zeros(backend.model.nv)
        self._mask[backend.all_arm_dofs] = 1.0
        # Freeze legs/waist INSIDE the QP via a zero velocity limit - masking
        # the returned velocity is not enough (the solver would route motion
        # through the frozen joints and the arm components stall far from the
        # target). The post-hoc mask stays as belt and braces.
        m = backend.model
        arm_names = {n for names in ARM_JOINTS.values() for n in names}
        vel = {m.joint(i).name: (100.0 if m.joint(i).name in arm_names else 1e-9)
               for i in range(m.njnt)}
        self._limits = [mink.ConfigurationLimit(m), mink.VelocityLimit(m, vel)]
        # Self-collision avoidance INSIDE the QP: the solver may never emit a
        # velocity that drives the distal arm into the trunk or the other
        # arm. This is what keeps sim FK states inside the deployment state
        # distribution - the real robot physically cannot self-penetrate, so
        # the kinematic replay must not either. Targets inside the body then
        # show up as large tracking error and are filtered, not silently
        # "achieved". Detection distance is generous (10 cm) so a fast
        # approach cannot tunnel through the constraint in one 33 ms step.
        if collision_min_dist is not None:
            groups = collision_geom_groups(m)
            pairs = [(groups["left"], groups["trunk"] + groups["right"]),
                     (groups["right"], groups["trunk"])]
            self._limits.append(mink.CollisionAvoidanceLimit(
                m, pairs,
                minimum_distance_from_collisions=collision_min_dist,
                collision_detection_distance=0.10))

    def _set_target(self, side, T):
        pos, quat = frames.pose_of(T)
        se3 = self._mink.SE3.from_rotation_and_translation(
            self._mink.SO3(wxyz=quat), pos)
        self.tasks[side].set_target(se3)

    def solve_tick(self, T_target_left, T_target_right, iters=5, dt=1 / 30.0):
        """Iterate the QP a few times toward the two targets; returns the
        full qpos vector (already applied to the backend)."""
        self._set_target("left", T_target_left)
        self._set_target("right", T_target_right)
        for _ in range(iters):
            try:
                v = self._mink.solve_ik(self.config, self._task_list, dt,
                                        solver=self.solver, damping=1e-3,
                                        limits=self._limits)
            except self._mink.NoSolutionFound:
                # QP infeasible (constraint corner case): hold the current
                # configuration - the tracking error stays large and the
                # tick is filtered downstream, never silently mis-solved
                break
            self.config.integrate_inplace(v * self._mask, dt)
        q = self.config.q.copy()
        self.backend.set_qpos(q)
        return q

    def solve_static(self, T_left, T_right, iters=300, dt=0.02, tol_pos=1e-3):
        """Converge onto a single pose pair (used for the t0 placement gate).
        Returns (pos_err_l, pos_err_r) in meters after convergence."""
        for i in range(iters):
            self.solve_tick(T_left, T_right, iters=1, dt=dt)
            errs = [np.linalg.norm(self.backend.flange_pose(s)[:3, 3] - T[:3, 3])
                    for s, T in (("left", T_left), ("right", T_right))]
            if max(errs) < tol_pos:
                break
        return errs
