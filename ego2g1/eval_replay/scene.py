"""Phase 2 (display machine): G1 rendering + eval-side IK reconstruction.

Reuses the tested sim from the outer repo's data_extraction (import via
--data-extraction-path added to sys.path by the viewer). Arms first; the revo2
hands are attached by `attach_hands` (added incrementally).
"""

import numpy as np


def _import_sim():
    """Return (g1, frames, hand_constants) from data_extraction (must be on sys.path)."""
    from data_extraction.common import frames
    from data_extraction.hand import constants as hand_constants
    from data_extraction.sim import g1
    return g1, frames, hand_constants


def load_mount_quats(data_extraction_root=None):
    """Per-side flange->Revo2-base rotation as a wxyz quat, from the extraction's
    b_calib stage (`mount_R_{side}`, data_extraction/s002_action_label/b_alignment.py:109
    — the rotation that makes the mounted hand's palm coincide with the mean human
    palm). This is the authoritative hand mount; identity is NOT correct.
    Returns None (identity, with a warning) if b_calib.npz cannot be found."""
    import pathlib

    from scipy.spatial.transform import Rotation

    roots = []
    if data_extraction_root is not None:
        roots.append(pathlib.Path(data_extraction_root))
    try:
        import data_extraction
        roots.append(pathlib.Path(data_extraction.__file__).resolve().parent.parent)
    except Exception:  # noqa: BLE001
        pass
    for root in roots:
        p = root / "data_extraction" / "work" / "_global" / "b_calib.npz"
        if p.exists():
            z = np.load(p)
            return {s: np.roll(Rotation.from_matrix(z[f"mount_R_{s}"]).as_quat(), 1)
                    for s in ("left", "right")}
    print("WARNING: b_calib.npz not found — falling back to an IDENTITY hand mount, which is "
          "visually wrong. Run the extraction's b_calib stage, or pass --data-extraction-path.")
    return None


class G1Renderer:
    """One G1 backend (arms) with offscreen rendering; optionally revo2 hands
    attached for finger rendering."""

    def __init__(self, width=480, height=480, cam=None, with_hands=False, hand_mount=None,
                 data_extraction_root=None):
        self._g1, self._frames, self._hc = _import_sim()
        self.with_hands = with_hands
        if with_hands:
            mount_quats = load_mount_quats(data_extraction_root)
            model = _build_combined_model(self._g1, self._hc, hand_mount, mount_quats)
            self.backend = self._g1.G1Backend(model=model)
            self._hand = _HandDrive(self.backend.model, self.backend.data, self._hc)
        else:
            self.backend = self._g1.G1Backend()
            self._hand = None
        self.arm_adr = np.concatenate(
            [self.backend.arm_qpos_adr["left"], self.backend.arm_qpos_adr["right"]])
        self._width, self._height = width, height
        self._cam = cam or {}

    def set_pose(self, arm_qpos14, hand_left6=None, hand_right6=None):
        self.backend.data.qpos[self.arm_adr] = arm_qpos14
        if self._hand is not None and hand_left6 is not None:
            self._hand.set(hand_left6, hand_right6)
        import mujoco
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def render(self):
        return self.backend.render(width=self._width, height=self._height, cam_kwargs=self._cam)


class EvalIK:
    """mink IK for the eval robot: pelvis-frame flange targets -> arm_qpos(14).
    Warm-started; reset to the ground-truth anchor at each re-plan."""

    def __init__(self, renderer: G1Renderer):
        self._g1, self._frames, _ = _import_sim()
        self.r = renderer
        self.ik = self._g1.DualArmIK(renderer.backend)

    def reset_to_arm(self, arm_qpos14):
        """Ground the IK at a known arm configuration (the GT anchor)."""
        self.r.backend.data.qpos[self.r.arm_adr] = arm_qpos14
        import mujoco
        mujoco.mj_forward(self.r.backend.model, self.r.backend.data)
        self.ik.config.update(self.r.backend.data.qpos.copy())

    def solve(self, T_left_pelvis, T_right_pelvis) -> np.ndarray:
        """Targets in pelvis frame -> world (constant fixed-base transform,
        because mink site targets are world-frame) -> solve -> arm_qpos(14)."""
        base = self.r.backend.base_pose()
        q = self.ik.solve_tick(base @ T_left_pelvis, base @ T_right_pelvis)
        return q[self.r.arm_adr].copy()


# --- combined G1 + revo2 hands (mjSpec attach) --------------------------------

def _build_combined_model(g1, hand_constants, hand_mount, mount_quats):
    """G1 with the built-in `rubber_hand` mesh REMOVED and the revo2 hands
    attached at the flange sites, compiled to one MjModel. hand_mount =
    optional dict(xyz=..., rpy=...) applied to both hands' base relative to the
    flange (default identity — the flange site sits at the wrist_yaw_link origin
    and the revo2 base is already flange-aligned there)."""
    return _build_combined_spec(g1, hand_constants, hand_mount, mount_quats).compile()


# Empirical correction applied on top of the calibrated mount_R: the hand needs a
# further 90deg CLOCKWISE spin about its own (finger) axis to sit like the real hand
# (palms inward / thumbs up, the bottle-grasp approach) rather than palm-flat-down.
# Likely a convention offset between the fk_tables' robot_palm frame (which mount_R
# is derived through) and the revo2 MJCF base_link frame. Override with --hand-mount-rpy.
DEFAULT_HAND_RPY = (0.0, 0.0, -np.pi / 2)


def _build_combined_spec(g1, hand_constants, hand_mount, mount_quats):
    """The uncompiled spec (rubber_hand removed, revo2 hands attached at the
    calibrated flange->Revo2 rotation) — two of these merge into the two-robot
    interactive scene. `hand_mount` (xyz/rpy) is an OPTIONAL extra correction
    applied on top of the calibrated mount."""
    import mujoco

    spec = mujoco.MjSpec.from_file(g1.MODEL_XML)
    for gm in list(spec.geoms):
        if "rubber_hand" in (gm.name or "") or "rubber_hand" in (getattr(gm, "meshname", "") or ""):
            spec.delete(gm)

    xyz = np.zeros(3) if hand_mount is None else np.asarray(hand_mount.get("xyz", np.zeros(3)), float)
    rpy = np.asarray(DEFAULT_HAND_RPY if (hand_mount is None or hand_mount.get("rpy") is None)
                     else hand_mount["rpy"], float)
    extra = _rpy_to_wxyz(rpy)
    for side in ("left", "right"):
        hand_spec = mujoco.MjSpec.from_file(str(hand_constants.MJCF_PATH[side]))
        root = hand_spec.worldbody.bodies[0]  # {side}_base_link
        site = spec.site(g1.EE_SITES[side])
        mount = np.array([1.0, 0, 0, 0]) if mount_quats is None else np.asarray(mount_quats[side], float)
        frame = site.parent.add_frame()
        frame.pos = np.asarray(site.pos, float) + xyz
        frame.quat = _quat_mul(_quat_mul(np.asarray(site.quat, float), mount), extra)
        frame.attach_body(root, f"{side}_hand_", "")
    return spec


class _RobotHandle:
    """qpos addressing + kinematic hand drive for one robot in a (possibly
    multi-robot) model, identified by a joint-name `prefix` ("" or "eval_")."""

    def __init__(self, model, data, hand_constants, g1, prefix=""):
        self.model, self.data = model, data
        adr = lambda n: model.joint(prefix + n).qposadr[0]
        self.arm_adr = np.array([adr(n) for s in ("left", "right") for n in g1.ARM_JOINTS[s]])
        self._hand = _HandDrive(model, data, hand_constants, prefix=prefix)

    def set(self, arm_qpos14, hand_left6, hand_right6):
        self.data.qpos[self.arm_adr] = arm_qpos14
        self._hand.set(hand_left6, hand_right6)


class TwoRobotScene:
    """GT + eval G1 (with hands) in ONE model, offset in y, for the interactive
    orbitable viewer. Display-only — IK reconstruction happens elsewhere."""

    def __init__(self, hand_mount=None, sep=0.9, data_extraction_root=None):
        import mujoco

        self._g1, self._frames, self._hc = _import_sim()
        mq = load_mount_quats(data_extraction_root)
        base = _build_combined_spec(self._g1, self._hc, hand_mount, mq)
        second = _build_combined_spec(self._g1, self._hc, hand_mount, mq)
        frame = base.worldbody.add_frame()
        frame.pos = [0.0, -sep, 0.0]
        frame.attach_body(second.body("pelvis"), "eval_", "")
        self.model = base.compile()
        self.data = mujoco.MjData(self.model)
        self.gt = _RobotHandle(self.model, self.data, self._hc, self._g1, prefix="")
        self.eval = _RobotHandle(self.model, self.data, self._hc, self._g1, prefix="eval_")
        # tint the EVAL robot blue so it is unmistakable (GT keeps the stock look)
        for g in range(self.model.ngeom):
            if self.model.body(self.model.geom_bodyid[g]).name.startswith("eval_"):
                self.model.geom_rgba[g] = [0.45, 0.62, 0.95, 1.0]

    def set_frame(self, gt_arm, gt_hl, gt_hr, ev_arm, ev_hl, ev_hr):
        import mujoco
        self.gt.set(gt_arm, gt_hl, gt_hr)
        self.eval.set(ev_arm, ev_hl, ev_hr)
        mujoco.mj_forward(self.model, self.data)


class _HandDrive:
    """Drive both attached revo2 hands kinematically from a 6-cmd [0,1] each,
    mirroring data_extraction.hand.screen.HandSim: proximal = cmd*ctrl_max, and
    the distal joint coupled = ratio*proximal (natural finger curl)."""

    # distal coupling ratios + which motor drives each finger's proximal
    _COUPLE = {"thumb": (1.0, "thumb_flex"), "index": (1.155, "index"),
               "middle": (1.155, "middle"), "ring": (1.155, "ring"), "pinky": (1.155, "pinky")}

    def __init__(self, model, data, hand_constants, prefix=""):
        self.model = model
        self.data = data
        MO = hand_constants.MOTOR_ORDER
        self.sides = {}
        for side in ("left", "right"):
            act_name = hand_constants.ACTUATOR_NAME[side]  # {motor: joint_name (side-prefixed)}
            qadr = np.empty(6, int)
            cmax = np.empty(6)
            for m, motor in enumerate(MO):
                jname = f"{prefix}{side}_hand_{act_name[motor]}"
                qadr[m] = self.model.joint(jname).qposadr[0]
                cmax[m] = _joint_ctrl_max(self.model, jname)
            prox_of = dict(zip(MO, qadr))
            couple = []  # (distal_qadr, ratio, source_prox_qadr)
            for finger, (ratio, motor) in self._COUPLE.items():
                dj = self.model.joint(f"{prefix}{side}_hand_{side}_{finger}_distal_joint")
                couple.append((dj.qposadr[0], ratio, prox_of[motor]))
            self.sides[side] = (qadr, cmax, couple)

    def set(self, cmd_left6, cmd_right6):
        for side, cmd in (("left", cmd_left6), ("right", cmd_right6)):
            qadr, cmax, couple = self.sides[side]
            self.data.qpos[qadr] = np.clip(cmd, 0.0, 1.0) * cmax
            for dadr, ratio, padr in couple:
                self.data.qpos[dadr] = ratio * self.data.qpos[padr]


def _joint_ctrl_max(model, jname):
    jid = model.joint(jname).id
    for i in range(model.nu):
        if model.actuator(i).trnid[0] == jid:
            return float(model.actuator(i).ctrlrange[1])
    return 1.0


def _rpy_to_wxyz(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r/2), np.sin(r/2), np.cos(p/2), np.sin(p/2), np.cos(y/2), np.sin(y/2))
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def _quat_mul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1,
                     w0*x1 + x0*w1 + y0*z1 - z0*y1,
                     w0*y1 - x0*z1 + y0*w1 + z0*x1,
                     w0*z1 + x0*y1 - y0*x1 + z0*w1])
