"""The 30-dim state/action layout, and the joint order behind it.

State and action share the same per-hand block structure — per hand
[eef vec9 (9) | hand cmd (6)], hands in ("left", "right") order — so one set of
slices serves both. They differ only in meaning: the state's eef block is an
ABSOLUTE measured-FK flange pose in the pelvis frame, while the action's is a
delta relative to the anchor. Hand dims are absolute [0, 1] in both.
"""

import numpy as np

HANDS = ("left", "right")

EEF_DIM = 9   # vec9: [tx, ty, tz, r00, r10, r20, r01, r11, r21]
HAND_DIM = 6
BLOCK_DIM = EEF_DIM + HAND_DIM  # 15 per hand
DIM = BLOCK_DIM * len(HANDS)    # 30, for both state and action

# Slices into a (..., 30) state or action vector.
EEF = {h: slice(i * BLOCK_DIM, i * BLOCK_DIM + EEF_DIM) for i, h in enumerate(HANDS)}
HAND = {h: slice(i * BLOCK_DIM + EEF_DIM, (i + 1) * BLOCK_DIM) for i, h in enumerate(HANDS)}

# Revo2 motor order within a hand block. Absolute, [0, 1], 0 = open, 1 = closed.
# Canonical source: data_extraction/hand/constants.py MOTOR_ORDER.
HAND_MOTOR_ORDER = ("thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky")

# The 14 IK/FK DOF, in the order DualArmIK returns them (left 7, then right 7).
# Canonical source: data_extraction/sim/g1.py ARM_JOINTS.
ARM_JOINTS = {
    "left": ("left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint",
             "left_wrist_yaw_joint"),
    "right": ("right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"),
}
ARM_DOF = sum(len(v) for v in ARM_JOINTS.values())  # 14
ARM_SLICE = {h: slice(i * 7, (i + 1) * 7) for i, h in enumerate(HANDS)}

# The flange the poses are defined at: the wrist_yaw_link origin, zero offset.
# NOT a palm or hand-mount frame.
EE_SITES = {"left": "left_ee_site", "right": "right_ee_site"}


def split(vec):
    """(..., 30) -> {"left": {"eef": (...,9), "hand": (...,6)}, "right": {...}}."""
    vec = np.asarray(vec)
    if vec.shape[-1] != DIM:
        raise ValueError(f"expected last dim {DIM}, got {vec.shape}")
    return {h: {"eef": vec[..., EEF[h]], "hand": vec[..., HAND[h]]} for h in HANDS}


def join(parts):
    """Inverse of `split`."""
    return np.concatenate(
        [np.concatenate([parts[h]["eef"], parts[h]["hand"]], axis=-1) for h in HANDS],
        axis=-1,
    )
