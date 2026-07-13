"""SE(3) / quaternion helpers for wrist_replay.

Conventions (the single source of truth for the whole tool):
- All internal rigid transforms are 4x4 homogeneous matrices (numpy float64).
- All quaternions inside this package are wxyz (scalar-first, MuJoCo order).
  The Pico HDF5 files store xyzw (scalar-last); `quat_wxyz_from_xyzw` is the
  only place that reordering happens — call it at ingest and never again.
"""

import numpy as np


def quat_wxyz_from_xyzw(q):
    """(..., 4) xyzw -> (..., 4) wxyz. The ingest boundary."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(a, b):
    """Hamilton product, wxyz."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def mat_from_quat(q):
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_mat(R):
    """3x3 rotation matrix -> wxyz quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def quat_slerp(qa, qb, u):
    """Spherical interpolation, wxyz, sign-corrected (shortest arc)."""
    qa = quat_normalize(qa)
    qb = quat_normalize(qb)
    d = float(np.dot(qa, qb))
    if d < 0.0:
        qb, d = -qb, -d
    if d > 0.9995:                      # nearly parallel: lerp
        return quat_normalize(qa + u * (qb - qa))
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - u) * th) * qa + np.sin(u * th) * qb) / np.sin(th)


def se3(pos, quat_wxyz):
    """position + wxyz quaternion -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = mat_from_quat(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def se3_from_rot(R, pos=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def se3_inv(T):
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def pose_of(T):
    """4x4 -> (pos (3,), quat wxyz (4,))."""
    return T[:3, 3].copy(), quat_from_mat(T[:3, :3])


def rot_geodesic_deg(Ra, Rb):
    """Angle (deg) of the rotation taking Ra to Rb."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
