"""SE(3) helpers for deployment: compose an action chunk onto an anchor, and
re-anchor a chunk from one anchor to another (needed by RTC).

The vec9/rot6d encoding is re-exported from `ego2g1.chunk_math` (pure numpy,
byte-pinned against the training loader) so there is exactly one definition of
it in the codebase. Only the deployment-specific math is new here.

Conventions, all inherited from training and none of them negotiable:
  vec9(T) = [translation(3), rot6d(6)]   -- TRANSLATION FIRST
  rot6d(R) = concat(R[:, 0], R[:, 1])    -- first two COLUMNS
  delta_k  = G(t)^-1 @ G(t+k)            -- right-multiplied, anchor-local frame
  target_k = T_anchor @ delta_k
"""

import numpy as np

from ego2g1.chunk_math import mat_to_6d, rot6d_to_mat, se3_to_vec9, vec9_to_se3
from ego2g1.common import layout as _layout

__all__ = ["mat_to_6d", "rot6d_to_mat", "se3_to_vec9", "vec9_to_se3",
           "se3_inv", "compose", "compose_chunk", "reanchor_chunk"]


def se3_inv(T):
    """Inverse of a (..., 4, 4) rigid transform, without a general solve."""
    T = np.asarray(T, dtype=np.float64)
    R, t = T[..., :3, :3], T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def compose(anchor_T, delta_vec9):
    """T_target = T_anchor @ vec9_to_se3(delta). Broadcasts over leading dims."""
    return np.asarray(anchor_T, dtype=np.float64) @ vec9_to_se3(delta_vec9)


def compose_chunk(anchor_T, actions):
    """Absolute flange targets for a whole chunk.

    anchor_T: {hand: (4, 4)} measured-FK anchor at the observation tick.
    actions:  (H, 30) raw action chunk from the policy.
    returns:  {hand: (H, 4, 4)} pelvis-frame targets.

    The hand-command dims are ignored here; read them straight off `actions`
    with layout.HAND[hand] — they are absolute, not deltas.
    """
    actions = np.asarray(actions, dtype=np.float64)
    return {h: compose(anchor_T[h], actions[..., _layout.EEF[h]]) for h in _layout.HANDS}


def reanchor_chunk(actions, anchor_old, anchor_new):
    """Re-express an action chunk against a new anchor. RTC needs this.

    A chunk's eef dims are deltas from the anchor that was current when it was
    generated. By the time we use it as an RTC guidance target, the robot has
    moved and the anchor is a different pose, so the deltas mean something
    different. Rewrite them to point at the same ABSOLUTE targets:

        T_abs   = T_old @ delta          (what the old chunk meant)
        delta'  = T_new^-1 @ T_abs       (the same target, from the new anchor)
                = (T_new^-1 @ T_old) @ delta

    Hand dims are absolute commands, not deltas, and pass through untouched.

    actions:    (H, 30) raw actions from the previous chunk (already sliced to
                the leftover, i.e. row 0 must correspond to new-chunk slot 0).
    anchor_old: {hand: (4, 4)} anchor the chunk was generated against.
    anchor_new: {hand: (4, 4)} anchor of the observation we are about to send.
    returns:    (H, 30) in the same layout.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape[-1] != _layout.DIM:
        raise ValueError(f"expected (..., {_layout.DIM}), got {actions.shape}")

    out = actions.copy()
    for h in _layout.HANDS:
        shift = se3_inv(anchor_new[h]) @ np.asarray(anchor_old[h], dtype=np.float64)
        deltas = vec9_to_se3(actions[..., _layout.EEF[h]])   # (H, 4, 4)
        out[..., _layout.EEF[h]] = se3_to_vec9(shift @ deltas)
    return out.astype(np.float32)
