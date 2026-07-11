"""Loader math: absolute stored poses -> anchor-relative action chunks.

Canonical source: the outer repo's data_extraction/loader/ (+ common/rot6d.py)
— keep byte-equivalent; pinned by the outer repo's loader-equivalence test
(data_extraction/tests/). numpy only. Copied (with history) from the reverted
fork commit 57b322f, which carried the same pinned copy.

Conventions (data_extraction/SPEC.md): datapoint at tick t has anchor = pose
at tick t; `pose.*` gathers H+1 rows (row 0 = anchor), `hand.*` gathers H
rows (commands t+1..t+H, absolute); per hand
delta_k = vec9_to_se3(pose_0)^-1 @ vec9_to_se3(pose_k), re-encoded vec9,
concatenated [eef 9 | hand 6] per hand -> (H, 30).
"""

import numpy as np

# ---------------------------------------------------------------------------
# rot6d / vec9 (copy of data_extraction/common/rot6d.py)
# 6d(R) = concat(R[:, 0], R[:, 1]); vec9(T) = [t (3), 6d(R) (6)].
# ---------------------------------------------------------------------------


def mat_to_6d(R):
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_mat(d6):
    d6 = np.asarray(d6, dtype=np.float64)
    a, b = d6[..., :3], d6[..., 3:6]
    x = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b - (x * b).sum(axis=-1, keepdims=True) * x
    y = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)  # columns


def se3_to_vec9(T):
    T = np.asarray(T)
    return np.concatenate([T[..., :3, 3], mat_to_6d(T[..., :3, :3])], axis=-1)


def vec9_to_se3(v):
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (4, 4))
    out[..., :3, :3] = rot6d_to_mat(v[..., 3:9])
    out[..., :3, 3] = v[..., :3]
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# loader pieces (copies of data_extraction/loader/{relative_actions,boundary}.py)
# ---------------------------------------------------------------------------


def make_delta_timestamps(action_horizon, fps):
    """delta_timestamps for LeRobotDataset: pose keys gather [0..H]/fps
    (anchor + chunk), hand keys gather [1..H]/fps (commands only)."""
    h = int(action_horizon)
    pose_ts = [k / fps for k in range(h + 1)]
    hand_ts = [k / fps for k in range(1, h + 1)]
    return {"pose.left": pose_ts, "pose.right": pose_ts,
            "hand.left": hand_ts, "hand.right": hand_ts}


class RelativeChunkActions:
    """Turn gathered pose/hand chunks into relative action chunks.

    Input sample: `pose.<hand>` (H+1, 9) vec9, `hand.<hand>` (H, 6) per hand.
    Output: same sample without those keys, plus `actions` (H, 15*n_hands)
    f32, per hand [eef delta vec9 | absolute hand cmd 6]. A sample without
    pose keys (e.g. at inference) passes through unchanged.
    """

    def __init__(self, hands=("left", "right")):
        self.hands = tuple(hands)

    def __call__(self, data: dict) -> dict:
        if not any(f"pose.{h}" in data for h in self.hands):
            return data
        out = dict(data)
        parts = []
        for hand in self.hands:
            pose = np.asarray(out.pop(f"pose.{hand}"), dtype=np.float64)
            hand_cmds = np.asarray(out.pop(f"hand.{hand}"), dtype=np.float64)
            if pose.ndim != 2 or pose.shape[-1] != 9:
                raise ValueError(f"pose.{hand}: expected (H+1, 9), got {pose.shape}")
            if hand_cmds.shape != (pose.shape[0] - 1, 6):
                raise ValueError(
                    f"hand.{hand}: expected ({pose.shape[0] - 1}, 6), got {hand_cmds.shape}")
            T = vec9_to_se3(pose)                              # (H+1, 4, 4)
            deltas = se3_to_vec9(np.linalg.inv(T[0]) @ T[1:])  # (H, 9)
            parts.append(np.concatenate([deltas, hand_cmds], axis=-1))
        out["actions"] = np.concatenate(parts, axis=-1).astype(np.float32)
        return out


class BoundaryAwareIndices:
    """Flat-index remap over a frame-indexed dataset laid out episode by
    episode (LeRobot order): global index = episode offset + t.

    Frame t is a VALID datapoint iff `t + H <= length - 1`, OR the episode is
    an `episode_real_end` sub-episode and `allow_terminal_padding` is on
    (repeat-padding then means "hold pose") - AND t is not an `anchor_bad`
    frame. pi0 ignores `action_is_pad`, so this is enforced by index
    remapping. (Strict splitting keeps anchor_bad empty today; honored anyway
    so semantics survive if bridging ever returns.)"""

    def __init__(self, episode_lengths, real_end_flags, action_horizon,
                 allow_terminal_padding, anchor_bad=None):
        lengths = [int(x) for x in episode_lengths]
        flags = [bool(x) for x in real_end_flags]
        if len(lengths) != len(flags):
            raise ValueError(f"{len(lengths)} lengths vs {len(flags)} real_end flags")
        bad = [set(int(t) for t in b) for b in anchor_bad] if anchor_bad \
            else [set()] * len(lengths)
        if len(bad) != len(lengths):
            raise ValueError(f"{len(lengths)} lengths vs {len(bad)} anchor_bad lists")
        h = int(action_horizon)
        valid = []
        offset = 0
        for length, real_end, ep_bad in zip(lengths, flags, bad):
            if real_end and allow_terminal_padding:
                n_valid = length
            else:
                n_valid = max(length - h, 0)
            valid.extend(offset + t for t in range(n_valid) if t not in ep_bad)
            offset += length
        self.total_frames = offset
        self.indices = np.asarray(valid, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


class BoundaryAwareDataset:
    """Wrap any frame-indexable dataset so only valid datapoints are visible."""

    def __init__(self, dataset, indices: BoundaryAwareIndices):
        self._dataset = dataset
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._dataset[self._indices[i]]
