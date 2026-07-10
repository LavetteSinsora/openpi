# canonical source: data_extraction/loader/ - keep in sync; pinned by
# data_extraction/tests/test_loader_equivalence.py
"""Ego-Pi (put_bottle_in_box) data pipeline pieces for openpi training.

Self-contained numpy copies of the extraction repo's rot6d/vec9 helpers,
RelativeChunkActions (as an openpi data-transform), boundary-aware indexing,
and the LeRobotEgoPiDataConfig factory. The dataset stores absolute per-tick
flange poses (`pose.left`/`pose.right`, vec9) and hand commands
(`hand.left`/`hand.right`); action chunks are built at load time by
differencing H+1 gathered poses against the chunk anchor.
"""

import dataclasses
import json
import pathlib

import einops
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model

# --------------------------------------------------------------------------
# rot6d / vec9 helpers (copy of data_extraction/common/rot6d.py)
# 6d(R) = concat(R[:, 0], R[:, 1]); vec9(T) = [t (3), 6d(R) (6)].
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# loader pieces (copies of data_extraction/loader/{relative_actions,boundary}.py)
# --------------------------------------------------------------------------


def make_delta_timestamps(action_horizon, fps):
    """delta_timestamps for LeRobotDataset: pose keys gather [0..H]/fps
    (anchor + chunk), hand keys gather [1..H]/fps (commands only)."""
    h = int(action_horizon)
    pose_ts = [k / fps for k in range(h + 1)]
    hand_ts = [k / fps for k in range(1, h + 1)]
    return {"pose.left": pose_ts, "pose.right": pose_ts,
            "hand.left": hand_ts, "hand.right": hand_ts}


@dataclasses.dataclass(frozen=True)
class RelativeChunkActions(_transforms.DataTransformFn):
    """Turn gathered pose/hand chunks into relative action chunks.

    Input sample (numpy): `pose.<hand>` (H+1, 9) vec9, `hand.<hand>` (H, 6)
    for each hand in `hands` order. Output: same sample without those keys,
    plus `actions` (H, (9+6)*len(hands)) f32 where per hand
    delta_k = vec9_to_se3(pose_0)^-1 @ vec9_to_se3(pose_k), k = 1..H.
    A sample without pose keys (e.g. at inference) passes through unchanged.
    """

    hands: tuple = ("left", "right")

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
    """Frame t of an episode is a valid datapoint iff t + H <= length - 1,
    OR the episode is an `episode_real_end` sub-episode and terminal padding
    is allowed (repeat-padding then means "hold pose"). pi0 ignores
    `action_is_pad`, so this is enforced by index remapping."""

    def __init__(self, episode_lengths, real_end_flags, action_horizon,
                 allow_terminal_padding):
        lengths = [int(x) for x in episode_lengths]
        flags = [bool(x) for x in real_end_flags]
        if len(lengths) != len(flags):
            raise ValueError(f"{len(lengths)} lengths vs {len(flags)} real_end flags")
        h = int(action_horizon)
        valid = []
        offset = 0
        for length, real_end in zip(lengths, flags):
            n_valid = length if (real_end and allow_terminal_padding) else max(length - h, 0)
            valid.extend(range(offset, offset + n_valid))
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


def load_extraction_meta(dataset_root) -> dict:
    """Read the extraction_meta.json sidecar written next to the dataset."""
    return json.loads((pathlib.Path(dataset_root) / "extraction_meta.json").read_text())


def make_boundary_aware(dataset, dataset_root, action_horizon):
    """Wrap `dataset` using lerobot meta episode lengths + sidecar real_end
    flags found under `dataset_root`."""
    root = pathlib.Path(dataset_root)
    sidecar = load_extraction_meta(root)
    lengths = {}
    with (root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            lengths[int(rec["episode_index"])] = int(rec["length"])
    n = len(lengths)
    episode_lengths = [lengths[i] for i in range(n)]
    real_end = [bool(sidecar["episodes"][str(i)]["episode_real_end"]) for i in range(n)]
    allow_pad = bool(sidecar["config"].get("allow_terminal_padding", True))
    idx = BoundaryAwareIndices(episode_lengths, real_end, action_horizon, allow_pad)
    return BoundaryAwareDataset(dataset, idx)


# --------------------------------------------------------------------------
# pi0 input/output adapters (modeled on openpi.policies.libero_policy)
# --------------------------------------------------------------------------


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EgoPiInputs(_transforms.DataTransformFn):
    """Repack an ego-pi sample (observation/image, observation/state (30,),
    actions (H, 30)) into pi0 model inputs. Single egocentric camera: it goes
    to base_0_rgb; both wrist slots are zero-padded and masked out."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        mask_padding = self.model_type == _model.ModelType.PI0  # pi0 masks padding images

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class EgoPiOutputs(_transforms.DataTransformFn):
    """Trim model padding back to the 30-dim ego-pi action space."""

    action_dim: int = 30

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


# --------------------------------------------------------------------------
# data config factory (modeled on LeRobotLiberoDataConfig)
# --------------------------------------------------------------------------

# imported lazily in config.py to avoid cycles; this module only depends on
# transforms + models.
def make_egopi_data_config_cls():
    """Deferred import shim: returns the LeRobotEgoPiDataConfig class.

    Defined via a function so this module does not import
    openpi.training.config at import time (config.py imports this module).
    """
    from openpi.training import config as _config

    @dataclasses.dataclass(frozen=True)
    class LeRobotEgoPiDataConfig(_config.DataConfigFactory):
        """Ego-pi put_bottle_in_box dataset (see data_extraction/SPEC.md).

        The dataset stores absolute poses; RelativeChunkActions builds the
        (H, 30) relative action chunks at load time from pose.*/hand.* chunks
        gathered via custom delta_timestamps.
        """

        # control-tick rate of the dataset (cfg.control_hz); used to build
        # the pose/hand delta_timestamps for the model's action horizon.
        fps: int = 30
        hands: tuple = ("left", "right")
        # local dataset root (the directory holding meta/, data/, videos|images/
        # and extraction_meta.json). Required for boundary_aware /
        # expected_config_hash; if None, the dataset comes from the HF cache.
        dataset_root: str | None = None

        @override
        def create(self, assets_dirs: pathlib.Path,
                   model_config: _model.BaseModelConfig) -> _config.DataConfig:
            # Repack: keep the pose/hand chunks (consumed by
            # RelativeChunkActions below) and map dataset keys onto the keys
            # the inference environment will send.
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/state": "state",
                            **{f"pose.{h}": f"pose.{h}" for h in self.hands},
                            **{f"hand.{h}": f"hand.{h}" for h in self.hands},
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = _transforms.Group(
                inputs=[
                    RelativeChunkActions(hands=tuple(self.hands)),
                    EgoPiInputs(model_type=model_config.model_type),
                ],
                outputs=[EgoPiOutputs(action_dim=15 * len(self.hands))],
            )

            model_transforms = _config.ModelTransformFactory()(model_config)

            base = self.create_base_config(assets_dirs, model_config)
            return dataclasses.replace(
                base,
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                custom_delta_timestamps=make_delta_timestamps(
                    model_config.action_horizon, self.fps),
                dataset_root=self.dataset_root or base.dataset_root,
            )

    return LeRobotEgoPiDataConfig
