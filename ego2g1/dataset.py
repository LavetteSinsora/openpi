"""Dataset construction: LeRobot dataset + sidecar assert + boundary remap +
train/val split by REAL episode.

Replaces the reverted fork commit's data_loader.py edits: we build the
dataset ourselves and hand it to stock transform_dataset/TorchDataLoader.
"""

import dataclasses
import json
import pathlib

import numpy as np

from ego2g1 import chunk_math


def load_extraction_meta(dataset_root) -> dict:
    """Read the extraction_meta.json sidecar written next to the dataset."""
    return json.loads((pathlib.Path(dataset_root) / "extraction_meta.json").read_text())


def assert_dataset_compatible(dataset_root, expected_config_hash: str | None, action_horizon: int, fps: int) -> dict:
    """Fail loud before GPU time: sidecar exists, hash matches, and the
    extraction config's horizon/fps agree with the training config."""
    meta = load_extraction_meta(dataset_root)
    if expected_config_hash is not None and meta["config_hash"] != expected_config_hash:
        raise ValueError(
            f"Dataset at {dataset_root} has extraction config_hash {meta['config_hash']}, "
            f"expected {expected_config_hash}. Regenerate the dataset or update the training config."
        )
    cfg = meta["config"]
    if int(cfg["action_horizon"]) < int(action_horizon):
        raise ValueError(f"extraction horizon {cfg['action_horizon']} < training horizon {action_horizon}")
    if float(cfg["control_hz"]) != float(fps):
        raise ValueError(f"extraction control_hz {cfg['control_hz']} != training fps {fps}")
    return meta


def _episode_lengths(dataset_root) -> list[int]:
    lengths = {}
    with (pathlib.Path(dataset_root) / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            lengths[int(rec["episode_index"])] = int(rec["length"])
    return [lengths[i] for i in range(len(lengths))]


@dataclasses.dataclass(frozen=True)
class SplitIndices:
    """Boundary-aware flat indices, split train/val by real source episode."""

    train: chunk_math.BoundaryAwareIndices
    val: chunk_math.BoundaryAwareIndices


def build_split_indices(dataset_root, action_horizon, val_real_episodes=(), allow_terminal_padding=True) -> SplitIndices:
    """Boundary-aware indices per subset. A LeRobot episode belongs to val iff
    its sidecar `source_episode` is in `val_real_episodes` — a real episode
    that was filter-split into several LeRobot episodes can never straddle
    the split."""
    meta = load_extraction_meta(dataset_root)
    lengths = _episode_lengths(dataset_root)
    n = len(lengths)
    eps = [meta["episodes"][str(i)] for i in range(n)]
    known_sources = {e["source_episode"] for e in eps}
    unknown = set(val_real_episodes) - known_sources
    if unknown:
        raise ValueError(f"val_real_episodes not in dataset: {sorted(unknown)}")

    def indices_for(subset_is_val: bool) -> chunk_math.BoundaryAwareIndices:
        # Zero out non-subset episodes' valid ranges by marking every frame
        # anchor_bad; lengths/offsets stay global so flat indices are correct.
        anchor_bad = []
        for e, length in zip(eps, lengths):
            in_val = e["source_episode"] in val_real_episodes
            if in_val == subset_is_val:
                anchor_bad.append(list(e.get("anchor_bad", [])))
            else:
                anchor_bad.append(list(range(length)))
        return chunk_math.BoundaryAwareIndices(
            lengths,
            [bool(e["episode_real_end"]) for e in eps],
            action_horizon,
            allow_terminal_padding,
            anchor_bad=anchor_bad,
        )

    return SplitIndices(train=indices_for(False), val=indices_for(True))


def raw_state_pool(train_config, *, split: str = "val") -> np.ndarray:
    """Every raw 30-dim state of a split's episodes, read straight from the
    parquet — no video decode, so it costs milliseconds. Feeds the shuffled-state
    val pool (ego2g1.transforms.ShuffleState)."""
    import pandas as pd

    root = pathlib.Path(train_config.dataset_root).resolve()
    meta = load_extraction_meta(root)
    val_sources = set(train_config.val_real_episodes)
    frames = []
    for idx_str, e in meta["episodes"].items():
        i = int(idx_str)
        if (e["source_episode"] in val_sources) != (split == "val"):
            continue
        path = root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.parquet"
        frames.append(np.stack(pd.read_parquet(path, columns=["state"])["state"].to_numpy()))
    if not frames:
        raise ValueError(f"split {split!r} has no episodes — cannot build a state pool")
    return np.concatenate(frames).astype(np.float32)


def create_dataset(train_config, model_config, *, split: str = "train"):
    """LeRobot dataset with pose/hand delta_timestamps, wrapped to expose only
    the boundary-valid datapoints of the requested split. Emits raw samples;
    all transforms (including RelativeChunkActions) are applied by stock
    transform_dataset from the DataConfig (ego2g1.data_config)."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    root = pathlib.Path(train_config.dataset_root).resolve()
    assert_dataset_compatible(root, train_config.expected_config_hash,
                              model_config.action_horizon, train_config.fps)

    dataset = lerobot_dataset.LeRobotDataset(
        train_config.repo_id,
        root=str(root),
        delta_timestamps=chunk_math.make_delta_timestamps(model_config.action_horizon, train_config.fps),
    )
    splits = build_split_indices(root, model_config.action_horizon, train_config.val_real_episodes)
    indices = splits.train if split == "train" else splits.val
    if len(indices) == 0:
        raise ValueError(f"split {split!r} has zero valid datapoints")

    # match stock create_torch_dataset's prompt_from_task behavior
    import openpi.transforms as _transforms
    import openpi.training.data_loader as _data_loader

    meta = lerobot_dataset.LeRobotDatasetMetadata(train_config.repo_id, root=str(root))
    wrapped = chunk_math.BoundaryAwareDataset(dataset, indices)
    return _data_loader.TransformedDataset(wrapped, [_transforms.PromptFromLeRobotTask(meta.tasks)])
