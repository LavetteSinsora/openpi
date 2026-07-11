"""Compute both norm-stats artifacts over the TRAIN split (TRAINING_PLAN.md §3.6).

Mirrors scripts/compute_norm_stats.py semantics (repack + data transforms,
i.e. exactly what precedes Normalize in the training stack) and additionally
accumulates the E001 per-slot sigma over the (H, D_real) action grid.

Run from the openpi root:
    uv run python -m ego2g1.compute_norm_stats
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as _normalize

from ego2g1 import config as _config
from ego2g1 import data_config as _data_config
from ego2g1 import dataset as _dataset
from ego2g1 import norm as _norm


class _PerSlotRunning:
    """Streaming per-(slot, dim) mean/variance (Chan parallel update)."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, batch: np.ndarray) -> None:  # (N, H, D)
        n = batch.shape[0]
        bmean = batch.mean(axis=0)
        bm2 = ((batch - bmean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, bmean, bm2
            return
        delta = bmean - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * (n / tot)
        self.m2 = self.m2 + bm2 + delta**2 * (self.count * n / tot)
        self.count = tot

    def sigma(self) -> np.ndarray:
        if self.count < 2:
            raise ValueError("need at least 2 samples")
        return np.sqrt(np.maximum(self.m2 / self.count, 0.0))


def main(config: _config.Ego2G1TrainConfig, batch_size: int = 128):
    model_config = config.model_config()
    dataset = _dataset.create_dataset(config, model_config, split="train")
    meta = _dataset.load_extraction_meta(config.dataset_root)

    data_config = _data_config.create_data_config(
        config, model_config, norm_assets_dir=config.assets_dirs / config.repo_id, skip_norm_stats=True
    )
    transforms = [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]

    pooled = {"state": _normalize.RunningStats(), "actions": _normalize.RunningStats()}
    per_slot = _PerSlotRunning()

    def flush(buf):
        actions = np.stack([s["actions"] for s in buf])  # (N, H, D_real)
        states = np.stack([s["state"] for s in buf])
        pooled["actions"].update(actions.reshape(-1, actions.shape[-1]))
        pooled["state"].update(states.reshape(-1, states.shape[-1]))
        per_slot.update(actions)

    buf = []
    for i in tqdm.tqdm(range(len(dataset)), desc="Computing stats (train split)"):
        sample = dataset[i]
        for t in transforms:
            sample = t(sample)
        buf.append({"actions": np.asarray(sample["actions"], dtype=np.float64),
                    "state": np.asarray(sample["state"], dtype=np.float64)})
        if len(buf) == batch_size:
            flush(buf)
            buf = []
    if buf:
        flush(buf)

    norm_stats = {k: v.get_statistics() for k, v in pooled.items()}
    per_slot_stats = _norm.PerSlotStats(
        sigma_slot=per_slot.sigma(),
        provenance={
            "extraction_config_hash": meta["config_hash"],
            "ego2g1_config_hash": config.config_hash(),
            "num_datapoints": per_slot.count,
            "val_real_episodes": list(config.val_real_episodes),
        },
    )

    problems = _norm.check_stats_sanity(norm_stats, per_slot_stats, config.degenerate_dim_allowlist)
    if problems:
        raise SystemExit("Stats sanity check FAILED:\n  " + "\n  ".join(problems))

    output_dir = config.assets_dirs / config.repo_id
    print(f"Writing pooled norm_stats.json and {_norm.PER_SLOT_FILENAME} to: {output_dir}")
    _normalize.save(output_dir, norm_stats)
    _norm.save_per_slot(output_dir, per_slot_stats)

    # eyeball report: per-slot sigma + effective E001 boost
    sigma_pooled = norm_stats["actions"].std[: config.action_dim_actual]
    gain = per_slot_stats.gain(config.per_slot_floor_c, sigma_pooled)
    with np.printoptions(precision=2, suppress=True, linewidth=200):
        print(f"sigma_slot (H, {config.action_dim_actual}), slots 0/9/24/49:")
        for k in (0, 9, 24, 49):
            print(f"  slot {k:2d}: {per_slot_stats.sigma_slot[k]}")
        print(f"gain grid (c={config.per_slot_floor_c}), max per slot:")
        print(f"  {gain.max(axis=1)}")


if __name__ == "__main__":
    main(tyro.cli(_config.Ego2G1TrainConfig))
