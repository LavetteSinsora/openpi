"""Serving-side policy construction. The ONLY supported way to serve an
ego2g1 checkpoint — it enforces the stamp guard and loads both stats
artifacts from the checkpoint itself (never from a live assets dir), so the
policy always runs the exact training-time transform stack
(TRAINING_PLAN.md §4): the checkpoint carries the per-slot gain grid whose
inverse MUST run before pooled Unnormalize (else early slots inflate up to
1/c in real units).
"""

import dataclasses
import json
import pathlib

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.training.checkpoints as _checkpoints

from ego2g1 import config as _config
from ego2g1 import data_config as _data_config
from ego2g1 import stamp as _stamp


def create_policy(checkpoint_dir: str | pathlib.Path, *, default_prompt: str | None = None) -> _policy.Policy:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    stamp = _stamp.check_supported(checkpoint_dir)

    # Rebuild the exact training config from the stamp (single source of truth).
    cfg_dict = dict(stamp["ego2g1_config"])
    for k in ("optimizer", "lr_schedule"):  # nested configs; not needed at serving
        cfg_dict.pop(k, None)
    for k in ("hands", "val_real_episodes", "degenerate_dim_allowlist"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    train_config = _config.Ego2G1TrainConfig(**cfg_dict)
    model_config = train_config.model_config()

    model = model_config.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    # Both stats artifacts from the checkpoint: pooled from the stock assets
    # dir, per-slot from the ego2g1 assets dir written by train.py.
    pooled_dir = checkpoint_dir / "assets" / train_config.repo_id
    per_slot_dir = checkpoint_dir / "assets_ego2g1"
    data_cfg = _data_config.create_data_config(
        train_config, model_config,
        norm_assets_dir=_merged_assets(pooled_dir, per_slot_dir),
    )

    import openpi.transforms as transforms

    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ],
        output_transforms=[
            *data_cfg.model_transforms.outputs,
            transforms.Unnormalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.data_transforms.outputs,
        ],
        metadata={"ego2g1_stamp": {k: stamp[k] for k in ("feature_flags", "ego2g1_config_hash",
                                                          "extraction_config_hash", "openpi_commit")}},
    )


def _merged_assets(pooled_dir: pathlib.Path, per_slot_dir: pathlib.Path) -> pathlib.Path:
    """data_config expects one dir with both artifacts; symlink them together."""
    from ego2g1 import norm as _norm

    if (pooled_dir / _norm.PER_SLOT_FILENAME).exists():
        return pooled_dir
    link = pooled_dir / _norm.PER_SLOT_FILENAME
    target = per_slot_dir / _norm.PER_SLOT_FILENAME
    if not target.exists():
        raise FileNotFoundError(
            f"per-slot stats not found at {target}; this checkpoint cannot be served with E001 semantics"
        )
    link.symlink_to(target)
    return pooled_dir
