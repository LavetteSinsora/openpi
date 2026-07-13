"""Serving-side policy construction. The ONLY supported way to serve an
ego2g1 checkpoint — it enforces the stamp guard and runs the exact
training-time transform stack (TRAINING_PLAN.md §4): the per-slot gain grid's
inverse MUST run before pooled Unnormalize (else early slots inflate up to
1/c in real units).

Norm-stats resolution (see `resolve_norm_assets`), in order:
1. an explicit `assets_dir=` (caller knows best);
2. the checkpoint's OWN copies — `<step>/assets/<repo_id>/norm_stats.json` +
   `<run>/assets_ego2g1/per_slot_stats.npz`. Preferred: they travel with the
   checkpoint and are provably the ones it trained with;
3. the training-time assets dir `<assets_base_dir>/<name>/<repo_id>` that
   compute_norm_stats writes and train.py reads (resolved from the CWD, so run
   from the openpi root). Used only when the checkpoint carries no copies —
   WARNS, because a live assets dir may have been recomputed since training.
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


def resolve_run_dir(checkpoint_dir: str | pathlib.Path) -> pathlib.Path:
    """Locate the run-level artifacts for a checkpoint.

    Orbax saves each step under <run>/<step>/{params,assets,...}, while
    train.py writes the stamp and per-slot artifact once at the run root.
    `create_policy` takes the STEP dir (it holds params); the stamp lives one
    level up. Accept either level so a flat copied-out checkpoint also works.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if (checkpoint_dir / _stamp.STAMP_FILENAME).exists():
        return checkpoint_dir
    if (checkpoint_dir.parent / _stamp.STAMP_FILENAME).exists():
        return checkpoint_dir.parent
    return checkpoint_dir  # let read_stamp raise its diagnostic


def config_from_stamp(stamp: dict) -> _config.Ego2G1TrainConfig:
    """Rebuild the training config from a stamp (single source of truth).
    optimizer/lr_schedule are dropped (not needed at serving); JSON lists are
    restored to the tuples the dataclass expects."""
    cfg_dict = dict(stamp["ego2g1_config"])
    for k in ("optimizer", "lr_schedule"):
        cfg_dict.pop(k, None)
    for k in ("hands", "val_real_episodes", "degenerate_dim_allowlist"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    # Checkpoints stamped before a feature existed were trained WITHOUT it —
    # missing keys must resolve to the legacy behavior, never to a new default.
    cfg_dict.setdefault("per_slot_center", False)
    cfg_dict.setdefault("model_space_clamp", None)
    return _config.Ego2G1TrainConfig(**cfg_dict)


NORM_STATS_FILENAME = "norm_stats.json"


def resolve_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir=None
                        ) -> tuple[pathlib.Path, pathlib.Path]:
    """-> (pooled_dir with norm_stats.json, per_slot_dir with per_slot_stats.npz).

    The two artifacts may live in different directories (the checkpoint keeps
    pooled stats per step but the per-slot grid once at the run root), so they
    are resolved independently and NOTHING is written into the checkpoint.
    Resolution order documented in the module docstring."""
    from ego2g1 import norm as _norm

    pooled_ck = checkpoint_dir / "assets" / train_config.repo_id
    per_slot_ck_run = run_dir / "assets_ego2g1"
    train_assets = train_config.assets_dirs / train_config.repo_id
    searched = []

    def pick(filename, candidates):
        for d in candidates:
            searched.append(d / filename)
            if (d / filename).exists():
                return d
        return None

    if assets_dir is not None:
        d = pathlib.Path(assets_dir)
        missing = [f for f in (NORM_STATS_FILENAME, _norm.PER_SLOT_FILENAME) if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"--assets-dir {d} is missing {missing}")
        return d, d

    pooled = pick(NORM_STATS_FILENAME, [pooled_ck, train_assets])
    per_slot = pick(_norm.PER_SLOT_FILENAME, [pooled_ck, per_slot_ck_run, train_assets])
    if pooled is None or per_slot is None:
        raise FileNotFoundError(
            "norm assets not found. Searched:\n  " + "\n  ".join(str(p) for p in searched) +
            "\nRun `python -m ego2g1.compute_norm_stats` from the openpi root, or pass --assets-dir."
        )
    if pooled == train_assets or per_slot == train_assets:
        print(f"WARNING: falling back to the training assets dir {train_assets} (the checkpoint does not "
              "carry its own copies). Confirm these stats are the ones this checkpoint trained with — "
              "they are not pinned to it.")
    return pooled, per_slot


def create_policy(checkpoint_dir: str | pathlib.Path, *, default_prompt: str | None = None,
                  assets_dir: str | pathlib.Path | None = None) -> _policy.Policy:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    run_dir = resolve_run_dir(checkpoint_dir)
    stamp = _stamp.check_supported(run_dir)

    train_config = config_from_stamp(stamp)
    model_config = train_config.model_config()

    model = model_config.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    pooled_dir, per_slot_dir = resolve_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir)
    data_cfg = _data_config.create_data_config(
        train_config, model_config, norm_assets_dir=pooled_dir, per_slot_dir=per_slot_dir,
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
