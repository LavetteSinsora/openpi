"""Ego2G1 training entrypoint. Run from the openpi root:
    uv run python -m ego2g1.train --exp-name my_run

Reuses scripts/train.py wholesale (init_logging, init_wandb,
init_train_state, sharding, checkpointing). Differences: our config
dataclass, our dataset/DataConfig construction, a train_step that also logs
the E001 per-slot loss decomposition, an in-loop validation eval on held-out
real episodes, and checkpoint stamping (feature flags + both stats artifacts).
"""

import dataclasses
import functools
import importlib.util
import logging
import pathlib
import sys

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import wandb
from flax.training import common_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _openpi_config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from ego2g1 import config as _config
from ego2g1 import data_config as _data_config
from ego2g1 import dataset as _dataset
from ego2g1 import stamp as _stamp

# per-slot loss buckets logged to wandb (E001 eval item 3): early slots are
# the mechanism check for the floored rescale, late slots the control.
SLOT_BUCKETS = {"slots_00_04": (0, 5), "slots_05_24": (5, 25), "slots_25_49": (25, 50)}


def _slot_bucket_means(chunked_loss: jnp.ndarray, action_horizon: int) -> dict[str, jnp.ndarray]:
    out = {}
    for name, (lo, hi) in SLOT_BUCKETS.items():
        hi = min(hi, action_horizon)
        if lo < action_horizon:
            out[name] = jnp.mean(chunked_loss[..., lo:hi])
    return out


@at.typecheck
def train_step(
    config: _openpi_config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Stock scripts/train.py train_step + per-slot loss decomposition in info."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), chunked_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, chunked_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # gradient decomposition by module: which part of the network training is
    # actually moving (SigLIP vision encoder vs 2B prefix expert vs 300M action
    # expert vs the small projection/time-MLP heads). Expert-1 params carry the
    # "_1" name suffix (cf. Pi0Config.get_freeze_filter).
    _img = nnx_utils.PathRegex(".*img.*")
    _llm = nnx_utils.PathRegex(".*llm.*")
    _expert1 = nnx_utils.PathRegex(".*llm.*_1.*")
    grad_groups = {
        "grad_norm/siglip": _img,
        "grad_norm/prefix_expert": nnx.All(_llm, nnx.Not(_expert1)),
        "grad_norm/action_expert": _expert1,
        "grad_norm/heads": nnx.All(nnx.Not(_img), nnx.Not(_llm)),
    }

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **{k: optax.global_norm(grads.filter(f)) for k, f in grad_groups.items()},
        **{f"loss/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.model.action_horizon).items()},
    }
    return new_state, info


@at.typecheck
def eval_step(
    config: _openpi_config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """compute_loss on a val batch, eval mode, FIXED rng: the same (t, noise)
    draws every eval, so the curve is comparable across steps. Uses EMA params
    when available (those are what get served)."""
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    chunked_loss = model.compute_loss(rng, observation, actions, train=False)
    return {
        "val/loss": jnp.mean(chunked_loss),
        **{f"val/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.model.action_horizon).items()},
    }


def _attention_probe(config: _config.Ego2G1TrainConfig, state: training_utils.TrainState, val_batch) -> dict:
    """Attention allocation of action tokens on a small fixed probe batch
    (ego2g1.diagnostics), reduced to wandb scalars: mass per token group,
    overall and at the first/last layer, plus mean attention entropy.
    Eager (un-jitted); uses EMA params like eval_step.

    Always run on the REAL-state val pool: on a blind batch there is no state to
    attend to and attn/text_state is 0 by construction."""
    from ego2g1 import diagnostics as _diagnostics

    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    obs, actions = jax.tree.map(lambda x: x[: config.probe_batch_size], val_batch)
    out = _diagnostics.attention_allocation(
        model, obs, actions, state_token_ids=_diagnostics.digit_token_ids()
    )
    per_layer = out["per_layer"]  # (L, G)
    payload = {}
    means = {}
    for g, name in enumerate(out["group_names"]):
        key = name.replace("/", "_")
        means[name] = float(per_layer[:, g].mean())
        payload[f"attn/{key}"] = means[name]
        payload[f"attn_first_layer/{key}"] = float(per_layer[0, g])
        payload[f"attn_last_layer/{key}"] = float(per_layer[-1, g])
    payload["attn/entropy"] = float(out["entropy_per_layer"].mean())
    # The attention-side counterpart of val/state_gap: of the attention the action
    # tokens spend on CONDITIONING (state digits vs task words vs pixels), what
    # share goes to the state? Climbing here + a widening val/state_gap is the
    # signature of a policy solving the task from proprioception.
    cond = {k: v for k, v in means.items() if k.startswith(("text/", "img/"))}
    denom = sum(cond.values())
    if denom > 0 and "text/state" in cond:
        payload["attn/state_share"] = cond["text/state"] / denom
    return payload


def _save_best(
    manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    val_loss: float,
) -> None:
    """Save params + assets ONLY (no train_state) to the best-checkpoint manager.

    The best checkpoint is for evaluation and serving, never for resuming, and
    ego2g1.serve.create_policy reads exactly <step>/params + the assets. Dropping
    the optimizer state (AdamW moments, ~2x the params) makes it about a third
    the size of a full checkpoint, which is what makes re-crowning it at every
    eval affordable. _split_params hands back the EMA params when EMA is on —
    the same weights eval_step scored, so the metric and the artifact agree."""

    def save_assets(directory: epath.Path):
        data_config = data_loader.data_config()
        if data_config.norm_stats is not None and data_config.asset_id is not None:
            import openpi.shared.normalize as _normalize_mod

            _normalize_mod.save(directory / data_config.asset_id, data_config.norm_stats)

    with at.disable_typechecking():
        _, params = _checkpoints._split_params(state)  # noqa: SLF001
    manager.save(
        step,
        {"assets": save_assets, "params": {"params": params}},
        metrics={"val_loss": float(val_loss)},
    )


def _load_stock_train_module():
    """Import scripts/train.py (not a package) for its init functions."""
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_scripts_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_openpi_train_config(config: _config.Ego2G1TrainConfig, data_cfg) -> _openpi_config.TrainConfig:
    """Adapter so stock init_train_state/checkpointing code can be reused.
    The `data` factory just returns our prebuilt DataConfig."""

    @dataclasses.dataclass(frozen=True)
    class _Fixed(_openpi_config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return data_cfg

    return _openpi_config.TrainConfig(
        name=config.name,
        exp_name=config.exp_name,
        model=config.model_config(),
        weight_loader=config.weight_loader(),
        data=_Fixed(repo_id=config.repo_id),
        optimizer=config.optimizer,
        lr_schedule=config.lr_schedule(),
        batch_size=config.batch_size,
        num_train_steps=config.num_train_steps,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        keep_period=config.keep_period,
        num_workers=config.num_workers,
        seed=config.seed,
        ema_decay=config.ema_decay,
        checkpoint_base_dir=config.checkpoint_base_dir,
        assets_base_dir=config.assets_base_dir,
        fsdp_devices=config.fsdp_devices,
        wandb_enabled=config.wandb_enabled,
        project_name=config.wandb_project,
        resume=config.resume,
        overwrite=config.overwrite,
    )


def main(config: _config.Ego2G1TrainConfig):
    stock = _load_stock_train_module()
    stock.init_logging()

    model_config = config.model_config()
    meta = _dataset.assert_dataset_compatible(
        config.dataset_root, config.expected_config_hash, model_config.action_horizon, config.fps
    )
    norm_assets_dir = config.assets_dirs / config.repo_id
    # TRAIN stack: "dropout" mode covers all three regimes — p=0.0 never masks
    # (byte-identical to the stock prompt), p=1.0 always masks, 0<p<1 flips a coin.
    data_cfg = _data_config.create_data_config(
        config, model_config, norm_assets_dir=norm_assets_dir,
        state_mode="dropout", dropout_p=config.state_dropout_p,
    )
    train_config = _to_openpi_train_config(config, data_cfg)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {config.batch_size} % devices {jax.device_count()} != 0")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        train_config.checkpoint_dir, keep_period=config.keep_period,
        overwrite=config.overwrite, resume=config.resume,
    )
    # Tier 3 of checkpoint retention. The other two come from the manager above:
    # max_to_keep=1 (latest, resumable) and keep_period (permanent archive). A
    # single manager cannot also track the best, because orbax's best_fn retargets
    # max_to_keep at the best checkpoints and the guaranteed-latest is lost — hence
    # a second manager in its own directory.
    # NOTE on --resume: this manager starts cold, so it can re-crown a checkpoint
    # worse than a pre-crash best. Acceptable; the archive still has the periodic ones.
    best_manager = ocp.CheckpointManager(
        epath.Path(train_config.checkpoint_dir) / "best",
        item_handlers={"assets": _checkpoints.CallbackHandler(), "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            best_fn=lambda metrics: float(metrics["val_loss"]),
            best_mode="min",
            keep_checkpoints_without_metrics=False,
            create=True,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    stock.init_wandb(train_config, resuming=resuming, enabled=config.wandb_enabled)

    # stamp before training starts so even a crashed run is identifiable;
    # stock save_assets writes pooled norm stats per checkpoint step, the
    # per-slot artifact is copied next to the stamp once (it is step-invariant)
    _stamp.write_stamp(train_config.checkpoint_dir, config, meta["config_hash"])
    from ego2g1 import norm as _norm
    per_slot = _norm.load_per_slot(norm_assets_dir)
    _norm.save_per_slot(train_config.checkpoint_dir / "assets_ego2g1", per_slot)
    # best/ is its own run dir as far as serving is concerned: resolve_run_dir
    # looks in the checkpoint dir and then its PARENT, so best/<step> resolves to
    # best/ — which needs its own stamp + per-slot stats or create_policy refuses it.
    _stamp.write_stamp(train_config.checkpoint_dir / "best", config, meta["config_hash"])
    _norm.save_per_slot(train_config.checkpoint_dir / "best" / "assets_ego2g1", per_slot)

    torch_dataset = _dataset.create_dataset(config, model_config, split="train")
    transformed = _data_loader.transform_dataset(torch_dataset, data_cfg)
    torch_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    data_loader = _data_loader.DataLoaderImpl(data_cfg, torch_loader)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Three fixed val pools, loaded once (the val split is small). All three draw
    # the SAME ticks in the same order (shuffle with the same seed), so the only
    # difference between the curves is what the prompt says about the state:
    #   val          real state digits          — the deployed condition
    #   val_blind    "State: unknown"           — how well it does with NO state
    #   val_shuffled a random other tick's state — how badly a WRONG state hurts
    # eval uses a fixed rng, so every curve is comparable across steps too.
    val_pools: dict[str, list] = {}
    if config.eval_interval > 0 and config.val_real_episodes:
        val_dataset = _dataset.create_dataset(config, model_config, split="val")
        pool_kwargs = {
            "val": {"state_mode": "real"},
            "val_blind": {"state_mode": "blind"},
            "val_shuffled": {
                "state_mode": "real",
                "shuffle_state_pool": _dataset.raw_state_pool(config, split="val"),
            },
        }
        for prefix, kwargs in pool_kwargs.items():
            pool_cfg = _data_config.create_data_config(
                config, model_config, norm_assets_dir=norm_assets_dir, **kwargs
            )
            val_loader = _data_loader.TorchDataLoader(
                _data_loader.transform_dataset(val_dataset, pool_cfg),
                local_batch_size=config.batch_size // jax.process_count(),
                sharding=data_sharding,
                shuffle=True,
                num_batches=config.eval_num_batches,
                num_workers=0,
                seed=config.seed,
            )
            val_pools[prefix] = list(iter(_data_loader.DataLoaderImpl(pool_cfg, val_loader)))
        logging.info(f"Loaded {config.eval_num_batches} fixed val batches x {len(val_pools)} pools "
                     f"({len(val_dataset)} datapoints from {len(config.val_real_episodes)} real episodes)")
    elif config.eval_interval > 0:
        logging.warning("eval_interval > 0 but val_real_episodes is empty — validation disabled")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = stock.init_train_state(train_config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(eval_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    eval_rng = jax.random.key(config.seed + 1)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step,
                     total=config.num_train_steps, dynamic_ncols=True)

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if val_pools and config.eval_interval > 0 and (step % config.eval_interval == 0 or step == config.num_train_steps - 1):
            val_reduced = {}
            for prefix, batches in val_pools.items():
                with sharding.set_mesh(mesh):
                    # same fold_in sequence for every pool: identical (t, noise)
                    # draws, so the pools differ ONLY in the prompt's state segment
                    val_infos = [peval_step(jax.random.fold_in(eval_rng, i), train_state, vb)
                                 for i, vb in enumerate(batches)]
                reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(val_infos)))
                val_reduced.update({k.replace("val/", f"{prefix}/", 1): v for k, v in reduced.items()})
            # state_gap: how much the policy LEANS on proprioception (big = it is
            # solving the task from state, which is what we are trying to break).
            # state_trust: how much a WRONG state hurts beyond having none at all
            # (big = it does not merely use the state, it is enslaved to it).
            if {"val/loss", "val_blind/loss", "val_shuffled/loss"} <= val_reduced.keys():
                val_reduced["val/state_gap"] = val_reduced["val_blind/loss"] - val_reduced["val/loss"]
                val_reduced["val/state_trust"] = val_reduced["val_shuffled/loss"] - val_reduced["val_blind/loss"]
            pbar.write(f"Step {step} [val]: " + ", ".join(f"{k}={v:.4f}" for k, v in val_reduced.items()))
            wandb.log(val_reduced, step=step)
            _save_best(best_manager, train_state, data_loader, step, float(val_reduced["val/loss"]))
        if val_pools and config.probe_interval > 0 and (step % config.probe_interval == 0 or step == config.num_train_steps - 1):
            wandb.log(_attention_probe(config, train_state, val_pools["val"][0]), step=step)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint managers to finish")
    checkpoint_manager.wait_until_finished()
    best_manager.wait_until_finished()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(_config.Ego2G1TrainConfig))
