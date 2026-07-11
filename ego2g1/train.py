"""Ego2G1 training entrypoint. Run from the openpi root:
    uv run python -m ego2g1.train --exp-name my_run

Reuses scripts/train.py wholesale (init_logging, init_wandb,
init_train_state, train_step, sharding, checkpointing) — the only differences
are: our config dataclass, our dataset/DataConfig construction, per-slot loss
decomposition in the logs, and stamping the checkpoint dir with feature flags
+ the two stats artifacts.
"""

import dataclasses
import functools
import importlib.util
import logging
import pathlib
import sys

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb
from flax.training import common_utils

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _openpi_config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from ego2g1 import config as _config
from ego2g1 import data_config as _data_config
from ego2g1 import dataset as _dataset
from ego2g1 import stamp as _stamp


def _load_stock_train_module():
    """Import scripts/train.py (not a package) for its init/step functions."""
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
        lr_schedule=config.lr_schedule,
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
    data_cfg = _data_config.create_data_config(config, model_config, norm_assets_dir=norm_assets_dir)
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
    stock.init_wandb(train_config, resuming=resuming, enabled=config.wandb_enabled)

    # stamp before training starts so even a crashed run is identifiable;
    # stock save_assets writes pooled norm stats per checkpoint step, the
    # per-slot artifact is copied next to the stamp once (it is step-invariant)
    _stamp.write_stamp(train_config.checkpoint_dir, config, meta["config_hash"])
    from ego2g1 import norm as _norm
    _norm.save_per_slot(train_config.checkpoint_dir / "assets_ego2g1", _norm.load_per_slot(norm_assets_dir))

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
        functools.partial(stock.train_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

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
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(_config.Ego2G1TrainConfig))
