"""Ego2G1TrainConfig: every knob in one frozen dataclass.

Produces stock openpi objects (DataConfig via ego2g1.data_config, TrainConfig
via to_train_config()) but is NOT registered in openpi's _CONFIGS — ego2g1
entrypoints take this dataclass directly through tyro.
"""

import dataclasses
import hashlib
import json
import pathlib
from typing import Literal

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders

from ego2g1 import model as _model
from ego2g1 import transforms as _transforms


@dataclasses.dataclass(frozen=True)
class Ego2G1TrainConfig:
    name: str = "ego2g1_pi05" # name of this config
    exp_name: str = "ego2g1" # name of specific experiment using this config

    # --- data ---
    dataset_root: str = "../../lerobot_datasets/ego2g1/put_bottle_in_box" # directory where dataset lives (data read directly from this)
    repo_id: str = "ego2g1/put_bottle_in_box" # norm stats are written to assets/<name>/<repo_id>/
    expected_config_hash: str | None = None # hash of the data_extraction config expected to use. copy that directly from the dataset you inspected and want to use (extraction_meta.json)
    fps: int = 30 # data's corresponding frequency (how many actions correspond to 1 second of expected execution)
    hands: tuple[str, ...] = ("left", "right") # order of hand in action label (i.e., which hand occupies the first 15-dim of the action)
    val_real_episodes: tuple[str, ...] = () # which episodes are validation episodes and should be held-out for norm stat calculation

    # --- model ---
    action_dim: int = 32  # pi05_base padded width
    action_dim_actual: int = 30 # actual dimension of the action (loss in padded dim is masked)
    action_horizon: int = 50
    control_mode: str = _transforms.CONTROL_MODE_EEF # pi0.5 pretraining appends "<control mode> joint/end effector <control mode>" as text tokens in thhe prompt

    # --- normalization ---
    per_slot_floor_c: float = 0.1 # parameter for per dim, per time-slot normalization
    # action dims allowed to have degenerate stats (norm.check_stats_sanity):
    # left-hand command dims 9..14 (left hand unused in put_bottle_in_box;
    # layout per hand [eef 9 | hand 6] in `hands` order, SPEC.md).
    degenerate_dim_allowlist: tuple[int, ...] = (9, 10, 11, 12, 13, 14)

    # --- train-time RTC ---
    rtc_training: bool = False
    rtc_d_max: int = 16  # maximum expected inference time (expressed in # of timesteps)

    # --- training ---
    batch_size: int = 32
    num_train_steps: int = 20000
    
    log_interval: int = 100 # interval of logging train loss, etc.
    save_interval: int = 1000 # interval of saving model checkpoint (for resuming training. new checkpoint saved, old deleted)
    keep_period: int = 5000 # interval of storing not-deleted checkpoints (for offline diagnostic)
    eval_interval: int = 1000 # interval of running eval (e.g., record loss on validation set), 0 disables 
    eval_num_batches: int = 4
    probe_interval: int = 1000 # interval of running attention allocation probe
    probe_batch_size: int = 2
    
    num_workers: int = 2
    seed: int = 42
    ema_decay: float | None = 0.99
    checkpoint_base_dir: str = "./checkpoints"
    assets_base_dir: str = "./assets"
    weight_loader_params_path: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # --- learning rate: cosine with warmup; the decay horizon is ALWAYS
    # num_train_steps (no separate decay_steps knob — changing the run length
    # automatically rescales the schedule so LR lands on final_lr at the end)
    peak_lr: float = 2.5e-5
    warmup_steps: int = 1_000
    final_lr: float = 2.5e-6  # LR at the last step; openpi convention: peak/10
    fsdp_devices: int = 1
    wandb_enabled: bool = True
    wandb_project: str = "ego2g1"
    resume: bool = False
    overwrite: bool = False

    def __post_init__(self):
        if self.action_dim_actual != 15 * len(self.hands):
            raise ValueError(
                f"action_dim_actual={self.action_dim_actual} != 15*len(hands)={15 * len(self.hands)}"
            )
        if self.warmup_steps >= self.num_train_steps:
            raise ValueError(f"warmup_steps={self.warmup_steps} >= num_train_steps={self.num_train_steps}")

    # --- derived ---

    def lr_schedule(self) -> _optimizer.CosineDecaySchedule:
        return _optimizer.CosineDecaySchedule(
            warmup_steps=self.warmup_steps,
            peak_lr=self.peak_lr,
            decay_steps=self.num_train_steps,
            decay_lr=self.final_lr,
        )

    def model_config(self) -> _model.Ego2G1Pi0Config:
        return _model.Ego2G1Pi0Config(
            pi05=True,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            action_dim_actual=self.action_dim_actual,
            rtc_training=self.rtc_training,
            rtc_d_max=self.rtc_d_max,
        )

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    def weight_loader(self) -> _weight_loaders.WeightLoader:
        return _weight_loaders.CheckpointWeightLoader(self.weight_loader_params_path)

    def config_hash(self) -> str:
        """Hash of every field that affects the produced training data/model."""
        payload = {k: v for k, v in dataclasses.asdict(self).items()
                   if k not in ("exp_name", "wandb_enabled", "wandb_project", "num_workers",
                                "log_interval", "save_interval", "keep_period", "resume", "overwrite",
                                "eval_interval", "eval_num_batches", "probe_interval", "probe_batch_size")}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def feature_flags(self) -> dict:
        """Checkpoint stamp (ego2g1.stamp): serving code must declare support
        for every flag with `required: True`."""
        return {
            "per_slot_rescale": {
                "required": self.per_slot_floor_c < 1.0,
                "floor_c": self.per_slot_floor_c,
            },
            "control_mode_prompt": {"required": True, "mode": self.control_mode},
            "relative_chunk_actions": {"required": True},
            **{k: {"required": False, "value": v}
               for k, v in self.model_config().feature_flags().items()},
        }
