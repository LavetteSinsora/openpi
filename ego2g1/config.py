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
    name: str = "ego2g1_pi05"
    exp_name: str = "ego2g1"

    # --- data ---
    # dataset root dir (holds meta/, data/, videos/ and extraction_meta.json)
    dataset_root: str = "../../lerobot_datasets/ego2g1/put_bottle_in_box"
    repo_id: str = "ego2g1/put_bottle_in_box"
    # data_extraction config hash this run expects; asserted against the
    # sidecar before anything else. ALWAYS read from the sidecar of the
    # dataset actually being trained on — never trust remembered values.
    expected_config_hash: str | None = None
    fps: int = 30
    hands: tuple[str, ...] = ("left", "right")
    # held-out REAL episodes (sidecar `source_episode` values); fixed list,
    # never re-rolled. Empty = no split yet (norm stats then use everything).
    val_real_episodes: tuple[str, ...] = ()

    # --- model ---
    action_dim: int = 32  # pi05_base padded width
    action_dim_actual: int = 30
    action_horizon: int = 50
    # prompt: π0.5 pretraining control-mode marker (appended, see transforms)
    control_mode: str = _transforms.CONTROL_MODE_EEF

    # --- E001 floored per-slot rescale ---
    # c=1.0 reproduces stock pooled behavior exactly; decided start: 0.1.
    per_slot_floor_c: float = 0.1
    # action dims allowed to have degenerate stats (norm.check_stats_sanity):
    # left-hand command dims 12..17 (left hand unused in put_bottle_in_box).
    degenerate_dim_allowlist: tuple[int, ...] = (12, 13, 14, 15, 16, 17)

    # --- train-time RTC (phase 1: off; code feature-complete) ---
    rtc_training: bool = False
    rtc_d_max: int = 16  # provisional 4060 estimate, TRAINING_PLAN.md §1

    # --- training ---
    batch_size: int = 32
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int = 5000
    num_workers: int = 2
    seed: int = 42
    ema_decay: float | None = 0.99
    checkpoint_base_dir: str = "./checkpoints"
    assets_base_dir: str = "./assets"
    weight_loader_params_path: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
        )
    )
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

    # --- derived ---

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
                                "log_interval", "save_interval", "keep_period", "resume", "overwrite")}
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
