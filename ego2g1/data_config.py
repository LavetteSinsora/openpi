"""Assemble the openpi DataConfig for ego2g1 (TRAINING_PLAN.md §3.5).

Stock transform_dataset fixes the order:
    repack -> data_transforms.inputs -> Normalize -> model_transforms.inputs
and inference applies outputs in reverse:
    model_transforms.outputs -> Unnormalize -> data_transforms.outputs.

Placement is normalization-critical:
- RelativeChunkActions and Ego2G1Inputs run BEFORE Normalize (and are exactly
  what compute_norm_stats sees).
- PerSlotRescale runs after Normalize (it is defined in pooled-normalized
  units) and before PadStatesAndActions (the gain grid is (H, 30)).
- AppendControlMode runs before TokenizePrompt.
- TokenizePrompt digitizes the NORMALIZED 30-dim state into the pi05 prompt.
"""

import dataclasses
import pathlib

import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

from ego2g1 import chunk_math
from ego2g1 import norm as _norm
from ego2g1 import transforms as _ego_transforms


def create_data_config(
    train_config,
    model_config,
    *,
    norm_assets_dir: pathlib.Path | str,
    skip_norm_stats: bool = False,
) -> _config.DataConfig:
    """Build the full DataConfig. `norm_assets_dir` is where the two stats
    artifacts live: the config assets dir at train time, the checkpoint's
    assets/<asset_id>/ dir at serving (policy.py passes that explicitly)."""
    norm_assets_dir = pathlib.Path(norm_assets_dir)

    norm_stats = None
    per_slot_gain = None
    if not skip_norm_stats:
        norm_stats = _normalize.load(norm_assets_dir)
        per_slot = _norm.load_per_slot(norm_assets_dir)
        sigma_pooled = norm_stats["actions"].std[: train_config.action_dim_actual]
        per_slot_gain = per_slot.gain(train_config.per_slot_floor_c, sigma_pooled)

    data_transforms = _transforms.Group(
        inputs=[
            chunk_math.RelativeChunkActions(hands=tuple(train_config.hands)),
            _ego_transforms.Ego2G1Inputs(model_type=model_config.model_type),
        ],
        outputs=[_ego_transforms.Ego2G1Outputs(action_dim=train_config.action_dim_actual)],
    )

    model_inputs = []
    model_outputs = []
    if per_slot_gain is not None:
        model_inputs.append(_ego_transforms.PerSlotRescale(gain=per_slot_gain))
        model_outputs.append(_ego_transforms.PerSlotRescaleInverse(gain=per_slot_gain))
    model_inputs += [
        _ego_transforms.AppendControlMode(control_mode=train_config.control_mode),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=model_config.discrete_state_input,
        ),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]
    model_transforms = _transforms.Group(inputs=model_inputs, outputs=model_outputs)

    # Repack (dataset-only, never at inference): make dataset samples look
    # like what the robot client sends — image/state under observation/*,
    # pose/hand chunks kept for RelativeChunkActions.
    repack_transforms = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "observation/image": "image",
                    "observation/state": "state",
                    **{f"pose.{h}": f"pose.{h}" for h in train_config.hands},
                    **{f"hand.{h}": f"hand.{h}" for h in train_config.hands},
                    "prompt": "prompt",
                }
            )
        ]
    )

    return _config.DataConfig(
        repo_id=train_config.repo_id,
        asset_id=train_config.repo_id,
        norm_stats=norm_stats,
        repack_transforms=repack_transforms,
        data_transforms=data_transforms,
        model_transforms=model_transforms,
        use_quantile_norm=True,  # pi05
        prompt_from_task=True,
    )
