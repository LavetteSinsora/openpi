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

import numpy as np

import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

from ego2g1 import chunk_math
from ego2g1 import norm as _norm
from ego2g1 import transforms as _ego_transforms


def center_dims_mask(hands: tuple[str, ...]) -> "np.ndarray":
    """Dims that per-slot centering applies to: the EEF-delta block (first 9)
    of each hand's [eef 9 | hand 6]. Hand commands are absolute — never centered."""
    return np.tile(np.repeat([True, False], [9, 6]), len(hands))


def build_per_slot_transforms(train_config, norm_stats, per_slot):
    """The E001 transform pair from stats artifacts (norm-critical, so in one
    place used by both training and serving):
    - degeneracy mask (norm.degenerate_action_dims) drives gain exemption AND
      data-path neutralization;
    - mu_n = per-slot mean mapped into normalized units, zeroed everywhere
      centering does not apply (hand commands, degenerate dims);
    - forward clamps to train_config.model_space_clamp."""
    d_real = train_config.action_dim_actual
    act = norm_stats["actions"]
    deg = _norm.degenerate_action_dims(act, d_real)
    gain = per_slot.gain(train_config.per_slot_floor_c, act.std[:d_real], degenerate_mask=deg)
    mu_n = None
    if train_config.per_slot_center:
        if per_slot.mu_slot is None:
            raise ValueError(
                "per_slot_center=True but the per-slot stats artifact has no mu_slot — "
                "re-run `python -m ego2g1.compute_norm_stats` with this code version"
            )
        q01, q99 = act.q01[:d_real], act.q99[:d_real]
        mu_n = (per_slot.mu_slot[:, :d_real] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mu_n = np.where(center_dims_mask(tuple(train_config.hands)) & ~deg, mu_n, 0.0)
        mu_n = mu_n.astype(np.float32)
    return (
        _ego_transforms.PerSlotRescale(gain=gain, mu_n=mu_n, degenerate_mask=deg,
                                       clamp=train_config.model_space_clamp),
        _ego_transforms.PerSlotRescaleInverse(gain=gain, mu_n=mu_n),
    )


def serve_state_mode(train_config) -> str:
    """The prompt state mode a checkpoint must be SERVED with, derived from the
    one field the stamp carries. A blind checkpoint (p >= 1.0) never saw a state
    digit and must not be shown one; a dropout checkpoint (0 < p < 1) was trained
    to cope without the state but is given the real one at inference — dropout is
    a train-time regularizer, not a deployment mode."""
    return "blind" if train_config.state_dropout_p >= 1.0 else "real"


def create_data_config(
    train_config,
    model_config,
    *,
    norm_assets_dir: pathlib.Path | str,
    per_slot_dir: pathlib.Path | str | None = None,
    skip_norm_stats: bool = False,
    state_mode: str = "real",
    dropout_p: float = 0.0,
    shuffle_state_pool: np.ndarray | None = None,
) -> _config.DataConfig:
    """Build the full DataConfig. `norm_assets_dir` holds norm_stats.json (the
    config assets dir at train time; the checkpoint's assets/<asset_id>/ dir at
    serving). `per_slot_dir` holds per_slot_stats.npz — defaults to
    norm_assets_dir, but serving passes the run-level assets_ego2g1 dir, so the
    two artifacts never have to live in the same directory.

    `state_mode` / `dropout_p` select how the prompt's state segment is built
    (ego2g1.transforms.Ego2G1TokenizePrompt). Callers:
      train loop   -> "dropout", p = train_config.state_dropout_p
      val (real)   -> "real"      | val (blind)    -> "blind"
      val (shuffled) -> "real" + shuffle_state_pool
      serving      -> serve_state_mode(train_config)
    `shuffle_state_pool` is diagnostic-only and must never be set for training."""
    norm_assets_dir = pathlib.Path(norm_assets_dir)
    per_slot_dir = pathlib.Path(per_slot_dir) if per_slot_dir is not None else norm_assets_dir

    norm_stats = None
    per_slot_transforms = None
    if not skip_norm_stats:
        norm_stats = _normalize.load(norm_assets_dir)
        per_slot = _norm.load_per_slot(per_slot_dir)
        per_slot_transforms = build_per_slot_transforms(train_config, norm_stats, per_slot)

    data_inputs = [
        chunk_math.RelativeChunkActions(hands=tuple(train_config.hands)),
        _ego_transforms.Ego2G1Inputs(model_type=model_config.model_type),
    ]
    if shuffle_state_pool is not None:
        # before Normalize: the pool holds RAW states, so the rest of the stack
        # (norm, per-slot neutralization, digitization) treats them identically
        # to a genuine state.
        data_inputs.append(_ego_transforms.ShuffleState(pool=shuffle_state_pool))
    data_transforms = _transforms.Group(
        inputs=data_inputs,
        outputs=[_ego_transforms.Ego2G1Outputs(action_dim=train_config.action_dim_actual)],
    )

    model_inputs = []
    model_outputs = []
    if per_slot_transforms is not None:
        forward, inverse = per_slot_transforms
        model_inputs.append(forward)
        model_outputs.append(inverse)
    if model_config.discrete_state_input:
        # Ego2G1TokenizePrompt owns the prompt string so the state segment can be
        # withheld; with state_mode="real" it is byte-identical to stock.
        tokenize = _ego_transforms.Ego2G1TokenizePrompt(
            tokenizer=_ego_transforms.Ego2G1Tokenizer(model_config.max_token_len),
            mode=state_mode,
            dropout_p=dropout_p,
        )
    else:
        # pi0-style prompt (no state, no Task:/State:/Action: scaffold) — a
        # different prompt TEMPLATE, not just a withheld state. state_mode is
        # moot here; config.__post_init__ forbids combining the two.
        tokenize = _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=False,
        )
    model_inputs += [
        _ego_transforms.AppendControlMode(control_mode=train_config.control_mode),
        _transforms.ResizeImages(224, 224),
        tokenize,
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
