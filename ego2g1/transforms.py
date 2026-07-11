"""Ego2G1 data transforms: pi05 input/output adapters, control-mode prompt,
and the E001 floored per-slot rescale pair.

Transform-stack placement (TRAINING_PLAN.md §3.5, enforced by data_config.py):
inputs:  repack -> [RelativeChunkActions, Ego2G1Inputs] -> Normalize ->
         [PerSlotRescale, AppendControlMode, ResizeImages, TokenizePrompt, Pad]
outputs: [PerSlotRescaleInverse] -> Unnormalize -> [Ego2G1Outputs]
PerSlotRescale is defined in pooled-quantile-normalized units, so it must sit
after Normalize (inputs) / before Unnormalize (outputs).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model

# π0.5 pretraining convention (arXiv 2504.16054: "we add '<<<control_mode>>>
# joint/end effector <<<control_mode>>>' to the text prompt"). openpi has no
# code for this; it must be part of the prompt string. The tokenizer's
# cleaning turns underscores into spaces downstream.
CONTROL_MODE_EEF = "end effector"
CONTROL_MODE_JOINT = "joint"


@dataclasses.dataclass(frozen=True)
class AppendControlMode(_transforms.DataTransformFn):
    """Append the pretraining control-mode marker to the prompt. Must run
    before TokenizePrompt, on both train and inference paths."""

    control_mode: str = CONTROL_MODE_EEF

    def __call__(self, data: dict) -> dict:
        if (prompt := data.get("prompt")) is None:
            raise ValueError("AppendControlMode requires a prompt")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        marker = f"<<<control_mode>>> {self.control_mode} <<<control_mode>>>"
        if marker not in prompt:
            data = {**data, "prompt": f"{prompt} {marker}"}
        return data


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Ego2G1Inputs(_transforms.DataTransformFn):
    """Repack an ego2g1 sample (observation/image, observation/state (30,),
    actions (H, 30)) into pi0 model inputs. Single egocentric camera goes to
    base_0_rgb; both wrist slots are zero-padded and masked out."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Ego2G1Outputs(_transforms.DataTransformFn):
    """Trim model padding back to the 30-dim ego2g1 action space."""

    action_dim: int = 30

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class PerSlotRescale(_transforms.DataTransformFn):
    """E001 forward: actions_out[k, d] = actions_in[k, d] * gain[k, d], where
    gain = sigma_pooled / max(sigma_slot, c * sigma_pooled) (ego2g1.norm).
    Operates on pooled-quantile-NORMALIZED actions of exactly (H, D_real)."""

    gain: np.ndarray  # (H, D_real) float32

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"])
        if actions.shape[-2:] != self.gain.shape:
            raise ValueError(f"actions {actions.shape} vs per-slot gain {self.gain.shape}")
        return {**data, "actions": (actions * self.gain).astype(np.float32)}


@dataclasses.dataclass(frozen=True)
class PerSlotRescaleInverse(_transforms.DataTransformFn):
    """E001 inverse: divide model output by the training gain grid. MANDATORY
    before pooled Unnormalize for E001-trained checkpoints (skipping it
    inflates early slots by up to 1/c in real units). Runs on the padded
    model output (H, D_model >= D_real); pad dims pass through."""

    gain: np.ndarray  # (H, D_real) float32

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        h, d_real = self.gain.shape
        if actions.shape[-2] != h or actions.shape[-1] < d_real:
            raise ValueError(f"actions {actions.shape} vs per-slot gain {self.gain.shape}")
        out = actions.astype(np.float32).copy()
        out[..., :d_real] = out[..., :d_real] / self.gain
        return {**data, "actions": out}
