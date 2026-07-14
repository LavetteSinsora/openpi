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
import random

import einops
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

# π0.5 pretraining convention (arXiv 2504.16054: "we add '<<<control_mode>>>
# joint/end effector <<<control_mode>>>' to the text prompt"). openpi has no
# code for this; it must be part of the prompt string. The tokenizer's
# cleaning turns underscores into spaces downstream.
CONTROL_MODE_EEF = "end effector"
CONTROL_MODE_JOINT = "joint"

# The state segment of the pi05 prompt, when the state is withheld. A WORD, not
# a number: the state reaches the model as 30 digit tokens, and every digit
# value (including the mid-range 128 a zeroed state would produce) is an
# ordinary token that occurs in real data — measured on put_bottle_in_box, bin
# 128 appears on 6.7% of ticks on dim 9 alone. A numeric sentinel is therefore
# only jointly rare, so "state absent" would be a 30-way conjunction the model
# must infer, and a partially-learned one reads as "every dim mid-range" — a
# plausible pose, the most misleading fallback available. A word has zero token
# overlap with any digit string and cannot be misread as a state.
STATE_SENTINEL = "unknown"


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


class Ego2G1Tokenizer(_tokenizer.PaligemmaTokenizer):
    """Stock PaligemmaTokenizer + the ability to put an arbitrary STRING in the
    state slot. Stock `tokenize` takes the state as a float array and formats it
    internally, so the state segment cannot be replaced from a preceding
    transform; and passing state=None switches to the pi0 template
    ("{text}\\n", no Task:/State:/Action: scaffold), which is a prompt-format
    change on top of the state ablation. Owning the string is the only way to
    withhold the state while keeping the pi05 template pi05_base pretrained on.
    """

    def tokenize_with_state_str(self, prompt: str, state_str: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)  # noqa: SLF001
        # pad/truncate exactly as stock PaligemmaTokenizer.tokenize does
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens), np.asarray(mask)

    @staticmethod
    def state_str(state: np.ndarray) -> str:
        """Stock digitization: normalized [-1, 1] state -> 256 bins -> digits."""
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized))


@dataclasses.dataclass(frozen=True)
class Ego2G1TokenizePrompt(_transforms.DataTransformFn):
    """Replaces openpi's TokenizePrompt. With mode="real" it is byte-identical
    to stock (pinned by tests/test_state_masking.py) — that identity is what
    keeps a state_dropout_p=0.0 run comparable to runs trained before this
    existed. With the state withheld, the digits become STATE_SENTINEL.

    mode:
      "real"    — always the digitized state (baseline)
      "blind"   — always the sentinel (state_dropout_p >= 1.0)
      "dropout" — per-sample coin flip at probability `dropout_p`
    """

    tokenizer: Ego2G1Tokenizer
    mode: str = "real"
    dropout_p: float = 0.0

    def __post_init__(self):
        if self.mode not in ("real", "blind", "dropout"):
            raise ValueError(f"unknown state mode {self.mode!r}")
        if self.mode == "dropout" and not 0.0 <= self.dropout_p <= 1.0:
            raise ValueError(f"dropout_p={self.dropout_p} must be in [0, 1]")

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        if (prompt := out.pop("prompt", None)) is None:
            raise ValueError("Ego2G1TokenizePrompt requires a prompt")
        if not isinstance(prompt, str):
            prompt = prompt.item()

        # stdlib `random`, NOT np.random: torch seeds Python's random per worker
        # per epoch but does NOT seed numpy, so numpy would hand every dataloader
        # worker the identical dropout sequence.
        masked = self.mode == "blind" or (self.mode == "dropout" and random.random() < self.dropout_p)
        if masked:
            state_str = STATE_SENTINEL
        else:
            if (state := out.get("state")) is None:
                raise ValueError("Ego2G1TokenizePrompt requires a state unless it is masked")
            state_str = self.tokenizer.state_str(state)

        tokens, token_masks = self.tokenizer.tokenize_with_state_str(prompt, state_str)
        return {**out, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class ShuffleState(_transforms.DataTransformFn):
    """Replace the state with a random other tick's. Diagnostic only (the
    val_shuffled pool): a policy that merely USES the state degrades to about
    the state-blind loss when fed a wrong one; a policy that has MEMORIZED the
    state->action map is destroyed by it. Sits before Normalize, so the pool
    holds raw states (ego2g1.dataset.raw_state_pool)."""

    pool: np.ndarray  # (N, 30) raw states

    def __call__(self, data: dict) -> dict:
        return {**data, "state": self.pool[random.randrange(len(self.pool))]}


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
    """E001 forward, on pooled-quantile-NORMALIZED data. In order:

    1. neutralize degenerate dims (degenerate_action_dims mask): actions AND
       state overwritten with -1.0 (where their constant resting value maps),
       so spike-tail outliers (|n| ~ 1e5 on raw dims 13/14) never reach the
       model, at train or serve time;
    2. center: actions -= mu_n (per-slot mean in normalized units; zeroed on
       non-centered dims — hand commands and degenerate dims);
    3. rescale: actions *= gain, gain = sigma_pooled / max(sigma_slot, c*sigma_pooled);
    4. clamp to +-clamp in model space — the airtight bound on target magnitude.

    Actions must be exactly (H, D_real); a sample without an actions key (the
    inference input path) still gets its state neutralized."""

    gain: np.ndarray  # (H, D_real) float32
    mu_n: np.ndarray | None = None  # (H, D_real) f32, zeros where not centered
    degenerate_mask: np.ndarray | None = None  # (D_real,) bool
    clamp: float | None = None

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        d_real = self.gain.shape[1]
        if self.degenerate_mask is not None and "state" in out:
            state = np.asarray(out["state"]).astype(np.float32).copy()
            if state.shape[-1] < d_real:
                raise ValueError(f"state {state.shape} vs degenerate mask ({d_real},)")
            state[..., :d_real][..., self.degenerate_mask] = -1.0
            out["state"] = state
        if "actions" not in out:
            return out
        actions = np.asarray(out["actions"]).astype(np.float64)
        if actions.shape[-2:] != self.gain.shape:
            raise ValueError(f"actions {actions.shape} vs per-slot gain {self.gain.shape}")
        if self.degenerate_mask is not None:
            actions[..., self.degenerate_mask] = -1.0
        if self.mu_n is not None:
            actions = actions - self.mu_n
        actions = actions * self.gain
        if self.clamp is not None:
            actions = np.clip(actions, -self.clamp, self.clamp)
        return {**out, "actions": actions.astype(np.float32)}


@dataclasses.dataclass(frozen=True)
class PerSlotRescaleInverse(_transforms.DataTransformFn):
    """E001 inverse: undo gain then centering on the model output. MANDATORY
    before pooled Unnormalize for E001-trained checkpoints (skipping the gain
    inflates early slots by up to 1/c in real units; skipping mu_n BIASES
    centered dims). Runs on the padded model output (H, D_model >= D_real);
    pad dims pass through. Neutralized dims have gain 1 / mu 0, so the model's
    learned constant flows back to the resting command via Unnormalize."""

    gain: np.ndarray  # (H, D_real) float32
    mu_n: np.ndarray | None = None  # (H, D_real) f32, zeros where not centered

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        h, d_real = self.gain.shape
        if actions.shape[-2] != h or actions.shape[-1] < d_real:
            raise ValueError(f"actions {actions.shape} vs per-slot gain {self.gain.shape}")
        out = actions.astype(np.float32).copy()
        out[..., :d_real] = out[..., :d_real] / self.gain
        if self.mu_n is not None:
            out[..., :d_real] = out[..., :d_real] + self.mu_n
        return {**data, "actions": out}
