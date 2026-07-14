"""State masking (state_dropout_p): the prompt-side state ablation.

The load-bearing test is the first one — with state_mode="real" the prompt must
be BYTE-IDENTICAL to stock openpi's. Every run trained before this existed used
the stock tokenizer, and a p=0.0 baseline is only comparable to them if the
prompt did not move. If that test fails, the whole experiment is invalid.
"""

import random

import jax
import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _openpi_transforms

from ego2g1 import config as _config
from ego2g1 import data_config as _data_config
from ego2g1 import diagnostics as diag
from ego2g1 import model as ego_model
from ego2g1 import transforms as ego_transforms

MAX_LEN = 200
PROMPT = "put the bottle in the box <<<control_mode>>> end effector <<<control_mode>>>"


def _state(seed: int = 0) -> np.ndarray:
    # normalized state, i.e. what the tokenizer sees (post-Normalize, post-PerSlotRescale)
    return np.clip(np.random.default_rng(seed).normal(0, 0.4, 30), -1, 1).astype(np.float32)


def _tokenize(mode: str, dropout_p: float = 0.0, state: np.ndarray | None = None) -> np.ndarray:
    t = ego_transforms.Ego2G1TokenizePrompt(
        tokenizer=ego_transforms.Ego2G1Tokenizer(MAX_LEN), mode=mode, dropout_p=dropout_p
    )
    state = _state() if state is None else state
    return t({"prompt": PROMPT, "state": state})["tokenized_prompt"]


def test_real_mode_is_byte_identical_to_stock():
    state = _state()
    stock = _openpi_transforms.TokenizePrompt(
        _tokenizer.PaligemmaTokenizer(MAX_LEN), discrete_state_input=True
    )({"prompt": PROMPT, "state": state})

    ours = ego_transforms.Ego2G1TokenizePrompt(
        tokenizer=ego_transforms.Ego2G1Tokenizer(MAX_LEN), mode="real"
    )({"prompt": PROMPT, "state": state})

    np.testing.assert_array_equal(ours["tokenized_prompt"], stock["tokenized_prompt"])
    np.testing.assert_array_equal(ours["tokenized_prompt_mask"], stock["tokenized_prompt_mask"])


def test_blind_prompt_carries_no_state_digits():
    digits = diag.digit_token_ids()
    real = _tokenize("real")
    blind = _tokenize("blind")
    # 30 values -> at least 30 digit tokens in the real prompt, none in the blind one
    assert np.isin(real, digits).sum() >= 30
    assert np.isin(blind, digits).sum() == 0
    # the state is the ONLY difference: two different states -> two different
    # real prompts, but the same blind prompt
    assert not np.array_equal(_tokenize("real", state=_state(1)), real)
    assert np.array_equal(_tokenize("blind", state=_state(1)), blind)


def test_blind_prompt_is_shorter_but_keeps_the_pi05_template():
    tok = ego_transforms.Ego2G1Tokenizer(MAX_LEN)
    blind_str = f"Task: {PROMPT}, State: {ego_transforms.STATE_SENTINEL};\nAction: "
    expected, _ = tok.tokenize_with_state_str(PROMPT, ego_transforms.STATE_SENTINEL)
    np.testing.assert_array_equal(_tokenize("blind"), expected)
    assert "Task:" in blind_str and "State:" in blind_str and "Action:" in blind_str


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 1.0])
def test_dropout_masks_at_the_requested_rate(p):
    blind = _tokenize("blind")
    random.seed(0)  # the transform uses stdlib random (torch seeds it per worker)
    n = 400
    masked = sum(np.array_equal(_tokenize("dropout", dropout_p=p), blind) for _ in range(n))
    assert abs(masked / n - p) < 0.08, f"p={p}: masked {masked}/{n}"


def test_serve_mode_is_derived_from_the_checkpoint_alone():
    def mode(p):
        return _data_config.serve_state_mode(_config.Ego2G1TrainConfig(state_dropout_p=p))

    assert mode(0.0) == "real"      # baseline: state at serve
    assert mode(0.5) == "real"      # dropout is a TRAIN-time regularizer only
    assert mode(1.0) == "blind"     # blind checkpoint must never see a state digit


def test_state_masking_feature_flag_required_only_when_blind():
    def flag(p):
        return _config.Ego2G1TrainConfig(state_dropout_p=p).feature_flags()["state_masking"]

    assert flag(0.0)["required"] is False
    assert flag(0.5)["required"] is False
    assert flag(1.0)["required"] is True
    assert flag(0.5)["p"] == 0.5


def test_dropout_with_discrete_state_input_off_is_rejected():
    # discrete_state_input=False removes the prompt state entirely AND changes the
    # template; masking it is meaningless and the combination is a config error.
    with pytest.raises(ValueError, match="discrete_state_input"):
        _config.Ego2G1TrainConfig(state_dropout_p=0.5, discrete_state_input=False)


def test_shuffle_state_replaces_the_state_from_the_pool():
    pool = np.arange(6 * 30, dtype=np.float32).reshape(6, 30)
    t = ego_transforms.ShuffleState(pool=pool)
    out = t({"state": np.zeros(30, np.float32), "image": "untouched"})
    assert out["image"] == "untouched"
    assert any(np.array_equal(out["state"], row) for row in pool)


def test_attention_probe_splits_task_from_state():
    cfg = ego_model.Ego2G1Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True,
        action_horizon=8, action_dim=6, max_token_len=16, dtype="float32",
    )
    m = cfg.create(jax.random.key(0))
    obs, act = cfg.fake_obs(2), cfg.fake_act(2)

    out = diag.attention_allocation(m, obs, act, state_token_ids=diag.digit_token_ids())
    assert "text/task" in out["group_names"]
    assert "text/state" in out["group_names"]
    assert "text" not in out["group_names"]
    # groups still partition the attention simplex (padding carries ~0 mass)
    np.testing.assert_allclose(out["per_layer"].sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(out["per_slot"].sum(-1), 1.0, atol=1e-5)

    # without the split the old single text group is preserved (back-compat)
    plain = diag.attention_allocation(m, obs, act)
    assert "text" in plain["group_names"]
