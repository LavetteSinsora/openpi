"""Ego2G1Policy: sampler dispatch, and the prefix's route into model space.

Built on the `dummy` gemma variant, so this runs on CPU with no checkpoint. What
it pins is the wiring, not the weights:

  * no prev_chunk -> the stock path, untouched;
  * prev_chunk -> it rides the SAME input transform chain as a training action
    array, so Normalize/PerSlotRescale land each row at its DESTINATION slot's
    constants. Reusing the previous chunk's model-space tensor instead would be
    wrong by the ratio of the two slots' sigmas — up to 10x near slot 0.
"""

import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms

from ego2g1 import model as ego_model
from ego2g1 import transforms as ego_transforms
from ego2g1.serve import policy as _policy
from ego2g1.serve import rtc as _rtc

H, D_REAL, D_MODEL = 8, 6, 8

_KW = dict(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    pi05=True,
    action_horizon=H,
    action_dim=D_MODEL,
    max_token_len=16,
    dtype="float32",
)


class _Inputs(_transforms.DataTransformFn):
    """Minimal stand-in for Ego2G1Inputs at this toy width. Crucially it preserves
    an `actions` key, which is what lets the RTC prefix ride the chain."""

    def __call__(self, data: dict) -> dict:
        img = ego_transforms._parse_image(data["observation/image"])
        out = {
            "state": np.asarray(data["observation/state"], np.float32),
            "image": {"base_0_rgb": img,
                      "left_wrist_0_rgb": np.zeros_like(img),
                      "right_wrist_0_rgb": np.zeros_like(img)},
            "image_mask": {"base_0_rgb": np.True_,
                           "left_wrist_0_rgb": np.False_,
                           "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            out["actions"] = data["actions"]
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


class _SpyPolicy(_policy.Ego2G1Policy):
    """Records which sampler ran and what prefix it was handed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []
        for name in ("_sample_actions", "_sample_guided", "_sample_pinned"):
            self._wrap(name)

    def _wrap(self, name):
        inner = getattr(self, name)

        def spy(*args, **kwargs):
            self.seen.append({"sampler": name, "args": args, "kwargs": kwargs})
            return inner(*args, **kwargs)

        setattr(self, name, spy)


def _make(rtc_training=False, rtc=None):
    extra = {"rtc_d_max": 4} if rtc_training else {}  # must be < action_horizon (8 here)
    cfg = ego_model.Ego2G1Pi0Config(
        **_KW, action_dim_actual=D_REAL, rtc_training=rtc_training, **extra
    )
    model = cfg.create(__import__("jax").random.key(0))

    gain = np.ones((H, D_REAL), np.float32)
    return _SpyPolicy(
        model,
        rtc=rtc or _rtc.RTCConfig(num_steps=2),
        rtc_training=rtc_training,
        action_horizon=H,
        transforms=[
            _Inputs(),
            ego_transforms.PerSlotRescale(gain=gain),
            ego_transforms.AppendControlMode(),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(_KW["max_token_len"]),
                discrete_state_input=True,   # as in the real chain (data_config.py)
            ),
            _transforms.PadStatesAndActions(D_MODEL),
        ],
        output_transforms=[ego_transforms.Ego2G1Outputs(action_dim=D_REAL)],
    )


def _obs():
    return {
        "observation/image": np.zeros((32, 32, 3), np.uint8),
        "observation/state": np.zeros(D_REAL, np.float32),
        "prompt": "put the bottle in the box",
    }


def test_no_prefix_takes_the_stock_path():
    p = _make()
    out = p.infer(_obs())
    assert out["rtc"]["sampler"] == "plain"
    assert [s["sampler"] for s in p.seen] == ["_sample_actions"]
    assert out["actions"].shape == (H, D_REAL)


def test_prefix_on_a_plain_checkpoint_selects_guided():
    p = _make(rtc_training=False)
    out = p.infer(_obs(), )  # warm up JIT on the plain path first
    p.seen.clear()

    out = p.infer({**_obs(), "prev_chunk": np.zeros((H, D_REAL), np.float32), "d": 3})
    assert out["rtc"]["sampler"] == "guided"
    assert out["rtc"]["d"] == 3
    assert [s["sampler"] for s in p.seen] == ["_sample_guided"]


def test_prefix_on_an_rtc_trained_checkpoint_selects_pinned():
    """Same client request, different checkpoint -> different sampler. This is what
    makes a future rtc_training retrain a server-side swap the robot never sees."""
    p = _make(rtc_training=True)
    out = p.infer({**_obs(), "prev_chunk": np.zeros((H, D_REAL), np.float32), "d": 2})
    assert out["rtc"]["sampler"] == "pinned"
    assert [s["sampler"] for s in p.seen] == ["_sample_pinned"]


def test_rtc_disabled_ignores_the_prefix():
    p = _make(rtc=_rtc.RTCConfig(enabled=False, num_steps=2))
    out = p.infer({**_obs(), "prev_chunk": np.zeros((H, D_REAL), np.float32), "d": 4})
    assert out["rtc"]["sampler"] == "plain"
    assert [s["sampler"] for s in p.seen] == ["_sample_actions"]


def test_prefix_reaches_the_sampler_in_padded_model_space():
    """The prefix must arrive as (batch, H, D_MODEL) — i.e. it went through
    PadStatesAndActions like a training action array, not raw off the wire."""
    p = _make()
    prefix = np.arange(H * D_REAL, dtype=np.float32).reshape(H, D_REAL)
    p.infer({**_obs(), "prev_chunk": prefix, "d": 3})

    call = next(s for s in p.seen if s["sampler"] == "_sample_guided")
    got = np.asarray(call["args"][2])
    assert got.shape == (1, H, D_MODEL), got.shape
    # real dims survived the trip; the pad dims are zeros
    assert got[0, :, :D_REAL] == pytest.approx(prefix, abs=1e-5)
    assert got[0, :, D_REAL:] == pytest.approx(0.0)


def test_prefix_is_not_returned_as_the_action():
    """A dumb-but-fatal wiring error: echoing the prefix back instead of sampling."""
    p = _make()
    prefix = np.full((H, D_REAL), 0.5, np.float32)
    out = p.infer({**_obs(), "prev_chunk": prefix, "d": 3})
    assert not np.allclose(out["actions"], prefix)


def test_weights_passed_to_guided_cover_the_splice_slot():
    """The band is [d, d+overlap) RELATIVE to d, so the slot the client splices at
    (d) always carries weight. This is the F2 fix: with an absolute horizon and the
    shipped d>=horizon, w[d] was 0 and RTC constrained nothing."""
    p = _make(rtc=_rtc.RTCConfig(overlap=4, num_steps=2))
    p.infer({**_obs(), "prev_chunk": np.zeros((H, D_REAL), np.float32), "d": 2})

    call = next(s for s in p.seen if s["sampler"] == "_sample_guided")
    w = np.asarray(call["args"][3])
    assert w.shape == (H,)
    assert (w[:2] == 1.0).all()      # committed slots fully constrained
    assert w[2] > 0.0                # THE splice slot must carry weight
