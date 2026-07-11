"""Model-side gates (TRAINING_PLAN.md §3.10): golden stock identity, gemma
patch safety, per-token/scalar equivalence, RTC loss and sampling pins.

CPU-runnable via the `dummy` gemma variant; dtype float32 for tight tolerances.
"""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config

from ego2g1 import gemma_patch
from ego2g1 import model as ego_model

_KW = dict(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    pi05=True,
    action_horizon=8,
    action_dim=6,
    max_token_len=16,
    dtype="float32",
)


def _stock():
    return _pi0_config.Pi0Config(**_KW)


def _ego(**kw):
    return ego_model.Ego2G1Pi0Config(**_KW, **kw)


def _loss(config, seed=0):
    m = config.create(jax.random.key(0))
    obs, act = config.fake_obs(2), config.fake_act(2)
    return np.asarray(m.compute_loss(jax.random.key(seed), obs, act))


def test_patch_is_applied_and_fingerprints_hold():
    assert gemma_patch.is_applied()
    gemma_patch.verify_fingerprints()  # must not raise on this checkout


def test_fingerprint_guard_raises_on_drift(monkeypatch):
    monkeypatch.setitem(gemma_patch._STOCK_FINGERPRINTS, "RMSNorm", "0" * 64)
    with pytest.raises(gemma_patch.StockSourceChangedError):
        gemma_patch.verify_fingerprints()


def test_patch_preserves_stock_bitwise():
    """Stock pi05 compute_loss is bitwise identical before/after the patch.

    Runs in a subprocess so the 'before' really is an unpatched process.
    """
    script = """
import jax, numpy as np
import openpi.models.pi0_config as pc
KW = dict(paligemma_variant="dummy", action_expert_variant="dummy", pi05=True,
          action_horizon=8, action_dim=6, max_token_len=16, dtype="float32")
cfg = pc.Pi0Config(**KW)
def loss():
    m = cfg.create(jax.random.key(0))
    return np.asarray(m.compute_loss(jax.random.key(1), cfg.fake_obs(2), cfg.fake_act(2)))
before = loss()
import ego2g1.gemma_patch as gp
gp.apply()
after = loss()
assert (before == after).all(), (before, after)
# non-pi05 (no adaRMS) path still runs under the patch
cfg = pc.Pi0Config(**{**KW, "pi05": False, "max_token_len": 48})
m = cfg.create(jax.random.key(0))
np.asarray(m.compute_loss(jax.random.key(1), cfg.fake_obs(2), cfg.fake_act(2)))
print("OK")
"""
    res = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_param_tree_matches_stock():
    import flax.nnx as nnx

    stock = nnx.state(_stock().create(jax.random.key(0))).to_pure_dict()
    ego = nnx.state(
        _ego(action_dim_actual=4, rtc_training=True, rtc_d_max=5).create(jax.random.key(0))
    ).to_pure_dict()

    def paths(d, prefix=()):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from paths(v, (*prefix, k))
            else:
                yield (*prefix, k), getattr(v, "shape", None)

    assert dict(paths(stock)) == dict(paths(ego))


def test_golden_stock_identity_when_features_off():
    a = _loss(_stock(), seed=3)
    b = _loss(_ego(), seed=3)  # action_dim_actual=None, rtc off -> super() delegation
    assert (a == b).all()


def test_masked_loss_full_width_is_stock_bitwise():
    # action_dim_actual == action_dim: the copied body must reproduce stock
    # exactly (same rng splits, same ops); the slice is a no-op.
    a = _loss(_stock(), seed=4)
    b = _loss(_ego(action_dim_actual=_KW["action_dim"]), seed=4)
    assert (a == b).all()


def test_masked_loss_shape_and_effect():
    full = _loss(_ego(action_dim_actual=None), seed=5)
    masked = _loss(_ego(action_dim_actual=3), seed=5)
    assert masked.shape == full.shape == (2, _KW["action_horizon"])
    assert not np.allclose(masked, full)  # the mask must change the value


def test_per_token_scalar_equivalence():
    """embed_suffix with a repeated-scalar per-token timestep matches the scalar
    path end to end through the llm (E002 eval 2), pi05 adaRMS branch."""
    cfg = _ego()
    m = cfg.create(jax.random.key(0))
    obs = jax.tree.map(jnp.asarray, cfg.fake_obs(2))
    import openpi.models.model as _model

    obs = _model.preprocess_observation(None, obs, train=False)
    x_t = jax.random.normal(jax.random.key(1), (2, cfg.action_horizon, cfg.action_dim))
    t_scalar = jnp.array([0.3, 0.8])
    t_tok = jnp.broadcast_to(t_scalar[:, None], (2, cfg.action_horizon))

    def suffix_out(timestep):
        prefix_tokens, prefix_mask, prefix_ar = m.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar, cond = m.embed_suffix(obs, x_t, timestep)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, out), _ = m.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, cond]
        )
        return np.asarray(m.action_out_proj(out[:, -cfg.action_horizon :]))

    a, b = suffix_out(t_scalar), suffix_out(t_tok)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_rtc_loss_masks_prefix_exactly():
    cfg = _ego(action_dim_actual=4, rtc_training=True, rtc_d_max=5)
    m = cfg.create(jax.random.key(0))
    obs, act = cfg.fake_obs(4), cfg.fake_act(4)
    rng = jax.random.key(6)
    loss = np.asarray(m.compute_loss(rng, obs, act))
    # reproduce the d draw (same derived stream as compute_loss)
    d = np.asarray(jax.random.randint(jax.random.fold_in(rng, 7), (4,), 0, cfg.rtc_d_max + 1))
    assert d.max() > 0, "test rng produced all-zero d; pick another seed"
    for i in range(4):
        assert (loss[i, : d[i]] == 0).all(), (i, d[i], loss[i])
        assert (loss[i, d[i] :] > 0).all(), (i, d[i], loss[i])


def test_rtc_d_zero_matches_non_rtc():
    # rtc_d_max=0 forces d=0 for every sample: no prefix, per-token timesteps
    # all equal the scalar -> must match the non-RTC masked path numerically.
    a = _loss(_ego(action_dim_actual=4), seed=7)
    b = _loss(_ego(action_dim_actual=4, rtc_training=True, rtc_d_max=0), seed=7)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_sample_actions_rtc_d0_matches_stock_sampling():
    cfg = _ego()
    m = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(2)
    noise = jax.random.normal(jax.random.key(8), (2, cfg.action_horizon, cfg.action_dim))
    stock_out = np.asarray(m.sample_actions(jax.random.key(9), obs, num_steps=4, noise=noise))
    prefix = jnp.zeros((2, cfg.action_horizon, cfg.action_dim))
    rtc_out = np.asarray(m.sample_actions_rtc(jax.random.key(9), obs, prefix, 0, num_steps=4, noise=noise))
    np.testing.assert_allclose(stock_out, rtc_out, rtol=1e-5, atol=1e-6)


def test_sample_actions_rtc_prefix_held_exactly():
    cfg = _ego(rtc_training=True, rtc_d_max=5)
    m = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(2)
    d = 3
    prefix = jax.random.normal(jax.random.key(10), (2, cfg.action_horizon, cfg.action_dim))
    out = np.asarray(m.sample_actions_rtc(jax.random.key(11), obs, prefix, d, num_steps=4))
    np.testing.assert_array_equal(out[:, :d], np.asarray(prefix[:, :d]))
    assert not np.allclose(out[:, d:], np.asarray(prefix[:, d:]))


def test_rtc_config_validation():
    with pytest.raises(ValueError):
        ego_model.Ego2G1Pi0Config(**{**_KW, "pi05": False, "max_token_len": 48}, rtc_training=True)
    with pytest.raises(ValueError):
        _ego(rtc_training=True, rtc_d_max=_KW["action_horizon"])
    with pytest.raises(NotImplementedError):
        _ego(num_flow_samples=2)
