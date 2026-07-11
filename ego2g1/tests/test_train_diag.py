"""Smoke gates for the training-loop pieces (train_step with per-slot buckets,
eval_step) and the offline diagnostics, on the CPU dummy model."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.config as _openpi_config
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils

from ego2g1 import diagnostics as diag
from ego2g1 import model as ego_model
from ego2g1 import train as ego_train

_KW = dict(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    pi05=True,
    action_horizon=8,
    action_dim=6,
    max_token_len=16,
    dtype="float32",
)


def _make_state(config: _openpi_config.TrainConfig):
    model = config.model.create(jax.random.key(0))
    params = nnx.state(model)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )


def test_train_step_and_eval_step():
    model_cfg = ego_model.Ego2G1Pi0Config(**_KW, action_dim_actual=4)
    tc = _openpi_config.TrainConfig(name="t", exp_name="t", model=model_cfg, batch_size=2)
    state = _make_state(tc)
    batch = (model_cfg.fake_obs(2), model_cfg.fake_act(2))

    new_state, info = ego_train.train_step(tc, jax.random.key(1), state, batch)
    assert int(new_state.step) == 1
    assert np.isfinite(float(info["loss"]))
    bucket_keys = [k for k in info if k.startswith("loss/slots_")]
    assert bucket_keys, info.keys()
    for k in bucket_keys:
        assert np.isfinite(float(info[k]))
    # gradient decomposition: groups present, finite, nonzero for the trained
    # parts, and consistent with the global norm (sum of squares)
    group_keys = ["grad_norm/siglip", "grad_norm/prefix_expert", "grad_norm/action_expert", "grad_norm/heads"]
    for k in group_keys:
        assert np.isfinite(float(info[k])), k
    total_sq = sum(float(info[k]) ** 2 for k in group_keys)
    np.testing.assert_allclose(total_sq, float(info["grad_norm"]) ** 2, rtol=1e-4)
    assert float(info["grad_norm/action_expert"]) > 0
    assert float(info["grad_norm/heads"]) > 0

    val_info = ego_train.eval_step(tc, jax.random.key(2), state, batch)
    assert np.isfinite(float(val_info["val/loss"]))
    # fixed rng => bitwise-repeatable val loss
    val_info2 = ego_train.eval_step(tc, jax.random.key(2), state, batch)
    assert float(val_info["val/loss"]) == float(val_info2["val/loss"])


def test_attention_allocation_shapes_and_normalization():
    cfg = ego_model.Ego2G1Pi0Config(**_KW)
    m = cfg.create(jax.random.key(0))
    obs, act = cfg.fake_obs(2), cfg.fake_act(2)
    out = diag.attention_allocation(m, obs, act)
    n_groups = len(out["group_names"])
    assert out["per_layer"].shape[1] == n_groups
    assert out["per_slot"].shape == (cfg.action_horizon, n_groups)
    # attention mass over groups partitions the simplex
    np.testing.assert_allclose(out["per_layer"].sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(out["per_slot"].sum(-1), 1.0, atol=1e-5)
    assert np.all(out["entropy_per_layer"] >= 0)


def test_image_patch_gap_shapes():
    cfg = ego_model.Ego2G1Pi0Config(**_KW)
    m = cfg.create(jax.random.key(0))
    img = np.asarray(cfg.fake_obs(1).images["base_0_rgb"][0])
    gap = diag.image_patch_gap(m, img, img)
    side = gap["cosine_grid"].shape[0]
    assert gap["cosine_grid"].shape == (side, side)
    for k in ("cosine_mean", "cosine_p05", "rel_l2_mean", "rel_l2_max"):
        assert np.isfinite(gap[k])