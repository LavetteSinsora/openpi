"""Gates for the degeneracy mask + selective per-slot centering + clamp
(2026-07 refactor): one degeneracy criterion owned by norm.py, consumed by the
sanity gate, the gain, the centering mask, and the data path itself.

All CPU, no lerobot / no model construction.
"""

import dataclasses
import json

import numpy as np
import pytest

import openpi.shared.normalize as _normalize

from ego2g1 import config as _config
from ego2g1 import data_config as _data_config
from ego2g1 import norm as _norm
from ego2g1.serve import policy as _policy
from ego2g1 import stamp as _stamp
from ego2g1 import transforms as ego_transforms

D = 30
H = 4


def _pooled_stats(spike_dims=(), tiny_span_dims=(), seed=0):
    """Healthy dims: span ~ 4*std. spike_dims: span=0 but std>0 (the killer
    shape). tiny_span_dims: span < 0.5*std (mask must also catch)."""
    rng = np.random.default_rng(seed)
    std = rng.uniform(0.05, 0.3, D)
    span = 4.0 * std
    q01 = rng.normal(size=D)
    for d in spike_dims:
        span[d] = 0.0
        std[d] = 0.004
    for d in tiny_span_dims:
        span[d] = 0.4 * std[d]
    return {
        "actions": _normalize.NormStats(mean=q01 + span / 2, std=std, q01=q01, q99=q01 + span),
        "state": _normalize.NormStats(mean=np.zeros(D), std=np.ones(D),
                                      q01=-np.ones(D), q99=np.ones(D)),
    }


def _per_slot(seed=1, with_mu=True):
    rng = np.random.default_rng(seed)
    # sigma constant across slots per dim -> per-dim constant gain, so tests
    # can distinguish centering drift from gain drift
    sigma = np.tile(rng.uniform(0.01, 0.3, D), (H, 1))
    mu = rng.normal(0.0, 0.1, (H, D)) if with_mu else None
    return _norm.PerSlotStats(sigma_slot=sigma, provenance={}, mu_slot=mu)


# ---------------------------------------------------------------------------
# the mask
# ---------------------------------------------------------------------------

def test_degenerate_mask_catches_both_shapes():
    pooled = _pooled_stats(spike_dims=[13, 14], tiny_span_dims=[7])
    mask = _norm.degenerate_action_dims(pooled["actions"], D)
    assert set(np.flatnonzero(mask)) == {7, 13, 14}


def test_degenerate_mask_healthy_dims_pass():
    mask = _norm.degenerate_action_dims(_pooled_stats()["actions"], D)
    assert not mask.any()


def test_gain_exempts_masked_dims():
    pooled = _pooled_stats(spike_dims=[13])
    mask = _norm.degenerate_action_dims(pooled["actions"], D)
    ps = _per_slot()
    g = ps.gain(0.1, pooled["actions"].std, degenerate_mask=mask)
    np.testing.assert_array_equal(g[:, 13], 1.0)
    # unmasked dims unchanged vs no-mask call
    g0 = ps.gain(0.1, pooled["actions"].std)
    np.testing.assert_array_equal(g[:, :13], g0[:, :13])


# ---------------------------------------------------------------------------
# the transform pair
# ---------------------------------------------------------------------------

def _mk_transforms(cfg=None, pooled=None, per_slot=None):
    cfg = cfg or _config.Ego2G1TrainConfig()
    pooled = pooled or _pooled_stats(spike_dims=[13, 14])
    per_slot = per_slot or _per_slot()
    # widen sigma grid to cfg horizon? tests use H=4 grids with a 4-slot config
    cfg = dataclasses.replace(cfg, action_horizon=H)
    return _data_config.build_per_slot_transforms(cfg, pooled, per_slot), pooled


def test_forward_neutralizes_actions_and_state_and_clamps():
    (fwd, _), pooled = _mk_transforms()
    actions = np.zeros((H, D), np.float32)
    actions[:, 13] = 2.0e5  # spike-tail monster (post-Normalize units)
    actions[:, 0] = 50.0    # heavy tail on a live dim
    state = np.zeros(D, np.float32)
    state[13] = 1.0e5
    out = fwd({"actions": actions, "state": state})
    assert (out["actions"][:, 13] == -1.0).all()      # neutralized, not 2e5
    assert (np.abs(out["actions"]) <= 10.0).all()     # clamped
    assert out["state"][13] == -1.0 and out["state"][0] == 0.0
    # inference path: no actions key, state still neutralized
    out = fwd({"state": state})
    assert "actions" not in out and out["state"][13] == -1.0


def test_forward_centers_only_eef_dims():
    (fwd, _), pooled = _mk_transforms()
    mu_n = fwd.mu_n
    center = _data_config.center_dims_mask(("left", "right"))
    assert mu_n is not None
    assert (mu_n[:, ~center] == 0.0).all()            # hand cmds never centered
    assert (mu_n[:, [13, 14]] == 0.0).all()           # degenerate never centered
    live_eef = center.copy(); live_eef[[13, 14]] = False
    assert np.abs(mu_n[:, live_eef]).max() > 0        # eef dims actually centered


def test_forward_inverse_roundtrip_with_centering():
    # clamp off: the roundtrip identity only holds for unclamped values
    cfg = dataclasses.replace(_config.Ego2G1TrainConfig(), model_space_clamp=None)
    (fwd, inv), _ = _mk_transforms(cfg=cfg)
    rng = np.random.default_rng(3)
    actions = rng.normal(0, 0.5, (H, D)).astype(np.float32)  # small: unclamped
    out = fwd({"actions": actions.copy(), "state": np.zeros(D)})
    padded = np.concatenate([out["actions"], rng.normal(size=(H, 2)).astype(np.float32)], axis=-1)
    back = inv({"actions": padded})["actions"]
    live = ~fwd.degenerate_mask
    np.testing.assert_allclose(back[:, :D][:, live], actions[:, live], rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(back[:, D:], padded[:, D:])  # pad dims untouched
    # neutralized dims come back as the constant -1 (resting), by design
    np.testing.assert_allclose(back[:, 13], -1.0, atol=1e-6)


def test_flat_command_stays_flat_on_uncentered_dims():
    """'hold still' on an absolute hand-command dim must be a FLAT target."""
    (fwd, _), _ = _mk_transforms()
    actions = np.zeros((H, D), np.float32)
    actions[:, 25] = -0.7  # constant right-hand command, normalized units
    out = fwd({"actions": actions, "state": np.zeros(D)})
    col = out["actions"][:, 25]
    assert np.ptp(col) < 1e-6, col  # no per-slot drift leaked in


def test_centering_requires_mu_artifact():
    cfg = dataclasses.replace(_config.Ego2G1TrainConfig(), action_horizon=H)
    with pytest.raises(ValueError, match="mu_slot"):
        _data_config.build_per_slot_transforms(cfg, _pooled_stats(), _per_slot(with_mu=False))
    # centering off: legacy artifact is fine
    cfg_off = dataclasses.replace(cfg, per_slot_center=False)
    fwd, inv = _data_config.build_per_slot_transforms(cfg_off, _pooled_stats(), _per_slot(with_mu=False))
    assert fwd.mu_n is None and inv.mu_n is None


# ---------------------------------------------------------------------------
# artifacts / stamp / serving compatibility
# ---------------------------------------------------------------------------

def test_mu_slot_save_load_and_legacy(tmp_path):
    ps = _per_slot()
    _norm.save_per_slot(tmp_path, ps)
    loaded = _norm.load_per_slot(tmp_path)
    np.testing.assert_array_equal(loaded.mu_slot, ps.mu_slot)
    legacy = tmp_path / "legacy"
    _norm.save_per_slot(legacy, _per_slot(with_mu=False))
    assert _norm.load_per_slot(legacy).mu_slot is None


def test_config_from_stamp_legacy_defaults(tmp_path):
    cfg = _config.Ego2G1TrainConfig()
    _stamp.write_stamp(tmp_path, cfg, "cafebabe00000000")
    p = tmp_path / _stamp.STAMP_FILENAME
    stamp = json.loads(p.read_text())
    # simulate a checkpoint stamped before centering/clamp existed
    del stamp["ego2g1_config"]["per_slot_center"]
    del stamp["ego2g1_config"]["model_space_clamp"]
    rebuilt = _policy.config_from_stamp(stamp)
    assert rebuilt.per_slot_center is False
    assert rebuilt.model_space_clamp is None
    # a current stamp round-trips the new fields
    rebuilt = _policy.config_from_stamp(json.loads(p.read_text()))
    assert rebuilt.per_slot_center is True and rebuilt.model_space_clamp == 10.0


def test_new_required_flags_are_supported(tmp_path):
    cfg = _config.Ego2G1TrainConfig()
    flags = cfg.feature_flags()
    assert flags["per_slot_center"]["required"] is True
    assert flags["degenerate_neutralization"]["required"] is True
    assert flags["model_space_clamp"]["required"] is False
    _stamp.write_stamp(tmp_path, cfg, "cafebabe00000000")
    _stamp.check_supported(tmp_path)  # must not raise


# ---------------------------------------------------------------------------
# sanity gate
# ---------------------------------------------------------------------------

def test_sanity_flags_spike_dim_not_allowlisted():
    pooled = _pooled_stats(spike_dims=[20])  # right-hand region: never allowlisted
    problems = _norm.check_stats_sanity(pooled, _per_slot(),
                                        _config.Ego2G1TrainConfig().degenerate_dim_allowlist)
    assert any("dim 20" in p and "degenerate" in p for p in problems), problems


def test_sanity_empirical_gate_catches_unmasked_monster():
    pooled = _pooled_stats()
    q01 = pooled["actions"].q01
    span = pooled["actions"].q99 - q01
    raw_min, raw_max = q01.copy(), (q01 + span).copy()
    raw_max[5] = q01[5] + 2000 * span[5]  # tail 2000 spans out on a live dim
    problems = _norm.check_stats_sanity(
        pooled, _per_slot(), _config.Ego2G1TrainConfig().degenerate_dim_allowlist,
        raw_min=raw_min, raw_max=raw_max,
    )
    assert any("dim 5" in p and "max |normalized|" in p for p in problems), problems
    # same extremes on a masked dim: fine (neutralized in the data path)
    pooled2 = _pooled_stats(spike_dims=[13])
    problems = _norm.check_stats_sanity(
        pooled2, _per_slot(), _config.Ego2G1TrainConfig().degenerate_dim_allowlist,
        raw_min=np.zeros(D), raw_max=np.full(D, 0.5),
    )
    assert problems == [], problems
