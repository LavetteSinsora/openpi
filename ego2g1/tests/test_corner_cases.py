"""Corner-case and integration gates found during the 2026-07 code review:
serving artifact layout (stamp at run root vs step dir), the degenerate-dim
allowlist vs the actual [eef 9 | hand 6] action layout, split partition
correctness, streaming-stats math, transform edge cases.

All CPU, no lerobot / no model construction needed.
"""

import dataclasses
import json

import numpy as np
import pytest

import openpi.shared.normalize as _normalize

from ego2g1 import chunk_math
from ego2g1 import config as _config
from ego2g1 import dataset as _dataset
from ego2g1 import norm as _norm
from ego2g1.serve import policy as _policy
from ego2g1 import stamp as _stamp
from ego2g1 import train as _train
from ego2g1 import transforms as ego_transforms
from ego2g1.compute_norm_stats import _PerSlotRunning


# ---------------------------------------------------------------------------
# chunk math corner cases
# ---------------------------------------------------------------------------

def _vec9(rng, n):
    pos = rng.normal(0, 0.3, (n, 3))
    A = rng.normal(size=(n, 3, 3))
    Q, _ = np.linalg.qr(A)
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1
    return np.concatenate([pos, chunk_math.mat_to_6d(Q)], axis=-1)


def test_relative_actions_numeric_against_explicit_se3():
    """delta_k must equal T_0^-1 @ T_k re-encoded, per hand independently."""
    rng = np.random.default_rng(11)
    h = 4
    sample = {f"pose.{hand}": _vec9(rng, h + 1) for hand in ("left", "right")}
    for hand in ("left", "right"):
        sample[f"hand.{hand}"] = rng.uniform(0, 1, (h, 6))
    out = chunk_math.RelativeChunkActions()(dict(sample))["actions"]
    for i, hand in enumerate(("left", "right")):
        T = chunk_math.vec9_to_se3(sample[f"pose.{hand}"])
        for k in range(h):
            expected = chunk_math.se3_to_vec9(np.linalg.inv(T[0]) @ T[k + 1])
            np.testing.assert_allclose(out[k, i * 15 : i * 15 + 9], expected, atol=1e-6)
            np.testing.assert_allclose(
                out[k, i * 15 + 9 : i * 15 + 15], sample[f"hand.{hand}"][k], atol=1e-6
            )


def test_relative_actions_single_hand_and_bad_shapes():
    rng = np.random.default_rng(12)
    t = chunk_math.RelativeChunkActions(hands=("right",))
    out = t({"pose.right": _vec9(rng, 3), "hand.right": rng.uniform(0, 1, (2, 6))})
    assert out["actions"].shape == (2, 15)

    with pytest.raises(ValueError, match="pose.right"):
        t({"pose.right": _vec9(rng, 3)[:, :8], "hand.right": np.zeros((2, 6))})
    with pytest.raises(ValueError, match="hand.right"):
        t({"pose.right": _vec9(rng, 3), "hand.right": np.zeros((3, 6))})


def test_boundary_indices_degenerate_lengths():
    h = 5
    # length == h, not real end: only t=0 has t+H == length-1+1 -> INVALID (needs t+H <= length-1)
    idx = chunk_math.BoundaryAwareIndices([h], [False], h, allow_terminal_padding=True)
    assert len(idx) == 0
    # length == h+1, not real end: exactly t=0 valid
    idx = chunk_math.BoundaryAwareIndices([h + 1], [False], h, allow_terminal_padding=True)
    assert list(idx.indices) == [0]
    # length < h, real end WITH terminal padding: every frame valid (hold pose)
    idx = chunk_math.BoundaryAwareIndices([3], [True], h, allow_terminal_padding=True)
    assert list(idx.indices) == [0, 1, 2]
    # real end but padding disabled: falls back to strict rule
    idx = chunk_math.BoundaryAwareIndices([3], [True], h, allow_terminal_padding=False)
    assert len(idx) == 0
    # every frame anchor_bad: nothing valid, offsets still advance
    idx = chunk_math.BoundaryAwareIndices(
        [4, 6], [True, True], 2, True, anchor_bad=[list(range(4)), []]
    )
    assert list(idx.indices) == [4, 5, 6, 7, 8, 9]
    assert idx.total_frames == 10


def test_boundary_indices_validation():
    with pytest.raises(ValueError, match="real_end"):
        chunk_math.BoundaryAwareIndices([3, 4], [True], 2, True)
    with pytest.raises(ValueError, match="anchor_bad"):
        chunk_math.BoundaryAwareIndices([3, 4], [True, False], 2, True, anchor_bad=[[0]])


def test_boundary_aware_dataset_remaps():
    idx = chunk_math.BoundaryAwareIndices([5, 5], [False, False], 3, True)
    wrapped = chunk_math.BoundaryAwareDataset(list(range(100, 110)), idx)
    assert len(wrapped) == 4
    assert [wrapped[i] for i in range(4)] == [100, 101, 105, 106]


def test_make_delta_timestamps_values():
    ts = chunk_math.make_delta_timestamps(2, 10)
    assert ts["pose.left"] == ts["pose.right"] == [0.0, 0.1, 0.2]
    assert ts["hand.left"] == ts["hand.right"] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# dataset split / sidecar gates (fabricated dataset dir, no lerobot)
# ---------------------------------------------------------------------------

def _fake_dataset(tmp_path, lengths, sources, real_ends, anchor_bad=None, horizon=5, hz=30.0):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    meta = {
        "config_hash": "cafebabe12345678",
        "config": {"action_horizon": horizon, "control_hz": hz},
        "episodes": {
            str(i): {
                "source_episode": sources[i],
                "episode_real_end": real_ends[i],
                **({"anchor_bad": anchor_bad[i]} if anchor_bad else {}),
            }
            for i in range(len(lengths))
        },
    }
    (root / "extraction_meta.json").write_text(json.dumps(meta))
    with (root / "meta" / "episodes.jsonl").open("w") as f:
        for i, length in enumerate(lengths):
            f.write(json.dumps({"episode_index": i, "length": length}) + "\n")
    return root


def test_assert_dataset_compatible_gates(tmp_path):
    root = _fake_dataset(tmp_path, [10], ["ep/0"], [True], horizon=50, hz=30.0)
    _dataset.assert_dataset_compatible(root, "cafebabe12345678", 50, 30)  # exact match
    _dataset.assert_dataset_compatible(root, None, 40, 30)  # hash unchecked, smaller horizon ok
    with pytest.raises(ValueError, match="config_hash"):
        _dataset.assert_dataset_compatible(root, "0000000000000000", 50, 30)
    with pytest.raises(ValueError, match="horizon"):
        _dataset.assert_dataset_compatible(root, None, 51, 30)
    with pytest.raises(ValueError, match="control_hz"):
        _dataset.assert_dataset_compatible(root, None, 50, 25)


def test_split_partitions_valid_indices(tmp_path):
    # real episode "A" was filter-split into lerobot episodes 0 and 2; "B" is ep 1.
    root = _fake_dataset(
        tmp_path,
        lengths=[10, 8, 6],
        sources=["task/A", "task/B", "task/A"],
        real_ends=[False, True, True],
        anchor_bad=[[1], [], []],
        horizon=5,
    )
    both = _dataset.build_split_indices(root, 5, val_real_episodes=())
    split = _dataset.build_split_indices(root, 5, val_real_episodes=("task/A",))

    all_idx = set(both.train.indices)
    train_idx, val_idx = set(split.train.indices), set(split.val.indices)
    assert train_idx | val_idx == all_idx  # nothing lost
    assert train_idx & val_idx == set()  # nothing shared
    # both sub-episodes of A land in val together
    assert val_idx == {t for t in all_idx if t < 10 or t >= 18}
    # empty val split when no episodes held out
    assert len(both.val) == 0

    with pytest.raises(ValueError, match="not in dataset"):
        _dataset.build_split_indices(root, 5, val_real_episodes=("task/NOPE",))


# ---------------------------------------------------------------------------
# E001 stats: unit consistency, allowlist vs actual action layout
# ---------------------------------------------------------------------------

def test_gain_is_invariant_to_per_dim_affine_rescaling():
    """The gain grid is computed from RAW sigmas but applied to pooled-quantile-
    NORMALIZED actions. That is only correct because gain is a ratio of sigmas,
    invariant under any per-dim affine map (which quantile norm is). Pin it."""
    rng = np.random.default_rng(21)
    h, d = 6, 4
    sigma_slot = rng.uniform(0.01, 2.0, (h, d))
    sigma_pooled = rng.uniform(0.5, 2.0, d)
    scale = rng.uniform(0.1, 10.0, d)  # arbitrary per-dim affine scale
    g_raw = _norm.PerSlotStats(sigma_slot, {}).gain(0.1, sigma_pooled)
    g_scaled = _norm.PerSlotStats(sigma_slot * scale, {}).gain(0.1, sigma_pooled * scale)
    np.testing.assert_allclose(g_raw, g_scaled, rtol=1e-6)


def test_gain_floor_c_validation_and_pooled_slicing():
    stats = _norm.PerSlotStats(np.ones((3, 2)), {})
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="floor_c"):
            stats.gain(bad, np.ones(2))
    # sigma_pooled longer than D_real (e.g. padded 32-dim stats) is sliced
    g = stats.gain(0.5, np.ones(5))
    assert g.shape == (3, 2)


def test_default_allowlist_matches_left_hand_command_dims():
    """Action layout (SPEC.md / RelativeChunkActions): per hand [eef 9 | hand 6]
    in ("left", "right") order -> left-hand commands are dims 9..14. The
    allowlist exists for exactly those dims; dims 15..17 are the RIGHT-hand EEF
    translation and must never be allowlisted by default."""
    allow = set(_config.Ego2G1TrainConfig().degenerate_dim_allowlist)
    assert allow == set(range(9, 15)), allow


def _stats_with_degenerate(dims, d_real=30, h=4):
    rng = np.random.default_rng(22)
    q01 = rng.normal(size=d_real)
    span = np.ones(d_real)
    sigma_slot = rng.uniform(0.1, 1.0, (h, d_real))
    for d in dims:
        span[d] = 0.0
        sigma_slot[:, d] = 0.0
    pooled = {
        "actions": _normalize.NormStats(
            mean=np.zeros(d_real), std=np.ones(d_real), q01=q01, q99=q01 + span
        )
    }
    return pooled, _norm.PerSlotStats(sigma_slot, {})


def test_stats_sanity_allowlisted_left_hand_passes():
    cfg = _config.Ego2G1TrainConfig()
    pooled, per_slot = _stats_with_degenerate(range(9, 15))
    assert _norm.check_stats_sanity(pooled, per_slot, cfg.degenerate_dim_allowlist) == []


def test_stats_sanity_degenerate_right_eef_fails():
    cfg = _config.Ego2G1TrainConfig()
    pooled, per_slot = _stats_with_degenerate([15])  # right EEF translation x
    problems = _norm.check_stats_sanity(pooled, per_slot, cfg.degenerate_dim_allowlist)
    assert any("dim 15" in p for p in problems), problems


def test_stats_sanity_detects_all_slot_degenerate_sigma():
    cfg = _config.Ego2G1TrainConfig()
    pooled, per_slot = _stats_with_degenerate([])
    sigma = per_slot.sigma_slot.copy()
    sigma[:, 3] = 0.0  # pooled span fine, but no per-slot signal anywhere
    problems = _norm.check_stats_sanity(
        pooled, _norm.PerSlotStats(sigma, {}), cfg.degenerate_dim_allowlist
    )
    assert any("dim 3" in p and "sigma_slot" in p for p in problems), problems


def test_per_slot_running_matches_direct_numpy():
    """Chan streaming update == direct population sigma, incl. uneven batches."""
    rng = np.random.default_rng(23)
    data = rng.normal(2.0, 3.0, (257, 5, 4))  # deliberately not a batch multiple
    running = _PerSlotRunning()
    for start in range(0, len(data), 100):  # batches of 100, 100, 57
        running.update(data[start : start + 100])
    np.testing.assert_allclose(running.sigma(), data.std(axis=0), rtol=1e-10)
    np.testing.assert_allclose(running.mean, data.mean(axis=0), rtol=1e-10)
    with pytest.raises(ValueError):
        fresh = _PerSlotRunning()
        fresh.update(data[:1])
        fresh.sigma()


# ---------------------------------------------------------------------------
# transforms corner cases
# ---------------------------------------------------------------------------

def test_parse_image_float_chw_and_uint8_hwc():
    import openpi.models.model as _model

    chw = np.random.default_rng(31).uniform(0, 1, (3, 8, 8)).astype(np.float32)
    out = ego_transforms.Ego2G1Inputs(model_type=_model.ModelType.PI05)(
        {"observation/image": chw, "observation/state": np.zeros(30, np.float32)}
    )
    img = out["image"]["base_0_rgb"]
    assert img.shape == (8, 8, 3) and img.dtype == np.uint8
    np.testing.assert_array_equal(img, (255 * chw).astype(np.uint8).transpose(1, 2, 0))
    # wrist slots zero + masked out; base masked in
    assert not out["image_mask"]["left_wrist_0_rgb"] and out["image_mask"]["base_0_rgb"]
    assert not out["image"]["right_wrist_0_rgb"].any()
    assert "actions" not in out  # inference sample: no actions key invented

    hwc = (np.random.default_rng(32).uniform(0, 255, (8, 8, 3))).astype(np.uint8)
    out = ego_transforms.Ego2G1Inputs(model_type=_model.ModelType.PI05)(
        {"observation/image": hwc, "observation/state": np.zeros(30), "actions": np.zeros((2, 30)),
         "prompt": "x"}
    )
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], hwc)
    assert "actions" in out and out["prompt"] == "x"


def test_outputs_trim_and_append_control_mode_array_prompt():
    out = ego_transforms.Ego2G1Outputs(action_dim=30)({"actions": np.ones((5, 32))})
    assert out["actions"].shape == (5, 30)
    # numpy 0-d prompt (what InjectDefaultPrompt produces) must work
    out = ego_transforms.AppendControlMode()({"prompt": np.asarray("grab it")})
    assert out["prompt"].startswith("grab it <<<control_mode>>>")


def test_per_slot_rescale_inference_passthrough_and_inverse_shape_guard():
    gain = np.full((4, 3), 2.0, np.float32)
    data = {"state": np.zeros(3)}
    assert ego_transforms.PerSlotRescale(gain=gain)(dict(data)) == data  # no actions at inference
    with pytest.raises(ValueError):  # wrong horizon on model output
        ego_transforms.PerSlotRescaleInverse(gain=gain)({"actions": np.zeros((5, 3), np.float32)})


# ---------------------------------------------------------------------------
# serving artifact layout (the RUNBOOK path: create_policy(<run>/<step>))
# ---------------------------------------------------------------------------

def _fake_run_dir(tmp_path, cfg, step="29999"):
    """Reproduce exactly what train.py + orbax leave on disk."""
    run_dir = tmp_path / "checkpoints" / cfg.name / cfg.exp_name
    step_dir = run_dir / step
    _stamp.write_stamp(run_dir, cfg, "cafebabe12345678")  # train.py:251
    _norm.save_per_slot(run_dir / "assets_ego2g1",  # train.py:253
                        _norm.PerSlotStats(np.full((2, 30), 0.5), {"k": "v"}))
    d = np.zeros(30)
    _normalize.save(step_dir / "assets" / cfg.repo_id,  # orbax save_assets
                    {"actions": _normalize.NormStats(mean=d, std=d + 1, q01=d - 1, q99=d + 1),
                     "state": _normalize.NormStats(mean=d, std=d + 1, q01=d - 1, q99=d + 1)})
    return run_dir, step_dir


def test_serving_resolves_run_level_stamp_from_step_dir(tmp_path):
    cfg = _config.Ego2G1TrainConfig()
    run_dir, step_dir = _fake_run_dir(tmp_path, cfg)
    # step dir has no stamp of its own -> must resolve to the run root
    assert _policy.resolve_run_dir(step_dir) == run_dir
    assert _policy.resolve_run_dir(run_dir) == run_dir  # run root passes through
    stamp = _stamp.check_supported(_policy.resolve_run_dir(step_dir))  # must not raise
    assert stamp["extraction_config_hash"] == "cafebabe12345678"


def test_config_from_stamp_roundtrip(tmp_path):
    cfg = _config.Ego2G1TrainConfig(
        val_real_episodes=("t/e1", "t/e2"), expected_config_hash="cafebabe12345678",
        per_slot_floor_c=0.2,
    )
    _stamp.write_stamp(tmp_path, cfg, "cafebabe12345678")
    rebuilt = _policy.config_from_stamp(_stamp.read_stamp(tmp_path))
    assert rebuilt == cfg
    assert rebuilt.config_hash() == cfg.config_hash()


def test_resolve_norm_assets_from_train_layout(tmp_path, monkeypatch):
    """The two artifacts live in DIFFERENT dirs in a real run (pooled per step,
    per-slot at the run root). Resolution must find each independently and never
    write into the checkpoint. Exercised with a RELATIVE checkpoint path — the
    shape that broke the old symlink-merge."""
    cfg = _config.Ego2G1TrainConfig()
    run_dir, step_dir = _fake_run_dir(tmp_path, cfg)
    monkeypatch.chdir(tmp_path)  # so a relative step path is realistic
    rel_step = step_dir.relative_to(tmp_path)

    pooled, per_slot = _policy.resolve_norm_assets(rel_step, _policy.resolve_run_dir(rel_step), cfg)
    assert pooled == rel_step / "assets" / cfg.repo_id
    assert per_slot == _policy.resolve_run_dir(rel_step) / "assets_ego2g1"
    gain = _norm.load_per_slot(per_slot).gain(
        cfg.per_slot_floor_c, _norm.load_pooled(pooled)["actions"].std[: cfg.action_dim_actual])
    assert gain.shape == (2, 30) and np.isfinite(gain).all()
    # nothing was written into the checkpoint's assets dir
    assert not (pooled / _norm.PER_SLOT_FILENAME).exists()


def test_resolve_norm_assets_falls_back_to_training_dir(tmp_path, monkeypatch):
    cfg = _config.Ego2G1TrainConfig()
    monkeypatch.chdir(tmp_path)
    train_assets = cfg.assets_dirs / cfg.repo_id  # assets/<name>/<repo_id>, from CWD
    d = np.zeros(30)
    _normalize.save(train_assets, {"actions": _normalize.NormStats(mean=d, std=d + 1, q01=d - 1, q99=d + 1),
                                   "state": _normalize.NormStats(mean=d, std=d + 1, q01=d - 1, q99=d + 1)})
    _norm.save_per_slot(train_assets, _norm.PerSlotStats(np.full((2, 30), 0.5), {}, mu_slot=np.zeros((2, 30))))

    empty = tmp_path / "ck" / "5000"
    pooled, per_slot = _policy.resolve_norm_assets(empty, empty.parent, cfg)
    assert pooled == train_assets and per_slot == train_assets


def test_resolve_norm_assets_explicit_and_missing(tmp_path, monkeypatch):
    cfg = _config.Ego2G1TrainConfig()
    monkeypatch.chdir(tmp_path)  # isolate from any real ./assets in the repo
    with pytest.raises(FileNotFoundError, match="missing"):
        _policy.resolve_norm_assets(tmp_path, tmp_path, cfg, assets_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Searched"):
        _policy.resolve_norm_assets(tmp_path / "nope", tmp_path / "nope", cfg)


def test_config_hash_scope():
    a = _config.Ego2G1TrainConfig()
    assert dataclasses.replace(a, exp_name="other").config_hash() == a.config_hash()
    assert dataclasses.replace(a, save_interval=1).config_hash() == a.config_hash()
    assert dataclasses.replace(a, per_slot_floor_c=0.5).config_hash() != a.config_hash()
    assert dataclasses.replace(a, action_horizon=25).config_hash() != a.config_hash()


# ---------------------------------------------------------------------------
# train helpers
# ---------------------------------------------------------------------------

def test_slot_bucket_means_clip_to_horizon():
    import jax.numpy as jnp

    loss = jnp.arange(8, dtype=jnp.float32)[None, :]  # (1, 8), horizon 8
    out = _train._slot_bucket_means(loss, 8)
    assert set(out) == {"slots_00_04", "slots_05_24"}  # 25+ bucket dropped
    np.testing.assert_allclose(float(out["slots_00_04"]), np.arange(0, 5).mean())
    np.testing.assert_allclose(float(out["slots_05_24"]), np.arange(5, 8).mean())
