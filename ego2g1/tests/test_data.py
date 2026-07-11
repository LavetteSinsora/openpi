"""Data-side gates: chunk-math equivalence vs data_extraction (when available),
per-slot rescale roundtrip/identity, control-mode prompt, stats sanity,
split correctness, stamp guard.
"""

import importlib.util

import numpy as np
import pytest

from ego2g1 import chunk_math
from ego2g1 import norm as _norm
from ego2g1 import transforms as ego_transforms


# ---------------------------------------------------------------------------
# chunk math
# ---------------------------------------------------------------------------

def _random_vec9(rng, n):
    # random valid poses: positions ~N(0, 0.3), rotations from QR
    pos = rng.normal(0, 0.3, (n, 3))
    A = rng.normal(size=(n, 3, 3))
    Q, R = np.linalg.qr(A)
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1
    return np.concatenate([pos, chunk_math.mat_to_6d(Q)], axis=-1)


def test_relative_chunk_actions_shapes_and_anchor_identity():
    rng = np.random.default_rng(0)
    h = 5
    sample = {}
    for hand in ("left", "right"):
        sample[f"pose.{hand}"] = _random_vec9(rng, h + 1)
        sample[f"hand.{hand}"] = rng.uniform(0, 1, (h, 6))
    out = chunk_math.RelativeChunkActions()(dict(sample))
    actions = out["actions"]
    assert actions.shape == (h, 30) and actions.dtype == np.float32
    # delta_1 for pose_1 == pose_0 is identity: zero translation, 6d of I
    same = dict(sample)
    same["pose.left"] = np.tile(sample["pose.left"][:1], (h + 1, 1))
    out2 = chunk_math.RelativeChunkActions()(same)
    ident = out2["actions"][:, :9]
    np.testing.assert_allclose(ident[:, :3], 0, atol=1e-9)
    np.testing.assert_allclose(ident[:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (h, 1)), atol=1e-9)
    # hand commands pass through absolutely
    np.testing.assert_allclose(out["actions"][:, 9:15], sample["hand.left"], atol=1e-6)


def test_relative_chunk_actions_passthrough_without_pose_keys():
    data = {"observation/state": np.zeros(30), "prompt": "x"}
    assert chunk_math.RelativeChunkActions()(dict(data)) == data


@pytest.mark.skipif(
    importlib.util.find_spec("data_extraction") is None,
    reason="outer-repo equivalence runs in the outer repo (data_extraction on sys.path)",
)
def test_chunk_math_equivalence_vs_data_extraction():
    """Byte-equivalence of the pinned copy against the canonical loader."""
    from data_extraction.common import rot6d as ref_rot6d
    from data_extraction.loader import boundary as ref_boundary
    from data_extraction.loader import relative_actions as ref_rel

    rng = np.random.default_rng(1)
    v = _random_vec9(rng, 64)
    np.testing.assert_array_equal(chunk_math.vec9_to_se3(v), ref_rot6d.vec9_to_se3(v))
    np.testing.assert_array_equal(
        chunk_math.se3_to_vec9(chunk_math.vec9_to_se3(v)), ref_rot6d.se3_to_vec9(ref_rot6d.vec9_to_se3(v))
    )
    assert chunk_math.make_delta_timestamps(50, 30) == ref_rel.make_delta_timestamps(50, 30)

    h = 7
    sample = {}
    for hand in ("left", "right"):
        sample[f"pose.{hand}"] = _random_vec9(rng, h + 1)
        sample[f"hand.{hand}"] = rng.uniform(0, 1, (h, 6))
    ours = chunk_math.RelativeChunkActions()(dict(sample))["actions"]
    theirs = ref_rel.RelativeChunkActions()(dict(sample))["actions"]
    np.testing.assert_array_equal(ours, theirs)

    lengths, ends, bad = [10, 8, 12], [True, False, True], [[0, 3], [], [11]]
    ours_idx = chunk_math.BoundaryAwareIndices(lengths, ends, 5, True, anchor_bad=bad)
    theirs_idx = ref_boundary.BoundaryAwareIndices(lengths, ends, 5, True, anchor_bad=bad)
    np.testing.assert_array_equal(ours_idx.indices, theirs_idx.indices)


def test_boundary_indices_semantics():
    # ep0: len 10, real end, padding allowed -> all 10 valid
    # ep1: len 8, not real end -> 8 - H valid
    # ep2: len 12, real end, anchor_bad [0, 11] -> 12 - 2
    idx = chunk_math.BoundaryAwareIndices([10, 8, 12], [True, False, True], 5, True,
                                          anchor_bad=[[], [], [0, 11]])
    assert idx.total_frames == 30
    got = list(idx.indices)
    assert got[:10] == list(range(10))
    assert got[10:13] == [10, 11, 12]  # 8 - 5 = 3 valid
    assert got[13:] == [19] + list(range(20, 29))  # ep2 offset 18: drop 18+0 and 18+11


# ---------------------------------------------------------------------------
# E001 per-slot rescale
# ---------------------------------------------------------------------------

def _fake_stats(h=6, d=4, seed=2):
    rng = np.random.default_rng(seed)
    sigma_pooled = rng.uniform(0.5, 2.0, d)
    # sigma_slot grows with slot index, like anchor-relative deltas
    sigma_slot = np.linspace(0.02, 1.0, h)[:, None] * sigma_pooled[None, :]
    return _norm.PerSlotStats(sigma_slot=sigma_slot, provenance={}), sigma_pooled


def test_gain_c1_is_identity():
    stats, sp = _fake_stats()
    np.testing.assert_allclose(stats.gain(1.0, sp), 1.0, atol=1e-12)


def test_gain_floor_caps_amplification():
    stats, sp = _fake_stats()
    g = stats.gain(0.1, sp)
    assert g.max() <= 10.0 + 1e-6 and g.min() > 0.0
    # smallest-sigma slot hits the floor exactly
    np.testing.assert_allclose(g[0], 10.0, rtol=1e-6)
    # a slot with sigma above pooled gets gain < 1 (shrunk toward unit scale)
    stats2 = dataclasses_replace_sigma(stats, stats.sigma_slot * 2.0)
    assert stats2.gain(0.1, sp)[-1].max() < 1.0


def dataclasses_replace_sigma(stats, sigma):
    import dataclasses

    return dataclasses.replace(stats, sigma_slot=sigma)


def test_degenerate_pooled_dim_gets_gain_one():
    stats, sp = _fake_stats()
    sp = sp.copy()
    sp[1] = 0.0
    g = stats.gain(0.1, sp)
    np.testing.assert_allclose(g[:, 1], 1.0)


def test_rescale_roundtrip_identity():
    stats, sp = _fake_stats()
    g = stats.gain(0.1, sp)
    rng = np.random.default_rng(3)
    actions = rng.normal(size=(6, 4)).astype(np.float32)
    fwd = ego_transforms.PerSlotRescale(gain=g)({"actions": actions})["actions"]
    # inverse runs on padded model output
    padded = np.concatenate([fwd, rng.normal(size=(6, 2)).astype(np.float32)], axis=-1)
    inv = ego_transforms.PerSlotRescaleInverse(gain=g)({"actions": padded})["actions"]
    np.testing.assert_allclose(inv[:, :4], actions, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(inv[:, 4:], padded[:, 4:])  # pad dims untouched


def test_rescale_shape_mismatch_raises():
    stats, sp = _fake_stats()
    g = stats.gain(0.1, sp)
    with pytest.raises(ValueError):
        ego_transforms.PerSlotRescale(gain=g)({"actions": np.zeros((5, 4), np.float32)})
    with pytest.raises(ValueError):
        ego_transforms.PerSlotRescaleInverse(gain=g)({"actions": np.zeros((6, 3), np.float32)})


def test_per_slot_save_load_roundtrip(tmp_path):
    stats, _ = _fake_stats()
    _norm.save_per_slot(tmp_path, dataclasses_replace_prov(stats, {"k": "v"}))
    loaded = _norm.load_per_slot(tmp_path)
    np.testing.assert_array_equal(loaded.sigma_slot, stats.sigma_slot)
    assert loaded.provenance == {"k": "v"}


def dataclasses_replace_prov(stats, prov):
    import dataclasses

    return dataclasses.replace(stats, provenance=prov)


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

def test_append_control_mode():
    t = ego_transforms.AppendControlMode()
    out = t({"prompt": "put the bottle in the box"})
    assert out["prompt"] == "put the bottle in the box <<<control_mode>>> end effector <<<control_mode>>>"
    # idempotent
    assert t(out)["prompt"] == out["prompt"]
    with pytest.raises(ValueError):
        t({})


def test_control_mode_survives_tokenizer_cleaning():
    """PaligemmaTokenizer replaces '_' with ' ': the marker the model sees is
    '<<<control mode>>> end effector <<<control mode>>>' — assert the pipeline
    produces the pretraining-format string, not something double-mangled."""
    prompt = ego_transforms.AppendControlMode()({"prompt": "x"})["prompt"]
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    assert cleaned == "x <<<control mode>>> end effector <<<control mode>>>"


# ---------------------------------------------------------------------------
# stamp guard
# ---------------------------------------------------------------------------

def test_stamp_roundtrip_and_guard(tmp_path):
    from ego2g1 import config as _config
    from ego2g1 import stamp as _stamp

    cfg = _config.Ego2G1TrainConfig(expected_config_hash="deadbeef00000000")
    _stamp.write_stamp(tmp_path, cfg, "deadbeef00000000")
    stamp = _stamp.check_supported(tmp_path)  # must not raise
    assert stamp["feature_flags"]["per_slot_rescale"]["required"] is True

    # unknown required feature -> refuse
    import json

    p = tmp_path / _stamp.STAMP_FILENAME
    data = json.loads(p.read_text())
    data["feature_flags"]["quantum_actions"] = {"required": True}
    p.write_text(json.dumps(data))
    with pytest.raises(_stamp.UnsupportedCheckpointError):
        _stamp.check_supported(tmp_path)

    # missing stamp -> refuse
    with pytest.raises(_stamp.UnsupportedCheckpointError):
        _stamp.check_supported(tmp_path / "nope")


def test_config_c1_not_required():
    from ego2g1 import config as _config

    cfg = _config.Ego2G1TrainConfig(per_slot_floor_c=1.0)
    assert cfg.feature_flags()["per_slot_rescale"]["required"] is False
