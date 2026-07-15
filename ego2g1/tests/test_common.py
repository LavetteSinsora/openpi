"""ego2g1.common: encoding round-trips, and the RTC re-anchor invariant.

These guard the two things that would fail silently on the robot: a vec9 layout
that disagrees with training (translation vs rot6d order, rows vs columns), and
a re-anchor that quietly moves the targets it was supposed to preserve.
"""

import numpy as np
import pytest

from ego2g1.common import layout, se3


def _rand_se3(rng, shape=()):
    """Random rigid transforms via QR (proper rotations only)."""
    q, r = np.linalg.qr(rng.normal(size=(*shape, 3, 3)))
    q = q * np.sign(np.einsum("...ii->...i", r))[..., None, :]
    det = np.linalg.det(q)
    q[..., :, 0] *= np.sign(det)[..., None]  # force det=+1
    T = np.zeros((*shape, 4, 4))
    T[..., :3, :3] = q
    T[..., :3, 3] = rng.normal(size=(*shape, 3))
    T[..., 3, 3] = 1.0
    return T


# --- encoding ---------------------------------------------------------------


def test_vec9_roundtrip():
    rng = np.random.default_rng(0)
    T = _rand_se3(rng, (32,))
    assert se3.vec9_to_se3(se3.se3_to_vec9(T)) == pytest.approx(T, abs=1e-12)


def test_vec9_layout_is_translation_first():
    """Guards against a rot6d-first / translation-first swap."""
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    v = se3.se3_to_vec9(T)
    assert v[:3] == pytest.approx([1.0, 2.0, 3.0])
    assert v[3:9] == pytest.approx([1, 0, 0, 0, 1, 0])  # columns of I


def test_rot6d_is_columns_not_rows():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90deg z
    d6 = se3.mat_to_6d(R)
    assert d6 == pytest.approx([0, 1, 0, -1, 0, 0])  # R[:,0], R[:,1]
    assert se3.rot6d_to_mat(d6) == pytest.approx(R, abs=1e-12)


def test_rot6d_gram_schmidt_repairs_nonorthogonal_input():
    """A regressed 6-vector is not orthonormal; decoding must still give SO(3)."""
    rng = np.random.default_rng(1)
    R = se3.rot6d_to_mat(rng.normal(size=6))
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_se3_inv_matches_dense_inverse():
    rng = np.random.default_rng(2)
    T = _rand_se3(rng, (8,))
    assert se3.se3_inv(T) == pytest.approx(np.linalg.inv(T), abs=1e-12)


# --- deployment math --------------------------------------------------------


def test_compose_inverts_the_training_delta():
    """Training does delta = G(t)^-1 @ G(t+k); deployment must undo exactly that."""
    rng = np.random.default_rng(3)
    anchor = _rand_se3(rng)
    absolute = _rand_se3(rng, (50,))
    deltas = se3.se3_to_vec9(se3.se3_inv(anchor) @ absolute)  # what the loader stores
    assert se3.compose(anchor, deltas) == pytest.approx(absolute, abs=1e-12)


def test_reanchor_preserves_absolute_targets():
    """THE RTC invariant: re-anchoring changes the deltas but not what they mean.

    If this fails, the guidance target pulls the new chunk toward poses the old
    chunk never intended, and the robot lunges at the chunk seam.
    """
    rng = np.random.default_rng(4)
    old = {h: _rand_se3(rng) for h in layout.HANDS}
    new = {h: _rand_se3(rng) for h in layout.HANDS}

    actions = np.zeros((50, layout.DIM), dtype=np.float32)
    for h in layout.HANDS:
        actions[:, layout.EEF[h]] = se3.se3_to_vec9(_rand_se3(rng, (50,)))
        actions[:, layout.HAND[h]] = rng.uniform(0, 1, size=(50, layout.HAND_DIM))

    rehomed = se3.reanchor_chunk(actions, old, new)

    for h in layout.HANDS:
        before = se3.compose(old[h], actions[:, layout.EEF[h]])
        after = se3.compose(new[h], rehomed[:, layout.EEF[h]])
        assert after == pytest.approx(before, abs=1e-6)


def test_reanchor_is_identity_when_the_anchor_did_not_move():
    rng = np.random.default_rng(5)
    anchor = {h: _rand_se3(rng) for h in layout.HANDS}
    actions = np.zeros((10, layout.DIM), dtype=np.float32)
    for h in layout.HANDS:
        actions[:, layout.EEF[h]] = se3.se3_to_vec9(_rand_se3(rng, (10,)))

    rehomed = se3.reanchor_chunk(actions, anchor, anchor)
    assert rehomed == pytest.approx(actions, abs=1e-6)


def test_reanchor_leaves_hand_dims_untouched():
    """Hand commands are absolute [0,1], not deltas. Transforming them is a bug."""
    rng = np.random.default_rng(6)
    old = {h: _rand_se3(rng) for h in layout.HANDS}
    new = {h: _rand_se3(rng) for h in layout.HANDS}

    actions = np.zeros((10, layout.DIM), dtype=np.float32)
    for h in layout.HANDS:
        actions[:, layout.EEF[h]] = se3.se3_to_vec9(_rand_se3(rng, (10,)))
        actions[:, layout.HAND[h]] = rng.uniform(0, 1, size=(10, layout.HAND_DIM))

    rehomed = se3.reanchor_chunk(actions, old, new)
    for h in layout.HANDS:
        assert rehomed[:, layout.HAND[h]] == pytest.approx(actions[:, layout.HAND[h]])


# --- layout -----------------------------------------------------------------


def test_slices_match_the_dataset_reader():
    """open_loop_eval/dataset_io.py has the authoritative, verified slices."""
    assert (layout.EEF["left"], layout.HAND["left"]) == (slice(0, 9), slice(9, 15))
    assert (layout.EEF["right"], layout.HAND["right"]) == (slice(15, 24), slice(24, 30))
    assert layout.DIM == 30
    assert layout.ARM_DOF == 14


def test_split_join_roundtrip():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(5, layout.DIM))
    assert layout.join(layout.split(v)) == pytest.approx(v)


def test_common_imports_without_jax():
    """ego2g1.deploy runs on a robot PC with no JAX. Keep common/ clean."""
    import sys
    for mod in ("ego2g1.common.se3", "ego2g1.common.layout", "ego2g1.chunk_math"):
        sys.modules.pop(mod, None)
    sys.modules.pop("jax", None)
    import ego2g1.common.se3  # noqa: F401
    assert "jax" not in sys.modules
