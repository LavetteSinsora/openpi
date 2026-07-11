"""E002: per-token adaRMS timestep conditioning without editing src/openpi.

`openpi.models.gemma` resolves `RMSNorm` (Block.__call__, gemma.py:303/318)
and `Module` (pi0.py:74) as module globals at call/construction time, so
rebinding the two symbols before any model is built is fully effective.
`apply()` does exactly that, guarded by a fingerprint of the stock source:
if an upstream pull changes either class, we refuse to patch instead of
silently patching drifted code.

Semantics (OPENPI_EDITS.md E002): `adarms_cond` may be `(b, emb)` (stock,
bit-identical code path) or `(b, s, emb)` (per-suffix-token). The modulation
Dense infers features from the last dim, so the param tree is unchanged;
`_gated_residual` and the layer scan broadcast rank-3 gates with no change.

Rationale for a runtime patch over a fork commit: keeps src/openpi bit-stock
(any openpi checkout works end to end), at the cost of monkeypatch
indirection. See TRAINING_PLAN.md §3.7a. Checkpoints trained with per-token
features must still be stamp-guarded (ego2g1.stamp): the param tree stays
stock-shaped, so unpatched code would run them with silently wrong semantics.
"""

import hashlib
import inspect
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.gemma as gemma
import openpi.shared.array_typing as at

# sha256 over inspect.getsource of the stock classes this module replaces
# (openpi commit: see `git -C third_party/openpi log -1 --format=%H upstream/main`).
# If these fail, re-derive PerTokenRMSNorm/PerTokenModule against the new
# stock code before updating the digests.
_STOCK_FINGERPRINTS = {
    "RMSNorm": "e17007080913b14e71ea380f1c2ae66f6b4528ee79e7df27a5276ed22d5168e7",
    "Module": "5c9b7e479c2e0a5dfe85018dc36590fc63007ad46aee6f830c2dbeba7712be94",
}

# Captured at first import, before apply() can have rebound them.
_STOCK_RMSNORM = gemma.RMSNorm
_STOCK_MODULE = gemma.Module


@at.typecheck
class PerTokenRMSNorm(nn.Module):
    """Stock gemma.RMSNorm plus a rank-3 (per-token) adaRMS cond branch.

    cond (b, emb): bit-identical to stock (same ops, same param creation).
    cond (b, s, emb): the same Dense (kernel (emb, 3*emb) — zero new params),
    split without the broadcast `[:, None]`; gate comes back (b, s, emb).
    """

    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            # regular RMSNorm (stock)
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        if cond.ndim == 2:
            # stock path: broadcast one modulation vector over the sequence
            modulation = modulation[:, None, :]
        elif cond.ndim != 3:
            raise ValueError(f"adarms cond must be rank 2 or 3, got shape {cond.shape}")
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate


# Flax derives default submodule scope names from the class name; every
# RMSNorm instance in gemma gets an explicit name=, but keep repr parity.
PerTokenRMSNorm.__name__ = "RMSNorm"
PerTokenRMSNorm.__qualname__ = "RMSNorm"


class PerTokenModule(_STOCK_MODULE):
    """gemma.Module with the adarms_cond typecheck widened to allow (b, s, emb).

    __call__ body is a copy of stock (gemma.py:389-412) — the layer scan
    passes adarms_cond through opaquely, so widening the annotation is the
    only change. Covered by the fingerprint guard on the stock class.
    """

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | at.Float[at.Array, "b _s _d"] | None] | None = None,
        *,
        kv_cache: gemma.KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], gemma.KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache


# The gemma Module is the top level of its linen tree (wrapped directly by
# nnx_bridge.ToNNX), so the class name never enters parameter paths; forced
# anyway for repr parity and safety against future nesting.
PerTokenModule.__name__ = "Module"
PerTokenModule.__qualname__ = "Module"


class StockSourceChangedError(RuntimeError):
    """The openpi checkout's gemma.py no longer matches the code this patch was derived from."""


def _fingerprint(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def verify_fingerprints() -> None:
    for name, obj in [("RMSNorm", _STOCK_RMSNORM), ("Module", _STOCK_MODULE)]:
        actual = _fingerprint(obj)
        expected = _STOCK_FINGERPRINTS[name]
        if actual != expected:
            raise StockSourceChangedError(
                f"gemma.{name} source digest {actual} != pinned {expected}. "
                "Upstream gemma.py changed: re-derive ego2g1/gemma_patch.py against the "
                "new stock code, re-run the golden tests, then update the pinned digest."
            )


def apply() -> None:
    """Idempotently rebind gemma.RMSNorm / gemma.Module to the per-token versions.

    Must run before any model construction; ego2g1.model applies it at import,
    and all ego2g1 entrypoints import through ego2g1.model.
    """
    if getattr(gemma, "_ego2g1_per_token_patch", False):
        return
    verify_fingerprints()
    gemma.RMSNorm = PerTokenRMSNorm
    gemma.Module = PerTokenModule
    gemma._ego2g1_per_token_patch = True  # noqa: SLF001


def is_applied() -> bool:
    return bool(getattr(gemma, "_ego2g1_per_token_patch", False))
