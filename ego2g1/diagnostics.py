"""Offline diagnostics: attention allocation and image-patch representation gap.

Not part of the train step — run against a checkpoint (or live model) on a few
probe samples every N steps / per checkpoint.

`attention_allocation`: for each transformer layer, how much attention mass the
action (suffix) tokens allocate to token groups (per-camera image tokens, text,
action tokens). Works by re-running the joint prefix+suffix forward with a
manual layer loop over the scanned/stacked layer params, using a prob-capturing
copy of stock gemma.Attention (sow can't cross the stock nn.scan+nn.remat, so
the loop replaces the scan; Block itself is reused, so norms/MLP/residuals are
the stock code paths).

`image_patch_gap`: SigLIP patch-token comparison between two images (e.g. the
same scene with a human hand vs a composited robot hand) — per-patch cosine
similarity grid + summary stats, to quantify the visual embodiment gap.
"""

import dataclasses
import functools

import einops
import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.gemma as gemma
import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.tokenizer as _paligemma_tokenizer


class SowingAttention(gemma.Attention):
    """Stock gemma.Attention with the softmax probs sown to `intermediates`.

    Body is a copy of stock (gemma.py Attention.__call__) with one added
    `self.sow` line; only ever used inside the manual layer loop below (never
    in training), so no rebind of gemma symbols is needed.
    """

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)

        import openpi.models.lora as lora

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=gemma._name("qkv_einsum", i),  # noqa: SLF001
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=gemma._name("q_einsum", i),  # noqa: SLF001
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=gemma._name("kv_einsum", i),  # noqa: SLF001
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = gemma._apply_rope(q, positions=positions)  # noqa: SLF001
        q *= self.configs[0].head_dim ** -0.5
        k = gemma._apply_rope(k, positions=positions)  # noqa: SLF001

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
        big_neg = -2.3819763e38
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # the one deviation from stock: keep the probs (float32 for stable sums)
        self.sow("intermediates", "probs", jax.nn.softmax(masked_logits, axis=-1))

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=gemma._name("attn_vec_einsum", i),  # noqa: SLF001
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)
        return out, (k, v)


@dataclasses.dataclass(frozen=True)
class TokenGroups:
    """Per-sample masks over the joint prefix+suffix sequence.

    Masks, not [start, end) ranges: on pi05 the state lives INSIDE the prompt as
    digit tokens, and its token count varies per sample (each of the 30 values is
    1-3 characters, packed differently by sentencepiece), so the task/state
    boundary is not shared across a batch."""

    names: tuple[str, ...]
    masks: np.ndarray  # (G, B, S) float32


@functools.lru_cache(maxsize=1)
def digit_token_ids() -> np.ndarray:
    """Vocab ids whose piece is a bare number — i.e. exactly the tokens the pi05
    state segment is made of. The task string and the <<<control_mode>>> marker
    contain no digits, so this splits prompt-from-state exactly on our template;
    under the "unknown" sentinel it selects nothing, and text/state reads a true 0."""
    sp = _paligemma_tokenizer.PaligemmaTokenizer(48)._tokenizer  # noqa: SLF001
    return np.asarray(
        [i for i in range(sp.get_piece_size()) if sp.id_to_piece(i).replace("▁", "").isdigit()]
    )


def token_groups(
    model: _pi0.Pi0, obs: _model.Observation, *, state_token_ids: np.ndarray | None = None
) -> TokenGroups:
    """Groups matching embed_prefix's concatenation order (images in obs.images
    order, then text) + the action suffix. With `state_token_ids`, the text group
    splits into text/task and text/state. Prompt PADDING belongs to no group —
    it is masked out of attention, so it carries ~0 mass and the groups still
    partition the attention simplex."""
    b = obs.state.shape[0]
    names: list[str] = []
    spans: list[tuple[int, int]] = []
    start = 0
    n = None  # all cameras share one resolution -> one SigLIP call to count tokens
    for name in obs.images:
        if n is None:
            image_tokens, _ = model.PaliGemma.img(obs.images[name], train=False)
            n = image_tokens.shape[1]
        names.append(f"img/{name}")
        spans.append((start, start + n))
        start += n

    text_start, text_len = None, 0
    if obs.tokenized_prompt is not None:
        text_start = start
        text_len = obs.tokenized_prompt.shape[1]
        start += text_len

    prefix_len = start
    state_span = None
    if not model.pi05:  # pi0 only: a continuous state token in the suffix
        state_span = (prefix_len, prefix_len + 1)
        start += 1
    action_span = (start, start + model.action_horizon)
    total = action_span[1]

    def const(a: int, z: int) -> np.ndarray:
        m = np.zeros((b, total), np.float32)
        m[:, a:z] = 1.0
        return m

    out_names, out_masks = [], []
    for name, (a, z) in zip(names, spans, strict=True):
        out_names.append(name)
        out_masks.append(const(a, z))

    if text_start is not None:
        ids = np.asarray(obs.tokenized_prompt)
        valid = (
            np.asarray(obs.tokenized_prompt_mask).astype(bool)
            if obs.tokenized_prompt_mask is not None
            else np.ones(ids.shape, bool)
        )
        if state_token_ids is not None:
            is_state = np.isin(ids, state_token_ids) & valid
            text_groups = [("text/task", valid & ~is_state), ("text/state", is_state)]
        else:
            text_groups = [("text", valid)]
        for name, sub in text_groups:
            m = np.zeros((b, total), np.float32)
            m[:, text_start : text_start + text_len] = sub.astype(np.float32)
            out_names.append(name)
            out_masks.append(m)

    if state_span is not None:
        out_names.append("state")
        out_masks.append(const(*state_span))
    out_names.append("action")
    out_masks.append(const(*action_span))
    return TokenGroups(names=tuple(out_names), masks=np.stack(out_masks))


def _llm_params(model: _pi0.Pi0) -> dict:
    """Pure linen param tree of the gemma Module inside the nnx bridge.
    The ToNNX bridge flattens the 'params' collection away: top-level keys are
    the module names (embedder, final_norm*, layers)."""
    state = nnx.state(model.PaliGemma.llm).to_pure_dict()
    return state.get("params", state)


def attention_allocation(
    model: _pi0.Pi0,
    obs: _model.Observation,
    actions,
    *,
    time_value: float = 0.5,
    rng=None,
    state_token_ids: np.ndarray | None = None,
) -> dict:
    """Train-style joint forward at one flow timestep; returns where the
    action-token queries put their attention mass.

    `state_token_ids` (pass diagnostics.digit_token_ids()) splits the text group
    into text/task and text/state — on pi05 the state IS text, so without the
    split the two are indistinguishable and the state-reliance question the probe
    exists to answer cannot be read off it.

    Returns dict with:
      group_names: (G,) names
      per_layer:  (L, G) — action-row attention mass per group, mean over
                  batch/heads/slots
      per_slot:   (ah, G) — mean over batch/heads/layers
      entropy_per_layer: (L,) — mean attention entropy of action rows (nats)
    """
    rng = rng if rng is not None else jax.random.key(0)
    obs = _model.preprocess_observation(None, obs, train=False)
    groups = token_groups(model, obs, state_token_ids=state_token_ids)

    noise = jax.random.normal(rng, actions.shape)
    t = jnp.full(actions.shape[0], time_value, dtype=jnp.float32)
    x_t = time_value * noise + (1 - time_value) * actions

    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar, adarms_cond = model.embed_suffix(obs, x_t, t)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)[:, None, :, :]
    positions = jnp.cumsum(input_mask, axis=1) - 1

    params = _llm_params(model)
    layer_params = params["layers"]
    depth = jax.tree.leaves(layer_params)[0].shape[0]

    llm_module = model.PaliGemma.llm.module  # the wrapped gemma.Module (patched)
    embed_dtype = jnp.dtype(llm_module.embed_dtype)
    xs = [prefix_tokens.astype(embed_dtype), suffix_tokens.astype(embed_dtype)]
    cond = [None, adarms_cond]

    block = _SowingBlock(configs=tuple(llm_module.configs))

    n_action = model.action_horizon
    per_layer, per_slot_layers, entropies = [], [], []
    for layer in range(depth):
        p_l = jax.tree.map(lambda x: x[layer], layer_params)
        (xs, _), inter = block.apply(
            {"params": p_l}, xs, None, positions, attn_mask, cond, True,
            mutable=["intermediates"],
        )
        probs = inter["intermediates"]["attn"]["probs"][0]  # (B, K, G, T, S) f32
        probs = einops.rearrange(probs, "B K G T S -> B (K G) T S")
        action_rows = probs[:, :, -n_action:, :]  # (B, H, ah, S)
        # (B,H,ah,S) x (G,B,S) -> (B,H,ah,G): per-sample masks, so the task/state
        # split can sit at a different token index in every row of the batch.
        mass = np.einsum("bhas,gbs->bhag", np.asarray(action_rows, np.float32), groups.masks)
        per_layer.append(mass.mean(axis=(0, 1, 2)))
        per_slot_layers.append(mass.mean(axis=(0, 1)))
        p = np.asarray(action_rows)
        entropies.append(float((-(p * np.log(np.clip(p, 1e-12, 1.0))).sum(-1)).mean()))

    return {
        "group_names": groups.names,
        "per_layer": np.stack(per_layer),               # (L, G)
        "per_slot": np.stack(per_slot_layers).mean(0),  # (ah, G)
        "entropy_per_layer": np.asarray(entropies),     # (L,)
    }


class _SowingBlock(gemma.Block):
    """Stock Block that instantiates SowingAttention instead of gemma.Attention.

    Body copy is avoided: we temporarily shadow the module-global Attention
    only for the duration of apply (the reference is resolved at trace time).
    """

    @nn.compact
    def __call__(self, *args, **kwargs):
        original = gemma.Attention
        gemma.Attention = SowingAttention
        try:
            return super().__call__(*args, **kwargs)
        finally:
            gemma.Attention = original


def image_patch_gap(model: _pi0.Pi0, image_a, image_b) -> dict:
    """SigLIP patch-token gap between two aligned images in [-1, 1] (H, W, 3).

    Returns per-patch cosine similarity on the SigLIP grid plus summary stats.
    Use on frame pairs that differ only in hand appearance (human vs composited
    robot) to localize and quantify the visual embodiment gap; track across
    checkpoints to see whether fine-tuning shrinks it.
    """
    a = jnp.asarray(image_a)[None]
    b = jnp.asarray(image_b)[None]
    tok_a, _ = model.PaliGemma.img(a, train=False)
    tok_b, _ = model.PaliGemma.img(b, train=False)
    ta = np.asarray(tok_a[0], dtype=np.float64)
    tb = np.asarray(tok_b[0], dtype=np.float64)
    cos = (ta * tb).sum(-1) / (np.linalg.norm(ta, axis=-1) * np.linalg.norm(tb, axis=-1) + 1e-12)
    n = cos.shape[0]
    side = int(round(np.sqrt(n)))
    grid = cos.reshape(side, side) if side * side == n else cos[None, :]
    l2 = np.linalg.norm(ta - tb, axis=-1) / (np.linalg.norm(ta, axis=-1) + 1e-12)
    return {
        "cosine_grid": grid,                    # (side, side)
        "cosine_mean": float(cos.mean()),
        "cosine_p05": float(np.quantile(cos, 0.05)),
        "rel_l2_mean": float(l2.mean()),
        "rel_l2_max": float(l2.max()),
    }
