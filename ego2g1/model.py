"""Ego2G1 Pi0 subclass: every model-behavior deviation in one place.

- action_dim_actual loss masking (OPENPI_EDITS.md, migrated from the old fork
  src edit): flow-matching loss only on the real action dims, padding excluded.
- Train-time RTC (arXiv 2512.05964, toggle `rtc_training`): per sample draw a
  prefix length d ~ Uniform{0..rtc_d_max}; the first d action tokens carry the
  ground-truth actions at per-token flow timestep t=0 (openpi convention:
  t=0 is CLEAN — the paper's τ=1-clean is the flipped convention); loss is
  masked to the postfix. Attention masks/positions are unchanged (the whole
  chunk stays one bidirectional suffix block, matching the paper).
- Per-token timestep `embed_suffix` (E002's pi0.py half): scalar timesteps
  delegate to the stock method untouched; only the per-token branch is new.
- `sample_actions_rtc`: inference-side RTC sampling (Phase 3; shipped now so
  the whole feature is toggleable). Stock `sample_actions` is inherited as-is.

With `action_dim_actual=None` and `rtc_training=False`, `compute_loss`
delegates to stock Pi0 — bitwise identity, pinned by tests/test_model.py.

Importing this module applies ego2g1.gemma_patch (required by the per-token
adaRMS path); all ego2g1 entrypoints import models through here.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config
from openpi.shared import array_typing as at

from ego2g1 import gemma_patch

gemma_patch.apply()


@dataclasses.dataclass(frozen=True)
class Ego2G1Pi0Config(_pi0_config.Pi0Config):
    # Loss is computed only on the first N action dims (rest are zero padding).
    action_dim_actual: int | None = None
    # Train-time RTC toggle (Phase 1 trains with False; code is feature-complete).
    rtc_training: bool = False
    # Max prefix length; d ~ Uniform{0..rtc_d_max} inclusive. Provisional from
    # the RTX 4060 latency estimate (TRAINING_PLAN.md §1); revisit after measuring.
    rtc_d_max: int = 16
    # E003 placeholder — gated on profiling evidence, only 1 is implemented.
    num_flow_samples: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.num_flow_samples != 1:
            raise NotImplementedError("E003 (num_flow_samples > 1) is gated on profiling; see OPENPI_EDITS.md")
        if self.rtc_training:
            if not self.pi05:
                raise ValueError("train-time RTC needs per-token adaRMS, i.e. the pi05 path")
            if not 0 <= self.rtc_d_max < self.action_horizon:
                raise ValueError(f"rtc_d_max={self.rtc_d_max} must be in [0, {self.action_horizon})")
        if self.action_dim_actual is not None and not 0 < self.action_dim_actual <= self.action_dim:
            raise ValueError(f"action_dim_actual={self.action_dim_actual} vs action_dim={self.action_dim}")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Ego2G1Pi0":
        return Ego2G1Pi0(self, rngs=nnx.Rngs(rng))

    def feature_flags(self) -> dict:
        """Model-side checkpoint flags (ego2g1.stamp adds data-side ones)."""
        return {
            # informational: loss-only, no serving-side requirement
            "action_dim_actual": self.action_dim_actual,
            # informational: an RTC-trained checkpoint degrades gracefully to d=0
            "rtc_training": self.rtc_training,
        }


class Ego2G1Pi0(_pi0.Pi0):
    def __init__(self, config: Ego2G1Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.action_dim_actual = config.action_dim_actual
        self.rtc_training = config.rtc_training
        self.rtc_d_max = config.rtc_d_max

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b s emb"] | None,
    ]:
        if timestep.ndim == 1:
            # scalar-per-sample timestep: the stock path, untouched.
            return super().embed_suffix(obs, noisy_actions, timestep)

        if not self.pi05:
            raise NotImplementedError("per-token timesteps are only implemented for the pi05 (adaRMS) path")
        b, s = timestep.shape
        if s != self.action_horizon:
            raise ValueError(f"per-token timestep length {s} != action_horizon {self.action_horizon}")

        action_tokens = self.action_in_proj(noisy_actions)
        # posemb_sincos is typechecked rank-1: flatten, embed, reshape back.
        time_emb = _pi0.posemb_sincos(
            timestep.reshape(-1), self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        ).reshape(b, s, -1)
        # pi05 time MLP (nnx.Linear acts on the last dim; works on (b, s, emb)).
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = nnx.swish(time_emb)

        input_mask = jnp.ones((b, s), dtype=jnp.bool_)
        # same block structure as stock: one suffix block, bidirectional inside.
        ar_mask = jnp.array([True] + [False] * (s - 1))
        return action_tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.action_dim_actual is None and not self.rtc_training:
            return super().compute_loss(rng, observation, actions, train=train)

        # Stock body (pi0.py:190-218) with two extensions. The rng splits and
        # every stock op are kept identical so that with rtc_training=False the
        # only difference from stock is the final dim slice (pinned by tests).
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.rtc_training:
            # RTC: freeze a random-length ground-truth prefix at t=0 (clean).
            # Independent rng stream so the (noise, time) draws above stay
            # aligned with the non-RTC path.
            d_rng = jax.random.fold_in(rng, 7)
            d = jax.random.randint(d_rng, batch_shape, 0, self.rtc_d_max + 1)
            slot = jnp.arange(self.action_horizon)
            prefix_mask = slot < d[..., None]  # (*b, ah)
            x_t = jnp.where(prefix_mask[..., None], actions, x_t)
            timestep_arg = jnp.where(prefix_mask, 0.0, time[..., None])  # (*b, ah)
        else:
            timestep_arg = time  # (*b,) -> stock scalar path in embed_suffix

        prefix_tokens, prefix_mask_tokens, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep_arg)
        input_mask = jnp.concatenate([prefix_mask_tokens, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.square(v_t - u_t)
        if self.action_dim_actual is not None:
            loss = loss[..., : self.action_dim_actual]
        loss = jnp.mean(loss, axis=-1)
        if self.rtc_training:
            # Loss on the postfix only (paper: prefix positions are conditioning,
            # not targets). Plain masking, no per-sample renormalization: samples
            # with larger d contribute (ah - d)/ah of a full sample's weight.
            loss = jnp.where(prefix_mask, 0.0, loss)
        return loss

    @at.typecheck
    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_actions: at.Float[at.Array, "b ah ad"],
        d: at.Int[at.Array, ""] | int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """RTC sampling: hold the first `d` actions of `prefix_actions` clean at
        t=0 through the whole Euler integration; integrate only the postfix.

        `prefix_actions` must already be in the MODEL action space (pooled
        quantile-normalized + per-slot rescaled + padded); rows >= d are
        ignored. With d=0 this reduces to plain sampling (same training support).
        Deployment must re-anchor the executing chunk's tail to the new anchor
        before transforming it (TRAINING_PLAN.md §1.2-1.3).
        """
        if not self.pi05:
            raise NotImplementedError("sample_actions_rtc requires the pi05 (adaRMS) path")
        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        slot_is_prefix = jnp.arange(self.action_horizon) < d  # (ah,)
        prefix_mask_bt = jnp.broadcast_to(slot_is_prefix, (batch_size, self.action_horizon))
        x_init = jnp.where(slot_is_prefix[None, :, None], prefix_actions, noise)

        # fill the KV cache with a forward pass of the prefix (stock, pi0.py:238-241)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # per-token timesteps: frozen prefix at 0, postfix at the current t
            time_tok = jnp.where(prefix_mask_bt, 0.0, jnp.broadcast_to(time, (batch_size, self.action_horizon)))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_tok)
            # attention plumbing: stock (pi0.py:248-263)
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = jnp.broadcast_to(
                prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_mask.shape[1])
            )
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            # prefix rows stay exactly the committed actions
            v_t = jnp.where(slot_is_prefix[None, :, None], 0.0, v_t)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (x_init, 1.0))
        return x_0

    @at.typecheck
    def sample_actions_guided(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_actions: at.Float[at.Array, "b ah ad"],
        weights: at.Float[at.Array, " ah"],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        max_guidance_weight: float = 10.0,
        use_vjp: bool = True,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Inference-time RTC (arXiv 2506.07339) for a checkpoint NOT trained with
        RTC. Adds an inpainting guidance term to the flow field, pulling the new
        chunk's early slots toward `prefix_actions` (the previous chunk's tail,
        re-anchored and pushed through the input transforms into MODEL space).

        Unlike `sample_actions_rtc`, this uses SCALAR timesteps — the stock
        `embed_suffix` path. That is the whole point: with `rtc_training=False`
        the model has never seen a per-token timestep, so feeding it one is
        out-of-distribution. Here the model is called exactly as it was trained
        and the entire RTC effect lives in the velocity correction.

        openpi flow convention (from compute_loss): x_t = t*noise + (1-t)*x_1 and
        v = noise - x_1, hence x_1 = x_t - t*v. LeRobot's RTC uses the same
        convention, so its formulas port directly; PI's Kinetix reference uses the
        flipped one (tau = 1 - t) and its signs must NOT be copied across.

        `weights` is the (ah,) soft mask from ego2g1.serve.rtc.prefix_weights:
        1.0 on slots already committed, decaying to 0 across the overlap, 0
        beyond it. Slots with weight 0 are generated freely.

        `use_vjp=False` selects the identity-Jacobian approximation (correction =
        error, no backward pass). That is what the LeRobot/unitree-deploy port
        actually computes, due to a `requires_grad_` ordering bug; it is free, so
        it is kept here as a deliberate A/B fallback rather than an accident.
        """
        if not self.pi05:
            raise NotImplementedError("sample_actions_guided requires the pi05 (adaRMS) path")
        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # KV cache over the (constant) prefix: x_t enters only the suffix, so the
        # VJP below traverses the action expert, never the 3B VLM.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Guide ONLY the real action dims. Dims [action_dim_actual, action_dim) are
        # zero padding: compute_loss slices them out, so the model gets no gradient
        # there and its output is unconstrained garbage. prefix_actions is zero on
        # those dims, so the error there is a meaningless residual — and the VJP's
        # J^T would smear it back across the REAL dims. Mask it at the source.
        dim_mask = jnp.ones((self.action_dim,), dtype=jnp.float32)
        if self.action_dim_actual is not None:
            dim_mask = dim_mask.at[self.action_dim_actual:].set(0.0)
        w = weights[None, :, None] * dim_mask[None, None, :]  # (1, ah, ad)

        def velocity(x_t, time):
            # scalar-per-sample timestep -> stock embed_suffix (trained path)
            t_vec = jnp.broadcast_to(time, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, t_vec)
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = jnp.broadcast_to(
                prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_mask.shape[1])
            )
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=pos,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry

            def clean(x):
                """x -> (predicted clean chunk, velocity). x_1 = x_t - t*v."""
                v = velocity(x, time)
                return x - time * v, v

            if use_vjp:
                x_1, vjp_fn, v_t = jax.vjp(clean, x_t, has_aux=True)
                err = (prefix_actions - x_1) * w
                correction = vjp_fn(err)[0]
            else:
                x_1, v_t = clean(x_t)
                err = (prefix_actions - x_1) * w
                correction = err  # identity-Jacobian approximation

            # guidance weight: (t^2 + (1-t)^2) / (t*(1-t)), clipped. U-shaped, so it
            # is large at both ends of the trajectory and the clip is what keeps it
            # finite there (the paper's beta; divergence at t->0 with few steps).
            denom = time * (1.0 - time)
            gw = jnp.where(
                denom > 1e-6,
                (time**2 + (1.0 - time) ** 2) / jnp.maximum(denom, 1e-6),
                max_guidance_weight,
            )
            gw = jnp.minimum(gw, max_guidance_weight)

            # v decreases -> x_1 increases (x_1 = x_t - t*v), so subtract to move
            # x_1 toward the target where err = (target - x_1) > 0.
            v_guided = v_t - gw * correction
            return x_t + dt * v_guided, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
