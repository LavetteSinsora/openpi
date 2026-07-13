"""Real-Time Chunking: the soft prefix mask, and which sampler a checkpoint gets.

Two different things are called RTC, and this repo can do both:

  inference-time RTC (arXiv 2506.07339) — no retraining. A guidance term is added
    to the flow field, pulling the new chunk's early slots toward the previous
    chunk's tail. Works on any flow-matching checkpoint. Costs a VJP through the
    action expert per denoising step. -> model.sample_actions_guided

  train-time RTC (arXiv 2512.05964) — the model is trained to condition on clean
    action prefixes, so at inference the prefix is simply pinned and there is no
    overhead at all. Requires rtc_training=True at training time.
    -> model.sample_actions_rtc

The client is identical for both: it always sends (re-anchored previous chunk, d).
The CHECKPOINT decides which sampler runs, via its stamp. A future retrain with
rtc_training=True is therefore a server-side swap the robot never notices.

Weight schedule ported from the LeRobot/PI reference (see get_prefix_weights in
policy_adapter/rtc/modeling_rtc.py). The guidance math itself lives in
ego2g1.model.sample_actions_guided, next to its train-time sibling, because it
shares the KV-cache and attention plumbing with it.
"""

import dataclasses
import enum
import math

import numpy as np


class AttentionSchedule(str, enum.Enum):
    ZEROS = "zeros"   # hard: 1 up to d, then nothing
    ONES = "ones"     # full weight across the whole overlap
    LINEAR = "linear"
    EXP = "exp"       # recommended


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """Inference-time RTC knobs.

    `overlap` is deliberately RELATIVE to d, not an absolute slot index.

    The reference implementations take an absolute `execution_horizon` and leave
    `d < execution_horizon` as an unwritten invariant. That is a trap: the client
    discards slots [0, d) (they elapsed during inference) and begins executing at
    slot d, so if d >= execution_horizon the mask is 1.0 on exactly the slots that
    get thrown away and 0.0 on the first slot that actually runs. RTC then costs a
    VJP per denoising step and constrains nothing. Making the band relative means
    the slot we splice at is always inside it, by construction.
    """

    enabled: bool = True
    # Slots of soft continuity BEYOND the committed prefix. The decay band is
    # [d, d + overlap). Bigger = smoother seams, less reactive.
    overlap: int = 10
    # Clip on the guidance weight, which diverges at both ends of the trajectory.
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: AttentionSchedule = AttentionSchedule.EXP
    # False => identity-Jacobian approximation: no backward pass, free, but not
    # the published algorithm. Kept as a deliberate A/B, see model.py.
    use_vjp: bool = True
    num_steps: int = 10

    def __post_init__(self):
        if self.overlap < 1:
            raise ValueError("overlap must be >= 1, else the splice slot is unconstrained")


def prefix_weights(d: int, overlap: int, horizon: int, n_real: int | None = None,
                   schedule: AttentionSchedule = AttentionSchedule.EXP) -> np.ndarray:
    """Soft mask over the new chunk's slots. Shape (horizon,).

      i < d                -> 1.0   already committed: these WILL execute while we
                                    infer, so the new chunk has no choice but to
                                    agree with them
      d <= i < d + overlap -> decay: the continuity band. The client splices at
                                    slot d, so this is where the seam actually is
                                    and it MUST carry weight.
      i >= d + overlap     -> 0.0   generated freely

    `n_real` is the number of genuine rows in the prefix. Everything past it is
    zero padding, and a zero vec9 is not a pose (rot6d_to_mat(zeros) is the zero
    matrix, det 0) — guiding toward it would drag the chunk toward a degenerate
    "pose". So the band is truncated at n_real, and d is capped there too. The
    LeRobot reference clamps exactly this; the original port dropped the clamp.
    """
    total = int(horizon)
    limit = total if n_real is None else max(0, min(int(n_real), total))

    start = int(np.clip(d, 0, limit))          # never freeze more than we actually have
    end = min(start + max(1, int(overlap)), limit)

    if schedule is AttentionSchedule.ZEROS:
        w = np.zeros(total, dtype=np.float32)
        w[:start] = 1.0
        return w
    if schedule is AttentionSchedule.ONES:
        w = np.zeros(total, dtype=np.float32)
        w[:end] = 1.0
        return w

    n_decay = max(end - start, 0)
    if n_decay > 0:
        decay = np.linspace(1.0, 0.0, n_decay + 2, dtype=np.float32)[1:-1]
        if schedule is AttentionSchedule.EXP:
            decay = decay * np.expm1(decay) / (math.e - 1.0)
    else:
        decay = np.zeros(0, dtype=np.float32)

    w = np.zeros(total, dtype=np.float32)
    w[:start] = 1.0
    w[start:start + n_decay] = decay
    return w


class Sampler(str, enum.Enum):
    PLAIN = "plain"      # no prefix supplied (first chunk, or --blocking)
    GUIDED = "guided"    # inference-time RTC
    PINNED = "pinned"    # train-time RTC


def select_sampler(*, rtc_training: bool, has_prefix: bool, rtc_enabled: bool) -> Sampler:
    """The dispatch. Driven by the checkpoint stamp, never by a user flag."""
    if not has_prefix or not rtc_enabled:
        return Sampler.PLAIN
    return Sampler.PINNED if rtc_training else Sampler.GUIDED
