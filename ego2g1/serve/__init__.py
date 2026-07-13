"""Policy server: the checkpoint half of deployment.

A pure function of the observation dict — it knows nothing about robots, DDS,
cameras, IK, or joints. Everything stateful and real-time lives in ego2g1.deploy,
in a separate process, so a JAX OOM here cannot drop robot control there.

    uv run python -m ego2g1.serve --checkpoint checkpoints/<name>/<exp>/<step>

Importing this package pulls in JAX (via ego2g1.model). ego2g1.deploy must never
import it.
"""

from ego2g1.serve.policy import Ego2G1Policy, create_policy
from ego2g1.serve.rtc import AttentionSchedule, RTCConfig, Sampler

__all__ = ["Ego2G1Policy", "create_policy", "RTCConfig", "AttentionSchedule", "Sampler"]
