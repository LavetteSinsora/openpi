"""Serve an ego2g1 checkpoint over websocket.

    uv run python -m ego2g1.serve --checkpoint checkpoints/ego2g1_pi05/<exp>/<step>

openpi's stock scripts/serve_policy.py cannot do this: it resolves configs
through openpi's _CONFIGS registry, and ego2g1 is deliberately not in it (the
config is rebuilt from the checkpoint's stamp instead). This is the entrypoint
TRAINING_PLAN/E002 calls for — it goes through ego2g1.serve.create_policy, which
applies the gemma patch (via the ego2g1.model import) and enforces the stamp
guard. Serving these checkpoints any other way runs them with silently wrong
semantics.

RTC is not a flag here. Whether a request gets guided sampling, pinned-prefix
sampling, or plain sampling is decided by the checkpoint's stamp and by whether
the client sent a prev_chunk. See ego2g1/serve/rtc.py.
"""

import dataclasses
import logging
import socket

import tyro

from openpi.serving import websocket_policy_server

from ego2g1.serve import policy as _policy
from ego2g1.serve import rtc as _rtc


@dataclasses.dataclass
class Args:
    # Checkpoint STEP dir (the one holding params/). The stamp is read from it
    # or from its parent.
    checkpoint: str

    port: int = 8000
    host: str = "0.0.0.0"

    # Used only when the request carries no "prompt".
    default_prompt: str | None = None

    # Override norm-asset resolution. Must contain BOTH norm_stats.json and
    # per_slot_stats.npz. Normally unnecessary — the checkpoint carries its own.
    assets_dir: str | None = None

    # --- RTC (inference-time; ignored when the client sends no prev_chunk) ---
    rtc: bool = True
    # Slots of soft continuity BEYOND the committed prefix, i.e. the decay band is
    # [d, d + overlap). Relative to d on purpose — see RTCConfig.
    overlap: int = 10
    max_guidance_weight: float = 10.0
    # False = identity-Jacobian approximation (free, but not the published
    # algorithm — it is what the LeRobot port accidentally computes). Kept for A/B.
    use_vjp: bool = True
    num_steps: int = 10

    # Dump every request/response for debugging.
    record: bool = False


def main(args: Args) -> None:
    policy = _policy.create_policy(
        args.checkpoint,
        default_prompt=args.default_prompt,
        assets_dir=args.assets_dir,
        rtc=_rtc.RTCConfig(
            enabled=args.rtc,
            overlap=args.overlap,
            max_guidance_weight=args.max_guidance_weight,
            use_vjp=args.use_vjp,
            num_steps=args.num_steps,
        ),
    )

    meta = policy.metadata
    flags = meta["ego2g1_stamp"]["feature_flags"]
    rtc_training = bool(flags.get("rtc_training", {}).get("value", False)) \
        if isinstance(flags.get("rtc_training"), dict) else bool(flags.get("rtc_training", False))

    logging.info("checkpoint: %s", args.checkpoint)
    logging.info("config hash: %s", meta["ego2g1_stamp"]["ego2g1_config_hash"])
    logging.info(
        "RTC: %s (checkpoint is %s-trained -> %s sampler when a prefix arrives)",
        "on" if args.rtc else "off",
        "rtc" if rtc_training else "plain",
        "pinned" if rtc_training else ("guided/vjp" if args.use_vjp else "guided/identity"),
    )

    if args.record:
        # Capture metadata BEFORE wrapping: PolicyRecorder is a bare BasePolicy and
        # has no .metadata, so reading it off the wrapper is an AttributeError at
        # startup. Stock serve_policy.py gets this ordering right; we had inverted it.
        from openpi.policies import policy as _openpi_policy
        policy = _openpi_policy.PolicyRecorder(policy, "policy_records")

    logging.info("serving on %s:%d (host %s)", args.host, args.port, socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=meta,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
