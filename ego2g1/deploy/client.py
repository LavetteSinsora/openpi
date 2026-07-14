"""Websocket client to the policy server, plus the delay budget.

The request is the plain openpi obs dict, optionally carrying the RTC prefix:

    {"observation/image": HWC uint8,
     "observation/state": (30,) float32,
     "prompt": str,
     "prev_chunk": (H, 30) float32   # optional: re-anchored leftover
     "d": int}                       # optional: inference delay, in ticks

and the reply is {"actions": (H, 30), "policy_timing": {...}, "rtc": {...}}.

The client does NOT choose the sampler. It always sends the prefix when it has
one; the checkpoint's stamp decides whether that becomes guided or pinned RTC.
That is what makes a future rtc_training retrain a server-side swap.

The image is resized to the model's 224x224 HERE, on the wire, not in the camera
(which keeps handing out the raw frame — that is what check/Rerun must see). We
send the full head frame otherwise: ~920 KB at 640x480, ~2.7 MB at 720p, raw
msgpack with websocket compression off, every inference. On the robot LAN that is
free; through an ssh tunnel to a remote server it is the dominant latency term,
and latency is `d`. Resizing first costs 150 KB instead.

Doing it twice is safe only because the target matches the server's
ResizeImages(224, 224) exactly: resize_with_pad early-returns identity on an image
that is already the target size, so the server's call is a literal no-op. Resize
to anything else here and the server WILL letterbox the letterbox — hence `resize`
is the full target, not a bool: changing it is changing what the model sees.
"""

import logging
import threading
import time

import numpy as np


class DelayBudget:
    """How many 30 Hz ticks an inference takes, `d` in the RTC papers.

    Deliberately a FIXED budget rather than a per-call prediction. `d` has to be
    sent WITH the request (the guidance mask depends on it) — so it is necessarily
    a forecast, and the cost of the two errors is asymmetric:

      d too small -> we execute past the frozen prefix while inferring, and the
                     new chunk's committed slots no longer match what the robot
                     actually did. That is a discontinuity: a lurch at the seam.
      d too large -> we over-constrain and lose a little reactivity. Harmless.

    So we take the high quantile of observed latency and hold an early chunk until
    tick d rather than splicing it in early. Deterministic beats adaptive.
    """

    def __init__(self, fps: int, *, initial: int = 12, quantile: float = 0.95,
                 window: int = 50, headroom: float = 1.15, max_d: int = 20):
        self.fps = fps
        self.quantile = quantile
        self.headroom = headroom
        # d MUST be bounded. Unbounded, a slow inference (GPU contention, memory
        # pressure — a 2 s call is not exotic) yields d=69 against H=50; the chunk
        # then installs with its consumption index clipped to H, i.e. ZERO usable
        # actions, and the replan trigger H-d-margin goes negative. The loop
        # silently stops planning. Saturating here instead means we merely lose
        # RTC's continuity guarantee (and say so) rather than the whole plan.
        self.max_d = int(max_d)
        self._d = min(int(initial), self.max_d)
        self._samples: list[float] = []
        self._window = window
        self._lock = threading.Lock()
        self.violations = 0
        self.saturated = 0

    @property
    def d(self) -> int:
        with self._lock:
            return self._d

    def observe(self, latency_s: float) -> None:
        with self._lock:
            self._samples.append(latency_s)
            if len(self._samples) > self._window:
                self._samples.pop(0)
            if len(self._samples) >= 5:
                q = float(np.quantile(self._samples, self.quantile))
                want = max(1, int(np.ceil(q * self.headroom * self.fps)))
                if want > self.max_d:
                    self.saturated += 1
                self._d = min(want, self.max_d)

    def note_violation(self) -> None:
        """The chunk landed LATER than d ticks: we executed past the frozen prefix
        and the seam is not guaranteed continuous. Counted, and surfaced — a
        steady stream of these means the budget is too tight."""
        with self._lock:
            self.violations += 1

    def stats(self) -> dict:
        with self._lock:
            base = {"d": self._d, "violations": self.violations, "saturated": self.saturated}
            if not self._samples:
                return {**base, "n": 0}
            s = np.asarray(self._samples)
            return {
                **base,
                "n": len(s),
                "mean_ms": float(s.mean() * 1000),
                "p95_ms": float(np.quantile(s, 0.95) * 1000),
            }


class PolicyClient:
    """Thin wrapper: connect, read the layout out of the handshake, infer."""

    def __init__(self, host: str, port: int, *, api_key: str | None = None,
                 resize: tuple[int, int] | None = (224, 224)):
        # Imported here, not at module scope: DelayBudget above is pure numpy, and
        # making the whole module require `websockets` would mean the timing logic
        # can't be imported or tested without the transport.
        from openpi_client import websocket_client_policy

        self._resize = None if resize is None else (int(resize[0]), int(resize[1]))
        self._ws = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port, api_key=api_key
        )
        meta = self._ws.get_server_metadata()
        self.metadata = meta

        # The client is config-free: the model's layout comes from the server, not
        # from ego2g1.config (which would drag JAX onto the robot PC).
        cfg = meta.get("ego2g1")
        if cfg is None:
            raise RuntimeError(
                "server did not advertise ego2g1 metadata — is it running "
                "`python -m ego2g1.serve`? Stock openpi serve_policy.py cannot "
                "serve these checkpoints correctly."
            )
        self.hands = tuple(cfg["hands"])
        self.action_horizon = int(cfg["action_horizon"])
        self.action_dim = int(cfg["action_dim"])
        self.fps = int(cfg["fps"])
        self.rtc_training = bool(cfg["rtc_training"])
        self.rtc = dict(cfg["rtc"])

        stamp = meta.get("ego2g1_stamp", {})
        logging.info("policy: horizon=%d dim=%d fps=%d hands=%s",
                     self.action_horizon, self.action_dim, self.fps, self.hands)
        logging.info("checkpoint config hash: %s", stamp.get("ego2g1_config_hash"))
        logging.info("RTC: %s (checkpoint rtc_training=%s)", self.rtc, self.rtc_training)
        logging.info("client-side resize: %s",
                     "off — sending the raw head frame" if self._resize is None else
                     "%dx%d (server's ResizeImages is then a no-op)" % self._resize)

    def _prepare_image(self, image):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if self._resize is None:
            return image
        from openpi_client import image_tools

        return np.ascontiguousarray(
            image_tools.resize_with_pad(image, *self._resize), dtype=np.uint8
        )

    def infer(self, image, state, prompt, *, prev_chunk=None, d: int = 0,
              n_prefix: int | None = None) -> dict:
        obs = {
            "observation/image": self._prepare_image(image),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": prompt,
        }
        if prev_chunk is not None:
            obs["prev_chunk"] = np.asarray(prev_chunk, dtype=np.float32)
            obs["d"] = int(d)
            # How many rows are real. The rest is zero padding, and a zero vec9 is
            # not a pose — the server must know where to stop.
            obs["n_prefix"] = int(len(prev_chunk) if n_prefix is None else n_prefix)

        t0 = time.monotonic()
        out = self._ws.infer(obs)
        out["client_latency_s"] = time.monotonic() - t0
        return out
