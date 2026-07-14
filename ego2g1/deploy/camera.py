"""The single egocentric camera the model consumes.

The model takes exactly ONE image (Ego2G1Inputs fills base_0_rgb and zero-fills +
attention-masks both wrist slots — it was trained having never seen a wrist view).

The G1's head streams a STEREO PAIR over ZMQ from the `image_server` running on
the robot's onboard board, so we must pick one eye. Which eye, and how well its
FOV/mount matches the Pico headset view the training video came from, is the
biggest open risk in this whole deployment — a policy fed a systematically
different viewpoint fails quietly and looks like a bad policy. Log the exact
array we send to Rerun and eyeball it against a training frame before believing
any rollout.

No resize here: read() hands back the RAW head frame, which is what check/Rerun
must see. The resize to the model's 224x224 happens once, at the wire, in
PolicyClient._prepare_image — see deploy/client.py for why it lives there and why
224x224 is the only safe value.
"""

import threading
import time

import numpy as np


class HeadCamera:
    """G1 head camera via the unitree_deploy ImageClient (ZMQ from the G1 board)."""

    def __init__(self, *, host: str = "192.168.123.164", eye: str = "left",
                 flip_bgr: bool = True):
        if eye not in ("left", "right"):
            raise ValueError(f"eye must be left|right, got {eye}")
        self.host = host
        self.eye = eye
        self.flip_bgr = flip_bgr
        self._client = None
        self._lock = threading.Lock()
        self._frame = None
        self._t = 0.0
        self._stop = threading.Event()
        self._thread = None

    def connect(self, *, timeout: float = 10.0) -> None:
        from unitree_deploy.robot_devices.cameras.configs import ImageClientCameraConfig
        from unitree_deploy.robot_devices.cameras.imageclient import ImageClientCamera

        self._client = ImageClientCamera(ImageClientCameraConfig(host_ip=self.host))
        self._client.connect()

        self._thread = threading.Thread(target=self._pump, name="camera", daemon=True)
        self._thread.start()

        t0 = time.monotonic()
        while self.age() > timeout:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"no frames from image_server at {self.host}. Is it running? "
                    "(ssh unitree@%s; conda activate tv; cd ~/image_server; python image_server.py)"
                    % self.host
                )
            time.sleep(0.05)

    def _pump(self) -> None:
        key = f"cam_{self.eye}_high"   # ImageClient splits the stereo frame into these
        while not self._stop.is_set():
            try:
                out = self._client.async_read()
            except Exception:
                time.sleep(0.01)
                continue
            img = out.get(key) if isinstance(out, dict) else out
            if img is not None:
                if self.flip_bgr:
                    img = img[..., ::-1]  # ImageClient yields BGR; the model wants RGB
                with self._lock:
                    self._frame = np.ascontiguousarray(img, dtype=np.uint8)
                    self._t = time.monotonic()
            time.sleep(1 / 60)

    def read(self):
        """Latest frame as (H, W, 3) uint8 RGB, or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._frame is None else time.monotonic() - self._t

    def close(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
