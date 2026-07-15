"""Read a LeRobot episode's per-frame arrays + egocentric video, and map
source episodes to LeRobot episode indices. No sim/jax dependency — usable in
both phases. Deps: numpy, pandas, opencv (cv2).

State layout (verified): per hand [eef vec9 (9) | hand motors (6)] in
("left","right") order -> state[0:9]=L eef, [9:15]=L hand, [15:24]=R eef,
[24:30]=R hand.
"""

import dataclasses
import json
import pathlib

import numpy as np

from ego2g1 import dataset as _dataset

# state dim slices
L_EEF, L_HAND = slice(0, 9), slice(9, 15)
R_EEF, R_HAND = slice(15, 24), slice(24, 30)
EEF = {"left": L_EEF, "right": R_EEF}
HAND = {"left": L_HAND, "right": R_HAND}


@dataclasses.dataclass
class EpisodeData:
    episode_index: int
    source_episode: str
    task: str
    n_frames: int
    state: np.ndarray       # (T, 30) f32
    arm_qpos: np.ndarray    # (T, 14) f32  [left 7, right 7]
    hand_left: np.ndarray   # (T, 6) f32
    hand_right: np.ndarray  # (T, 6) f32
    video_path: pathlib.Path
    real_end: bool
    fps: float


def _info(root: pathlib.Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _tasks(root: pathlib.Path) -> dict[int, str]:
    out = {}
    with (root / "meta" / "tasks.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["task_index"])] = rec["task"]
    return out


def _chunk(episode_index: int, root: pathlib.Path) -> int:
    return episode_index // int(_info(root).get("chunks_size", 1000))


def parquet_path(root, episode_index: int) -> pathlib.Path:
    root = pathlib.Path(root)
    return root / "data" / f"chunk-{_chunk(episode_index, root):03d}" / f"episode_{episode_index:06d}.parquet"


def video_path(root, episode_index: int, video_key: str = "image") -> pathlib.Path:
    root = pathlib.Path(root)
    return (root / "videos" / f"chunk-{_chunk(episode_index, root):03d}"
            / video_key / f"episode_{episode_index:06d}.mp4")


def source_to_episodes(root, source_episode: str) -> list[int]:
    """LeRobot episode indices produced from a given real source episode."""
    meta = _dataset.load_extraction_meta(root)
    eps = meta["episodes"]
    idx = [int(i) for i, e in eps.items() if e["source_episode"] == source_episode]
    if not idx:
        known = sorted({e["source_episode"] for e in eps.values()})
        raise ValueError(f"source_episode {source_episode!r} not found. Known: {known[:5]}... ({len(known)} total)")
    return sorted(idx)


def load_episode(root, episode_index: int) -> EpisodeData:
    import pandas as pd

    root = pathlib.Path(root)
    df = pd.read_parquet(parquet_path(root, episode_index))
    meta = _dataset.load_extraction_meta(root)
    ep_meta = meta["episodes"][str(episode_index)]
    tasks = _tasks(root)
    task_idx = int(np.asarray(df["task_index"].iloc[0]))
    stack = lambda col: np.stack(df[col].to_numpy()).astype(np.float32)
    return EpisodeData(
        episode_index=episode_index,
        source_episode=ep_meta["source_episode"],
        task=tasks.get(task_idx, ""),
        n_frames=len(df),
        state=stack("state"),
        arm_qpos=stack("arm_qpos"),
        hand_left=stack("hand.left"),
        hand_right=stack("hand.right"),
        video_path=video_path(root, episode_index),
        real_end=bool(ep_meta["episode_real_end"]),
        fps=float(_info(root).get("fps", 30)),
    )


# The datasets are AV1-encoded (meta/info.json: video.codec = av1). Decoders are
# tried in the order below so we use whatever already works on the machine:
#   1. lerobot's own decoder — the exact path TRAINING uses, so it works wherever
#      training works (whichever backend that box has: torchcodec/pyav/...);
#   2. PyAV directly (its wheels bundle libdav1d);
#   3. OpenCV (many builds cannot decode AV1: "Failed to get pixel format").

def _decode_lerobot(path, indices, fps) -> np.ndarray:
    import torch
    from lerobot.common.datasets.video_utils import decode_video_frames

    timestamps = [float(i) / fps for i in indices]
    frames = decode_video_frames(pathlib.Path(path), timestamps, 1.0 / fps / 2.0)  # (N,C,H,W) float [0,1]
    arr = (frames.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return np.ascontiguousarray(arr)


def _decode_pyav(path, indices) -> np.ndarray:
    import av

    want, got, last = {int(i) for i in indices}, {}, max(int(i) for i in indices)
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in want:
                got[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
    if set(got) != want:
        raise RuntimeError(f"pyav decoded {len(got)}/{len(want)} requested frames")
    return np.stack([got[int(i)] for i in indices])


def _decode_cv2(path, indices) -> np.ndarray:
    import cv2

    want, got, last = {int(i) for i in indices}, {}, max(int(i) for i in indices)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    try:
        i = 0
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
    if set(got) != want:
        raise RuntimeError(f"OpenCV decoded {len(got)}/{len(want)} frames (AV1 unsupported in this build?)")
    return np.stack([got[int(i)] for i in indices])


def read_video_frames_at(path, indices, fps: float = 30.0) -> np.ndarray:
    """Decode just `indices` -> (len(indices), H, W, 3) uint8 RGB, trying the
    decoders above in order. Same pixels the policy saw in training."""
    errors = []
    for name, fn in (("lerobot", lambda: _decode_lerobot(path, indices, fps)),
                     ("pyav", lambda: _decode_pyav(path, indices)),
                     ("opencv", lambda: _decode_cv2(path, indices))):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — try the next backend
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"could not decode {path} (AV1). Tried:\n  " + "\n  ".join(errors) +
        "\nInstall a working decoder, e.g. `pip install --user av`."
    )


def read_video_frames(path, n_expected: int | None = None, fps: float = 30.0) -> np.ndarray:
    """Decode the whole video -> (T, H, W, 3) uint8 RGB."""
    if n_expected is None:
        raise ValueError("n_expected (episode frame count) is required")
    return read_video_frames_at(path, range(n_expected), fps)
