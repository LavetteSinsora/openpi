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
    )


def read_video_frames(path, n_expected: int | None = None) -> np.ndarray:
    """Decode the whole mp4 -> (T, H, W, 3) uint8 RGB. Uses OpenCV's bundled
    ffmpeg (sidesteps the torchcodec breakage)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    arr = np.stack(frames)
    if n_expected is not None and len(arr) != n_expected:
        # LeRobot videos are 1:1 with frames; a mismatch is worth surfacing but
        # not fatal (clip to the shorter of the two at the call site).
        print(f"WARNING: video {path.name} has {len(arr)} frames, expected {n_expected}")
    return arr


def read_video_frame(path, index: int) -> np.ndarray:
    """Decode a single frame (T,H,W,3-less) -> (H, W, 3) uint8 RGB, by seeking.
    Used in Phase 1 where only the query-tick frames are needed."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
