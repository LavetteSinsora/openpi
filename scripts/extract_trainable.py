#!/usr/bin/env python3
"""
Extract ALL trainable weights from a full openpi checkpoint into a single .npz.

The full training checkpoint (~4.8 GB) holds frozen base params + trained
params + optimizer state. Only the trained leaves are needed to reconstruct the
model: full params = pi05_base params overlaid with this archive.

What is actually trainable under pi05_libero_object_lora — the freeze filter
(Pi0Config.get_freeze_filter) freezes `.*llm.*` EXCEPT `.*lora.*`, so the
trained set is:
  - the LoRA adapters inside both transformers (~84 MB), AND
  - everything OUTSIDE the llm path: the SigLIP vision tower (PaliGemma/img,
    ~400M params) and the action/time projection layers.
A LoRA-only archive would silently drop the trained vision tower, making the
checkpoint unreconstructable — hence this script keeps the full trainable set.
Optimizer state is intentionally dropped (only needed to resume training).

Usage
-----
  .venv/bin/python scripts/extract_trainable.py \
    --checkpoint-dir checkpoints/pi05_libero/pi05_libero_object_lora/masked_loss_summed_subsampling/5000 \
    --out artifacts/trainable/step_5000.npz

Run from the openpi repo root.
"""

import dataclasses
import pathlib
import sys

import numpy as np
import tyro

# Allow running with a bare interpreter (openpi is already importable inside the venv).
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import openpi.models.model as _model  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402


@dataclasses.dataclass
class Args:
    checkpoint_dir: str  # path to the checkpoint step dir (containing params/)
    out: str             # destination .npz path


def _is_trainable(key: str) -> bool:
    # Complement of the freeze filter: frozen = ".*llm.*" AND NOT ".*lora.*",
    # so trained = anything outside llm, plus the LoRA adapters inside it.
    return "llm" not in key or "lora" in key


def run(args: Args) -> None:
    ckpt = pathlib.Path(args.checkpoint_dir)
    params_path = ckpt / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"No params/ under {ckpt}")

    # Restore as numpy so we don't allocate GPU memory.
    params = _model.restore_params(params_path, restore_type=np.ndarray)

    # Flatten to path-keyed leaves; "/"-join the tuple path for a readable key.
    flat = flatten_dict(params, sep="/")
    trainable = {k: np.asarray(v) for k, v in flat.items() if _is_trainable(k)}

    n_lora = sum(1 for k in trainable if "lora" in k)
    if n_lora == 0:
        raise RuntimeError(
            f"No LoRA params found in {params_path}. "
            "Was this checkpoint trained with a *_lora model variant?"
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # "/" is illegal in npz member names → encode as "__", decode on reload.
    np.savez(out, **{k.replace("/", "__"): v for k, v in trainable.items()})

    lora_mb = sum(v.nbytes for k, v in trainable.items() if "lora" in k) / 1e6
    other_mb = sum(v.nbytes for k, v in trainable.items() if "lora" not in k) / 1e6
    print(
        f"Saved {len(trainable)} trainable tensors → {out}\n"
        f"  LoRA adapters: {n_lora} tensors, {lora_mb:.1f} MB\n"
        f"  non-llm (img tower + projections): {len(trainable) - n_lora} tensors, {other_mb:.1f} MB"
    )


if __name__ == "__main__":
    run(tyro.cli(Args))
