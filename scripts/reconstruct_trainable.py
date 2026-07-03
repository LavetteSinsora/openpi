#!/usr/bin/env python3
"""Rebuild a full params checkpoint from pi05_base + a trainable_step_N.npz.

Inverse of extract_trainable.py: starts from the frozen pi05_base params and
overlays every archived trainable leaf (LoRA adapters, SigLIP tower,
projections). The result loads with create_trained_policy / benchmark.py
exactly like the original checkpoint's params (optimizer state is not
recovered — evaluation/deployment only, no training resume).

Usage
-----
  .venv/bin/python scripts/reconstruct_trainable.py \
    --trainable-npz artifacts/trainable/step_999.npz \
    --out-dir /workspace/ckpt_rebuilt_999

  then evaluate with:  benchmark.py --checkpoint-dir /workspace/ckpt_rebuilt_999
"""

import dataclasses
import pathlib
import shutil
import sys

import numpy as np
import orbax.checkpoint as ocp
import tyro
from flax.traverse_util import flatten_dict, unflatten_dict

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import openpi.models.model as _model  # noqa: E402
import openpi.shared.download as download  # noqa: E402


@dataclasses.dataclass
class Args:
    trainable_npz: str   # archive produced by extract_trainable.py
    out_dir: str         # checkpoint step dir to create (params/ written inside)
    base_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    overwrite: bool = False


def run(args: Args) -> None:
    base_path = download.maybe_download(args.base_checkpoint)
    params = _model.restore_params(base_path, restore_type=np.ndarray)
    flat = flatten_dict(params, sep="/")

    npz = np.load(args.trainable_npz)
    n_replaced = n_new = 0
    for enc_key in npz.files:
        key = enc_key.replace("__", "/")
        val = npz[enc_key]
        if key in flat:
            if flat[key].shape != val.shape:
                raise ValueError(f"shape mismatch at {key}: base {flat[key].shape} vs npz {val.shape}")
            n_replaced += 1
        else:
            n_new += 1  # lora_a / lora_b leaves don't exist in the base checkpoint
        flat[key] = val
    if n_new == 0:
        raise RuntimeError("no LoRA leaves added — is this really an extract_trainable.py archive?")
    print(f"overlaid {n_replaced} trained leaves, added {n_new} LoRA leaves onto base")

    out = pathlib.Path(args.out_dir).resolve() / "params"
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists — pass --overwrite to replace")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Same layout restore_params expects: a PyTree checkpoint with a top-level
    # "params" entry (see openpi.models.model.restore_params).
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(out, {"params": unflatten_dict(flat, sep="/")})
    print(f"wrote {out}")


if __name__ == "__main__":
    run(tyro.cli(Args))
