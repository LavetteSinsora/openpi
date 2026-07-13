"""Vendor the G1 arm-IK trio from the training repo into `deploy/_g1_sim/`.

The deployment kinematics MUST be the kinematics that generated the training
labels (see kinematics.py). The clean way to guarantee that is to import the
outer repo's `data_extraction` at runtime — but then `ego2g1/` is not a single
shippable folder. This script makes a controlled COPY instead, and writes a
MANIFEST of content hashes so `test_deploy.py` can prove the copy has not drifted
from the source (on any machine that has both).

    python -m ego2g1.deploy.vendor_g1_sim --source /path/to/repo-with-data_extraction

Re-run it whenever data_extraction's sim/frames/assets change. The manifest diff
is the signal that a re-vendor is due; the test is what makes forgetting loud.

Only what the deploy path's FK/IK actually touches is copied — sim/g1.py,
common/frames.py, and the G1 MJCF assets. NOT data_extraction.hand (hand commands
deploy as pass-through [0,1], no retargeting) and NOT the rest of the sim package.
"""

import argparse
import hashlib
import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
VENDOR = HERE / "_g1_sim"

# Paths relative to the data_extraction package root. The vendored tree mirrors
# this layout under _g1_sim/, so g1.py's `from ..common import frames` and its
# `../assets/unitree_g1/...` model path both resolve unchanged.
FILES = ["sim/__init__.py", "sim/g1.py", "common/__init__.py", "common/frames.py"]
TREES = ["assets/unitree_g1"]


def _sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _iter_tree(root: pathlib.Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            yield p


def vendor(source: pathlib.Path) -> dict:
    de = source / "data_extraction" if (source / "data_extraction").is_dir() else source
    if not (de / "sim" / "g1.py").exists():
        raise SystemExit(f"no data_extraction/sim/g1.py under {source}")

    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    VENDOR.mkdir(parents=True)

    manifest = {}
    for rel in FILES:
        src = de / rel
        dst = VENDOR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[rel] = _sha(src)
    for rel in TREES:
        for src in _iter_tree(de / rel):
            r = src.relative_to(de).as_posix()
            dst = VENDOR / r
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest[r] = _sha(src)

    (VENDOR / "__init__.py").write_text(
        '"""Vendored copy of data_extraction\'s G1 arm sim (see vendor_g1_sim.py).\n\n'
        "Do NOT edit by hand — edits belong in the training repo, then re-vendor.\n"
        'test_deploy.py::test_vendored_g1_sim_matches_source guards against drift."""\n'
    )
    (VENDOR / "MANIFEST.json").write_text(
        json.dumps({"source_layout": "data_extraction", "files": manifest}, indent=2,
                   sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="../..",
                    help="repo root holding data_extraction/ (default: ../.. from openpi root)")
    args = ap.parse_args()
    m = vendor(pathlib.Path(args.source).expanduser().resolve())
    print(f"vendored {len(m)} files into {VENDOR}")
    print(f"  wrote {VENDOR/'MANIFEST.json'}")
