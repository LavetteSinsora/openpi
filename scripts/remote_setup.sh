#!/usr/bin/env bash
# One-time environment setup on a fresh GPU pod (RunPod PyTorch/CUDA template,
# Ubuntu 22.04, run as root). Idempotent — safe to re-run after a failure.
#
# Required env:
#   DATASET_REPO   HF dataset repo holding the dataset tarball,
#                  e.g. <your-hf-user>/libero_object_summed_subsampling
#   HF_TOKEN       only if DATASET_REPO is private
#
# Usage:  bash scripts/remote_setup.sh
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/remote_env.sh"

: "${DATASET_REPO:?export DATASET_REPO=<hf-user>/<repo> (HF dataset repo with the tarball)}"
export DATASET_TAR="${DATASET_TAR:-libero_object_summed_subsampling.tar}"
# 500 parquet + 1000 mp4 + 4 meta files. A partial extraction MUST fail here —
# an incomplete dataset dir silently triggers a confusing HF Hub fetch later.
DATASET_EXPECTED_FILES=1504

mkdir -p "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME" "$CKPT_BASE" \
         "$EXPERIMENTS_DIR" "$ARTIFACTS_DIR/trainable" "$STATUS_DIR" "$LOG_DIR"

echo "=== [1/6] system packages"
export DEBIAN_FRONTEND=noninteractive
# Ubuntu 24.04 marks the system python "externally managed" (PEP 668); we do
# want gsutil/crcmod in it (step 4), since openpi shells out to `gsutil`.
export PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update -qq
apt-get install -y -qq ffmpeg libgl1 libegl1 libosmesa6 libglib2.0-0 \
    build-essential python3-dev python3-pip git curl tmux > /dev/null

echo "=== [2/6] uv + openpi venv (exact locked deps)"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
(cd "$REPO_DIR" && uv sync --no-dev)
[ -x "$PY" ] || { echo "venv python missing at $PY"; exit 1; }

echo "=== [3/6] LIBERO (vendored submodule, pinned) + sim deps"
git -C "$REPO_DIR" submodule update --init third_party/libero
# LIBERO is NOT pip-installed (broken packaging) — it is imported via
# PYTHONPATH=$LIBERO_DIR. Its sim deps are installed explicitly; robosuite goes
# in with --no-deps to skip pynput/evdev, whose C build fails headless.
uv pip install --python "$PY" -q \
    numpy==1.26.4 mujoco==3.2.3 bddl==1.0.1 future easydict gym==0.25.2 \
    cloudpickle matplotlib numba scipy termcolor h5py
uv pip install --python "$PY" -q --no-deps robosuite==1.4.1
# libero/libero/__init__.py calls input() on first import if this file is
# missing → EOFError in subprocesses. Pre-write it.
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $LIBERO_DIR/libero/libero
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
init_states: $LIBERO_DIR/libero/libero/init_files
datasets: $LIBERO_DIR/libero/datasets
assets: $LIBERO_DIR/libero/libero/assets
EOF

echo "=== [4/6] gsutil for the public gs://openpi-assets bucket"
# openpi routes gs://openpi-assets through the gsutil CLI (anonymous access
# works — the bucket is public). Composite objects in pi05_base/params need
# *compiled* crcmod, hence the --no-binary reinstall.
# --ignore-installed: gsutil deps (e.g. cryptography) may already exist as
# apt-installed packages that pip cannot uninstall (no RECORD file).
command -v gsutil > /dev/null || python3 -m pip install -q --ignore-installed gsutil
python3 -m pip install -q --no-binary :all: --force-reinstall crcmod
gsutil version -l 2> /dev/null | grep -qi "compiled crcmod: True" \
    || { echo "compiled crcmod missing — pi05_base download would fail"; exit 1; }

echo "=== [5/6] dataset (single tarball from HF Hub → local extract)"
if [ ! -f "$STATUS_DIR/dataset.done" ]; then
    rm -rf "$DATASET_DIR"  # never trust a partial dir (see completeness note above)
    "$PY" - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id=os.environ["DATASET_REPO"],
    filename=os.environ["DATASET_TAR"],
    repo_type="dataset",
    local_dir=os.path.join(os.environ["WORKSPACE"], "downloads"),
)
print(f"downloaded: {path}")
PYEOF
    # --no-same-owner: as root, tar tries to restore the archive's original
    # uid/gid, which the pod filesystem forbids.
    tar --no-same-owner -xf "$WORKSPACE/downloads/$DATASET_TAR" -C "$HF_LEROBOT_HOME"
    n_files=$(find "$DATASET_DIR" -type f | wc -l)
    [ "$n_files" -eq "$DATASET_EXPECTED_FILES" ] \
        || { echo "dataset incomplete: $n_files files (expected $DATASET_EXPECTED_FILES)"; exit 1; }
    touch "$STATUS_DIR/dataset.done"
fi
echo "dataset OK: $DATASET_DIR"

echo "=== [6/6] preflight"
bash "$REPO_DIR/scripts/remote_preflight.sh"

echo
echo "setup complete. Next:  tmux new -s train  →  bash scripts/remote_run.sh"
