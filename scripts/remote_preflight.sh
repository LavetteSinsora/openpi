#!/usr/bin/env bash
# Fail-fast environment checks. Runs every known failure mode in seconds so a
# broken pod is caught before it burns GPU-hours. Run after setup and again at
# the start of every remote_run.sh.
#
# Deliberately not `set -e`: all checks run, all failures are reported at once.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/remote_env.sh"

FAIL=0
check() {
    local name="$1"; shift
    local out
    if out=$("$@" 2>&1); then
        echo "  ok   $name"
    else
        echo "  FAIL $name"
        echo "$out" | tail -n 8 | sed 's/^/         /'
        FAIL=1
    fi
}

echo "preflight:"

check "GPU visible (nvidia-smi)" nvidia-smi

check "jax sees the GPU" "$PY" -c \
    'import jax; d = jax.devices(); assert d and d[0].platform == "gpu", d; print(d)'

# The classic container trap: NVIDIA_DRIVER_CAPABILITIES without "graphics"
# breaks EGL. Catch it here, not 10 hours in during the first eval.
check "MuJoCo EGL off-screen rendering" "$PY" -c \
    'import os; os.environ.setdefault("MUJOCO_GL", "egl"); import mujoco; m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"box\" size=\".1 .1 .1\"/></worldbody></mujoco>"); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m); r.update_scene(d); img = r.render(); assert img.any(), "rendered an empty frame"; print("rendered", img.shape)'

check "LIBERO + robosuite import (PYTHONPATH + ~/.libero/config.yaml)" "$PY" -c \
    'import os, robosuite; from libero.libero import benchmark, get_libero_path; p = get_libero_path("bddl_files"); assert os.path.isdir(p), p; print(p)'

check "dataset complete (1504 files)" bash -c \
    'n=$(find "$DATASET_DIR" -type f 2>/dev/null | wc -l); [ "$n" -eq 1504 ] || { echo "$n files at $DATASET_DIR"; exit 1; }'

check "norm stats committed in repo" test -f \
    "$REPO_DIR/assets/pi05_libero/$CONFIG_NAME/libero_object_summed_subsampling/norm_stats.json"

check "anonymous gsutil access to gs://openpi-assets" bash -c \
    'gsutil ls gs://openpi-assets/checkpoints/pi05_base/ | grep -q params'

check "compiled crcmod (composite-object downloads)" bash -c \
    'gsutil version -l 2>/dev/null | grep -qi "compiled crcmod: True"'

check "wandb API key valid" bash -c \
    '[ -n "${WANDB_API_KEY:-}" ] && "$REPO_DIR/.venv/bin/wandb" login --verify'

# Need ~20GB headroom: a segment transition holds old + new + orbax tmp
# checkpoints (~27GB total, 9GB each) minus the 9GB already-counted old one.
# RunPod NETWORK volumes report the whole storage cluster to df (petabytes)
# while the per-volume quota is invisible until EDQUOT — so when
# VOLUME_SIZE_GB is set (the size you picked at volume creation), headroom is
# computed as quota minus used instead of trusting df.
check "disk: >=20GB headroom on $WORKSPACE (set VOLUME_SIZE_GB for network volumes)" bash -c '
    if [ -n "${VOLUME_SIZE_GB:-}" ]; then
        used_kb=$(du -sk "$WORKSPACE" 2>/dev/null | cut -f1)
        avail_kb=$((VOLUME_SIZE_GB * 1048576 - used_kb))
    else
        avail_kb=$(df -k --output=avail "$WORKSPACE" | tail -1)
    fi
    [ "$avail_kb" -ge 20971520 ] || { echo "only $((avail_kb / 1048576))GB headroom"; exit 1; }'

if [ "${AUTO_TERMINATE:-1}" = "1" ]; then
    check "runpodctl present (auto-terminate; else AUTO_TERMINATE=0)" command -v runpodctl
    check "RUNPOD_POD_ID set" bash -c '[ -n "${RUNPOD_POD_ID:-}" ]'
    # End-to-end: configure the CLI (it needs its config file pre-created) and
    # make an authenticated API call, so a broken self-stop surfaces here — not
    # after a 30-hour run fails.
    check "runpodctl authenticated API access" bash -c \
        '[ -n "${RUNPOD_API_KEY:-}" ] && mkdir -p ~/.runpod && touch ~/.runpod/.runpod.yaml && runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null && runpodctl get pod "$RUNPOD_POD_ID" >/dev/null'
fi

if [ "$FAIL" -ne 0 ]; then
    echo "preflight FAILED — fix the items above before running the experiment"
    exit 1
fi
echo "preflight passed"
