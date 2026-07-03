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

check "disk: >=60GB free on $WORKSPACE" bash -c \
    'avail_kb=$(df -Pk --output=avail "$WORKSPACE" | tail -1); [ "$avail_kb" -ge 62914560 ] || { echo "only $((avail_kb / 1048576))GB free"; exit 1; }'

if [ "${AUTO_TERMINATE:-1}" = "1" ]; then
    check "runpodctl present (auto-terminate; else AUTO_TERMINATE=0)" command -v runpodctl
    check "RUNPOD_POD_ID set" bash -c '[ -n "${RUNPOD_POD_ID:-}" ]'
    check "RUNPOD_API_KEY set (runpod.io Settings -> API Keys)" bash -c \
        '[ -n "${RUNPOD_API_KEY:-}" ] || [ -f ~/.runpod/config.toml ]'
fi

if [ "$FAIL" -ne 0 ]; then
    echo "preflight FAILED — fix the items above before running the experiment"
    exit 1
fi
echo "preflight passed"
