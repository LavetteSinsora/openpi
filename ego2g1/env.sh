# ego2g1 session setup for a machine with a shared/borrowed venv (e.g. the PPU box).
# MUST be sourced, not executed:   source ego2g1/env.sh
#
# What it does (nothing is written to the borrowed venv):
#   1. activates the venv that has the accelerator (PPU) packages
#   2. puts THIS checkout's src/ on PYTHONPATH so `import openpi` resolves here,
#      shadowing the venv's own editable openpi install
#   3. sets the XLA memory fraction used for training
#
# Override the venv location per machine:   EGO2G1_VENV=/path/to/.venv source ego2g1/env.sh

EGO2G1_VENV="${EGO2G1_VENV:-$HOME/openpi/.venv}"
_EGO2G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

if [ ! -f "$EGO2G1_VENV/bin/activate" ]; then
    echo "ERROR: no venv at $EGO2G1_VENV (set EGO2G1_VENV=/path/to/.venv)" >&2
    return 1 2>/dev/null || exit 1
fi

source "$EGO2G1_VENV/bin/activate"
export PYTHONPATH="$_EGO2G1_ROOT/src"
# extra packages the shared venv lacks (e.g. gcsfs), installed via
# `pip install --target ~/pypath-extra <pkg>` so the venv itself stays untouched
if [ -d "$HOME/pypath-extra" ]; then
    export PYTHONPATH="$PYTHONPATH:$HOME/pypath-extra"
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
cd "$_EGO2G1_ROOT"

# verification: openpi must resolve to THIS checkout; jax must see accelerators
EGO2G1_ROOT="$_EGO2G1_ROOT" python - <<'PY'
import os
import pathlib
import jax
import openpi

root = pathlib.Path(os.environ["EGO2G1_ROOT"]).resolve()
src = pathlib.Path(openpi.__file__).resolve()
print(f"openpi     : {src}")
print(f"jax backend: {jax.default_backend()}  devices: {jax.device_count()}")
if not src.is_relative_to(root):
    print(f"WARNING: openpi does NOT resolve to this checkout ({root}) — PYTHONPATH shadowing failed")
if jax.default_backend() == "cpu":
    print("WARNING: jax is on CPU — training will not use the accelerators")
PY
