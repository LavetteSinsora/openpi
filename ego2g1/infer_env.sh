# ego2g1 INFERENCE session setup — pin serving to a SINGLE PPU with
# on-demand memory, instead of grabbing 0.9 of every one of the 16 cards.
# MUST be sourced, not executed:   source ego2g1/infer_env.sh
#
# Why this exists:
#   env.sh is the TRAINING profile. It sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.9,
#   which makes JAX preallocate 90% of memory on EVERY visible device the moment
#   it initializes — and JAX auto-detects all 16 PPUs. Serving one pi0.5 replica
#   only computes on device 0, so the other 15 are wasted and every card reads
#   0.9. This wrapper flips two knobs BEFORE sourcing env.sh:
#     1. grow memory on demand instead of preallocating 0.9  (PREALLOCATE=false)
#     2. expose only ONE PPU to JAX                          (<VENDOR>_VISIBLE_DEVICES)
#
# Usage:
#   source ego2g1/infer_env.sh                 # uses PPU 0
#   EGO2G1_PPU=3 source ego2g1/infer_env.sh    # uses PPU 3
#
# One-time setup: set EGO2G1_PPU_VAR to your PPU plugin's visible-devices env
# var (the CUDA_VISIBLE_DEVICES analog). Find it with the discovery snippet in
# ego2g1/README, e.g.  strings <pjrt_plugin.so> | grep -i visible_devices.
#   EGO2G1_PPU_VAR=PPU_VISIBLE_DEVICES EGO2G1_PPU=0 source ego2g1/infer_env.sh

: "${EGO2G1_PPU:=0}"
: "${EGO2G1_PPU_VAR:=PPU_VISIBLE_DEVICES}"   # <-- adjust once, per Step 1 discovery

# 1) memory: don't grab 90% up front; allocate as the model needs it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Cap prealloc too, in case the PPU plugin ignores PREALLOCATE. env.sh reads
# this with `:-0.9`, so exporting it here wins over env.sh's default.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"

# 2) devices: expose only one PPU. Set the discovered var name and index.
export "$EGO2G1_PPU_VAR"="$EGO2G1_PPU"
echo "infer_env: ${EGO2G1_PPU_VAR}=${EGO2G1_PPU}  PREALLOCATE=false  MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

# reuse the training env for venv + PYTHONPATH + verification print
# (env.sh's trailing python prints `devices: N` — expect 1, not 16).
source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/env.sh"
