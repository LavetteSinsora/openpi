# ego2g1 INFERENCE session setup — pin serving to a SINGLE device so it stops
# occupying all 16 cards. MUST be sourced, not executed:
#   source ego2g1/infer_env.sh
#
# The "PPU" box is 16 Alibaba/T-Head PPU-ZW810E NPUs (name=PPU-ZW810E), exposed
# to JAX through a CUDA-compatible shim (backend reports "gpu", plugin xla_cuda12).
#
# Why this exists:
#   env.sh is the TRAINING profile: XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 preallocates
#   ~90% on EVERY visible device, and JAX auto-detects all 16. Serving one pi0.5
#   replica only needs one card, so the other 15 are wasted. This wrapper hides
#   the rest BEFORE sourcing env.sh, so JAX sees exactly one device.
#
# IMPORTANT — allocator flags: XLA_PYTHON_CLIENT_PREALLOCATE=false and
#   ALLOCATOR=platform are NVIDIA-oriented and have segfaulted this NPU's driver
#   on the first infer call. Do NOT set them here. We keep the SAME allocator as
#   the (working) training profile and only (a) restrict the device and
#   (b) optionally lower the memory fraction.
#
# Usage:
#   source ego2g1/infer_env.sh                       # device 0, mem 0.9
#   EGO2G1_PPU=15 source ego2g1/infer_env.sh         # device 15
#   EGO2G1_PPU=15 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 source ego2g1/infer_env.sh
#
# EGO2G1_PPU_VAR must be the env var this NPU actually honors — confirm with the
# probe loop in ego2g1/README (some cards ignore CUDA_VISIBLE_DEVICES).

: "${EGO2G1_PPU:=0}"
: "${EGO2G1_PPU_VAR:=CUDA_VISIBLE_DEVICES}"   # <-- set to whatever the probe proves works

# devices: expose exactly one card to JAX.
export "$EGO2G1_PPU_VAR"="$EGO2G1_PPU"

# memory: keep the training allocator (proven stable on this NPU); just cap the
# fraction. env.sh reads this with `:-0.9`, so an override here wins. With one
# device visible, 0.9 sits on ONE card instead of all 16 — lower it to share.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "infer_env: ${EGO2G1_PPU_VAR}=${EGO2G1_PPU}  MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}  (allocator: default)"

# reuse the training env for venv + PYTHONPATH + verification print
# (env.sh's trailing python prints `devices: N` — expect 1, not 16).
source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/env.sh"
