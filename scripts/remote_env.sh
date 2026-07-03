# Shared environment for the remote (RunPod/any GPU box) training + eval scripts.
# Sourced by remote_setup.sh / remote_preflight.sh / remote_run.sh — not run directly.
#
# Everything lives under $WORKSPACE (default /workspace = the RunPod volume disk,
# which survives pod stop/restart; only pod *termination* deletes it).

export WORKSPACE="${WORKSPACE:-/workspace}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR
export PY="$REPO_DIR/.venv/bin/python"
export LIBERO_DIR="$REPO_DIR/third_party/libero"

# ── experiment identity (must match the TrainConfig in libero_object_configs.py)
export CONFIG_NAME="${CONFIG_NAME:-pi05_libero_object_lora}"
export EXP_NAME="${EXP_NAME:-masked_loss_summed_subsampling}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-pi05_libero_replication}"

# ── storage layout
export HF_LEROBOT_HOME="$WORKSPACE/lerobot"                    # dataset lives here
export DATASET_DIR="$HF_LEROBOT_HOME/libero_object_summed_subsampling"
export OPENPI_DATA_HOME="$WORKSPACE/openpi_cache"              # gs:// checkpoint cache
export CKPT_BASE="$WORKSPACE/checkpoints/pi05_libero"          # full ckpts (ephemeral)
export EXPERIMENTS_DIR="$WORKSPACE/experiments"                # eval outputs + videos
export ARTIFACTS_DIR="$WORKSPACE/artifacts"                    # LoRA .npz before upload
export STATUS_DIR="$WORKSPACE/status"                          # stage .done markers
export LOG_DIR="$WORKSPACE/logs"

# ── runtime knobs (verified on the Colab bring-up; see repo docs)
export MUJOCO_GL="${MUJOCO_GL:-egl}"                 # GPU off-screen rendering
export PYTHONPATH="$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}"  # LIBERO is a namespace pkg, not pip-installed
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1            # torch>=2.6 breaks LIBERO .pruned_init loads
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MPLBACKEND=Agg                                # never inherit a GUI/inline backend

export PATH="$HOME/.local/bin:$PATH"                 # uv installs here
