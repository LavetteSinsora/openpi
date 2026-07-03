#!/usr/bin/env bash
# Full unattended π0.5 → LIBERO-OBJECT experiment:
#
#   preflight → canary eval (10 rollouts) → canary train (20 steps)
#   → baseline eval → 30k-step LoRA train → eval every kept checkpoint
#   → LoRA extract + wandb artifact upload → summary alert → pod self-terminate
#
# Run inside tmux so an SSH disconnect doesn't kill it:
#   tmux new -s train
#   bash scripts/remote_run.sh
#
# Knobs (env vars):
#   RUN_CANARY=1        cheap end-to-end smoke tests before the real run
#   RUN_BASELINE=1      full 500-rollout eval of the pretrained base model
#   TRIALS_PER_TASK=50
#   AUTO_TERMINATE=1    terminate the RunPod pod when done OR on failure
#                       (set 0 while debugging interactively)
#
# Stages record .done markers in $STATUS_DIR — re-running after a crash skips
# completed work, and an interrupted training run resumes from its checkpoints.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/remote_env.sh"

RUN_CANARY="${RUN_CANARY:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
AUTO_TERMINATE="${AUTO_TERMINATE:-1}"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$EXPERIMENTS_DIR" "$ARTIFACTS_DIR/lora" "$CKPT_BASE"
LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO_DIR"

# ── helpers ──────────────────────────────────────────────────────────────────

notify() {  # title, text — best-effort, never fails the run
    "$PY" scripts/wandb_notify.py alert --title "$1" --text "${2:-}" || true
}

upload_artifact() {  # path, name, type — retried; artifacts are the permanent outputs
    local attempt
    for attempt in 1 2 3; do
        "$PY" scripts/wandb_notify.py artifact --path "$1" --name "$2" --type "$3" && return 0
        echo "artifact upload failed (attempt $attempt/3), retrying in 30s"
        sleep 30
    done
    return 1
}

terminate_pod() {
    if [ "$AUTO_TERMINATE" != "1" ]; then
        echo "AUTO_TERMINATE=0 — pod left running (remember: it keeps billing!)"
        return 0
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl > /dev/null; then
        echo "cannot self-terminate: RUNPOD_POD_ID/runpodctl missing — TERMINATE THE POD MANUALLY"
        return 0
    fi
    [ -n "${RUNPOD_API_KEY:-}" ] && runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null
    echo "terminating pod $RUNPOD_POD_ID in 60s (Ctrl-C to abort)"
    sleep 60
    runpodctl remove pod "$RUNPOD_POD_ID"
}

on_error() {
    local rc=$?
    trap - ERR
    echo "RUN FAILED (exit $rc) — log: $LOG"
    local tail_txt
    tail_txt=$(tail -n 15 "$LOG" 2>/dev/null || true)
    notify "pi05 LIBERO run FAILED" "exit=$rc on $(hostname). Last log lines:
$tail_txt"
    upload_artifact "$LOG" "run_log_failed" "log" || true
    terminate_pod
    exit "$rc"
}
trap on_error ERR

stage() {  # stage <name> <fn...> — skips if a previous run already completed it
    local name="$1"; shift
    local marker="$STATUS_DIR/$name.done"
    if [ -f "$marker" ]; then
        echo "── $name: already done, skipping"
        return 0
    fi
    echo "── $name: started $(date -u '+%F %T UTC')"
    "$@"
    touch "$marker"
    echo "── $name: done $(date -u '+%F %T UTC')"
}

# ── stages ───────────────────────────────────────────────────────────────────

# 1 trial/task against the base checkpoint: exercises the entire eval path
# (gs:// download incl. crcmod, policy load, EGL rollouts, video encode, wandb)
# in ~minutes, and warms the pi05_base cache that training reuses.
stage_canary_eval() {
    "$PY" scripts/benchmark.py \
        --config-name "$CONFIG_NAME" \
        --checkpoint-dir gs://openpi-assets/checkpoints/pi05_base \
        --exp-dir "$EXPERIMENTS_DIR/canary_eval" \
        --num-trials-per-task 1
    notify "canary eval passed" "Full eval pipeline OK (10 rollouts, base checkpoint)."
}

# 20 steps at the real batch size: catches OOM, dataloader/norm-stats problems,
# and exercises a checkpoint save — without touching the real checkpoint dir.
stage_canary_train() {
    "$PY" scripts/train.py "$CONFIG_NAME" \
        --exp-name canary \
        --num-train-steps 20 \
        --save-interval 10 \
        --checkpoint-base-dir "$WORKSPACE/checkpoints_canary" \
        --overwrite
    rm -rf "$WORKSPACE/checkpoints_canary"
    notify "canary train passed" "20 steps at batch 32 incl. checkpoint save. Starting the real run."
}

stage_baseline() {
    "$PY" scripts/benchmark.py \
        --config-name "$CONFIG_NAME" \
        --checkpoint-dir gs://openpi-assets/checkpoints/pi05_base \
        --exp-dir "$EXPERIMENTS_DIR/pi05_base_benchmark" \
        --num-trials-per-task "$TRIALS_PER_TASK"
    upload_artifact "$EXPERIMENTS_DIR/pi05_base_benchmark/results.json" "results_baseline" "eval_results"
}

stage_train() {
    local ckpt_dir="$CKPT_BASE/$CONFIG_NAME/$EXP_NAME"
    local extra=()
    # A checkpoint dir from an interrupted run → resume (needs the wandb_id.txt
    # that train.py wrote next to the step dirs on the first attempt).
    if [ -d "$ckpt_dir" ] && [ -n "$(ls -A "$ckpt_dir" 2>/dev/null)" ]; then
        echo "existing checkpoints found — resuming"
        extra+=(--resume)
    fi
    notify "training started" "$CONFIG_NAME / $EXP_NAME, 30k steps."
    "$PY" scripts/train.py "$CONFIG_NAME" \
        --exp-name "$EXP_NAME" \
        --checkpoint-base-dir "$CKPT_BASE" \
        ${extra[@]+"${extra[@]}"}
    notify "training finished" "Kept checkpoints: $(ls "$ckpt_dir" | tr '\n' ' ')"
}

# Evaluate every kept checkpoint (5000..25000 + final 29999), then immediately
# extract + upload its LoRA adapters so a late crash never loses finished work.
stage_evals() {
    local ckpt_dir="$CKPT_BASE/$CONFIG_NAME/$EXP_NAME"
    local step_dir step marker
    for step in $(find "$ckpt_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n); do
        step_dir="$ckpt_dir/$step"
        marker="$STATUS_DIR/eval_step_$step.done"
        [ -f "$marker" ] && { echo "step $step already evaluated, skipping"; continue; }

        "$PY" scripts/benchmark.py \
            --config-name "$CONFIG_NAME" \
            --checkpoint-dir "$step_dir" \
            --exp-dir "$EXPERIMENTS_DIR/$EXP_NAME/step_$step" \
            --num-trials-per-task "$TRIALS_PER_TASK" \
            --train-step "$step"

        "$PY" scripts/extract_lora.py \
            --checkpoint-dir "$step_dir" \
            --out "$ARTIFACTS_DIR/lora/step_$step.npz"
        upload_artifact "$ARTIFACTS_DIR/lora/step_$step.npz" "lora_step_$step" "lora_weights"
        upload_artifact "$EXPERIMENTS_DIR/$EXP_NAME/step_$step/results.json" "results_step_$step" "eval_results"

        touch "$marker"
    done
}

stage_summary() {
    "$PY" - <<'PYEOF'
import json
import os
import pathlib

exp = pathlib.Path(os.environ["EXPERIMENTS_DIR"])
rows = []
baseline = exp / "pi05_base_benchmark" / "results.json"
if baseline.exists():
    rows.append(("baseline (pi05_base)", json.loads(baseline.read_text())["aggregate_success_rate"]))
steps = sorted(
    (exp / os.environ["EXP_NAME"]).glob("step_*/results.json"),
    key=lambda p: int(p.parent.name.removeprefix("step_")),
)
for r in steps:
    rows.append((r.parent.name, json.loads(r.read_text())["aggregate_success_rate"]))
text = "\n".join(f"{name}: {rate:.1%}" for name, rate in rows)
print(text)
pathlib.Path(os.environ["ARTIFACTS_DIR"], "summary.txt").write_text(text)
PYEOF
    notify "pi05 LIBERO experiment COMPLETE" "$(cat "$ARTIFACTS_DIR/summary.txt")

Pod will now self-terminate."
    upload_artifact "$LOG" "run_log" "log"
}

# ── main ─────────────────────────────────────────────────────────────────────

bash scripts/remote_preflight.sh   # always runs — cheap, catches env drift

if [ "$RUN_CANARY" = "1" ]; then
    stage canary_eval stage_canary_eval
    stage canary_train stage_canary_train
fi
if [ "$RUN_BASELINE" = "1" ]; then
    stage baseline stage_baseline
fi
stage train stage_train
stage_evals                         # per-step markers handle idempotency inside
stage summary stage_summary

echo "ALL DONE — log: $LOG"
terminate_pod
