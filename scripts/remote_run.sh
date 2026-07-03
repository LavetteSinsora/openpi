#!/usr/bin/env bash
# Full unattended π0.5 → LIBERO-OBJECT experiment, with training and evaluation
# INTERLEAVED in 5k-step segments:
#
#   preflight → canary eval (10 rollouts) → canary train (20 steps)
#   → baseline eval → [train 5k → eval → extract+upload trainable weights] × 6
#   → summary alert → pod self-terminate
#
# Why segments are equivalent to one continuous 30k run:
#   - the LR schedule (CosineDecaySchedule, decay_steps=30000) follows the
#     global step restored from the checkpoint, not --num-train-steps;
#   - train.py saves/restores optimizer + data-loader state, and resumes the
#     same wandb run via wandb_id.txt → one continuous loss curve;
#   - orbax (max_to_keep=1) deletes an old checkpoint only AFTER the next one
#     is committed, so there is never a window without a resumable checkpoint.
# Each segment saves only its final checkpoint (step 4999/9999/.../29999); the
# previous segment's checkpoint is auto-deleted after we've already evaluated
# and archived it. Disk peak ~2 checkpoints instead of 6.
#
# Run inside tmux so an SSH disconnect doesn't kill it:
#   tmux new -s train
#   bash scripts/remote_run.sh
#
# Knobs (env vars):
#   TOTAL_STEPS=30000  EVAL_EVERY=5000
#   RUN_CANARY=1        cheap end-to-end smoke tests before the real run
#   RUN_BASELINE=1      full eval of the pretrained base model (before training,
#                       so a broken eval setup is caught early)
#   TRIALS_PER_TASK=50
#   AUTO_TERMINATE=1    on success: terminate the pod (everything is on wandb).
#                       on FAILURE: stop the pod instead — /workspace survives a
#                       stop, so training progress is kept; restart the pod and
#                       re-run this script to resume. Set 0 while debugging.
#
# Stages record .done markers in $STATUS_DIR — re-running after a crash skips
# completed work and resumes training from the latest checkpoint.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/remote_env.sh"

TOTAL_STEPS="${TOTAL_STEPS:-30000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
RUN_CANARY="${RUN_CANARY:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
AUTO_TERMINATE="${AUTO_TERMINATE:-1}"

CKPT_DIR="$CKPT_BASE/$CONFIG_NAME/$EXP_NAME"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$EXPERIMENTS_DIR" "$ARTIFACTS_DIR/trainable" "$CKPT_BASE"
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

terminate_pod() {  # success path: everything permanent is on wandb
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

stop_pod() {  # failure path: STOP, don't terminate — /workspace (checkpoints,
    # dataset, venv) survives a stop at storage-only cost, so the run can resume.
    if [ "$AUTO_TERMINATE" != "1" ]; then
        echo "AUTO_TERMINATE=0 — pod left running for debugging (it keeps billing!)"
        return 0
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl > /dev/null; then
        echo "cannot self-stop: RUNPOD_POD_ID/runpodctl missing — STOP THE POD MANUALLY"
        return 0
    fi
    [ -n "${RUNPOD_API_KEY:-}" ] && runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null
    echo "stopping pod $RUNPOD_POD_ID in 60s (Ctrl-C to abort)"
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
}

on_error() {
    local rc=$?
    trap - ERR
    echo "RUN FAILED (exit $rc) — log: $LOG"
    local tail_txt
    tail_txt=$(tail -n 15 "$LOG" 2>/dev/null || true)
    notify "pi05 LIBERO run FAILED" "exit=$rc on $(hostname). Pod will be STOPPED (state preserved) — restart it and re-run remote_run.sh to resume. Last log lines:
$tail_txt"
    upload_artifact "$LOG" "run_log_failed" "log" || true
    stop_pod
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

# Train up to a global step target. Huge --save-interval → only the segment's
# final step (target-1) is saved. Resuming an already-finished segment is a
# no-op: train.py restores step==target and exits without training.
train_to() {
    local target="$1"
    local extra=()
    if find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | grep -q .; then
        extra+=(--resume)
    elif [ -d "$CKPT_DIR" ]; then
        # dir exists but holds no finished checkpoint (crash before first save)
        extra+=(--overwrite)
    fi
    "$PY" scripts/train.py "$CONFIG_NAME" \
        --exp-name "$EXP_NAME" \
        --checkpoint-base-dir "$CKPT_BASE" \
        --num-train-steps "$target" \
        --save-interval 1000000 \
        ${extra[@]+"${extra[@]}"}
}

# Evaluate a saved checkpoint, then archive its trainable weights (LoRA +
# SigLIP tower + projections — see extract_trainable.py) to wandb immediately,
# BEFORE the next segment's save causes orbax to delete this checkpoint.
eval_and_archive() {
    local step="$1"
    local step_dir="$CKPT_DIR/$step"
    [ -d "$step_dir" ] || { echo "expected checkpoint missing: $step_dir"; return 1; }

    "$PY" scripts/extract_trainable.py \
        --checkpoint-dir "$step_dir" \
        --out "$ARTIFACTS_DIR/trainable/step_$step.npz"
    upload_artifact "$ARTIFACTS_DIR/trainable/step_$step.npz" "trainable_step_$step" "trainable_weights"

    "$PY" scripts/benchmark.py \
        --config-name "$CONFIG_NAME" \
        --checkpoint-dir "$step_dir" \
        --exp-dir "$EXPERIMENTS_DIR/$EXP_NAME/step_$step" \
        --num-trials-per-task "$TRIALS_PER_TASK" \
        --train-step "$step"
    upload_artifact "$EXPERIMENTS_DIR/$EXP_NAME/step_$step/results.json" "results_step_$step" "eval_results"
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

for target in $(seq "$EVAL_EVERY" "$EVAL_EVERY" "$TOTAL_STEPS"); do
    step=$((target - 1))   # openpi saves the final checkpoint at num_train_steps-1
    stage "train_to_$target" train_to "$target"
    stage "eval_step_$step" eval_and_archive "$step"
done

stage summary stage_summary

echo "ALL DONE — log: $LOG"
terminate_pod
