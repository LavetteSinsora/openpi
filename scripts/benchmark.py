#!/usr/bin/env python3
"""
Evaluate a π0.5 policy on LIBERO-OBJECT (10 tasks × 50 trials).

Usage
-----
# Baseline (pretrained π0.5, no fine-tuning):
  .venv/bin/python scripts/benchmark.py \
    --config-name pi05_libero_object_lora \
    --checkpoint-dir gs://openpi-assets/checkpoints/pi05_base \
    --exp-dir experiments/pi05_base_benchmark

# Fine-tuned checkpoint at step 5000:
  .venv/bin/python scripts/benchmark.py \
    --config-name pi05_libero_object_lora \
    --checkpoint-dir checkpoints/pi05_libero/pi05_libero_object_lora/masked_loss_summed_subsampling/5000 \
    --exp-dir experiments/masked_loss_summed_subsampling/step_5000 \
    --train-step 5000

Run from the openpi repo root with PYTHONPATH pointing at third_party/libero
(see scripts/remote_env.sh).
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import subprocess
import sys
import typing

import imageio
import numpy as np
import tyro
from libero.libero import benchmark as libero_benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

# Allow running with a bare interpreter (openpi is already importable inside the venv).
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import openpi.training.config as _config
import openpi.policies.policy_config as _policy_config

LIBERO_ENV_RESOLUTION = 256
TASK_SUITE = "libero_object"
MAX_STEPS = 280          # longest libero_object demo has 254 steps
NUM_STEPS_WAIT = 10      # let objects settle before acting
REPLAN_STEPS = 5         # execute this many actions from each predicted chunk
RESIZE = 224
MAX_FAILURE_VIDEOS = 5
MAX_SUCCESS_VIDEOS = 3
DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    exp_dir: str
    num_trials_per_task: int = 50
    seed: int = 7
    wandb_project: str = "pi05_libero_replication"
    wandb_enabled: bool = True
    train_step: int | None = None  # for wandb x-axis when mid-training eval
    # Evaluate with the norm stats stored inside the checkpoint itself instead of
    # this repo's assets. Needed for official PI checkpoints (e.g. pi05_libero),
    # which were trained with their own normalization.
    use_checkpoint_norm_stats: bool = False
    # Where the instruction fed to the policy comes from:
    #   "env"      — the BDDL (:language ...) annotation via env.language_instruction,
    #                the SAME source the training dataset was built from
    #                ("pick the ketchup and place it in the basket").
    #   "filename" — LIBERO's task.language, reconstructed from the .bddl filename
    #                ("pick UP the ketchup..."), i.e. a phrasing never seen in
    #                training. Useful as a paraphrase-robustness probe.
    prompt_source: typing.Literal["env", "filename"] = "env"
    # Comma-separated task ids to evaluate (e.g. "0,1,2,3,4"); None = all tasks.
    # Lets several benchmark processes run disjoint task shards in parallel —
    # give each its own --exp-dir and a reduced XLA_PYTHON_CLIENT_MEM_FRACTION.
    task_ids: str | None = None


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_env(task, seed: int):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _write_config_snapshot(exp_dir: pathlib.Path, train_config, checkpoint_dir: str) -> None:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT),
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    snapshot = {
        "config": dataclasses.asdict(train_config),
        "checkpoint_dir": checkpoint_dir,
        "git_commit": git_commit,
    }
    (exp_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, default=str)
    )


def run(args: Args) -> None:
    np.random.seed(args.seed)
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── load policy ──────────────────────────────────────────────────────────
    train_config = _config.get_config(args.config_name)
    # Always load OUR norm stats from the config's assets dir and pass them
    # explicitly. Otherwise create_trained_policy reads norm stats from the
    # checkpoint's own assets/ — which the pretrained pi05_base checkpoint does
    # NOT have for our asset_id, so the baseline eval would crash. Passing them
    # explicitly also guarantees baseline and fine-tuned runs use identical
    # normalization, which is what makes the comparison fair.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = data_config.norm_stats
    if args.use_checkpoint_norm_stats:
        # None → create_trained_policy loads them from the checkpoint's assets.
        norm_stats = None
    elif norm_stats is None:
        raise FileNotFoundError(
            f"norm stats for config {args.config_name} not found under "
            f"{train_config.assets_dirs} — run compute_norm_stats.py and commit "
            "assets/, or pass --use-checkpoint-norm-stats."
        )
    try:
        policy = _policy_config.create_trained_policy(
            train_config, args.checkpoint_dir, norm_stats=norm_stats
        )
    except ValueError:
        # Pretrained base checkpoints (e.g. pi05_base) contain no lora_a/lora_b
        # params, so loading them with a *_lora model config fails the pytree
        # structure check. Retry with the non-LoRA variants; every other model
        # setting and the norm stats above stay identical, keeping the
        # baseline/fine-tuned comparison fair.
        logging.info("checkpoint has no LoRA params — retrying with base (non-LoRA) variants")
        base_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model,
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
            ),
        )
        policy = _policy_config.create_trained_policy(
            base_config, args.checkpoint_dir, norm_stats=norm_stats
        )
    _write_config_snapshot(exp_dir, train_config, args.checkpoint_dir)

    # ── wandb ────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_enabled:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=exp_dir.name,
            config={
                "config_name": args.config_name,
                "checkpoint_dir": args.checkpoint_dir,
                "num_trials_per_task": args.num_trials_per_task,
                "prompt_source": args.prompt_source,
            },
        )

    # ── task suite ───────────────────────────────────────────────────────────
    suite = libero_benchmark.get_benchmark_dict()[TASK_SUITE]()
    if args.task_ids is not None:
        selected_tasks = [int(x) for x in args.task_ids.split(",")]
    else:
        selected_tasks = list(range(suite.n_tasks))
    logging.info(f"Evaluating tasks {selected_tasks} × {args.num_trials_per_task} trials")

    all_results = {}
    total_ep, total_succ = 0, 0
    video_log: dict = {}  # accumulated for a single wandb.log call at the end

    for task_id in selected_tasks:
        task = suite.get_task(task_id)
        # task_key stays filename-based regardless of prompt_source so wandb
        # metric keys and video paths are stable across eval variants.
        task_name = task.language.replace(" ", "_")
        task_key = f"task_{task_id:02d}_{task_name}"
        initial_states = suite.get_task_init_states(task_id)

        env = _get_env(task, args.seed)
        if args.prompt_source == "env":
            prompt = getattr(env, "language_instruction", None) or task.language
        else:
            prompt = task.language
        logging.info(f"  prompt ({args.prompt_source}): {prompt}")

        video_dir = exp_dir / "videos" / task_key
        (video_dir / "success").mkdir(parents=True, exist_ok=True)
        (video_dir / "failure").mkdir(parents=True, exist_ok=True)

        task_succ = 0
        saved_success = 0
        saved_failure = 0

        for ep_idx in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])
            action_plan: collections.deque = collections.deque()
            frames = []
            done = False

            for t in range(MAX_STEPS + NUM_STEPS_WAIT):
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    continue

                # [::-1] (vertical flip only) matches convert_hdf5_to_lerobot.py:
                # robosuite offscreen frames arrive bottom-up (OpenGL row order).
                # openpi's LIBERO example uses [::-1, ::-1] because PI's RLDS
                # dataset is 180°-rotated — OURS IS NOT; that extra horizontal
                # flip fed the model a mirrored world at eval.
                img = np.ascontiguousarray(obs["agentview_image"][::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
                img_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
                wrist_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
                frames.append(img_r)

                if not action_plan:
                    element = {
                        "observation/image": img_r,
                        "observation/wrist_image": wrist_r,
                        "observation/state": np.concatenate([
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                        "prompt": prompt,
                    }
                    chunk = policy.infer(element)["actions"]
                    action_plan.extend(chunk[:REPLAN_STEPS])

                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    task_succ += 1
                    break

            suffix = "success" if done else "failure"
            save_video = (done and saved_success < MAX_SUCCESS_VIDEOS) or (
                not done and saved_failure < MAX_FAILURE_VIDEOS
            )
            if save_video and frames:
                ep_label = f"ep_{ep_idx:02d}.mp4"
                video_path = video_dir / suffix / ep_label
                imageio.mimwrite(
                    video_path,
                    [np.asarray(f) for f in frames],
                    fps=10,
                )
                video_log[f"video/{task_key}/{suffix}/{ep_label}"] = str(video_path)
                if done:
                    saved_success += 1
                else:
                    saved_failure += 1

            if (ep_idx + 1) % 10 == 0:
                logging.info(
                    f"  {task_key}: {ep_idx + 1}/{args.num_trials_per_task} trials, "
                    f"{task_succ} successes"
                )

        env.close()
        success_rate = task_succ / args.num_trials_per_task
        all_results[task_key] = {
            "success_rate": success_rate,
            "successes": task_succ,
            "trials": args.num_trials_per_task,
        }
        total_ep += args.num_trials_per_task
        total_succ += task_succ
        logging.info(f"[task {task_id}, {len(all_results)}/{len(selected_tasks)}] {task_key}: {success_rate:.1%}")
        if wandb_run is not None:
            # incremental progress so the run is inspectable long before it ends
            wandb.log({
                f"{task_key}/success_rate": success_rate,
                "tasks_completed": task_id + 1,
                "running_success_rate": total_succ / total_ep,
            })

    # ── aggregate + write results ─────────────────────────────────────────────
    aggregate_rate = total_succ / total_ep
    results = {
        "aggregate_success_rate": aggregate_rate,
        "prompt_source": args.prompt_source,
        "per_task": all_results,
    }
    (exp_dir / "results.json").write_text(json.dumps(results, indent=2))
    logging.info(f"Aggregate success rate: {aggregate_rate:.1%}")

    if wandb_run is not None:
        log_dict = {"aggregate_success_rate": aggregate_rate}
        for k, v in all_results.items():
            log_dict[f"{k}/success_rate"] = v["success_rate"]
        log_dict.update({k: wandb.Video(v, fps=10, format="mp4") for k, v in video_log.items()})
        if args.train_step is not None:
            wandb.log(log_dict, step=args.train_step)
        else:
            wandb.log(log_dict)
        try:
            # Email/Slack notification (requires "Scriptable run alerts" enabled
            # in the wandb user settings).
            wandb_run.alert(
                title=f"Eval done: {exp_dir.name}"[:64],
                text=(
                    f"aggregate success rate {aggregate_rate:.1%} "
                    f"over {total_ep} episodes\ncheckpoint: {args.checkpoint_dir}"
                ),
            )
        except Exception as e:
            logging.warning(f"wandb alert failed: {e}")
        wandb_run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(tyro.cli(Args))
