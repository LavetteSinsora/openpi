#!/usr/bin/env python3
"""
Language-instruction-following probe on the LIBERO-OBJECT task 9 scene.

Fixes the scene ("pick up the orange juice" — objects, locations, init states)
and varies ONLY the prompt, then classifies which object ends up in the basket:

  instructed      — policy grounded the object word and acted on it
  trained_target  — policy ignored language, followed the visual scene prior
                    (picked the orange juice it was trained to pick here)
  other / none    — neither; see approach distances + videos for partial credit

The env's own `done` only tracks the BDDL goal (orange juice), so success for
arbitrary objects is evaluated with LIBERO's general predicate machinery:
`env.env._eval_predicate(["in", obj, "basket_1_contain_region"])` — a
geometric point-in-box test that works for any scene object.

Usage
-----
# 1. Survey the scene first (no GPU model needed) — verify the spatial wording
#    of the detailed prompts against rendered frames + printed positions:
  .venv/bin/python scripts/benchmark_language.py --survey --exp-dir /workspace/experiments/lang_probe

# 2. Run (single process, all conditions):
  .venv/bin/python scripts/benchmark_language.py \
    --checkpoint-dir /workspace/ckpt_keep_1999 \
    --exp-dir /workspace/experiments/lang_probe_step1999

# 2'. Or sharded (parallel) + merged into one wandb run:
  ... --condition-ids 0,1,2 --exp-dir .../lang_probe_shard0 --no-wandb-enabled &
  ... --condition-ids 3,4   --exp-dir .../lang_probe_shard1 --no-wandb-enabled &
  wait
  .venv/bin/python scripts/benchmark_language.py --merge-dirs .../lang_probe_shard0 .../lang_probe_shard1 \
    --exp-dir .../lang_probe_step1999 --wandb-run-name lang_probe_step1999
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys

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

# ── scene / rollout constants (identical to benchmark.py) ────────────────────
TASK_SUITE = "libero_object"
TASK_ID = 9  # "pick up the orange juice and place it in the basket"
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = 280
NUM_STEPS_WAIT = 10
REPLAN_STEPS = 5
RESIZE = 224
DUMMY_ACTION = [0.0] * 6 + [-1.0]

TRAINED_TARGET = "orange_juice_1"
BASKET_SITE = "basket_1_contain_region"
MOVABLE_OBJECTS = [
    "orange_juice_1",
    "butter_1",
    "chocolate_pudding_1",
    "bbq_sauce_1",
    "ketchup_1",
    "salad_dressing_1",
]


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    prompt: str
    instructed_object: str


# Swap prompts use the exact training template ("pick the X and place it in the
# basket") so only the object word varies. Detailed variants add two spatial
# anchors — wording verified against --survey renders before the full run.
CONDITIONS = [
    Condition("swap_bbq", "pick the bbq sauce and place it in the basket", "bbq_sauce_1"),
    Condition("swap_pudding", "pick the chocolate pudding and place it in the basket", "chocolate_pudding_1"),
    Condition(
        "swap_bbq_detailed",
        "pick the bbq sauce in front of the butter, to the left of the orange juice, and place it in the basket",
        "bbq_sauce_1",
    ),
    Condition(
        "swap_pudding_detailed",
        "pick the chocolate pudding behind the ketchup, to the right of the orange juice, and place it in the basket",
        "chocolate_pudding_1",
    ),
    Condition("paraphrase_oj", "get the orange container and place it in the basket", "orange_juice_1"),
]


@dataclasses.dataclass
class Args:
    exp_dir: str
    checkpoint_dir: str = ""  # required unless --survey or --merge-dirs
    config_name: str = "pi05_libero_object_lora"
    num_trials_per_condition: int = 5
    # Comma-separated indices into CONDITIONS (e.g. "0,1,2") for parallel shards.
    condition_ids: str | None = None
    seed: int = 7
    wandb_project: str = "pi05_libero_replication"
    wandb_enabled: bool = True
    wandb_run_name: str = "lang_probe"
    use_checkpoint_norm_stats: bool = False
    # Render the first init states + print object positions, then exit.
    survey: bool = False
    # Merge shard exp-dirs into one wandb run, then exit.
    merge_dirs: list[str] | None = None


# ── env helpers ───────────────────────────────────────────────────────────────


def _get_env(seed: int):
    suite = libero_benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    init_states = suite.get_task_init_states(TASK_ID)
    return env, init_states


def _agentview(obs) -> np.ndarray:
    # [::-1] (vertical flip only) — matches the training dataset. See benchmark.py.
    return np.ascontiguousarray(obs["agentview_image"][::-1])


def _wrist(obs) -> np.ndarray:
    return np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])


def _object_positions(env) -> dict[str, np.ndarray]:
    sim = env.env.sim
    return {
        name: np.array(sim.data.body_xpos[env.env.obj_body_id[name]])
        for name in MOVABLE_OBJECTS
    }


def _objects_in_basket(env) -> list[str]:
    return [
        name
        for name in MOVABLE_OBJECTS
        if env.env._eval_predicate(["in", name, BASKET_SITE])
    ]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ── survey mode ───────────────────────────────────────────────────────────────


def survey(args: Args) -> None:
    """Render the trial init states and print object positions/relations."""
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    env, init_states = _get_env(args.seed)

    for idx in range(args.num_trials_per_condition):
        env.reset()
        obs = env.set_init_state(init_states[idx])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        positions = _object_positions(env)
        oj = positions[TRAINED_TARGET]
        print(f"\n── init state {idx} (world frame; +y = image-RIGHT in the agentview)")
        for name, p in positions.items():
            side = "right of OJ" if p[1] > oj[1] else "left of OJ"
            marker = "  <- trained target" if name == TRAINED_TARGET else f"  ({side}, Δy={p[1] - oj[1]:+.2f}, Δx={p[0] - oj[0]:+.2f})"
            print(f"  {name:24s} x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}{marker}")
        frame_path = exp_dir / f"survey_init{idx}.png"
        imageio.imwrite(frame_path, _agentview(obs))
        print(f"  frame -> {frame_path}")
    env.close()
    print("\nCheck the frames against the detailed-condition prompts before the full run:")
    for i, c in enumerate(CONDITIONS):
        print(f"  [{i}] {c.name}: {c.prompt!r}")


# ── rollout ───────────────────────────────────────────────────────────────────


def run_trial(env, policy, condition: Condition, init_state, video_path: pathlib.Path) -> dict:
    env.reset()
    obs = env.set_init_state(init_state)
    action_plan: collections.deque = collections.deque()
    frames = []
    min_dist = {name: float("inf") for name in MOVABLE_OBJECTS}
    in_basket: list[str] = []
    steps_used = 0

    for t in range(MAX_STEPS + NUM_STEPS_WAIT):
        if t < NUM_STEPS_WAIT:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            continue

        img_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(_agentview(obs), RESIZE, RESIZE))
        wrist_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(_wrist(obs), RESIZE, RESIZE))
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
                "prompt": condition.prompt,
            }
            chunk = policy.infer(element)["actions"]
            action_plan.extend(chunk[:REPLAN_STEPS])

        obs, _, _, _ = env.step(action_plan.popleft().tolist())
        steps_used = t - NUM_STEPS_WAIT + 1

        eef = np.asarray(obs["robot0_eef_pos"])
        for name, pos in _object_positions(env).items():
            d = float(np.linalg.norm(eef - pos))
            if d < min_dist[name]:
                min_dist[name] = d

        in_basket = _objects_in_basket(env)
        if in_basket:
            break

    if condition.instructed_object in in_basket:
        outcome = "instructed"
    elif TRAINED_TARGET in in_basket:
        outcome = "trained_target"
    elif in_basket:
        outcome = "other"
    else:
        outcome = "none"

    if frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, [np.asarray(f) for f in frames], fps=10)

    closest = min(min_dist, key=min_dist.get)
    return {
        "condition": condition.name,
        "prompt": condition.prompt,
        "instructed_object": condition.instructed_object,
        "outcome": outcome,
        "objects_in_basket": in_basket,
        "steps": steps_used,
        "closest_object": closest,
        "min_dist_instructed": round(min_dist[condition.instructed_object], 3),
        "min_dist_trained_target": round(min_dist[TRAINED_TARGET], 3),
        "min_dist": {k: round(v, 3) for k, v in min_dist.items()},
        "video": str(video_path),
    }


OUTCOMES = ["instructed", "trained_target", "other", "none"]


def _condition_summary(trials: list[dict]) -> dict:
    n = len(trials)
    summary = {f"{o}_rate": sum(t["outcome"] == o for t in trials) / n for o in OUTCOMES}
    summary["mean_min_dist_instructed"] = round(float(np.mean([t["min_dist_instructed"] for t in trials])), 3)
    summary["mean_min_dist_trained_target"] = round(float(np.mean([t["min_dist_trained_target"] for t in trials])), 3)
    return summary


# ── wandb logging (used by both direct runs and merge mode) ───────────────────


def _log_wandb(args: Args, conditions: dict) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "checkpoint_dir": args.checkpoint_dir,
            "num_trials_per_condition": args.num_trials_per_condition,
        },
    )
    log: dict = {}
    columns = ["condition", "prompt", "trial", "outcome", "objects_in_basket",
               "steps", "closest_object", "min_dist_instructed", "min_dist_trained_target"]
    table = wandb.Table(columns=columns)
    for name, cond in conditions.items():
        for key, val in cond["summary"].items():
            log[f"{name}/{key}"] = val
        for trial_idx, t in enumerate(cond["trials"]):
            table.add_data(name, t["prompt"], trial_idx, t["outcome"],
                           ",".join(t["objects_in_basket"]), t["steps"], t["closest_object"],
                           t["min_dist_instructed"], t["min_dist_trained_target"])
            video = pathlib.Path(t["video"])
            if video.exists():
                log[f"video/{name}/{video.name}"] = wandb.Video(str(video), fps=10, format="mp4")
    log["trials_table"] = table
    wandb.log(log)

    lines = [
        f"{name}: instructed {c['summary']['instructed_rate']:.0%}, "
        f"trained-target {c['summary']['trained_target_rate']:.0%}, "
        f"none {c['summary']['none_rate']:.0%}"
        for name, c in conditions.items()
    ]
    try:
        run.alert(title="Language probe done"[:64], text="\n".join(lines))
    except Exception as e:
        logging.warning(f"wandb alert failed: {e}")
    run.finish()


# ── merge mode ────────────────────────────────────────────────────────────────


def merge(args: Args) -> None:
    conditions: dict = {}
    for d in map(pathlib.Path, args.merge_dirs):
        res = json.loads((d / "results.json").read_text())
        overlap = conditions.keys() & res["conditions"].keys()
        if overlap:
            raise ValueError(f"condition(s) {sorted(overlap)} in multiple shards — check --condition-ids")
        conditions.update(res["conditions"])
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results.json").write_text(json.dumps({"conditions": conditions}, indent=2))
    for name, cond in conditions.items():
        logging.info(f"{name}: {cond['summary']}")
    if args.wandb_enabled:
        _log_wandb(args, conditions)


# ── main ──────────────────────────────────────────────────────────────────────


def run(args: Args) -> None:
    if args.merge_dirs:
        merge(args)
        return
    if args.survey:
        survey(args)
        return
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required (unless --survey or --merge-dirs)")

    np.random.seed(args.seed)
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        [CONDITIONS[int(i)] for i in args.condition_ids.split(",")]
        if args.condition_ids is not None
        else list(CONDITIONS)
    )

    # Policy loading identical to benchmark.py (explicit norm stats + non-LoRA fallback).
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = None if args.use_checkpoint_norm_stats else data_config.norm_stats
    if not args.use_checkpoint_norm_stats and norm_stats is None:
        raise FileNotFoundError(f"norm stats not found under {train_config.assets_dirs}")
    try:
        policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir, norm_stats=norm_stats)
    except ValueError:
        logging.info("checkpoint has no LoRA params — retrying with base (non-LoRA) variants")
        base_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
            ),
        )
        policy = _policy_config.create_trained_policy(base_config, args.checkpoint_dir, norm_stats=norm_stats)

    env, init_states = _get_env(args.seed)
    conditions_out: dict = {}
    for condition in selected:
        logging.info(f"── condition {condition.name}: {condition.prompt!r}")
        trials = []
        for trial_idx in range(args.num_trials_per_condition):
            video_path = exp_dir / "videos" / condition.name / f"trial_{trial_idx:02d}.mp4"
            row = run_trial(env, policy, condition, init_states[trial_idx], video_path)
            # include the outcome in the filename for easy browsing
            final_path = video_path.with_name(f"trial_{trial_idx:02d}_{row['outcome']}.mp4")
            video_path.rename(final_path)
            row["video"] = str(final_path)
            trials.append(row)
            logging.info(
                f"  trial {trial_idx}: {row['outcome']:<14s} in_basket={row['objects_in_basket']} "
                f"steps={row['steps']} d(instructed)={row['min_dist_instructed']} d(OJ)={row['min_dist_trained_target']}"
            )
        conditions_out[condition.name] = {
            "prompt": condition.prompt,
            "instructed_object": condition.instructed_object,
            "summary": _condition_summary(trials),
            "trials": trials,
        }
        logging.info(f"  summary: {conditions_out[condition.name]['summary']}")
    env.close()

    (exp_dir / "results.json").write_text(json.dumps({"conditions": conditions_out}, indent=2))
    if args.wandb_enabled:
        _log_wandb(args, conditions_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(tyro.cli(Args))
