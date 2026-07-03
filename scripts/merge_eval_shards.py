#!/usr/bin/env python3
"""Merge parallel benchmark.py task shards into ONE wandb run.

Run the shards with wandb disabled, then merge:

  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python scripts/benchmark.py ... \
      --task-ids 0,1,2,3,4 --exp-dir .../step_999_fixed_shard0 --no-wandb-enabled &
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python scripts/benchmark.py ... \
      --task-ids 5,6,7,8,9 --exp-dir .../step_999_fixed_shard1 --no-wandb-enabled &
  wait
  python scripts/merge_eval_shards.py \
      --exp-dirs .../step_999_fixed_shard0 .../step_999_fixed_shard1 \
      --name step_999_fixed --train-step 999

Produces a single wandb run (per-task rates, one aggregate, all videos, one
alert) — indistinguishable from an unsharded benchmark run.
"""

import argparse
import json
import pathlib

import wandb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-dirs", nargs="+", required=True, help="shard --exp-dir paths")
    p.add_argument("--name", required=True, help="merged wandb run name, e.g. step_999_fixed")
    p.add_argument("--train-step", type=int, default=None)
    p.add_argument("--wandb-project", default="pi05_libero_replication")
    args = p.parse_args()

    per_task: dict = {}
    prompt_source = None
    total_succ = total_ep = 0
    video_log: dict = {}
    for d in map(pathlib.Path, args.exp_dirs):
        res = json.loads((d / "results.json").read_text())
        prompt_source = prompt_source or res.get("prompt_source")
        overlap = per_task.keys() & res["per_task"].keys()
        if overlap:
            raise ValueError(f"task(s) {sorted(overlap)} appear in multiple shards — check --task-ids")
        for k, v in res["per_task"].items():
            per_task[k] = v
            total_succ += v["successes"]
            total_ep += v["trials"]
        # videos/<task_key>/<success|failure>/ep_XX.mp4
        for mp4 in sorted(d.glob("videos/*/*/*.mp4")):
            video_log["video/" + "/".join(mp4.parts[-3:])] = str(mp4)

    aggregate = total_succ / total_ep
    merged = {
        "aggregate_success_rate": aggregate,
        "prompt_source": prompt_source,
        "per_task": dict(sorted(per_task.items())),
    }
    out = pathlib.Path(args.exp_dirs[0]).parent / f"{args.name}_results.json"
    out.write_text(json.dumps(merged, indent=2))
    print(f"aggregate over {total_ep} episodes ({len(per_task)} tasks): {aggregate:.1%} → {out}")

    run = wandb.init(project=args.wandb_project, name=args.name, config={"train_step": args.train_step})
    log = {"aggregate_success_rate": aggregate}
    for k, v in per_task.items():
        log[f"{k}/success_rate"] = v["success_rate"]
    log.update({k: wandb.Video(v, fps=10, format="mp4") for k, v in video_log.items()})
    if args.train_step is not None:
        wandb.log(log, step=args.train_step)
    else:
        wandb.log(log)
    try:
        run.alert(
            title=f"Eval done: {args.name}"[:64],
            text=f"aggregate success rate {aggregate:.1%} over {total_ep} episodes",
        )
    except Exception:
        pass
    run.finish()


if __name__ == "__main__":
    main()
