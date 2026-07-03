#!/usr/bin/env python3
"""Send wandb alerts / upload artifacts from the remote orchestration script.

All calls attach to a single per-experiment "orchestrator" run (resumed by id),
so the whole experiment's notifications and permanent artifacts live in one
place in the wandb UI.

Usage
-----
  python scripts/wandb_notify.py alert --title "training done" --text "..."
  python scripts/wandb_notify.py artifact --path artifacts/lora/step_5000.npz \
      --name lora_step_5000 --type lora_weights

Alerts are delivered by email and/or Slack only if enabled in wandb:
  wandb.ai → User settings → Alerts → turn on "Scriptable run alerts".
"""

import argparse
import os
import re

import wandb


def _init_run() -> "wandb.sdk.wandb_run.Run":
    exp = os.environ.get("EXP_NAME", "experiment")
    run_id = re.sub(r"[^a-zA-Z0-9_-]", "-", f"orchestrator-{exp}")[:64]
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT_NAME", "pi05_libero_replication"),
        id=run_id,
        name=run_id,
        resume="allow",
        job_type="orchestration",
        settings=wandb.Settings(silent=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    alert = sub.add_parser("alert", help="send a wandb alert (email/Slack)")
    alert.add_argument("--title", required=True)
    alert.add_argument("--text", default="")

    artifact = sub.add_parser("artifact", help="upload a file as a wandb artifact")
    artifact.add_argument("--path", required=True)
    artifact.add_argument("--name", required=True)
    artifact.add_argument("--type", default="artifact")

    args = parser.parse_args()
    run = _init_run()
    if args.cmd == "alert":
        # wandb caps alert titles at 64 chars.
        run.alert(title=args.title[:64], text=args.text[:2048])
        print(f"alert sent: {args.title[:64]}")
    else:
        art = wandb.Artifact(name=args.name, type=args.type)
        art.add_file(args.path)
        run.log_artifact(art)
        print(f"artifact logged: {args.name} ({args.path})")
    run.finish()


if __name__ == "__main__":
    main()
