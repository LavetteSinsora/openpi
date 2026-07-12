# ego2g1 training
Train pi0.5 on PPU with already-configured venv in `~/openpi`.

## Clone repo

```bash
git clone https://github.com/LavetteSinsora/openpi ~/openpi-ego2g1
cd ~/openpi-ego2g1 && git switch ego2g1-data
rm -rf .venv                      # this directory's .venv isn't used. use the already configured venv in /openpi
```

## Training

Edit `ego2g1/config.py`:

| field | set to |
|---|---|
| `dataset_root` | ABSOLUTE path to the dataset dir |
| `expected_config_hash` | dataset's `config_hash`; read from dataset's `extraction_meta.json` |
| `num_workers` | ~8 on a big box (video decode is the loader bottleneck) |
| `val_real_episodes` | episodes treated as validation set |

1. Setup venv:
```bash
source ~/openpi-ego2g1/ego2g1/env.sh  
python -m pytest ego2g1/tests -q         
```

2. Fetch GCS assets over plain HTTPS (venv has no gcsfs):
```bash
python ego2g1/fetch_assets.py        # paligemma tokenizer (4 MB) + pi05_base params (12.4 GB)
```
Files land in `~/.cache/openpi/`.

3. Compute normalization stats:
```bash
python -m ego2g1.compute_norm_stats
```
Normalization stats are written to `assets/<cfg.name>/<repo_id>/`.
Episodes in `cfg.val_real_episodes` are excluded when calculating norm stats.

4. Train:

```bash
wandb login                       
python -m ego2g1.train --exp-name exp_name    
```
Checkpoints are written to `checkpoints/<cfg.name>/<cfg.exp_name>/<step>/`.
Use `--resume` to resume training from most recent checkpoint.

## Serve policy

Run inside tmux (the server blocks until killed). Pick a checkpoint step that
exists on disk: `ls checkpoints/ego2g1_pi05/run1/` (keepers = multiples of 5k,
plus the latest).

```bash
source ~/openpi-ego2g1/ego2g1/env.sh
python - <<'EOF'
from ego2g1 import policy
from openpi.serving import websocket_policy_server

p = policy.create_policy("checkpoints/ego2g1_pi05/run1/15000",
                         default_prompt="put the bottle in the box")
print("stamp:", sorted(p.metadata["ego2g1_stamp"]["feature_flags"]))  # loaded via the guard

websocket_policy_server.WebsocketPolicyServer(p, host="0.0.0.0", port=8000).serve_forever()
EOF
```

Notes:
- serving must run from this repo with env.sh sourced: the custom transforms +
  their mandatory inverses live in `ego2g1/`, and `create_policy` is the only
  loader that applies them (stamp-guarded; stock openpi serving is refused).
- the FIRST infer request triggers XLA compilation (minutes) — warm up before
  connecting the robot.
- robot client: `openpi_client` websocket to `<server>:8000`; send
  `{"observation/image", "observation/state" (30,), "prompt"}`, receive
  `{"actions": (50, 30)}` in raw units — anchor-relative EEF deltas (compose
  with the obs-tick flange pose via ego2g1.chunk_math, then IK) + absolute
  hand commands.