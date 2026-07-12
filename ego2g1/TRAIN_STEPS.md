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

2. Fetch GCS assets over plain HTTPS (one-time; the venv has no gcsfs — do NOT pip install):
```bash
python ego2g1/fetch_assets.py        # paligemma tokenizer (4 MB) + pi05_base params (12.4 GB)
```
Files land in `~/.cache/openpi/` (override: `OPENPI_DATA_HOME`); skip-if-cached, safe to re-run.

3. Compute normalization stats:
```bash
python -m ego2g1.compute_norm_stats
```
Normalization stats are written to `assets/<cfg.name>/<repo_id>/`.
Episodes in `cfg.val_real_episodes` are excluded when calculating norm stats.

4. Train (inside tmux — an SSH drop must not kill the run):

```bash
wandb login                       
python -m ego2g1.train --exp-name exp_name     # env.sh already sets XLA_PYTHON_CLIENT_MEM_FRACTION
```
Checkpoints are written to `checkpoints/<cfg.name>/<cfg.exp_name>/<step>/`.
Only keeper steps survive on disk: every `keep_period` (5k) plus the latest.
Use `--resume` to resume training from most recent checkpoint.

## Serve policy

```python
from ego2g1 import policy
p = policy.create_policy("checkpoints/ego2g1_pi05/run1/15000",
                         default_prompt="put the bottle in the box")

from openpi.serving import websocket_policy_server
websocket_policy_server.WebsocketPolicyServer(p, host="0.0.0.0", port=8000).serve_forever()
```
Serving must run from this repo (`source ego2g1/env.sh`, then python from the repo
root): the custom transforms + their mandatory inverses live in `ego2g1/`, and
`create_policy` is the only loader that applies them (stamp-guarded). The step
passed must be a keeper that exists on disk (multiple of 5k, or the latest).