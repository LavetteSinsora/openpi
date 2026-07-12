# ego2g1 training
Train pi0.5 on PPU with already-configured venv in /openpi.

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

2. Compute normalization stats:
```bash
python -m ego2g1.compute_norm_stats
```
Normalization stats are written to  `assets/<cfg.name>/<repo_id>/`.

## 3. Train

```bash
python -m ego2g1.train --exp-name run1
```

- first run downloads pi05_base (~10 GB, cached in `~/.cache/openpi`)
- first train step takes minutes (XLA compilation) — not a hang
- one 96 GB device fits the model; all visible devices are used data-parallel.
  For a shakedown run on one device, restrict visibility (vendor equivalent of
  CUDA_VISIBLE_DEVICES). To shard the model itself: `--fsdp-devices N`.
- checkpoints: `checkpoints/ego2g1_pi05/run1/<step>/`, every 1k steps,
  keepers every 5k, stamp at the run root. Resume after a crash: same command
  + `--resume`.

## 4. Health checklist

Startup (first minute):
- [ ] hash assert passes; no config/sidecar errors
- [ ] `Loaded 4 fixed val batches (... from 9 real episodes)` — MUST appear
- [ ] wandb `camera_views`: egocentric image present, both wrist slots black

Curves (wandb):
- [ ] `loss` starts O(1-2), drops steeply first ~500 steps, then grinds down
- [ ] `loss/slots_00_04` ~ same order as `slots_05_24` / `slots_25_49`, all
      declining (early-slot bucket stuck high = the E001/centering diagnostic)
- [ ] `val/loss` every 1k steps tracks train down; where it bottoms out is the
      checkpoint to serve
- [ ] `grad_norm` stable O(1-10), no spikes/NaN; all four `grad_norm/*`
      components nonzero
- [ ] utilization: `watch -n1 nvidia-smi` (or vendor smi) — high util with
      brief dips at save/eval; sawtooth-to-zero = input-bound, raise num_workers

Any NaN, flat-from-start loss, or missing val line: stop, keep the log.

## 5. Serve / inspect a checkpoint

```python
from ego2g1 import policy
p = policy.create_policy("checkpoints/ego2g1_pi05/run1/19999")  # step dir; stamp-guarded
```

The stamp guard is mandatory: this run's checkpoints require per_slot_center +
degenerate_neutralization; stock openpi serving code will (correctly) be
refused, and skipping the inverse transforms would execute biased/mis-scaled
actions on the robot.
