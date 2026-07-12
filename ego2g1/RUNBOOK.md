# ego2g1 training runbook (training machine)

Assumes: an openpi checkout with a working uv env, the LeRobot dataset
(with `extraction_meta.json` sidecar) somewhere on disk, a GPU that fits a
pi05 full fine-tune (openpi README: > 70 GB single-GPU; use `--fsdp-devices N`
across GPUs otherwise), and network access for the pi05_base weights + wandb.

## 0. Code

```bash
cd <openpi-checkout>
git remote add mine https://github.com/LavetteSinsora/openpi   # once
git fetch mine && git switch ego2g1-data
uv sync    # no-op if env already matches the lock
# the env must resolve THIS checkout (stale editable pointers are the silent killer):
uv run python -c "import openpi, pathlib; print(pathlib.Path(openpi.__file__).resolve())"
```

`src/openpi` on this branch is byte-identical to upstream/main — everything
custom is in `ego2g1/`. Do **not** hand-install extra packages into this env
without pins: an unpinned install can bump jax (0.5.3 → 0.10.x) and break
openpi's flax scan (this bit us locally).

## 1. Gates before GPU time

```bash
uv run python -m pytest ego2g1/tests -q     # fingerprint guard + golden identity on THIS box
```

All must pass (1 skip is normal: the chunk-math equivalence test runs in
the outer repo). This validates the gemma-patch fingerprint against the
checkout and bitwise-stock behavior with features off.

## 2. Dataset

Copy the **regenerated** dataset (the pre-cleanliness-layer one is stale) to
e.g. `~/data/put_bottle_in_box`, then read its hash — never trust a
remembered value:

```bash
HASH=$(uv run python -c "import json;print(json.load(open('$HOME/data/put_bottle_in_box/extraction_meta.json'))['config_hash'])")
```

## 3. Shared flags

`compute_norm_stats` and `train` must see the SAME config (val split, c,
horizon...). Keep one flags file:

```bash
FLAGS=(--dataset-root ~/data/put_bottle_in_box
       --expected-config-hash $HASH
       --val-real-episodes put_bottle_in_box/episode_7 put_bottle_in_box/episode_23 put_bottle_in_box/episode_41 put_bottle_in_box/episode_58 put_bottle_in_box/episode_76)
```

(Val list: pick ~10% of real episodes, fixed forever; entries are sidecar
`source_episode` values. The list above is a placeholder — settle it once.)

## 4. Norm stats (train split only — val is excluded automatically)

```bash
uv run python -m ego2g1.compute_norm_stats "${FLAGS[@]}"
```

Writes `assets/ego2g1_pi05/<repo_id>/{norm_stats.json,per_slot_stats.npz}`
(run from the openpi root so `./assets` matches training), prints the
per-slot sigma grid + E001 gains for eyeballing, and hard-fails on
non-allowlisted degenerate dims and on unmasked spike-tail dims (max
|normalized| gate).

Since the centering/degeneracy refactor the per-slot artifact also carries
`mu_slot` — stats computed by older code will be refused at train time when
`per_slot_center` is on (the default). Regenerate rather than reuse.

## 5. Train

```bash
wandb login                       # or add --no-wandb-enabled
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python -m ego2g1.train \
    "${FLAGS[@]}" --exp-name run1
```

First run downloads `gs://openpi-assets/checkpoints/pi05_base/params`
(cached under `~/.cache/openpi`). Multi-GPU: `--fsdp-devices N`.
Phase 1 keeps the defaults `--per-slot-floor-c 0.1` and no RTC
(`--rtc-training` turns it on later — nothing else changes).

Logged: `loss`, `grad_norm`, `param_norm`, per-slot buckets
`loss/slots_00_04|05_24|25_49` every `log_interval`; `val/loss` + the same
buckets every `eval_interval` (fixed rng + fixed val batches ⇒ comparable
across steps; EMA params, i.e. what gets served).

## 6. Serve / analyze a checkpoint

```python
from ego2g1 import policy
p = policy.create_policy("checkpoints/ego2g1_pi05/run1/29999")   # stamp-guarded

from ego2g1 import diagnostics
out = diagnostics.attention_allocation(model, obs, actions)      # per-layer/per-slot allocation
gap = diagnostics.image_patch_gap(model, human_img, robot_img)   # SigLIP patch gap
```

The stamp guard is not optional: serving an E001 checkpoint without the
per-slot inverse rescale executes early slots up to 10× too large.
