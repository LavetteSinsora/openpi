# eval_in_sim — teacher-forced checkpoint replay in MuJoCo

Watch a checkpoint behave on a held-out episode: two G1s side by side —
**ground truth** (recorded motion) and the **evaluated checkpoint** (policy
rollout) — beside the **real egocentric video**, all on one scrubbable timeline.

**Loop = open-loop teacher-forced.** Each query feeds the policy the *real*
recorded image + state; it predicts 50 actions; the first K (default 25) are
executed on the eval robot, then the next query re-reads the real observation —
re-anchoring the eval robot to ground truth every K steps (a visible "snap").
This is in-distribution and needs no rendered observation; it shows local
per-waypoint accuracy, not compounding drift.

Two phases, because the PPU box is headless:

## Phase 1 — rollout (on the box, needs jax + checkpoint + dataset)

```bash
python -m ego2g1.open_loop_eval.eval_in_sim.rollout \
    --checkpoint checkpoints/ego2g1_pi05/run1/10000 \
    --source-episode put_bottle_in_box/episode_10 \
    --dataset-root /path/to/put_bottle_in_box \
    --stride 25 --num-steps 10 --out eval_rollout.npz
```

Runs the exact deployment stack (`create_policy` → `infer` → `sample_actions`
Euler integration) and dumps a tiny `eval_rollout.npz` (raw actions + anchor
states per query). No mujoco/mink/display. Needs a video decoder:
`pip install --user opencv-python` (additive `--user`; dry-run first).

`scp eval_rollout.npz` to the display machine.

## Phase 2 — viewer (on a machine with a display: mujoco/mink + data_extraction)

```bash
python -m ego2g1.open_loop_eval.eval_in_sim.viewer \
    --rollout eval_rollout.npz \
    --dataset-root /path/to/put_bottle_in_box \
    --data-extraction-path /path/to/ego-pi-replication \
    --hands            # attach revo2 dexterous hands (omit for arms + grip bars)
```

Reconstructs the eval trajectory (compose deltas with the anchor → mink IK),
reads GT `arm_qpos`/hand/video locally, renders both robots (offscreen) + video,
and opens a live OpenCV window. Below the three panels is a **timeline** plotting
the per-frame EEF proprioception MSE (eval vs GT), with green markers at the
teacher-forcing re-syncs — expect a sawtooth (grows within each K-window, drops
to ~0 at each re-sync). `--data-extraction-path` must contain the `data_extraction`
package (sim/hand/`assets/unitree_g1`/`assets/revo2`) — clone the outer repo or
copy that subtree.

Controls (live window): **SPACE** play/pause · **a/d** or **,/.** step · drag the
**frame** trackbar to scrub · **q/ESC** quit.

Flags:
- `--hands` attach the revo2 dexterous hands (the G1's own `rubber_hand` mesh is
  removed so they don't double up); omit for arms + grip-bar overlay only.
- `--interactive` open an **orbitable mujoco 3D viewer** with both robots in one
  scene (GT left, eval right) — drag to rotate/zoom/pan, SPACE play/pause,
  left/right arrow step. No video/timeline panel in this mode. Needs a display.
- `--mp4 --out replay.mp4` write a composited video instead of a window
  (auto-selected when headless).

## Transform sanity (no checkpoint needed)

```bash
python -m ego2g1.open_loop_eval.eval_in_sim.rollout --synthetic-gt \
    --source-episode put_bottle_in_box/episode_10 \
    --dataset-root /path/to/put_bottle_in_box --out gt.npz
python -m ego2g1.open_loop_eval.eval_in_sim.viewer --rollout gt.npz --mp4 ...
```

`--synthetic-gt` writes a dump whose composed targets exactly reproduce the
recorded trajectory, so the eval robot should **overlay** the GT robot. Verified:
IK reconstruction reproduces recorded arm joints to ~0.3° mean / 4° max.
