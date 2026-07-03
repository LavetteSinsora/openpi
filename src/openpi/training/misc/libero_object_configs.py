"""
Fine-tuning config for LIBERO-OBJECT with π0.5, LoRA, and masked loss.

How this differs from the officially provided pi05_libero fine-tuning config
===========================================================================

1. Frequency mismatch and summed-subsampling action labels
-----------------------------------------------------------
The LIBERO demonstrations were recorded at 20 Hz, but our training and
deployment target is 10 Hz.  The naïve approach is to take every other frame
and keep the action label that was recorded at that timestep.  This is wrong
for two reasons:

  (a) Speed: a 20 Hz action command is designed to move the end-effector in
      50 ms.  If we apply it for 100 ms (the 10 Hz window), the robot reaches
      its target in the first half of the window and then sits idle.  Effective
      speed is halved.

  (b) Distribution shift: at 10 Hz the state the robot is in *after* executing
      one step is the state after two 20 Hz steps, but in the naive-subsampled
      dataset the model only ever saw states that are one 20 Hz step apart.
      At deployment the robot will be in states it was never trained on.

Our fix is summed subsampling:

    action_10hz[i] = action_20hz[2i] + action_20hz[2i+1]

This works because LIBERO actions are *relative* delta commands (OSC
position/orientation deltas), so the sum gives exactly the displacement needed
to cover the full 100 ms window.  The robot then moves continuously throughout
the 100 ms window and arrives at the state the model was trained to expect.
Gripper commands are NOT deltas (they are target positions), so we take only
the more recent of the two: action_20hz[2i+1].

The conversion is implemented in scripts/convert_hdf5_to_lerobot.py.
The official config uses a plain LeRobot dataset from physical-intelligence/libero
which does not perform this correction.

2. Masked loss on real action dimensions only
---------------------------------------------
The π0 / π0.5 model internally uses action_dim=32 for all tasks.  The 7-DOF
LIBERO actions (6 EEF delta + 1 gripper) are right-padded to 32 dimensions
with zeros by PadStatesAndActions in the transform pipeline.

During training with the standard loss:
    loss = mean_over_dims(||v_t - u_t||^2)

the 25 padded dimensions contribute to the loss.  Their noisy actions are pure
Gaussian noise (x_t = t*noise + (1-t)*0), so u_t = noise - 0 = noise.  The
model learns to output 0 for those dims (which minimises the expected loss of
the padded channels), but the gradient signal from 25/32 ≈ 78% of the
dimensions is wasted on a task that doesn't matter.

Our fix is to set action_dim_actual=7 in Pi0Config (a field added to openpi
for this project) and apply the loss only to the first 7 dims:
    loss = mean_over_7_dims(||v_t[:7] - u_t[:7]||^2)

This is implemented via the action_dim_actual field added to Pi0Config and
Pi0.__init__/compute_loss in third_party/openpi/src/openpi/models/pi0{_config}.py.

3. LoRA fine-tuning instead of full fine-tuning
------------------------------------------------
The official pi05_libero config fine-tunes all parameters.  We use LoRA
(Low-Rank Adaptation) adapters on all attention projections (Q/K/V/O) and all
FFN projections in both the PaliGemma 2B backbone and the 300M action expert.

  - paligemma_variant="gemma_2b_lora":  rank=16, alpha=16 (scaling=1.0)
  - action_expert_variant="gemma_300m_lora": rank=32, alpha=32 (scaling=1.0)

The freeze_filter returned by Pi0Config.get_freeze_filter() freezes the
transformer weights (paths matching .*llm.*) except the LoRA adapters. Note
that everything OUTSIDE the llm path stays trainable: the SigLIP vision tower
(PaliGemma/img, ~400M params) and the action/time projection layers. This
matches upstream openpi LoRA behavior — but it means a checkpoint is
reconstructed from base + ALL trainable leaves, not base + LoRA alone
(see scripts/extract_trainable.py).
EMA is disabled (ema_decay=None) because EMA would allocate a full copy of
the 2.3B-parameter model for weights that are mostly frozen — wasteful.

The pre-trained π0.5 checkpoint is loaded from GCS and LoRA adapter weights
are randomly initialised (CheckpointWeightLoader fills missing .*lora.* keys
from the random-init model via its missing_regex=".*lora.*" logic).

4. discrete_state_input=True
-----------------------------
The official pi05_libero config sets discrete_state_input=False, meaning the
robot's proprioceptive state is completely unused (neither tokenised into the
language prefix nor passed to the action expert).  We set it to True so the
state is tokenised and prepended to the language prompt.  This gives the model
access to the current end-effector position and gripper state, which is useful
for the relatively precise manipulation tasks in LIBERO-OBJECT.

5. Dataset location and HF_LEROBOT_HOME
-----------------------------------------
The repo_id "libero_object_summed_subsampling" is a *local* identifier.
LeRobot resolves the on-disk path as:

    HF_LEROBOT_HOME / repo_id
    = <project_root>/data/lerobot/pi05_libero / libero_object_summed_subsampling

Set HF_LEROBOT_HOME to <project_root>/data/lerobot/pi05_libero before running
compute_norm_stats.py or train.py.  See the project README for the exact
export command.

If you later upload the dataset to HuggingFace Hub under an org (e.g.
"pi05_libero/libero_object_summed_subsampling"), change repo_id to the full
org/dataset string and point HF_LEROBOT_HOME one level higher
(<project_root>/data/lerobot) so the paths still align.  Also add
AssetsConfig(asset_id="libero_object_summed_subsampling") to avoid the nested
path in the assets directory.

Norm stats are written to:
    assets/pi05_libero/pi05_libero_object_lora/libero_object_summed_subsampling/norm_stats.json

The pi05_libero/ grouping layer lets future experiments (e.g. pi05_libero_spatial_*)
live alongside this one under the same project umbrella without colliding.

6. Training-step budget
------------------------
We have ~12,500 datapoints and use batch_size=32, giving ≈391 steps per epoch.
30,000 steps ≈ 77 epochs.  The default CosineDecaySchedule is set up with
decay_steps=30,000, so the learning rate is designed to reach ~0 exactly at
the end of training — the schedule and step count are coupled.  With LoRA
(~42M trainable params out of 2.3B) and a strong pretrained prior, convergence
should be visible within the first 5k–15k steps; the eval-every-5k protocol
lets you stop early if the success rate plateaus.
"""


def get_libero_object_configs():
    # Import here to avoid circular imports (same pattern as polaris_config.py).
    from openpi.models import pi0_config
    from openpi.training import weight_loaders
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotLiberoDataConfig
    from openpi.training.config import TrainConfig

    return [
        TrainConfig(
            name="pi05_libero_object_lora",
            project_name="pi05_libero_replication",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=10,
                # Tokenise the 8-dim state into the language prompt so the model
                # can condition on the current EEF pose and gripper state.
                discrete_state_input=True,
                # Only compute flow-matching loss on the 7 real dims; dims 8-32
                # are zero-padding and contribute no useful gradient signal.
                action_dim_actual=7,
                # LoRA variants add rank-16 / rank-32 adapters to all attention
                # and FFN projections; all other weights are frozen.
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotLiberoDataConfig(
                # Local name only. LeRobot resolves this to:
                #   HF_LEROBOT_HOME/libero_object_summed_subsampling/
                # Set HF_LEROBOT_HOME=<project_root>/data/lerobot/pi05_libero
                # before running any training or norm-stats scripts.
                repo_id="libero_object_summed_subsampling",
                base_config=DataConfig(prompt_from_task=True),
                # LIBERO actions are already relative deltas — no extra
                # delta-conversion needed (unlike some older pi0 checkpoints).
                extra_delta_transform=False,
            ),
            # Load frozen base weights; LoRA adapters are randomly initialised
            # (CheckpointWeightLoader fills missing .*lora.* keys from the
            # random-init model, see weight_loaders.CheckpointWeightLoader).
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            # Freeze everything in .*llm.* EXCEPT .*lora.* params.
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            # EMA over a 2.3B-param model where 99.9% is frozen wastes ~4.6 GB.
            ema_decay=None,
            batch_size=32,
            # 30k steps × 32 batch ≈ 77 epochs over 12,500 datapoints.
            # The default CosineDecaySchedule also has decay_steps=30,000,
            # so the LR curve and step count are aligned by design.
            # Eval every 5k steps; stop early if success rate plateaus.
            num_train_steps=30_000,
            save_interval=5_000,
            # max_to_keep=1 is hardcoded in checkpoints.py, but keep_period
            # overrides it: orbax permanently preserves any checkpoint whose
            # step % keep_period == 0. With keep_period=5000 a single 30k run
            # keeps checkpoints at 5000/10000/15000/20000/25000 (plus the final
            # step, 29999, kept as the latest by max_to_keep=1). This lets us
            # evaluate every periodic checkpoint after one continuous training
            # run — no restart-and-resume blocks needed. ~6 × 4.8 GB on local
            # Colab disk, which is well within the A100 runtime's storage.
            keep_period=5_000,
            # Relative to the openpi repo root (the cwd when running
            # scripts/train.py / scripts/benchmark.py). pi05_libero/ groups all
            # LIBERO experiments; config name provides the next level
            # (pi05_libero_object_lora/), then asset_id below that.
            assets_base_dir="./assets/pi05_libero",
            checkpoint_base_dir="./checkpoints/pi05_libero",
        ),
    ]
