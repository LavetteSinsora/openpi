"""Teacher-forced eval replay: two G1s (ground truth + evaluated checkpoint) +
egocentric video, scrubbable.

Two phases (see the plan / ego2g1/eval_replay/README):
- rollout (Phase 1, headless box): run the policy through the exact deployment
  stack, dump eval_rollout.npz. Needs jax + the checkpoint + a video decoder;
  NO mujoco/mink/display.
- viewer (Phase 2, a machine with a display): reconstruct the eval trajectory
  (compose + IK) and render both robots + video. Needs mujoco/mink + the
  data_extraction sim/hand/assets.

dataset_io is shared and has no sim dependency.
"""
