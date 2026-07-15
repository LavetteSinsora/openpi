"""Eval in sim (two phases; see README):

- rollout (Phase 1, headless box): run the policy through the exact deployment
  stack IN-PROCESS, dump eval_rollout.npz. Needs jax + the checkpoint + a video
  decoder; NO mujoco/mink/display.
- viewer (Phase 2, a machine with a display): reconstruct the eval trajectory
  (compose + IK) and render both robots + video. Needs mujoco/mink + the
  data_extraction sim/hand/assets.

Bare on purpose: importing this package must not pull in jax or mujoco.
"""
