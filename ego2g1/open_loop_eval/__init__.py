"""Open-loop, teacher-forced checkpoint eval.

Same algorithm everywhere: every K ticks, snap to the recording's ground-truth
posture, feed the policy the RECORDED image + state, execute the first K of the 50
predicted actions, measure drift, repeat. In-distribution by construction; measures
local per-waypoint accuracy, not compounding drift.

    eval_in_sim/    the policy runs IN-PROCESS (create_policy), no serve/websocket.
      rollout.py    PPU box   — jax + checkpoint + dataset -> eval_rollout.npz
      viewer.py     display   — mujoco/mink + data_extraction; reads the npz, renders
    eval_in_real/   the policy runs as a SEPARATE `ego2g1.serve` websocket process;
                    the driver reuses the `ego2g1.deploy` hardware stack; no jax.

    dataset_io.py   shared by both halves (pure numpy/pandas/cv2; no sim/jax dep).

This __init__ MUST stay a bare docstring: `ego2g1.deploy` and the Mac reach into
this package without jax/mujoco on the box, and that only holds because importing
the package pulls in nothing.
"""
