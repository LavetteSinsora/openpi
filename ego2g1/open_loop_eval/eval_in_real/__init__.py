"""Eval in real: the teacher-forced loop on the actual robot (see README).

`eval_real.py` is a thin driver, not a standalone client: it reuses the shared
`ego2g1.deploy` hardware stack (client/dds/kinematics/ramp/safety/trajectory/chunk)
and talks to a SEPARATELY-run `ego2g1.serve` over a websocket. That server's code
is not vendored here.

MUST NOT import jax or mujoco — this runs on the Mac that drives the G1, which has
neither. Holds because this __init__ and eval_real's imports touch only the
jax-free deploy stack + open_loop_eval.dataset_io.
"""
