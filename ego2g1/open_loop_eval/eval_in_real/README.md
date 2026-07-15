# eval_in_real — teacher-forced eval on the real robot

The hardware twin of [`eval_in_sim`](../eval_in_sim/README.md), which runs the same
loop in MuJoCo. Every K ticks: snap the arm back to the recording's ground-truth
posture, feed the policy the **recorded** frame for that tick, execute the first K of
the 50 actions it predicts, repeat. It prints the drift — `max |q − q_gt|` after each
segment — and writes `eval_real.npz`.

`eval_real.py` is a thin driver: it reuses the `ego2g1.deploy` hardware stack
(dds/kinematics/ramp/safety/trajectory/chunk) and talks to a **separately-run**
`ego2g1.serve` over a websocket. It imports no jax/mujoco.

## Run it

Bring up `ego2g1.serve` and (if remote) the ssh tunnel exactly as for a live
deployment — see [deploy/README.md](../../deploy/README.md) "Split deployment" and
warm the server up (first infer triggers a minutes-long XLA compile) before the arm
is energised. Then:

```bash
python -m ego2g1.open_loop_eval.eval_in_real.eval_real \
    --dataset ../../lerobot_datasets/ego2g1/put_bottle_in_box \
    --episode 0 --host 127.0.0.1 --port 8000 --k 25
```

> **SAFETY** — the G1-D lowcmd path has no balance controller; every joint is held by
> our position PD. Robot on a stand or suspended, remote in hand. ctrl-C damps.

## Why this is the rung to run before the first live rollout

A bad live rollout has two explanations that nothing else separates: the policy is
bad, or the G1's head camera does not show what the Pico headset showed in training.
That viewpoint risk fails quietly — a shifted FOV just looks like a mediocre
checkpoint. Feeding the policy the recording's own frames removes the camera from the
equation entirely, so anything that goes wrong is the policy, the transforms, or the
robot.

The converse is worth saying out loud: **this rung tells you nothing about whether the
head camera is usable.** It is the control, not the experiment.

## The snap-back is a ramp, not a teleport

`eval_in_sim` can move a MuJoCo robot instantly and a real arm cannot, and after K
ticks of drift the snap can be a large motion. So it goes through the rate limiter and
then **settles** (`--settle-s`) before the next query: the anchor is the measured FK,
and reading it while the arm is still coasting anchors the chunk on a pose the robot is
not in. The pause costs nothing — a teacher-forced timeline is already discontinuous at
every snap.

No RTC, no async, no delay budget here. Those exist to make chunk seams continuous;
this rung breaks the timeline at every seam on purpose.

## `--image-resize`

Defaults to `(224, 224)` (on), same as the live deploy client — see
[deploy/README.md](../../deploy/README.md) "Client-side image resize" for why 224×224
is the only safe value.
