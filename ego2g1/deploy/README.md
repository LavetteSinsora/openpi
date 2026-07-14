# ego2g1 deploy

The robot half. It never imports JAX or `ego2g1.config`: the model's layout
(horizon, dim, fps, hands, RTC mode) arrives in the server's websocket handshake,
so the machine driving the G1 needs nothing but `openpi_client`, numpy, mink and
the Unitree SDK.

Walk `python -m ego2g1.deploy.check --help` before every session. Rungs 1-7 catch
most deployment bugs with the model out of the loop.

> **SAFETY** — the G1-D lowcmd path has no balance controller: every joint, legs
> included, is held by our position PD. Robot on a stand or suspended, remote in
> hand. ctrl-C damps; the remote is the thing that always works.

## Split deployment: serve on the PPU cluster, deploy on the Mac

Nothing in the code assumes co-location. `ego2g1.serve` binds `0.0.0.0:8000`;
`ego2g1.deploy` takes `--host/--port` for the policy server, entirely separate
from the DDS/camera args that point at the robot. So the PPU box can hold the
GPU and the checkpoint while the Mac — the only machine on the G1-D's
192.168.123.x subnet — runs the control loop.

**On the cluster:**

```bash
source ego2g1/env.sh                  # or EGO2G1_VENV=/path/to/.venv source ego2g1/env.sh
python -m ego2g1.serve --checkpoint checkpoints/ego2g1_pi05/<exp>/<step>
```

**On the Mac** — tunnel rather than exposing the port. `PolicyClient` accepts an
`api_key` but the deploy entrypoint never plumbs one through, so the server has
no auth in practice; the tunnel *is* the auth.

```bash
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
    -L 8000:localhost:8000 user@ppu

python -m ego2g1.deploy --host 127.0.0.1 --port 8000 --task "..." --blocking
```

If the serve job lands on a compute node rather than the box you SSH into,
forward to that node from the login node: `-L 8000:node42:8000 user@login`.

Warm the server up before connecting the robot: the first infer request triggers
an XLA compile that takes minutes, and the loop's starvation watchdog is armed
only after the first chunk lands — precisely so that compile can't damp a robot
that has not yet moved.

### The wire budget is the delay budget

`DelayBudget.max_d` is 20 ticks — **~667 ms at 30 Hz, and that is a hard ceiling**.
Round trip (upload + inference + download) has to stay under it at p95, or `d`
saturates: the loop keeps planning, but chunks splice at slot `d` without RTC's
continuity guarantee and the seams rest on the joint clamp alone. The loop logs
`budget={'d':…, 'p95_ms':…, 'saturated':…, 'violations':…}` every 2 s — that is
the go/no-go readout. Run `--blocking` first and watch it before enabling the
async/RTC path.

If the tunnel dies mid-episode the websocket call in T5 just blocks (TCP will not
fail fast), no chunks arrive, the trajectory drains, and the starvation watchdog
trips at 1.0 s and damps. Correct behaviour — but with a WAN in the loop it is a
live risk, not a theoretical one.

## Client-side image resize (`--image-resize`)

**Default: `(224, 224)`, i.e. on.** `PolicyClient._prepare_image` resizes the
frame at the wire, and it is the single change that makes a remote server viable.

The camera hands out the raw head frame — ~920 KB at 640x480, ~2.7 MB at 720p —
and `openpi_client` ships it as raw msgpack with websocket compression off, on
every inference. Over the robot LAN that is free. Through an ssh tunnel to the
cluster it is the dominant latency term, and latency is `d` (see above).
Resizing first puts 150 KB on the wire instead of 2.7 MB.

**224x224 is the only safe value.** The server's pipeline still runs
`ResizeImages(224, 224)`, an aspect-preserving pad; `resize_with_pad`
early-returns identity on an image already at the target size, so at 224x224 the
server's call is a literal no-op and the model sees exactly one letterbox. Resize
to anything *else* here and the server will letterbox the letterbox — a quiet
train/serve viewpoint mismatch, which is the failure mode that looks like a bad
policy rather than a bug.

So the flag is the full target, not a bool: changing it changes what the model
sees.

```bash
--image-resize 224 224     # default
--image-resize None        # send the raw frame; server does the resize, as before
```

Turn it off (`None`) if you want the server to own the resize again — e.g. when
you switch to a different resize method or a checkpoint trained at another
resolution. If you do, do it on the LAN, or expect `saturated` to climb.

`HeadCamera.read()` is unaffected and still returns the raw frame: `check camera`
and any Rerun logging must see what the sensor actually produced, not what we
chose to transmit.
