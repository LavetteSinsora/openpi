"""Robot client: the hardware half of deployment.

Owns everything stateful and real-time — cameras, DDS, FK/IK, the trajectory
buffer, the threads, the e-stop. It talks to the policy over a websocket and has
no idea what a flow-matching model is.

    python -m ego2g1.deploy --host 127.0.0.1 --task "put the bottle in the box"

MUST NOT import jax or openpi. The robot PC has neither. This holds today because
ego2g1/__init__.py is a bare docstring and the gemma patch fires at
ego2g1.model import time, which only train and serve touch. ego2g1.common is
pure numpy and safe to import from here; ego2g1.config is NOT (it pulls in jax).

The client is deliberately config-free about the model: it reads the action
layout, horizon, and fps out of the server's metadata handshake. The checkpoint
decides what it is; the client just supplies observations.
"""
