"""Run the policy on the G1.

    python -m ego2g1.deploy --host 127.0.0.1 --task "put the bottle in the box"

Async + RTC by default. `--blocking` pauses at chunk boundaries instead (the
bring-up mode: no concurrency, no RTC, one thing to debug at a time).

Before running this, walk the ladder in `python -m ego2g1.deploy.check --help`.
Steps 1-4 there catch most deployment bugs with the model out of the loop.

SAFETY: the G1-D lowcmd path has no balance controller — every joint, legs
included, is held by our position PD. The robot must be on a stand or suspended.
Keep the remote in hand: ctrl-C damps, but the remote is the thing that always works.
"""

import dataclasses
import logging
import signal
import sys
import time

import tyro

from ego2g1.deploy import camera as _camera
from ego2g1.deploy import client as _client
from ego2g1.deploy import dds as _dds
from ego2g1.deploy import kinematics as _kin
from ego2g1.deploy import loop as _loop
from ego2g1.deploy import safety as _safety


@dataclasses.dataclass
class Args:
    task: str

    # --- policy server ---
    host: str = "127.0.0.1"
    port: int = 8000
    # Resize the frame to this (h, w) BEFORE it goes on the wire — 20x less payload,
    # which is what makes a remote server (ssh -L to the PPU box) viable at all. The
    # default matches the server's ResizeImages(224, 224), so the server's own resize
    # becomes a no-op. `--image-resize None` sends the raw head frame instead; any
    # OTHER value gets letterboxed twice. See deploy/client.py and deploy/README.md.
    image_resize: tuple[int, int] | None = (224, 224)

    # --- robot ---
    network_interface: str | None = None   # None => join the existing DDS domain
    domain: int = 0
    camera_host: str = "192.168.123.164"
    # Which eye of the G1 head stereo pair feeds the model. OPEN ITEM: this must
    # match the Pico egocentric viewpoint the training video came from.
    eye: str = "left"
    hands: bool = True

    # --- kinematics ---
    # The IK and the G1 model are the SAME ones that generated the labels. By
    # default they come from the vendored copy in deploy/_g1_sim/ (so this folder
    # ships standalone); pass a repo holding data_extraction/ to override with an
    # external copy — e.g. when developing against an edited training sim.
    data_extraction_path: str | None = None
    ik_iters: int = 5
    collision_min_dist: float = 0.005

    # --- loop ---
    blocking: bool = False
    lookahead_s: float = 0.10
    replan_margin: int = 8
    initial_d: int = 12                    # ticks; refined from measured latency

    # --- safety ---
    max_joint_step: float = 0.15           # rad per 30 Hz tick
    max_tracking_error_m: float = 0.10

    duration_s: float = 0.0                # 0 => run until interrupted


def main(args: Args) -> None:
    limits = _safety.SafetyLimits(
        max_joint_step=args.max_joint_step,
        max_tracking_error_m=args.max_tracking_error_m,
    )

    logging.info("connecting to policy server %s:%d ...", args.host, args.port)
    client = _client.PolicyClient(args.host, args.port, resize=args.image_resize)

    logging.info("loading kinematics from %s ...", args.data_extraction_path)
    kin = _kin.Kinematics(
        args.data_extraction_path,
        collision_min_dist=args.collision_min_dist,
        ik_iters=args.ik_iters,
        fps=client.fps,
    )

    logging.info("connecting DDS ...")
    dds = _dds.G1DDS(network_interface=args.network_interface, domain=args.domain,
                     enable_hands=args.hands)
    dds.connect()
    logging.info("lowstate OK. arm q = %s", dds.arm_q().round(3))

    logging.info("connecting camera (%s eye) ...", args.eye)
    cam = _camera.HeadCamera(host=args.camera_host, eye=args.eye)
    cam.connect()
    logging.info("camera OK: frame %s", cam.read().shape)

    budget = _client.DelayBudget(client.fps, initial=args.initial_d)

    loop = _loop.DeployLoop(
        _loop.LoopConfig(
            task=args.task, fps=client.fps, blocking=args.blocking,
            lookahead_s=args.lookahead_s, replan_margin=args.replan_margin,
        ),
        dds=dds, camera=cam, kinematics=kin, client=client, budget=budget, limits=limits,
    )

    def shutdown(signum=None, frame=None):
        logging.warning("shutting down -> damping")
        loop.estop("interrupted")
        loop.stop()
        cam.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logging.warning("STARTING. ctrl-C damps. Keep the remote in hand.")
    loop.start()

    t0 = time.monotonic()
    try:
        while True:
            time.sleep(2.0)
            if loop.watchdog.tripped:
                logging.error("watchdog tripped: %s", loop.watchdog.reason)
                break
            logging.info("chunks=%d ticks=%d late=%d | budget=%s | clamped=%d",
                         loop.stats["chunks"], loop.stats["ticks"], loop.stats["late"],
                         budget.stats(), loop.clamp.clamped_ticks)
            if args.duration_s and time.monotonic() - t0 > args.duration_s:
                break
    finally:
        shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
