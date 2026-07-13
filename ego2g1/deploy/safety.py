"""The safety layer. The vendor's lowcmd path ships with none, so this is not
optional garnish — it is the thing standing between an IK glitch and the hardware.

Three independent gates, because they fail for different reasons:

  1. Joint-delta clamp  — between IK and the trajectory. Catches a bad IK solve
     (unreachable target, QP corner case, a garbage action from a mis-normalized
     chunk) BEFORE it becomes a 500 Hz setpoint. A single 30 Hz knot that jumps
     1 rad is a violent motion; clamping it to a survivable step turns a lurch
     into a lag.
  2. Watchdog          — state staleness and trajectory starvation. If we cannot
     SEE the robot, we must not command it.
  3. E-stop            — one latched call to dds.damp(). Remember that stopping
     the publisher is NOT a stop: the firmware holds the last setpoint forever.

Everything here trips to `damp()`, never to "stop sending".
"""

import dataclasses
import logging
import threading
import time

import numpy as np

from ego2g1.common import layout

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SafetyLimits:
    # Max joint motion per 30 Hz action tick. 0.15 rad @ 30 Hz ~ 4.5 rad/s, which
    # is brisk but not violent for the G1 arm.
    max_joint_step: float = 0.15
    # Absolute joint-velocity ceiling used for the same check when ticks are late.
    max_joint_vel: float = 5.0
    # Trip the e-stop if lowstate goes quiet for this long.
    max_state_age: float = 0.2
    # Trip if the camera stops delivering frames — otherwise the policy is fed a
    # frozen image forever while the arm keeps moving on it.
    max_camera_age: float = 0.5
    # Trip if the emitter runs out of plan for this long (it holds the last knot,
    # which is safe, but it means the planner has died).
    max_starvation: float = 1.0
    # Trip if the IK cannot reach targets by this much — the policy is asking for
    # something the arm physically cannot do, or our frames are wrong.
    max_tracking_error_m: float = 0.10
    # Consecutive violations tolerated before tripping (a single spike is noise).
    trip_after: int = 3


class Clamp:
    """Rate-limits the joint knots leaving the IK."""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self._last = None
        self.clamped_ticks = 0
        self.max_seen = 0.0

    def reset(self, q14) -> None:
        self._last = np.asarray(q14, dtype=np.float64).copy()

    def __call__(self, q14, dt: float):
        """Return a joint vector no further than the limit from the previous one."""
        q = np.asarray(q14, dtype=np.float64)
        if self._last is None:
            self._last = q.copy()
            return q.copy()

        step = q - self._last
        mag = float(np.abs(step).max())
        self.max_seen = max(self.max_seen, mag)

        cap = min(self.limits.max_joint_step, self.limits.max_joint_vel * max(dt, 1e-3))
        if mag > cap:
            self.clamped_ticks += 1
            step = step * (cap / mag)
            logger.warning("joint step %.3f rad clamped to %.3f rad", mag, cap)

        out = self._last + step
        self._last = out.copy()
        return out


class Watchdog:
    """Trips the e-stop when the world stops making sense."""

    def __init__(self, limits: SafetyLimits, on_trip):
        self.limits = limits
        self._on_trip = on_trip
        self._strikes = {}
        self._starving_since: float | None = None
        self._tripped = False
        self._lock = threading.Lock()
        self.reason: str | None = None

    @property
    def tripped(self) -> bool:
        return self._tripped

    def trip(self, reason: str) -> None:
        with self._lock:
            if self._tripped:
                return
            self._tripped = True
            self.reason = reason
        logger.error("E-STOP: %s", reason)
        try:
            self._on_trip()
        except Exception:
            logger.exception("e-stop handler raised")

    def _strike(self, key: str, bad: bool, reason: str) -> None:
        n = self._strikes.get(key, 0)
        if not bad:
            self._strikes[key] = 0
            return
        n += 1
        self._strikes[key] = n
        if n >= self.limits.trip_after:
            self.trip(reason)

    def check_state_age(self, age: float) -> None:
        self._strike("state", age > self.limits.max_state_age,
                     f"rt/lowstate stale for {age:.3f}s — cannot see the robot")

    def check_starvation(self, runway: float, now: float) -> None:
        """Empty trajectory means the emitter is holding its last knot — safe, but
        it means we have no plan.

        This is DURATION-based, not strike-based, because a brief empty buffer is
        normal and expected: at episode start, and after a chunk is exhausted, we
        legitimately have nothing queued while the first/next inference runs. Only
        a SUSTAINED absence of plan means the planner has actually died. (A strike
        counter here trips in `trip_after` control ticks — tens of milliseconds —
        and would fire on every startup.)
        """
        if runway > 0.0:
            self._starving_since = None
            return
        if self._starving_since is None:
            self._starving_since = now
        elif now - self._starving_since > self.limits.max_starvation:
            self.trip(f"no trajectory for {now - self._starving_since:.1f}s — planner is dead")

    def check_tracking(self, errors: dict) -> None:
        worst = max(errors.values()) if errors else 0.0
        self._strike("track", worst > self.limits.max_tracking_error_m,
                     f"IK tracking error {worst*1000:.0f} mm — target unreachable "
                     "or the frames are wrong")


def sanity_check_action(action) -> bool:
    """Cheap guard on a chunk row before it ever reaches the IK.

    A mis-normalized or corrupted chunk usually shows up as non-finite values or a
    delta that is metres long. Catching it here means it never becomes a pose.
    """
    a = np.asarray(action)
    if a.shape != (layout.DIM,) or not np.all(np.isfinite(a)):
        return False
    for h in layout.HANDS:
        trans = a[layout.EEF[h]][:3]
        if np.linalg.norm(trans) > 1.5:   # a 1.5 m single-tick delta is nonsense
            return False
    return True
