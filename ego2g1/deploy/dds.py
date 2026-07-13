"""The only module that talks to the robot. Everything else is pure Python.

Wire contract (G1-D, direct rt/lowcmd — no whole-body control):

  rt/lowstate   robot -> us,  LowState_ (unitree_hg), 35 motor slots, ~500 Hz
  rt/lowcmd     us -> robot,  LowCmd_   (unitree_hg), 35 motor slots + CRC
  rt/brainco/{left,right}/cmd    us -> hand bridge, MotorCmds_   (unitree_go)
  rt/brainco/{left,right}/state  bridge -> us,      MotorStates_ (unitree_go)

The firmware runs `tau = kp*(q* - q) + kd*(dq* - dq) + tau_ff` on the LAST
message it received. It does NOT interpolate and it does NOT time out: if we stop
publishing, it holds the last setpoint forever. So:

  * "stop publishing" is NOT a stop. The only real stop is to publish a damping
    command (kp=0, small kd) — see `damp()`. That is the e-stop.
  * A step change in q_target is a torque spike proportional to the jump. The
    30 Hz action stream must be interpolated up to 500 Hz before it gets here;
    that is TrajectoryBuffer's job, not the firmware's.
  * Every message needs a CRC or the firmware silently drops it.

Joint indices (G1_29_JointIndex): legs 0-11, waist 12-14, arms 15-28, unused
29-34. We command the 14 arms and pin the 3 waist joints at 0 (training froze
them there). The legs are held at whatever they measured at connect time —
harmless on a fixed base, and irrelevant to our pose math since waist==0 means
pelvis->flange depends only on the arm joints.
"""

import dataclasses
import threading
import time

import numpy as np

from ego2g1.common import layout

# --- joint index map (mirrors unitree_deploy.robot_devices.robot_devices_index) ---
N_MOTORS = 35
WAIST_IDX = [12, 13, 14]           # yaw, roll, pitch
ARM_IDX = {
    "left": [15, 16, 17, 18, 19, 20, 21],   # shoulder p/r/y, elbow, wrist r/p/y
    "right": [22, 23, 24, 25, 26, 27, 28],
}
# Flat, in the order layout.ARM_JOINTS / DualArmIK use: left 7 then right 7.
ARM_IDX_FLAT = ARM_IDX["left"] + ARM_IDX["right"]
LEG_IDX = list(range(0, 12))

# Brainco: 6 motors per hand. MotorCmd_.q is a normalized [0, 1] position
# (0 = open, 1 = closed) — same units the retargeting produces, so hand commands
# pass through untransformed.
#
# Index order is [Thumb, ThumbAux, Index, Middle, Ring, Pinky] against our
# HAND_MOTOR_ORDER of [thumb_flex, thumb_rot, index, middle, ring, pinky]. The 1:1
# mapping is PLAUSIBLE but UNVERIFIED on hardware — see check.py hand-sweep.
BRAINCO_N = 6


@dataclasses.dataclass(frozen=True)
class Gains:
    """Vendor defaults (unitree_deploy G1ArmConfig). Shoulders/elbows are the
    'weak' motors and get the lower gain; wrists lower still."""
    kp_shoulder_elbow: float = 80.0
    kd_shoulder_elbow: float = 3.0
    kp_wrist: float = 40.0
    kd_wrist: float = 1.5
    kp_waist: float = 300.0
    kd_waist: float = 3.0
    kp_leg: float = 300.0
    kd_leg: float = 3.0
    # e-stop: zero stiffness, pure damping -> the arm goes limp but does not
    # freefall. Never 0/0, which would drop the arm.
    kd_damp: float = 2.0


_WRIST = {19, 20, 21, 26, 27, 28}


class G1DDS:
    """LowCmd publisher + LowState/Brainco subscribers. Owns the wire, nothing else."""

    def __init__(self, *, network_interface: str | None = None, domain: int = 0,
                 gains: Gains = Gains(), enable_hands: bool = True):
        self.gains = gains
        self.enable_hands = enable_hands
        self._domain = domain
        self._iface = network_interface

        self._lock = threading.Lock()          # guards the state buffers
        self._cmd_lock = threading.Lock()      # guards the shared LowCmd _msg + CRC
        self._lowstate = None          # raw last LowState_
        self._lowstate_t = 0.0
        self._hand_state = {h: np.zeros(BRAINCO_N, np.float32) for h in layout.HANDS}
        self._hand_state_t = {h: 0.0 for h in layout.HANDS}

        self._msg = None
        self._crc = None
        self._pub = None
        self._sub = None
        self._hand_pub = {}
        self._hand_sub = {}
        self._hand_msg = {}
        self._connected = False
        self._estopped = False

    # --- lifecycle ----------------------------------------------------------

    def connect(self, *, timeout: float = 5.0) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        # No interface arg when unset: joins an existing DDS domain instead of
        # fighting a running ROS 2 for it.
        if self._iface:
            ChannelFactoryInitialize(self._domain, self._iface)
        else:
            ChannelFactoryInitialize(self._domain)

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_lowstate, 10)
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

        if self.enable_hands:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
            for h in layout.HANDS:
                self._hand_pub[h] = ChannelPublisher(f"rt/brainco/{h}/cmd", MotorCmds_)
                self._hand_pub[h].Init()
                self._hand_sub[h] = ChannelSubscriber(f"rt/brainco/{h}/state", MotorStates_)
                self._hand_sub[h].Init(self._make_hand_cb(h), 10)
                msg = MotorCmds_()
                msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(BRAINCO_N)]
                for i in range(BRAINCO_N):
                    msg.cmds[i].q = 0.0
                    msg.cmds[i].dq = 1.0   # vendor uses dq as a speed field here
                self._hand_msg[h] = msg

        # Wait for the first state before we are allowed to command anything: the
        # LowCmd needs mode_machine from it, and we must know where the robot IS
        # before we tell it where to go.
        t0 = time.monotonic()
        while self.lowstate_age() > timeout:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    "no rt/lowstate within %.1fs — check the Ethernet link, the DDS "
                    "domain, and that nothing else owns the bus." % timeout
                )
            time.sleep(0.01)

        self._crc = CRC()
        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._msg.mode_pr = 0
        self._msg.mode_machine = self._raw().mode_machine

        # Seed every slot at its MEASURED position with its gains, so the first
        # published message is a no-op hold rather than a jump.
        q = self.motor_q()
        for i in range(N_MOTORS):
            self._msg.motor_cmd[i].mode = 1
            self._msg.motor_cmd[i].q = float(q[i])
            self._msg.motor_cmd[i].dq = 0.0
            self._msg.motor_cmd[i].tau = 0.0
            kp, kd = self._gains_for(i)
            self._msg.motor_cmd[i].kp = kp
            self._msg.motor_cmd[i].kd = kd

        self._connected = True

    def _gains_for(self, i: int) -> tuple[float, float]:
        g = self.gains
        if i in _WRIST:
            return g.kp_wrist, g.kd_wrist
        if i in ARM_IDX_FLAT:
            return g.kp_shoulder_elbow, g.kd_shoulder_elbow
        if i in WAIST_IDX:
            return g.kp_waist, g.kd_waist
        return g.kp_leg, g.kd_leg

    # --- subscribers --------------------------------------------------------

    def _on_lowstate(self, msg) -> None:
        with self._lock:
            self._lowstate = msg
            self._lowstate_t = time.monotonic()

    def _make_hand_cb(self, hand: str):
        def cb(msg):
            try:
                q = np.array([msg.states[i].q for i in range(BRAINCO_N)], dtype=np.float32)
            except Exception:
                return
            with self._lock:
                self._hand_state[hand] = q
                self._hand_state_t[hand] = time.monotonic()
        return cb

    def _raw(self):
        with self._lock:
            return self._lowstate

    def lowstate_age(self) -> float:
        """Seconds since the last rt/lowstate. The watchdog reads this — a stale
        state means we are commanding a robot we cannot see."""
        with self._lock:
            return float("inf") if self._lowstate is None else time.monotonic() - self._lowstate_t

    def hand_state_age(self, hand: str) -> float:
        with self._lock:
            t = self._hand_state_t[hand]
            return float("inf") if t == 0.0 else time.monotonic() - t

    def motor_q(self) -> np.ndarray:
        """(35,) measured positions."""
        s = self._raw()
        return np.array([s.motor_state[i].q for i in range(N_MOTORS)], dtype=np.float64)

    def motor_dq(self) -> np.ndarray:
        s = self._raw()
        return np.array([s.motor_state[i].dq for i in range(N_MOTORS)], dtype=np.float64)

    def arm_q(self) -> np.ndarray:
        """(14,) measured arm joints, in DualArmIK order (left 7, right 7)."""
        s = self._raw()
        return np.array([s.motor_state[i].q for i in ARM_IDX_FLAT], dtype=np.float64)

    def arm_dq(self) -> np.ndarray:
        s = self._raw()
        return np.array([s.motor_state[i].dq for i in ARM_IDX_FLAT], dtype=np.float64)

    def hand_q(self, hand: str) -> np.ndarray:
        """Measured Brainco motors. NOT for the model — the training state's hand
        block is the last COMMAND (see kinematics.state). Log this, don't feed it."""
        with self._lock:
            return self._hand_state[hand].copy()

    # --- publishers ---------------------------------------------------------

    def send_arm(self, arm_q14, *, waist=None) -> None:
        """Publish one LowCmd: 14 arm targets + the waist pinned (default 0 rad,
        which is where training froze it). Legs keep their seeded hold."""
        if not self._connected:
            raise RuntimeError("send_arm before connect()")
        if self._estopped:
            return  # latched: only damp() publishes from here on

        arm_q14 = np.asarray(arm_q14, dtype=np.float64).reshape(-1)
        if arm_q14.shape != (layout.ARM_DOF,):
            raise ValueError(f"expected ({layout.ARM_DOF},), got {arm_q14.shape}")

        waist = np.zeros(3) if waist is None else np.asarray(waist, dtype=np.float64)
        # The mutate + CRC + Write must be atomic wrt damp(), which touches the same
        # _msg from another thread. Interleaved, the CRC could be computed over one
        # field set and the message published carrying a different one — the
        # firmware silently drops any message whose CRC does not match.
        with self._cmd_lock:
            if self._estopped:
                return
            for k, i in enumerate(ARM_IDX_FLAT):
                self._msg.motor_cmd[i].q = float(arm_q14[k])
            for k, i in enumerate(WAIST_IDX):
                self._msg.motor_cmd[i].q = float(waist[k])
            self._msg.crc = self._crc.Crc(self._msg)
            self._pub.Write(self._msg)

    def send_hands(self, cmds: dict) -> None:
        """cmds: {hand: (6,)} absolute motor commands in [0, 1].

        Brainco's MotorCmd_.q is normalized [0, 1] (0=open, 1=closed), which is what
        the retargeting emits — no conversion. What is still UNVERIFIED is the motor
        ORDER: run `check.py hand-sweep` and watch which finger actually moves.
        """
        if not self.enable_hands or self._estopped:
            return
        for h in layout.HANDS:
            q = np.clip(np.asarray(cmds[h], dtype=np.float64), 0.0, 1.0)
            msg = self._hand_msg[h]
            for i in range(BRAINCO_N):
                msg.cmds[i].q = float(q[i])
            self._hand_pub[h].Write(msg)

    def damp(self) -> None:
        """THE E-STOP. Zero stiffness, pure damping, on every joint.

        Latches: once damped, send_arm/send_hands become no-ops, so a control
        thread that has not noticed yet cannot re-energise the arm. Publishing
        (rather than merely stopping) is essential — the firmware holds the last
        command forever, so silence would leave the robot stiff at its last target.
        """
        self._estopped = True   # set first: latches send_arm/send_hands off
        if not self._connected:
            return
        q = self.motor_q()
        with self._cmd_lock:
            for i in range(N_MOTORS):
                self._msg.motor_cmd[i].q = float(q[i])
                self._msg.motor_cmd[i].dq = 0.0
                self._msg.motor_cmd[i].tau = 0.0
                self._msg.motor_cmd[i].kp = 0.0
                self._msg.motor_cmd[i].kd = self.gains.kd_damp
            self._msg.crc = self._crc.Crc(self._msg)
            for _ in range(5):  # a few times, in case one is dropped
                self._pub.Write(self._msg)
            time.sleep(0.002)

    @property
    def estopped(self) -> bool:
        return self._estopped
