from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List,Sequence

import math
import numpy as np
import rclpy
from rclpy.node import Node
import time
from xhand_control_interfaces.msg import XHandCommand, XHandStateArray



finger_sensor_locations = {
    "thumb": "thumb",
    "index": "index",
    "middle": "middle",
    "ring": "ring",
    "pinky": "pinky",
}

finger_joints = {
    "thumb":  ["thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2"],
    "index":  ["index_bend_joint", "index_joint1", "index_joint2"],
    "middle": ["mid_joint1", "mid_joint2"],
    "ring":   ["ring_joint1", "ring_joint2"],
    "pinky":  ["pinky_joint1", "pinky_joint2"],
}


@dataclass(frozen=True)
class XHandClientConfig:
    hand_id: int = 1

    cmd_topic: str = "/xhand_control/xhand_command"
    state_topic: str = "/xhand_control/xhand_state"
    joint_names: List[str] = field(
        default_factory=lambda: [
            "thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2",
            "index_bend_joint", "index_joint1", "index_joint2",
            "mid_joint1", "mid_joint2",
            "ring_joint1", "ring_joint2",
            "pinky_joint1", "pinky_joint2",
        ]
    )
    mode: int = 3
    kp: float = 100.0
    ki: float = 0.0
    kd: float = 0.0
    effort_limit: float = 350.0


_CFG = XHandClientConfig()

_NODE: Optional[Node] = None
_PUB = None
_SUB = None

_HAVE_STATE: bool = False
_STATE_JOINT_NAMES: List[str] = []
_STATE_POS: Optional[np.ndarray] = None
_NAME_TO_INDEX: Dict[str, int] = {}

# sensor forces: location -> (fx, fy, fz)
_STATE_FORCES: Dict[str, Tuple[float, float, float]] = {}

# last sent command cache
_LAST_SENT: Optional[Dict[str, float]] = None


def _ensure_rclpy():
    if not rclpy.ok():
        rclpy.init()


def _state_cb(msg: XHandStateArray):
    global _HAVE_STATE, _STATE_JOINT_NAMES, _STATE_POS, _NAME_TO_INDEX, _STATE_FORCES

    idx = None
    for i, hid in enumerate(msg.hand_id):
        if int(hid) == int(_CFG.hand_id):
            idx = i
            break
    if idx is None:
        return

    hand_state = msg.hand_states[idx]
    sensor_state = msg.sensor_states[idx]

    names = list(hand_state.name)
    pos = np.array(hand_state.position, dtype=np.float32)

    _STATE_JOINT_NAMES = names
    _STATE_POS = pos
    _NAME_TO_INDEX = {n: i for i, n in enumerate(names)}

    forces: Dict[str, Tuple[float, float, float]] = {}
    for fs in sensor_state.finger_sensor_states:
        loc = str(fs.location)
        fx = float(fs.calc_force.x)
        fy = float(fs.calc_force.y)
        fz = float(fs.calc_force.z)
        forces[loc] = (fx, fy, fz)
    _STATE_FORCES = forces

    _HAVE_STATE = True


def get_xhand_node() -> Node:
    global _NODE, _PUB, _SUB

    if _NODE is not None:
        return _NODE

    _ensure_rclpy()
    node = Node("xhand_client_singleton")

    _PUB = node.create_publisher(XHandCommand, _CFG.cmd_topic, 10)
    _SUB = node.create_subscription(XHandStateArray, _CFG.state_topic, _state_cb, 10)

    _NODE = node
    return _NODE


def spin_once(timeout_sec: float = 0.05) -> None:
    node = get_xhand_node()
    rclpy.spin_once(node, timeout_sec=float(timeout_sec))


def wait_for_state(timeout_sec: float = 2.0) -> None:
    import time

    t0 = time.time()
    while True:
        if _HAVE_STATE:
            return
        if (time.time() - t0) > float(timeout_sec):
            raise RuntimeError(
                f"wait_for_state timeout after {timeout_sec}s. "
                f"Check topic='{_CFG.state_topic}' and hand_id={_CFG.hand_id}."
            )
        spin_once(0.05)


def get_joint_names_order() -> List[str]:
    return list(_CFG.joint_names)


def get_hand_positions_by_name() -> Dict[str, float]:
    if not _HAVE_STATE or _STATE_POS is None:
        raise RuntimeError("No xhand state yet. Call wait_for_state() first.")
    out: Dict[str, float] = {}
    for n, i in _NAME_TO_INDEX.items():
        out[n] = float(_STATE_POS[i])
    return out


def get_finger_forces() -> Dict[str, Tuple[float, float, float]]:
    if not _HAVE_STATE:
        raise RuntimeError("No xhand state yet. Call wait_for_state() first.")
    return dict(_STATE_FORCES)


def get_finger_force_mag(location: str) -> float:
    if not _HAVE_STATE:
        raise RuntimeError("No xhand state yet. Call wait_for_state() first.")
    loc = str(location)
    if loc not in _STATE_FORCES:
        return 0.0
    fx, fy, fz = _STATE_FORCES[loc]
    return float(math.sqrt(fx * fx + fy * fy + fz * fz))


def set_last_sent_target(target_by_name: Dict[str, float]) -> None:
    names = get_joint_names_order()
    if len(names) != 12:
        raise RuntimeError(f"xhand joint_names order must be 12, got {len(names)}")
    for n in names:
        if n not in target_by_name:
            raise RuntimeError(f"set_last_sent_target missing joint '{n}'")
    global _LAST_SENT
    _LAST_SENT = {n: float(target_by_name[n]) for n in names}


def get_last_sent_target() -> Dict[str, float]:
    if _LAST_SENT is None:
        raise RuntimeError("_LAST_SENT is None. Call set_last_sent_target() or init_relative_target_from_state() first.")
    return dict(_LAST_SENT)


def init_relative_target_from_state() -> Dict[str, float]:
    wait_for_state(timeout_sec=2.0)
    cur = get_hand_positions_by_name()
    names = get_joint_names_order()
    if len(names) != 12:
        raise RuntimeError(f"xhand joint_names order must be 12, got {len(names)}")
    tgt = {n: float(cur[n]) for n in names}
    set_last_sent_target(tgt)
    return tgt


def publish_joint_positions(target_by_name: Dict[str, float]) -> None:
    if _PUB is None:
        get_xhand_node()

    if not _HAVE_STATE or _STATE_POS is None:
        raise RuntimeError("No xhand state yet. Call wait_for_state() before publish.")

    # 校验：确保 joint 在 state 里出现过
    for n in _CFG.joint_names:
        if n not in _NAME_TO_INDEX:
            raise RuntimeError(f"Joint '{n}' not found in XHandState. Please align joint_names.")

    msg = XHandCommand()
    msg.hand_id = int(_CFG.hand_id)
    msg.mode = int(_CFG.mode)

    for n in _CFG.joint_names:
        if n not in target_by_name:
            raise RuntimeError(f"Target missing joint '{n}'")
        v = float(target_by_name[n])

        msg.name.append(n)
        msg.position.append(v)
        msg.kp.append(float(_CFG.kp))
        msg.ki.append(float(_CFG.ki))
        msg.kd.append(float(_CFG.kd))
        msg.effort_limit.append(float(_CFG.effort_limit))

    _PUB.publish(msg)

    set_last_sent_target(target_by_name)


def publish_joint_positions_relative(delta_by_name: Dict[str, float]) -> Dict[str, float]:
    base = get_last_sent_target()
    names = get_joint_names_order()
    if len(names) != 12:
        raise RuntimeError(f"xhand joint_names order must be 12, got {len(names)}")

    out: Dict[str, float] = {}
    for n in names:
        dv = float(delta_by_name.get(n, 0.0))
        out[n] = float(base[n]) + dv

    publish_joint_positions(out)
    return out


def _force_mag_from_tuple(v: tuple[float, float, float]) -> float:
    fx, fy, fz = float(v[0]), float(v[1]), float(v[2])
    return float(math.sqrt(fx * fx + fy * fy + fz * fz))


def safe_relative_move_with_force_guard(
    delta_by_joint: Dict[str, float],
    *,
    finger_order: Sequence[str] = ("thumb", "index", "middle", "ring", "pinky"),
    force_thresh: float = 1.0,
    spin_before_send_sec: float = 0.10,
) -> List[bool]:
    wait_for_state(timeout_sec=2.0)

    t_end = time.time() + float(spin_before_send_sec)
    while time.time() < t_end:
        spin_once(0.02)

    forces = get_finger_forces()  # location -> (fx,fy,fz)

    reached: Dict[str, bool] = {}
    for f in finger_order:
        loc = finger_sensor_locations.get(str(f), None)
        if loc is None or loc not in forces:
            reached[str(f)] = False
            continue
        reached[str(f)] = _force_mag_from_tuple(forces[loc]) >= float(force_thresh)

    guarded_delta: Dict[str, float] = {}
    for jn, dv in delta_by_joint.items():
        guarded_delta[str(jn)] = float(dv)

    for f in finger_order:
        if not reached[str(f)]:
            continue
        for jn in finger_joints.get(str(f), []):
            if jn in guarded_delta:
                guarded_delta[jn] = 0.0

    names = get_joint_names_order()
    delta12 = {n: 0.0 for n in names}
    for jn, dv in guarded_delta.items():
        if jn not in delta12:
            raise RuntimeError(f"delta_by_joint contains unknown joint '{jn}' (not in get_joint_names_order())")
        delta12[jn] = float(dv)

    publish_joint_positions_relative(delta12)


    return [bool(reached[str(f)]) for f in finger_order]


def set_joint_positions_direct(
    target_by_joint: Dict[str, float],
    *,
    fill_missing: str = "last",  # "last" or "state"
) -> Dict[str, float]:
    wait_for_state(timeout_sec=2.0)
    names = get_joint_names_order()
    if len(names) != 12:
        raise RuntimeError(f"xhand joint_names order must be 12, got {len(names)}")

    if fill_missing == "last":
        try:
            base = get_last_sent_target()
        except Exception:
            base = init_relative_target_from_state()
    elif fill_missing == "state":
        cur = get_hand_positions_by_name()
        base = {n: float(cur[n]) for n in names}
    else:
        raise ValueError("fill_missing must be 'last' or 'state'")

    out = dict(base)
    for jn, v in target_by_joint.items():
        jn = str(jn)
        if jn not in out:
            raise RuntimeError(f"Unknown joint '{jn}' (not in get_joint_names_order())")
        out[jn] = float(v)

    publish_joint_positions(out) 
    return out


def set_joint_positions_direct_with_force_guard(
    target_by_joint: Dict[str, float],
    *,
    finger_order: Sequence[str] = ("thumb", "index", "middle", "ring", "pinky"),
    force_thresh: float = 1.0,
    spin_before_send_sec: float = 0.10,
    fill_missing: str = "last",  
) -> List[bool]:
    wait_for_state(timeout_sec=2.0)

    t_end = time.time() + float(spin_before_send_sec)
    while time.time() < t_end:
        spin_once(0.02)

    names = get_joint_names_order()
    if len(names) != 12:
        raise RuntimeError(f"xhand joint_names order must be 12, got {len(names)}")

    if fill_missing == "last":
        try:
            base = get_last_sent_target()
        except Exception:
            base = init_relative_target_from_state()
    elif fill_missing == "state":
        cur = get_hand_positions_by_name()
        base = {n: float(cur[n]) for n in names}
    else:
        raise ValueError("fill_missing must be 'last' or 'state'")

    forces = get_finger_forces()  # location -> (fx,fy,fz)
    reached: Dict[str, bool] = {}
    for f in finger_order:
        loc = finger_sensor_locations.get(str(f), None)
        if loc is None or loc not in forces:
            reached[str(f)] = False
            continue
        reached[str(f)] = _force_mag_from_tuple(forces[loc]) >= float(force_thresh)

    out = dict(base)
    for jn, v in target_by_joint.items():
        jn = str(jn)
        if jn not in out:
            raise RuntimeError(f"Unknown joint '{jn}' (not in get_joint_names_order())")
        out[jn] = float(v)

    for f in finger_order:
        if not reached[str(f)]:
            continue
        for jn in finger_joints.get(str(f), []):
            if jn in out:
                out[jn] = float(base[jn])

    publish_joint_positions(out)
    return [bool(reached[str(f)]) for f in finger_order]

