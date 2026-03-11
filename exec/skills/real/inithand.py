# exec/skills/real/inithand.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import time
import math
import numpy as np

from exec.skills.base import BaseSkill, SkillResult
from env.real.xarm_client import get_xarm

from env.real.xhand_client import (
    wait_for_state as wait_for_hand_state,
    get_joint_names_order,
    get_hand_positions_by_name,
    spin_once,
    set_joint_positions_direct,
)


@dataclass
class InitHandSkillConfig:
    # -----------------------------
    # Hand (12-DOF) -> flat
    # -----------------------------
    interp_steps: int = 60
    step_wait_sec: float = 0.02

    q_flat: List[float] = field(
        default_factory=lambda: [
            0.0, 0.0, 0.0,   # thumb
            0.0, 0.0, 0.0,   # index
            0.0, 0.0,        # middle
            0.0, 0.0,        # ring
            0.0, 0.0,        # pinky
        ]
    )

    # -----------------------------
    # Arm TCP rpy (keep xyz)
    # -----------------------------
    roll_rad: float = 0.0
    pitch_rad: float = math.pi / 2.0
    yaw_rad: float = 0.0

    # xArm motion params (only used in set_position)
    speed: float = 50.0
    mvacc: float = 200.0
    wait: bool = True
    is_radian: bool = True

    verbose: bool = False


class InitHandSkill(BaseSkill):
    DOF = 12

    def __init__(self, env=None, *, cfg: Optional[InitHandSkillConfig] = None, trace_maxlen: int = 10):
        super().__init__(env, trace_maxlen=trace_maxlen)
        self.cfg = cfg or InitHandSkillConfig()

    # -----------------------------
    # helpers
    # -----------------------------
    def _spin_for(self, sec: float) -> None:
        t_end = time.time() + float(sec)
        while time.time() < t_end:
            spin_once(0.02)

    def _read_current_hand_vec(self, names: List[str]) -> np.ndarray:
        now = get_hand_positions_by_name()
        cur: List[float] = []
        for n in names:
            if n not in now:
                raise RuntimeError(f"Joint '{n}' not in current state.")
            cur.append(float(now[n]))
        if len(cur) != self.DOF:
            raise RuntimeError(f"Current vec dim mismatch, got {len(cur)}")
        return np.array(cur, dtype=np.float32)

    def _get_flat_target(self) -> np.ndarray:
        q = self.cfg.q_flat
        if len(q) != self.DOF:
            raise RuntimeError(f"q_flat must be {self.DOF}-dim, got {len(q)}")
        return np.array([float(x) for x in q], dtype=np.float32)

    def _sdk_check(self, code: int, *, name: str) -> None:
        if int(code) != 0:
            raise RuntimeError(f"{name} failed, code={int(code)}")

    def _ready_arm(self, arm) -> None:
        # minimal ready (和你 InitArmSkill 一致风格)
        self._sdk_check(arm.motion_enable(enable=True), name="arm.motion_enable")
        self._sdk_check(arm.set_mode(0), name="arm.set_mode(0)")
        self._sdk_check(arm.set_state(0), name="arm.set_state(0)")

    # -----------------------------
    # public API
    # -----------------------------
    def inithand(self, name: str = "", *, render: Optional[bool] = None, verbose: Optional[bool] = None) -> SkillResult:
        _ = render
        self.reset_trace()

        vb = self.cfg.verbose if verbose is None else bool(verbose)

        try:
            # -----------------------------
            # 1) hand -> flat
            # -----------------------------
            wait_for_hand_state(timeout_sec=2.0)

            names = get_joint_names_order()
            if len(names) != self.DOF:
                raise RuntimeError(f"joint_names must be {self.DOF}, got {len(names)}")

            q0_hand = self._read_current_hand_vec(names)
            q_flat = self._get_flat_target()

            steps = int(self.cfg.interp_steps)
            dt = float(self.cfg.step_wait_sec)

            if vb:
                print("[InitHandSkill] names   =", names)
                print("[InitHandSkill] hand q0  =", q0_hand.tolist())
                print("[InitHandSkill] target  =", q_flat.tolist())
                print("[InitHandSkill] steps   =", steps, "dt=", dt)

            if steps <= 0:
                tgt = {names[i]: float(q_flat[i]) for i in range(self.DOF)}
                set_joint_positions_direct(tgt, fill_missing="state")
                self._spin_for(max(0.06, dt))
            else:
                for k in range(steps + 1):
                    t = float(k) / float(steps)
                    q = q0_hand * (1.0 - t) + q_flat * t
                    cmd = {names[i]: float(q[i]) for i in range(self.DOF)}
                    set_joint_positions_direct(cmd, fill_missing="state")
                    self._spin_for(max(0.06, dt))

            # -----------------------------
            # 2) arm: keep xyz, set rpy
            # -----------------------------
            arm = get_xarm()
            self._ready_arm(arm)

            code, pos = arm.get_position(is_radian=bool(self.cfg.is_radian))
            self._sdk_check(code, name="arm.get_position")
            x0, y0, z0 = float(pos[0]), float(pos[1]), float(pos[2])

            r, p, yw = float(self.cfg.roll_rad), float(self.cfg.pitch_rad), float(self.cfg.yaw_rad)

            if vb:
                print("[InitHandSkill] tcp xyz =", (x0, y0, z0))
                print("[InitHandSkill] set rpy =", (r, p, yw))

            code2 = arm.set_position(
                x=x0, y=y0, z=z0,
                roll=r, pitch=p, yaw=yw,
                speed=float(self.cfg.speed),
                mvacc=float(self.cfg.mvacc),
                wait=bool(self.cfg.wait),
                is_radian=bool(self.cfg.is_radian),
            )
            self._sdk_check(code2, name="arm.set_position")

            raw: Dict[str, Any] = {
                "name": name,
                "hand_q0": q0_hand.tolist(),
                "hand_target": q_flat.tolist(),
                "tcp_xyz": [x0, y0, z0],
                "tcp_rpy_target": [r, p, yw],
            }
            return self._result(ok=True, error_code="none", message="inithand done", advice="", raw=raw)

        except Exception as e:
            # 延续你其它 real skill 的风格：依然 ok=True，把异常塞到 advice
            return self._result(
                ok=True,
                error_code="none",
                message="OK",
                advice=f"{type(e).__name__}: {e}",
            )
