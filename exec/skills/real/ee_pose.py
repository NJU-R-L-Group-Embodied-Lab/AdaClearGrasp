# exec/skills/real/ee_pose.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List

import time
import numpy as np

from env.real.xhand_client import (
    wait_for_state,
    get_joint_names_order,
    get_hand_positions_by_name,
    spin_once,
    set_joint_positions_direct,
)

from exec.skills.base import BaseSkill, SkillResult

Pose = Literal["flat", "work"]


@dataclass
class HandPoseConfig:
    # 线性插值步数与步间隔
    interp_steps: int = 60
    step_wait_sec: float = 0.02

    # Preset poses (12-DOF)
    q_flat: List[float] = field(
        default_factory=lambda: [
            0.0, 0.0, 0.0,   # thumb
            0.0, 0.0, 0.0,   # index
            0.0, 0.0,        # middle
            0.0, 0.0,        # ring
            0.0, 0.0,        # pinky
        ]
    )

    q_work: List[float] = field(
        default_factory=lambda: [
            1.655, -0.5, 0.5,   # thumb
            0.0, 1.2, 0.2,   # index
            1.2, 0.2,        # middle
            1.2, 0.2,        # ring
            1.2, 0.2,        # pinky
        ]
    )

    verbose: bool = False


class EEPoseSkill(BaseSkill):
    DOF = 12

    def __init__(self, env=None, cfg: Optional[HandPoseConfig] = None, *, trace_maxlen: int = 10):
        super().__init__(env, trace_maxlen=trace_maxlen)
        self.cfg = cfg or HandPoseConfig()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_target(self, pose: Pose) -> np.ndarray:
        if pose == "flat":
            q = self.cfg.q_flat
        elif pose == "work":
            q = self.cfg.q_work
        else:
            raise ValueError(pose)

        if len(q) != self.DOF:
            raise RuntimeError(f"Target pose must be {self.DOF}-dim, got {len(q)}")
        return np.array([float(x) for x in q], dtype=np.float32)

    def _read_current_vec(self, names: List[str]) -> np.ndarray:
        now = get_hand_positions_by_name()
        cur: List[float] = []
        for n in names:
            if n not in now:
                raise RuntimeError(f"Joint '{n}' not in current state.")
            cur.append(float(now[n]))
        if len(cur) != self.DOF:
            raise RuntimeError(f"Current vec dim mismatch, got {len(cur)}")
        return np.array(cur, dtype=np.float32)

    def _spin_for(self, sec: float) -> None:
        t_end = time.time() + float(sec)
        while time.time() < t_end:
            spin_once(0.02)

    # -----------------------------
    # Public API
    # -----------------------------
    def set_pose(self, pose: Pose, *, render: bool = False, verbose: bool = False) -> SkillResult:
        _ = render
        self.reset_trace()

        try:
            wait_for_state(timeout_sec=2.0)

            names = get_joint_names_order()
            if len(names) != self.DOF:
                raise RuntimeError(f"joint_names must be {self.DOF}, got {len(names)}")

            q_target = self._get_target(pose)
            q0 = self._read_current_vec(names)

            steps = int(self.cfg.interp_steps)
            dt = float(self.cfg.step_wait_sec)

            if self.cfg.verbose or verbose:
                print("[EEPoseSkillRealDirect] pose   =", pose)
                print("[EEPoseSkillRealDirect] names  =", names)
                print("[EEPoseSkillRealDirect] q0     =", q0.tolist())
                print("[EEPoseSkillRealDirect] target =", q_target.tolist())
                print("[EEPoseSkillRealDirect] steps  =", steps, "dt=", dt)

            if steps <= 0:
                tgt_by_joint = {names[i]: float(q_target[i]) for i in range(self.DOF)}
                set_joint_positions_direct(tgt_by_joint, fill_missing="state")
                self._spin_for(max(0.06, dt))
                return self._result(ok=True, error_code="none", message="OK", advice="")

            for k in range(steps + 1):
                t = float(k) / float(steps)
                q = q0 * (1.0 - t) + q_target * t
                cmd_by_joint = {names[i]: float(q[i]) for i in range(self.DOF)}
                set_joint_positions_direct(cmd_by_joint, fill_missing="state")
                self._spin_for(max(0.06, dt))

            return self._result(ok=True, error_code="none", message="OK", advice="")

        except Exception as e:
            # 按你的要求：依然 ok=True
            return self._result(
                ok=True,
                error_code="none",
                message="OK",
                advice=f"{type(e).__name__}: {e}",
            )
