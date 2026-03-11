# exec/skills/real/initarm.py
from __future__ import annotations

from dataclasses import dataclass,field
from typing import Optional, Dict, Any, List

from env.real.xarm_client import get_xarm
from exec.skills.base import BaseSkill, SkillResult


@dataclass
class InitArmSkillConfig:
    arm_init_7: List[float] = field(
        default_factory=lambda: [
            -0.77920043, 0.4921763, 0.74077785,
            1.2483665, 2.8245757, 0.6868453, -0.050859887
        ]
    )

    speed: float = 0.30
    mvacc: float = 1.0
    wait: bool = True
    is_radian: bool = True
    verbose: bool = False


class InitArmSkill(BaseSkill):
    def __init__(self, env=None, *, cfg: Optional[InitArmSkillConfig] = None, trace_maxlen: int = 10):
        super().__init__(env, trace_maxlen=trace_maxlen)
        self.cfg = cfg or InitArmSkillConfig()

    def _sdk_check(self, code: int, *, name: str) -> None:
        if int(code) != 0:
            raise RuntimeError(f"{name} failed, code={int(code)}")

    def initarm(self, name: str = "", *, render: Optional[bool] = None, verbose: Optional[bool] = None) -> SkillResult:
        self.reset_trace()
        self.cfg.verbose = self.cfg.verbose if verbose is None else bool(verbose)

        arm = get_xarm()

        target = list(self.cfg.arm_init_7)
        if len(target) != 7:
            raise RuntimeError(f"arm_init_7 must be 7 dims, got {len(target)}")

        code, q0 = arm.get_servo_angle(is_radian=bool(self.cfg.is_radian))  # :contentReference[oaicite:5]{index=5}
        self._sdk_check(code, name="arm.get_servo_angle")

        if self.cfg.verbose:
            print("[InitArmSkill] q0 =", q0)
            print("[InitArmSkill] target =", target)

        self._sdk_check(arm.motion_enable(enable=True), name="arm.motion_enable")
        self._sdk_check(arm.set_mode(0), name="arm.set_mode(0)")
        self._sdk_check(arm.set_state(0), name="arm.set_state(0)")

        code = arm.set_servo_angle(
            angle=target,
            speed=float(self.cfg.speed),
            mvacc=float(self.cfg.mvacc),
            wait=bool(self.cfg.wait),
            is_radian=bool(self.cfg.is_radian),
        ) 
        self._sdk_check(code, name="arm.set_servo_angle")

        raw: Dict[str, Any] = {"q0": q0, "target": target}
        return self._result(
            ok=True,
            error_code="none",
            message="initarm done",
            advice="",
        )
