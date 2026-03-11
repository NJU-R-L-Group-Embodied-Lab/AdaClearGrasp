# exec/skills/real/clear.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np

from env.real.xarm_client import get_xarm
from exec.skills.base import BaseSkill, SkillResult
from exec.skills.real.ee_pose import EEPoseSkill

Mode = Literal["push", "pull"]
Side = Literal["left", "center", "right", "middle"]


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class PushPullConfig:
    # xArm set_position motion params
    speed: float = 200.0
    mvacc: float = 2000.0
    wait: bool = True
    is_radian: bool = True

    # workspace safety (mm)
    clamp_xy: bool = True
    safe_x_min: float = 315.0
    safe_x_max: float = 915.0
    safe_y_min: float = -300.0
    safe_y_max: float = 300.0

    #   local Z -> world +X
    #   local X -> world -Z
    #   local Y -> world +Y
    #   roll = 0, pitch = +pi/2, yaw = 0
    roll0: float = 0.0
    pitch0: float = math.pi / 2.0

    side_angle: float = math.pi / 6.0
    yaw_offset: float = 0.0


class _SimplePushPull(BaseSkill):
    """
    Real push/pull (one-shot):
      - Force attitude: roll=0, pitch=+pi/2
      - Yaw sets finger direction in XY plane (local Z)
      - push: move along finger direction
      - pull: move opposite finger direction (retreat), finger yaw unchanged
    """

    def __init__(self, env=None, cfg: Optional[PushPullConfig] = None, *, trace_maxlen: int = 10):
        super().__init__(env, trace_maxlen=trace_maxlen)
        self.cfg = cfg or PushPullConfig()

    # -----------------------------
    # Low-level arm helpers
    # -----------------------------
    def _sdk_check(self, code: int, *, name: str) -> None:
        if int(code) != 0:
            raise RuntimeError(f"{name} failed, code={int(code)}")

    def _ensure_motion_ready(self, arm) -> None:
        self._sdk_check(arm.motion_enable(enable=True), name="arm.motion_enable")
        if hasattr(arm, "set_mode"):
            self._sdk_check(arm.set_mode(0), name="arm.set_mode(0)")
        if hasattr(arm, "set_state"):
            self._sdk_check(arm.set_state(0), name="arm.set_state(0)")

    def _get_tcp6(self, arm) -> Tuple[float, float, float, float, float, float]:
        code, pos = arm.get_position(is_radian=bool(self.cfg.is_radian))
        self._sdk_check(code, name="arm.get_position")
        if not isinstance(pos, (list, tuple)) or len(pos) != 6:
            raise RuntimeError(f"arm.get_position returned invalid pos: {pos}")
        return tuple(float(v) for v in pos)  # x,y,z,roll,pitch,yaw

    def _set_tcp6(self, arm, x: float, y: float, z: float, r: float, p: float, yw: float) -> None:
        code = arm.set_position(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(r),
            pitch=float(p),
            yaw=float(yw),
            speed=float(self.cfg.speed),
            mvacc=float(self.cfg.mvacc),
            wait=bool(self.cfg.wait),
            is_radian=bool(self.cfg.is_radian),
        )
        self._sdk_check(code, name="arm.set_position")

    def _side_to_yaw(self, side: Side) -> float:
        if side == "middle":
            side = "center"
        if side == "left":
            return +float(self.cfg.side_angle)
        if side == "center":
            return 0.0
        if side == "right":
            return -float(self.cfg.side_angle)
        raise RuntimeError(f"Invalid side: {side}")

    def _compute(self, mode: Mode, side: Side) -> tuple[float, np.ndarray]:
        """
        yaw_finger: finger direction (local Z) in world XY plane
        motion_dir:
          push => +finger_dir
          pull => -finger_dir  (retreat along finger direction)
        """
        yaw_finger = _wrap_to_pi(self._side_to_yaw(side) + float(self.cfg.yaw_offset))
        finger_dir = np.array([math.cos(yaw_finger), math.sin(yaw_finger)], dtype=np.float32)

        if mode == "push":
            motion_dir = finger_dir
        elif mode == "pull":
            motion_dir = -finger_dir
        else:
            raise RuntimeError(f"Invalid mode: {mode}")

        return float(yaw_finger), motion_dir

    def _run(self, mode: Mode, side: Side, dist_m: float, *, render: bool, verbose: bool) -> SkillResult:
        _ = render
        try:
            self.reset_trace()

            dist_m = float(dist_m)
            if dist_m <= 0.0:
                raise RuntimeError("dist_m must be > 0")

            arm = get_xarm()
            self._ensure_motion_ready(arm)

            _ = EEPoseSkill(self.env).set_pose("work", render=False, verbose=verbose)

            yaw_finger, motion_dir = self._compute(mode=mode, side=side)

            x, y, z, _r, _p, _yw = self._get_tcp6(arm)

            r = float(self.cfg.roll0)
            p = float(self.cfg.pitch0)
            yw = float(yaw_finger)

            if verbose:
                side_norm = "center" if side == "middle" else side
                print(
                    f"[real_{mode}] side={side_norm} dist_m={dist_m:.3f} "
                    f"target_rpy_deg=({r*180/math.pi:.1f},{p*180/math.pi:.1f},{yw*180/math.pi:.1f}) "
                    f"motion_dir={motion_dir.tolist()} yaw_offset_deg={self.cfg.yaw_offset*180/math.pi:.1f}"
                )

            self._set_tcp6(arm, x, y, z, r, p, yw)

            dist_mm = dist_m * 1000.0
            x2 = float(x) + float(motion_dir[0]) * dist_mm
            y2 = float(y) + float(motion_dir[1]) * dist_mm

            if self.cfg.clamp_xy:
                x2 = _clamp(x2, float(self.cfg.safe_x_min), float(self.cfg.safe_x_max))
                y2 = _clamp(y2, float(self.cfg.safe_y_min), float(self.cfg.safe_y_max))

            self._set_tcp6(arm, x2, y2, z, r, p, yw)

            return self._result(ok=True, error_code="none", message="OK", advice="")

        except Exception as e:
            return self._result(ok=False, error_code="call_error", message="", advice=f"{type(e).__name__}: {e}")


class PushSkill(_SimplePushPull):
    def push(self, side: Side, dist_m: float, *, render: bool = False, verbose: bool = False) -> SkillResult:
        return self._run(mode="push", side=side, dist_m=float(dist_m), render=render, verbose=verbose)


class PullSkill(_SimplePushPull):
    def pull(self, side: Side, dist_m: float, *, render: bool = False, verbose: bool = False) -> SkillResult:
        return self._run(mode="pull", side=side, dist_m=float(dist_m), render=render, verbose=verbose)
