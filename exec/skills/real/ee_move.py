from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from env.real.xarm_client import get_xarm
from env.real.camera import get_camera


@dataclass
class EEMoveCfg:
    # heights in mm
    z_lift: float = 400.0
    z_work: float = 130.0

    speed: float = 50.0
    mvacc: float = 200.0

    wait: bool = True
    is_radian: bool = True
    lift_step_mm: float = 30.0


class EEMoveSkill:
    def __init__(self, env=None, cfg: Optional[EEMoveCfg] = None):
        self.env = env
        self.cfg = cfg or EEMoveCfg()

    def _check(self, code: int, name: str) -> None:
        if int(code) != 0:
            raise RuntimeError(f"{name} failed, code={int(code)}")

    def _get_state(self, arm) -> int:
        code, state = arm.get_state()
        self._check(code, "arm.get_state")
        return int(state)

    def _get_err_warn(self, arm) -> Tuple[int, int]:
        code, data = arm.get_err_warn_code()
        self._check(code, "arm.get_err_warn_code")
        err, warn = data
        return int(err), int(warn)

    def _ready(self, arm) -> None:
        self._check(arm.motion_enable(enable=True), "arm.motion_enable")
        self._check(arm.set_mode(0), "arm.set_mode(0)")

        st = self._get_state(arm)
        if st == 4:  # STOP
            err, warn = self._get_err_warn(arm)
            print(f"[EEMove] xarm STOP before move, err={err}, warn={warn}. cleaning...")
            arm.clean_error()
            arm.clean_warn()

        self._check(arm.set_state(0), "arm.set_state(0)")

    def _get_tcp(self, arm) -> Tuple[float, float, float, float, float, float]:
        code, pos = arm.get_position(is_radian=self.cfg.is_radian)
        self._check(code, "arm.get_position")
        if not isinstance(pos, (list, tuple)) or len(pos) != 6:
            raise RuntimeError(f"arm.get_position returned invalid pos: {pos}")
        return tuple(float(v) for v in pos)

    def _set_tcp(
        self,
        arm,
        x: float,
        y: float,
        z: float,
        r: float,
        p: float,
        yw: float,
        *,
        speed: Optional[float] = None,
        mvacc: Optional[float] = None,
        wait: Optional[bool] = None,
    ) -> None:
        sp = float(self.cfg.speed if speed is None else speed)
        acc = float(self.cfg.mvacc if mvacc is None else mvacc)
        wt = bool(self.cfg.wait if wait is None else wait)

        code = arm.set_position(
            x=float(x), y=float(y), z=float(z),
            roll=float(r), pitch=float(p), yaw=float(yw),
            speed=sp, mvacc=acc, wait=wt, is_radian=bool(self.cfg.is_radian),
        )

        if int(code) == 0:
            return

        # On failure, print state & err/warn for diagnosis
        try:
            st = self._get_state(arm)
            err, warn = self._get_err_warn(arm)
            print(f"[EEMove] set_position failed code={int(code)}; state={st}; err={err}; warn={warn}")
        except Exception:
            pass

        raise RuntimeError(f"arm.set_position failed, code={int(code)}")

    def move_xy(self, x: float, y: float) -> None:
        """Move only x/y (mm), keep z & orientation."""
        arm = get_xarm()
        self._ready(arm)

        x0, y0, z0, r0, p0, yw0 = self._get_tcp(arm)
        self._set_tcp(arm, float(x), float(y), z0, r0, p0, yw0)

    def lift(self) -> None:
        """Lift z to cfg.z_lift in one shot."""
        arm = get_xarm()
        self._ready(arm)

        x0, y0, z0, r0, p0, yw0 = self._get_tcp(arm)
        self._set_tcp(arm, x0, y0, float(self.cfg.z_lift), r0, p0, yw0)


    def lower(self) -> None:
        """Lower z to cfg.z_work."""
        arm = get_xarm()
        self._ready(arm)

        x0, y0, z0, r0, p0, yw0 = self._get_tcp(arm)
        self._set_tcp(arm, x0, y0, float(self.cfg.z_work), r0, p0, yw0)


    def move_to(self, obj_name: str) -> None:
        """
        Move TCP x/y to the object's (x,y) from camera (base frame, mm),
        keep current z and orientation.
        """
        if not obj_name or not isinstance(obj_name, str):
            raise ValueError("move_to requires obj_name (non-empty string)")

        cam = get_camera(self.env)
        tx, ty = cam.get_object_xy(obj_name, frame="base")
        arm = get_xarm()
        self._ready(arm)

        x0, y0, z0, r0, p0, yw0 = self._get_tcp(arm)
        self._set_tcp(arm, float(tx), float(ty), z0, r0, p0, yw0)
