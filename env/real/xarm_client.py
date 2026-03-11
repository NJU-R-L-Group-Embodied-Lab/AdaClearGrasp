from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from xarm.wrapper import XArmAPI


@dataclass(frozen=True)
class XArmClientConfig:
    ip: str = "192.168.1.196"
    is_radian: bool = True
    do_not_open: bool = False

    # TCP offset: [x, y, z, roll, pitch, yaw]
    tcp_offset: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 85.0, 0.0, 0.0, 0.0)


_CFG = XArmClientConfig()
_ARM: Optional[XArmAPI] = None


def _check(code: int, name: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{name} failed, code={int(code)}")


def get_xarm() -> XArmAPI:
    global _ARM
    if _ARM is not None:
        return _ARM

    arm = XArmAPI(_CFG.ip, is_radian=_CFG.is_radian, do_not_open=_CFG.do_not_open)

    if _CFG.do_not_open:
        _check(arm.connect(), "arm.connect")

    # basic ready
    _check(arm.motion_enable(enable=True), "arm.motion_enable")
    _check(arm.set_mode(0), "arm.set_mode(0)")
    _check(arm.set_state(0), "arm.set_state(0)")

    # set tcp offset (NOTE: may enter state=5; restore state=0 after)
    _check(arm.set_tcp_offset(list(_CFG.tcp_offset), is_radian=_CFG.is_radian), "arm.set_tcp_offset")
    _check(arm.set_state(0), "arm.set_state(0) after set_tcp_offset")

    _ARM = arm
    return _ARM
