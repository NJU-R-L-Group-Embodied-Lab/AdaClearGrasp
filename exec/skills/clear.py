# exec/skills/clear.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import math
import numpy as np
import torch

from exec.skills.base import BaseSkill, SkillResult, StepEvent
from exec.skills.ee_pose import EEPoseSkill


Mode = Literal["push", "pull"]
Side = Literal["left", "center", "right", "middle"]


@dataclass
class PushPullConfig:
    # Action normalization
    pos_upper: float = 0.03
    rot_upper: float = 0.05

    # Yaw alignment
    yaw_tol_deg: float = 5.0
    yaw_step: float = 0.08
    yaw_max_steps: int = 200

    # Motion
    step_m: float = 0.01         # commanded delta per env step (m)
    max_steps: int = 500         # safety budget (counts motion steps only)

    # Stuck detection (tcp not moving)
    stuck_window: int = 10
    stuck_min_moved: float = 1e-4  # m


def _get_tcp_R_w(env) -> torch.Tensor:
    tcp = env.unwrapped.agent.tcp
    R = tcp.pose.to_transformation_matrix()[..., :3, :3]
    if R.ndim == 2:
        R = R.unsqueeze(0)
    return R.to(device=env.unwrapped.device, dtype=torch.float32)


def _get_tcp_p_w(env) -> torch.Tensor:
    tcp = env.unwrapped.agent.tcp
    p = tcp.pose.p
    if not isinstance(p, torch.Tensor):
        p = torch.as_tensor(p, device=env.unwrapped.device, dtype=torch.float32)
    p = p[..., :3]
    if p.ndim == 1:
        p = p.unsqueeze(0)
    if p.shape[0] != 1:
        p = p[:1]
    return p


class _SimplePushPull(BaseSkill):
    def __init__(self, env, cfg: Optional[PushPullConfig] = None, *, trace_maxlen: int = 10):
        super().__init__(
            env,
            trace_maxlen=trace_maxlen,
            eps_stuck=float((cfg or PushPullConfig()).stuck_min_moved),
        )
        self.cfg = cfg or PushPullConfig()

    # -----------------------------
    # Action helpers
    # -----------------------------
    def _pack_action(self, dpos_w: np.ndarray, dyaw: float = 0.0) -> np.ndarray:
        a = np.zeros(18, dtype=np.float32)
        a[0:3] = np.clip(dpos_w / float(self.cfg.pos_upper), -1.0, 1.0).astype(np.float32)
        a[5] = float(np.clip(dyaw / float(self.cfg.rot_upper), -1.0, 1.0))  # rot_z
        return a

    def _tcp_xy(self) -> np.ndarray:
        p = _get_tcp_p_w(self.env)[0]
        return np.array([float(p[0].item()), float(p[1].item())], dtype=np.float32)

    def _msg_xy(self) -> str:
        xy = self._tcp_xy()
        return f"x={float(xy[0]):.4f}, y={float(xy[1]):.4f}"

    def _call_error(self, e: Exception) -> SkillResult:
        msg = ""
        try:
            msg = self._msg_xy()
        except Exception:
            msg = ""
        return self._result(
            ok=False,
            error_code="call_error",
            message=msg,
            advice=f"{type(e).__name__}: {e}",
        )

    # -----------------------------
    # Yaw alignment (closed-loop on finger heading only)
    # -----------------------------
    def _current_finger_dir_xy(self) -> np.ndarray:
        R = _get_tcp_R_w(self.env)[0]  # (3,3)
        zw = R[:, 2]                   # local +Z in world

        vx = float(zw[0].item())
        vy = float(zw[1].item())
        n = math.sqrt(vx * vx + vy * vy)
        if n < 1e-9:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([vx / n, vy / n], dtype=np.float32)

    def _yaw_align(self, yaw_t: float, *, render: bool, verbose: bool) -> tuple[bool, int]:
        tol = float(self.cfg.yaw_tol_deg) * math.pi / 180.0
        steps = 0

        tx = math.cos(float(yaw_t))
        ty = math.sin(float(yaw_t))
        t = np.array([tx, ty], dtype=np.float32)

        while steps < int(self.cfg.yaw_max_steps):
            u = self._current_finger_dir_xy()

            dot = float(np.clip(u[0] * t[0] + u[1] * t[1], -1.0, 1.0))
            cross = float(u[0] * t[1] - u[1] * t[0])
            err = float(math.atan2(cross, dot))  # signed angle

            if abs(err) <= tol:
                return True, steps

            dyaw = -float(np.clip(err, -float(self.cfg.yaw_step), float(self.cfg.yaw_step)))

            self._step(
                self._pack_action(np.array([0.0, 0.0, 0.0], dtype=np.float32), dyaw=dyaw),
                i=steps,
                render=render,
            )

            self._trace.add(StepEvent(i=int(steps), moved=abs(dyaw), event=None))

            if verbose and (steps < 10 or (steps + 1) % 30 == 0):
                print(
                    f"[YawAlign] step={steps+1} "
                    f"err_deg={err*180/math.pi:+.1f} "
                    f"dyaw_deg={dyaw*180/math.pi:+.1f}"
                )

            steps += 1

        return False, steps

    # -----------------------------
    # Direction logic
    # -----------------------------
    def _side_to_angle(self, side: Side) -> float:
        if side == "middle":
            side = "center"
        if side == "left":
            return math.pi / 6.0
        if side == "center":
            return 0.0
        if side == "right":
            return -math.pi / 6.0
        raise RuntimeError(f"Invalid side: {side}")

    def _compute_yaws_and_motion(self, mode: Mode, side: Side) -> tuple[float, np.ndarray, float]:
        ang = float(self._side_to_angle(side))

        # Final motion yaw in world:
        # push: +x rotated by ang
        # pull: -x rotated by -ang => yaw = pi - ang
        yaw_motion = ang if mode == "push" else (math.pi - ang)

        motion = np.array([math.cos(yaw_motion), math.sin(yaw_motion)], dtype=np.float32)
        nrm = float(np.linalg.norm(motion))
        if nrm < 1e-9:
            raise RuntimeError("motion direction too small")
        motion = (motion / np.float32(nrm)).astype(np.float32)

        # Finger yaw:
        # push: finger == motion
        # pull: finger opposite motion (so retreat along -finger == motion)
        yaw_finger = yaw_motion if mode == "push" else (yaw_motion + math.pi)
        while yaw_finger > math.pi:
            yaw_finger -= 2.0 * math.pi
        while yaw_finger < -math.pi:
            yaw_finger += 2.0 * math.pi

        return float(yaw_finger), motion, float(yaw_motion)

    # -----------------------------
    # Main routine
    # -----------------------------
    def _run(self, mode: Mode, side: Side, dist_m: float, *, render: bool, verbose: bool) -> SkillResult:
        try:
            self.reset_trace()

            if float(dist_m) <= 0.0:
                raise RuntimeError("dist_m must be > 0")

            yaw_finger, motion_dir, yaw_motion = self._compute_yaws_and_motion(mode=mode, side=side)

            side_norm = "center" if side == "middle" else side

            if verbose:
                print(
                    f"[{mode}] side={side_norm} "
                    f"yaw_motion_deg={yaw_motion*180/math.pi:.1f} "
                    f"yaw_finger_deg={yaw_finger*180/math.pi:.1f} "
                    f"motion_dir={motion_dir.tolist()} dist_m={float(dist_m):.3f}"
                )

            # 1) Yaw align
            ok_yaw, _yaw_steps = self._yaw_align(yaw_finger, render=render, verbose=verbose)

            # 1.5) Set finger pose to push or pull
            pose_skill = EEPoseSkill(self.env)
            pose_res = pose_skill.set_pose(mode, render=render, verbose=verbose)

            # 2) Move until moved_total reaches dist_m (step budget is just safety)
            step_m = float(self.cfg.step_m)
            max_steps = int(self.cfg.max_steps)

            moved_total = 0.0
            moved_hist: list[float] = []

            for i in range(max_steps):
                remain = float(dist_m) - moved_total
                if remain <= 0.0:
                    return self._result(
                        ok=True,
                        error_code="none",
                        message=self._msg_xy(),
                        advice="",
                    )

                this_step = step_m if remain > step_m else remain

                dpos = np.array(
                    [float(motion_dir[0]) * this_step, float(motion_dir[1]) * this_step, 0.0],
                    dtype=np.float32,
                )

                pa = self._tcp_xy()
                self._step(self._pack_action(dpos, dyaw=0.0), i=i, render=render)
                pb = self._tcp_xy()

                moved = float(np.linalg.norm(pb - pa))
                moved_total += moved

                moved_hist.append(moved)
                if len(moved_hist) > int(self.cfg.stuck_window):
                    moved_hist = moved_hist[-int(self.cfg.stuck_window):]

                self._trace.add(StepEvent(i=int(i), moved=moved, event=None))

                if verbose and (i < 10 or (i + 1) % 30 == 0):
                    print(
                        f"[{mode}] step={i+1}/{max_steps} "
                        f"moved={moved:.4f} moved_total={moved_total:.4f}/{float(dist_m):.4f}"
                    )

                # Stuck criterion (same as ee_move): window-full and all small moves.
                if len(moved_hist) == int(self.cfg.stuck_window) and max(moved_hist) < float(self.cfg.stuck_min_moved):
                    return self._result(
                        ok=False,
                        error_code="stuck",
                        message=self._msg_xy(),
                        advice="Consider changing direction or using a different motion.",
                    )

            return self._result(
                ok=False,
                error_code="max_step",
                message=self._msg_xy(),
                advice="Consider reducing step size or retrying the action.",
            )
        except Exception as e:
            return self._call_error(e)


class PushSkill(_SimplePushPull):
    def push(self, side: Side, dist_m: float, *, render: bool = False, verbose: bool = False) -> SkillResult:
        return self._run(mode="push", side=side, dist_m=float(dist_m), render=render, verbose=verbose)


class PullSkill(_SimplePushPull):
    def pull(self, side: Side, dist_m: float, *, render: bool = False, verbose: bool = False) -> SkillResult:
        return self._run(mode="pull", side=side, dist_m=float(dist_m), render=render, verbose=verbose)
