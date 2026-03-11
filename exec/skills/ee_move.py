# exec/skills/ee_move.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch
import math

from exec.skills.base import BaseSkill, SkillResult, StepEvent


def get_tcp_p_w(env) -> torch.Tensor:
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


def _get_pose_p_xy(obj: Any, device: torch.device) -> tuple[float, float]:
    if not hasattr(obj, "pose"):
        raise RuntimeError("Object must have attribute `.pose`")
    pose = obj.pose
    if not hasattr(pose, "p"):
        raise RuntimeError("Object pose must have attribute `.p`")

    p = pose.p
    if not isinstance(p, torch.Tensor):
        p = torch.as_tensor(p, device=device, dtype=torch.float32)
    p = p.reshape(-1)[:3]
    return float(p[0].item()), float(p[1].item())


@dataclass
class EEMoveSimpleConfig:
    # Controller normalization
    pos_upper: float = 0.03

    # Step size
    step_xy: float = 0.004  # m per step
    step_z: float = 0.004

    # Hard-coded heights (world Z)
    z_lift: float = 0.40  # safe travel height
    z_work: float = 0.13  # push / grasp height

    eps: float = 0.003
    max_steps: int = 500

    action_dim: int = 18

    # Minimal stuck detector
    stuck_window: int = 10
    stuck_min_moved: float = 1e-4  # m

    # Object moved detector (via env.evaluate()["height"])
    moved_height_eps: float = 1e-3


class EEMoveSkill(BaseSkill):
    def __init__(self, env, cfg: Optional[EEMoveSimpleConfig] = None, *, trace_maxlen: int = 10):
        super().__init__(
            env,
            trace_maxlen=trace_maxlen,
            eps_stuck=float((cfg or EEMoveSimpleConfig()).stuck_min_moved),
        )
        self.cfg = cfg or EEMoveSimpleConfig()

    # ------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------
    def _pack_action(self, dpos_w: np.ndarray) -> np.ndarray:
        a = np.zeros(self.cfg.action_dim, dtype=np.float32)
        a[0:3] = np.clip(dpos_w / float(self.cfg.pos_upper), -1.0, 1.0).astype(np.float32)
        return a

    def _tcp_np(self) -> np.ndarray:
        p = get_tcp_p_w(self.env)[0].detach().cpu().numpy()
        return np.asarray(p, dtype=np.float32).reshape(3)

    def _msg_xy(self) -> str:
        p = self._tcp_np()
        return f"x={float(p[0]):.4f}, y={float(p[1]):.4f}"

    def _get_target_height(self) -> float:
        ev = self.env.evaluate()
        if not isinstance(ev, dict) or "height" not in ev:
            raise RuntimeError("env.evaluate() must return dict with key 'height'")
        h = ev["height"]
        if isinstance(h, torch.Tensor):
            return float(h.reshape(-1)[0].item())
        return float(h)

    def _object_moved(self, h0: float) -> bool:
        h1 = self._get_target_height()
        return abs(h1 - h0) > float(self.cfg.moved_height_eps)

    def _resolve_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            name = obj
            u = self.env.unwrapped

            # Common places (single actor)
            for attr in ("target_object", "all_objects"):
                if hasattr(u, attr):
                    a = getattr(u, attr)
                    if getattr(a, "name", None) == name:
                        return a

            # target_actors list
            if hasattr(u, "target_actors"):
                for a in getattr(u, "target_actors"):
                    if getattr(a, "name", None) == name:
                        return a

            # clutter_objects merged
            if hasattr(u, "clutter_objects"):
                for a in getattr(u, "clutter_objects"):
                    if getattr(a, "name", None) == name:
                        return a

            # clutter_actors_per_item nested
            if hasattr(u, "clutter_actors_per_item"):
                for per_item in getattr(u, "clutter_actors_per_item"):
                    for a in per_item:
                        if getattr(a, "name", None) == name:
                            return a

            raise RuntimeError(f"Cannot resolve object name: {name}")

        return obj

    def _call_error(self, e: Exception) -> SkillResult:
        xy = ""
        try:
            xy = self._msg_xy()
        except Exception:
            xy = ""
        return self._result(
            ok=False,
            error_code="call_error",
            message=xy,
            advice=f"{type(e).__name__}: {e}",
        )

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------
    def lift(self, *, render: bool = False, verbose: bool = False) -> SkillResult:
        try:
            return self._move_to_z(self.cfg.z_lift, name="lift", render=render, verbose=verbose)
        except Exception as e:
            return self._call_error(e)

    def lower(self, *, render: bool = False, verbose: bool = False) -> SkillResult:
        try:
            return self._lower_to_z_or_stuck(z_min=self.cfg.z_work, name="lower", render=render, verbose=verbose)
        except Exception as e:
            return self._call_error(e)

    def move_xy(self, dx: float, dy: float, *, render: bool = False, verbose: bool = False) -> SkillResult:
        try:
            return self._move_xy_closed_loop(dx=dx, dy=dy, name="move_xy", render=render, verbose=verbose)
        except Exception as e:
            return self._call_error(e)

    def move_to_xy(self, x: float, y: float, *, render: bool = False, verbose: bool = False) -> SkillResult:
        try:
            p0 = self._tcp_np()
            dx = float(x) - float(p0[0])
            dy = float(y) - float(p0[1])
            return self._move_xy_closed_loop(dx=dx, dy=dy, name="move_to_xy", render=render, verbose=verbose)
        except Exception as e:
            return self._call_error(e)

    def move_to(self, obj: Any, *, render: bool = False, verbose: bool = False) -> SkillResult:
        try:
            self.reset_trace()
            h0 = self._get_target_height()

            o = self._resolve_obj(obj)
            x_t, y_t = _get_pose_p_xy(o, device=self.env.unwrapped.device)

            steps = 0
            moved_hist: list[float] = []

            while steps < int(self.cfg.max_steps):
                p = get_tcp_p_w(self.env)  # (1,3)
                px = float(p[0, 0].item())
                py = float(p[0, 1].item())

                ex = float(x_t - px)
                ey = float(y_t - py)

                if abs(ex) <= float(self.cfg.eps) and abs(ey) <= float(self.cfg.eps):
                    return self._result(
                        ok=True,
                        error_code="none",
                        message=self._msg_xy(),
                        advice="",
                    )

                sx = float(np.clip(ex, -float(self.cfg.step_xy), float(self.cfg.step_xy)))
                sy = float(np.clip(ey, -float(self.cfg.step_xy), float(self.cfg.step_xy)))
                sx = float(np.clip(sx, -float(self.cfg.pos_upper), float(self.cfg.pos_upper)))
                sy = float(np.clip(sy, -float(self.cfg.pos_upper), float(self.cfg.pos_upper)))

                dpos = np.array([sx, sy, 0.0], dtype=np.float32)

                pa = self._tcp_np()
                _, _, _, _, _info = self._step(self._pack_action(dpos), i=steps, render=render)
                pb = self._tcp_np()

                moved = float(np.linalg.norm((pb - pa)[:2]))
                moved_hist.append(moved)
                if len(moved_hist) > int(self.cfg.stuck_window):
                    moved_hist = moved_hist[-int(self.cfg.stuck_window):]

                self._trace.add(StepEvent(i=int(steps), moved=moved, event=None))

                if len(moved_hist) == int(self.cfg.stuck_window) and max(moved_hist) < float(self.cfg.stuck_min_moved):
                    return self._result(
                        ok=False,
                        error_code="stuck",
                        message=self._msg_xy(),
                        advice="Consider changing direction or using a different motion.",
                    )

                steps += 1
                if verbose and steps % 50 == 0:
                    err = math.sqrt(ex * ex + ey * ey)
                    print(f"[MoveTo] step={steps} err_xy={err:.4f} moved={moved:.4f}")

            return self._result(
                ok=False,
                error_code="max_step",
                message=self._msg_xy(),
                advice="Consider reducing step size or retrying the action.",
            )
        except Exception as e:
            return self._call_error(e)

    # ------------------------------------------------
    # Internal
    # ------------------------------------------------
    def _move_to_z(self, z_target: float, *, name: str, render: bool, verbose: bool) -> SkillResult:
        self.reset_trace()
        h0 = self._get_target_height()

        steps = 0
        moved_hist: list[float] = []

        while steps < int(self.cfg.max_steps):
            p = self._tcp_np()
            dz = float(z_target - float(p[2]))

            if abs(dz) <= float(self.cfg.eps):
                return self._result(
                    ok=True,
                    error_code="none",
                    message=self._msg_xy(),
                    advice="",
                )

            step = float(np.clip(dz, -float(self.cfg.step_z), float(self.cfg.step_z)))
            dpos = np.array([0.0, 0.0, step], dtype=np.float32)

            pa = self._tcp_np()
            _, _, _, _, _info = self._step(self._pack_action(dpos), i=steps, render=render)
            pb = self._tcp_np()

            moved = float(abs(pb[2] - pa[2]))
            moved_hist.append(moved)
            if len(moved_hist) > int(self.cfg.stuck_window):
                moved_hist = moved_hist[-int(self.cfg.stuck_window):]

            self._trace.add(StepEvent(i=int(steps), moved=moved, event=None))

            if len(moved_hist) == int(self.cfg.stuck_window) and max(moved_hist) < float(self.cfg.stuck_min_moved):
                return self._result(
                    ok=False,
                    error_code="stuck",
                    message=self._msg_xy(),
                    advice="Consider changing direction or using a different motion.",
                )

            steps += 1
            if verbose and steps % 50 == 0:
                print(f"[MoveZ] step={steps} z_err={dz:+.4f} moved={moved:.4f}")

        return self._result(
            ok=False,
            error_code="max_step",
            message=self._msg_xy(),
            advice="Consider reducing step size or retrying the action.",
        )

    def _move_xy_closed_loop(self, *, dx: float, dy: float, name: str, render: bool, verbose: bool) -> SkillResult:
        self.reset_trace()
        h0 = self._get_target_height()

        p0 = self._tcp_np()
        desired = np.array([float(dx), float(dy), 0.0], dtype=np.float32)

        steps = 0
        moved_hist: list[float] = []

        while steps < int(self.cfg.max_steps):
            p_now = self._tcp_np()
            achieved = (p_now - p0).astype(np.float32)
            err = desired - achieved
            pos_err = float(np.linalg.norm(err[:2]))

            if pos_err <= float(self.cfg.eps):
                return self._result(
                    ok=True,
                    error_code="none",
                    message=self._msg_xy(),
                    advice="",
                )

            step_xy = err[:2].copy()
            norm = float(np.linalg.norm(step_xy))
            if norm > float(self.cfg.step_xy):
                step_xy *= float(self.cfg.step_xy) / norm

            dpos = np.array([float(step_xy[0]), float(step_xy[1]), 0.0], dtype=np.float32)

            pa = self._tcp_np()
            _, _, _, _, _info = self._step(self._pack_action(dpos), i=steps, render=render)
            pb = self._tcp_np()

            moved = float(np.linalg.norm((pb - pa)[:2]))
            moved_hist.append(moved)
            if len(moved_hist) > int(self.cfg.stuck_window):
                moved_hist = moved_hist[-int(self.cfg.stuck_window):]

            self._trace.add(StepEvent(i=int(steps), moved=moved, event=None))

            if len(moved_hist) == int(self.cfg.stuck_window) and max(moved_hist) < float(self.cfg.stuck_min_moved):
                return self._result(
                    ok=False,
                    error_code="stuck",
                    message=self._msg_xy(),
                    advice="Consider changing direction or using a different motion.",
                )

            steps += 1
            if verbose and steps % 50 == 0:
                print(f"[{name}] step={steps} pos_err_xy={pos_err:.4f} moved={moved:.4f}")

        return self._result(
            ok=False,
            error_code="max_step",
            message=self._msg_xy(),
            advice="Consider reducing step size or retrying the action.",
        )
    

    def _lower_to_z_or_stuck(self, *, z_min: float, name: str, render: bool, verbose: bool) -> SkillResult:
        self.reset_trace()
        h0 = self._get_target_height()

        steps = 0

        while steps < int(self.cfg.max_steps):
            p = self._tcp_np()
            pz = float(p[2])
            dz_to_min = float(z_min) - pz  # <= 0 when above z_min

            # Already at / below minimum height (within eps)
            if abs(dz_to_min) <= float(self.cfg.eps) or pz <= float(z_min):
                p_end = self._tcp_np()
                msg = f"{self._msg_xy()}, z={float(p_end[2]):.4f}"
                return self._result(
                    ok=True,
                    error_code="none",
                    message=msg,
                    advice="",
                )

            # Move down, but do NOT go below z_min
            step = max(dz_to_min, -float(self.cfg.step_z))  # negative; clamp magnitude and prevent overshoot
            dpos = np.array([0.0, 0.0, float(step)], dtype=np.float32)


            pa = self._tcp_np()
            _, _, _, _, _info = self._step(self._pack_action(dpos), i=steps, render=render)
            pb = self._tcp_np()

            dz_moved = float(pb[2] - pa[2])      
            moved = float(abs(dz_moved))        

            self._trace.add(StepEvent(i=int(steps), moved=moved, event=None))

            if moved < float(self.cfg.stuck_min_moved):
                p_end = self._tcp_np()
                msg = f"{self._msg_xy()}, z={float(p_end[2]):.4f}"
                return self._result(
                    ok=True,
                    error_code="none",
                    message=msg,
                    advice="",
                )

            steps += 1
            if verbose and steps % 50 == 0:
                p_now = self._tcp_np()
                print(f"[{name}] step={steps} z={float(p_now[2]):.4f} dz_moved={dz_moved:+.6f}")

        p_end = self._tcp_np()
        msg = f"{self._msg_xy()}, z={float(p_end[2]):.4f}"
        return self._result(
            ok=True,
            error_code="none",
            message=msg,
            advice="",
        )

    def move_to_xyz(self, x: float, y: float, z: float, *, render: bool = False, verbose: bool = False) -> SkillResult:
        try:
            return self._move_xyz_closed_loop(
                x=float(x),
                y=float(y),
                z=float(z),
                name="move_to_xyz",
                render=render,
                verbose=verbose,
            )
        except Exception as e:
            return self._call_error(e)
        

    def _move_xyz_closed_loop(self, *, x: float, y: float, z: float, name: str, render: bool, verbose: bool) -> SkillResult:
        self.reset_trace()
        h0 = self._get_target_height()

        steps = 0
        moved_hist: list[float] = []

        while steps < int(self.cfg.max_steps):
            p = get_tcp_p_w(self.env)  # (1,3)
            px = float(p[0, 0].item())
            py = float(p[0, 1].item())
            pz = float(p[0, 2].item())

            ex = float(x - px)
            ey = float(y - py)
            ez = float(z - pz)

            if abs(ex) <= float(self.cfg.eps) and abs(ey) <= float(self.cfg.eps) and abs(ez) <= float(self.cfg.eps):
                p_end = self._tcp_np()
                msg = f"x={float(p_end[0]):.4f}, y={float(p_end[1]):.4f}, z={float(p_end[2]):.4f}"
                return self._result(ok=True, error_code="none", message=msg, advice="")

            sx = float(np.clip(ex, -float(self.cfg.step_xy), float(self.cfg.step_xy)))
            sy = float(np.clip(ey, -float(self.cfg.step_xy), float(self.cfg.step_xy)))
            sz = float(np.clip(ez, -float(self.cfg.step_z), float(self.cfg.step_z)))

            sx = float(np.clip(sx, -float(self.cfg.pos_upper), float(self.cfg.pos_upper)))
            sy = float(np.clip(sy, -float(self.cfg.pos_upper), float(self.cfg.pos_upper)))
            sz = float(np.clip(sz, -float(self.cfg.pos_upper), float(self.cfg.pos_upper)))

            dpos = np.array([sx, sy, sz], dtype=np.float32)

            pa = self._tcp_np()
            _, _, _, _, _info = self._step(self._pack_action(dpos), i=steps, render=render)
            pb = self._tcp_np()

            moved = float(np.linalg.norm(pb - pa))
            moved_hist.append(moved)
            if len(moved_hist) > int(self.cfg.stuck_window):
                moved_hist = moved_hist[-int(self.cfg.stuck_window):]

            self._trace.add(StepEvent(i=int(steps), moved=moved, event=None))

            if len(moved_hist) == int(self.cfg.stuck_window) and max(moved_hist) < float(self.cfg.stuck_min_moved):
                msg = f"x={float(pb[0]):.4f}, y={float(pb[1]):.4f}, z={float(pb[2]):.4f}"
                return self._result(
                    ok=False,
                    error_code="stuck",
                    message=msg,
                    advice="Consider changing waypoint spacing or using a safer height.",
                )

            steps += 1
            if verbose and steps % 50 == 0:
                err = math.sqrt(ex * ex + ey * ey + ez * ez)
                print(f"[{name}] step={steps} err_xyz={err:.4f} moved={moved:.4f}")

        p_end = self._tcp_np()
        msg = f"x={float(p_end[0]):.4f}, y={float(p_end[1]):.4f}, z={float(p_end[2]):.4f}"
        return self._result(
            ok=False,
            error_code="max_step",
            message=msg,
            advice="Consider increasing max_steps or reducing waypoint distances.",
        )
