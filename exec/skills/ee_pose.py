# exec/skills/ee_pose.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Any

import numpy as np
import torch

from exec.skills.base import BaseSkill, SkillResult, StepEvent

Pose = Literal["flat", "work"]


@dataclass
class FingerPoseConfig:
    max_steps: int = 300
    hand_delta_upper: float = 0.2
    kp: float = 0.08
    u_cap: float = 0.3
    tol: float = 2e-3

    stuck_window: int = 10
    stuck_min_moved: float = 0.005

    q_flat: Optional[torch.Tensor] = field(
        default_factory=lambda: torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
    )
    q_push_pull: Optional[torch.Tensor] = field(
        default_factory=lambda: torch.tensor(
            [0.0, 0.0, 1.4, 1.4, 1.4, 0.7, 1.4, 0.2, 0.2, 0.2, 0.2, 0.2],
            dtype=torch.float32,
        )
    )

    moved_height_eps: float = 1e-3


class EEPoseSkill(BaseSkill):
    HAND_DOF = 12

    # qpos[7:19] index k is mainly affected by action[QPOS7_19_TO_ACTION[k]]
    QPOS7_19_TO_ACTION = [6, 9, 12, 14, 16, 7, 10, 13, 15, 17, 8, 11]

    def __init__(self, env, cfg: Optional[FingerPoseConfig] = None, *, trace_maxlen: int = 10):
        self.cfg = cfg or FingerPoseConfig()
        super().__init__(env, trace_maxlen=trace_maxlen, eps_stuck=float(self.cfg.stuck_min_moved))

    def _get_target_height(self) -> float:
        ev = self.env.evaluate()
        if not isinstance(ev, dict) or "height" not in ev:
            raise RuntimeError("env.evaluate() must return dict with key 'height'")
        h = ev["height"]
        if isinstance(h, torch.Tensor):
            return float(h.reshape(-1)[0].item())
        return float(h)

    def _get_hand_qpos(self) -> torch.Tensor:
        qpos = self.env.unwrapped.agent.robot.get_qpos()
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos, device=self.env.device, dtype=torch.float32)
        qpos = qpos.to(self.env.device, torch.float32)

        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.ndim != 2:
            raise RuntimeError(f"robot qpos must be 2D, got {tuple(qpos.shape)}")
        if qpos.shape[1] < 19:
            raise RuntimeError(f"robot qpos dof < 19, got {qpos.shape[1]}")

        hand = qpos[:, 7:19]  # (B,12)
        if hand.shape[1] != self.HAND_DOF:
            raise RuntimeError(f"hand qpos must be (B,12), got {tuple(hand.shape)}")
        return hand

    def _get_target_hand_qpos(self, pose: Pose) -> torch.Tensor:
        dev = self.env.device
        if pose == "flat":
            q = self.cfg.q_flat
        elif pose == "work":
            q = self.cfg.q_push_pull
        else:
            raise ValueError(pose)

        if q is None:
            raise RuntimeError(f"Target finger qpos for pose='{pose}' is None. Please set cfg.q_flat/q_push_pull.")

        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q, device=dev, dtype=torch.float32)
        else:
            q = q.to(device=dev, dtype=torch.float32)

        q = q.reshape(-1)
        if q.numel() != self.HAND_DOF:
            raise RuntimeError(f"Target finger qpos must be 12-dim, got {q.numel()}")
        return q.unsqueeze(0)  # (1,12)

    def _pack_hand_action(self, u12: np.ndarray) -> np.ndarray:
        action_dim = int(self.env.action_space.shape[0])
        if action_dim < 18:
            raise RuntimeError(f"env.action_space dim < 18, got {action_dim}")
        if u12.shape != (self.HAND_DOF,):
            raise RuntimeError(f"u12 must be (12,), got {u12.shape}")

        a = np.zeros((action_dim,), dtype=np.float32)
        u12 = np.clip(u12.astype(np.float32), -1.0, 1.0)

        for k in range(self.HAND_DOF):
            a[self.QPOS7_19_TO_ACTION[k]] = u12[k]

        return a

    def _msg_pose_state(self, pose: Pose, q: torch.Tensor, q_t: torch.Tensor, stuck: torch.Tensor) -> str:
        # q, q_t: (1,12); stuck: (1,12)
        err_abs = (q_t - q).abs()
        not_stuck = ~stuck
        if bool(not_stuck.any().item()):
            err_max = float(err_abs[not_stuck].max().item())
        else:
            err_max = float(err_abs.max().item()) if err_abs.numel() > 0 else 0.0
        stuck_cnt = int(stuck.sum().item())
        return f"pose={pose} err_max={err_max:.6f} stuck={stuck_cnt}/12"

    def _call_error(self, e: Exception) -> SkillResult:
        msg = ""
        try:
            q = self._get_hand_qpos()
            msg = f"hand_qpos_max={float(q.abs().max().item()):.6f}"
        except Exception:
            msg = ""
        return self._result(
            ok=False,
            error_code="call_error",
            message=msg,
            advice=f"{type(e).__name__}: {e}",
        )

    def set_pose(self, pose: Pose, *, render: bool = False, verbose: bool = False) -> SkillResult:
        try:
            self.reset_trace()

            h0 = self._get_target_height()
            q_target_1 = self._get_target_hand_qpos(pose)  # (1,12)

            steps = 0
            moved_hist: list[float] = []

            delta_u = float(self.cfg.hand_delta_upper)
            kp = float(self.cfg.kp)
            tol = float(self.cfg.tol)
            u_cap = float(self.cfg.u_cap)

            # Joint-level stuck handling
            stuck = torch.zeros((1, self.HAND_DOF), device=self.env.device, dtype=torch.bool)

            for i in range(int(self.cfg.max_steps)):
                q = self._get_hand_qpos()  # (B,12) (assume B=1)
                if q.shape[0] != 1:
                    raise RuntimeError(f"EEPoseSkill expects num_envs=1, got batch={q.shape[0]}")
                q_t = q_target_1.expand(q.shape[0], -1)

                err = q_t - q
                err_abs = err.abs()

                # Only consider non-stuck joints for termination
                not_stuck = ~stuck
                active = (err_abs > tol) & not_stuck

                if not bool(active.any().item()):
                    h1 = self._get_target_height()
                    object_moved = abs(h1 - h0) > float(self.cfg.moved_height_eps)
                    _ = object_moved  # keep evaluation side effects unchanged

                    all_stuck = bool(stuck.all().item())
                    # If all joints are stuck and still not at target, treat as stuck failure.
                    if all_stuck:
                        return self._result(
                            ok=False,
                            error_code="stuck",
                            message=self._msg_pose_state(pose, q, q_t, stuck),
                            advice="Consider changing direction or using a different motion.",
                        )

                    return self._result(
                        ok=True,
                        error_code="none",
                        message=self._msg_pose_state(pose, q, q_t, stuck),
                        advice="",
                    )

                # Normalized action directly (avoid always hitting +/-1)
                u = torch.zeros_like(err)
                u[active] = (kp * err[active]) / delta_u
                u = u.clamp(min=-u_cap, max=u_cap)

                q_before = q

                action = self._pack_hand_action(u[0].detach().cpu().numpy().astype(np.float32))
                _, _, _, _, _info = self._step(action, i=i, render=render)

                q_after = self._get_hand_qpos()

                dq = (q_after - q_before).abs()  # (1,12)

                # Mark newly stuck joints: commanded (active) but didn't move enough this step
                newly_stuck = active & (dq < float(self.cfg.stuck_min_moved))
                if bool(newly_stuck.any().item()):
                    stuck = stuck | newly_stuck

                moved = float(dq.max().item())
                self._trace.add(StepEvent(i=int(i), moved=moved, event=None))

                moved_hist.append(moved)
                if len(moved_hist) > int(self.cfg.stuck_window):
                    moved_hist = moved_hist[-int(self.cfg.stuck_window):]

                steps += 1

                if verbose and (i < 10 or (i + 1) % 30 == 0):
                    err_max_active = float(err_abs[active].max().item()) if bool(active.any().item()) else 0.0
                    umax = float(torch.abs(u).max().item())
                    stuck_cnt = int(stuck.sum().item())
                    print(
                        f"[SetPose] step={i+1}/{int(self.cfg.max_steps)} "
                        f"err_max_active={err_max_active:.6f} u_max={umax:.3f} "
                        f"moved={moved:.6f} stuck={stuck_cnt}/12"
                    )
                    if bool(newly_stuck.any().item()):
                        idxs = (
                            torch.nonzero(newly_stuck[0], as_tuple=False)
                            .reshape(-1)
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                        print(f"[SetPose] newly_stuck joints (in qpos[7:19] order): {idxs}")

                if len(moved_hist) == int(self.cfg.stuck_window) and max(moved_hist) < float(self.cfg.stuck_min_moved):
                    return self._result(
                        ok=False,
                        error_code="stuck",
                        message=self._msg_pose_state(pose, q_after, q_t, stuck),
                        advice="Consider changing direction or using a different motion.",
                    )

                if bool(stuck.all().item()):
                    q_now = self._get_hand_qpos()
                    return self._result(
                        ok=False,
                        error_code="stuck",
                        message=self._msg_pose_state(pose, q_now, q_t, stuck),
                        advice="Consider changing direction or using a different motion.",
                    )

            # Max steps reached
            q_now = self._get_hand_qpos()
            q_t = q_target_1.expand(q_now.shape[0], -1)

            return self._result(
                ok=False,
                error_code="max_step",
                message=self._msg_pose_state(pose, q_now, q_t, stuck),
                advice="Consider reducing step size or retrying the action.",
            )
        except Exception as e:
            return self._call_error(e)
