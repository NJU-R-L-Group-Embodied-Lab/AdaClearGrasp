from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from exec.skills.base import BaseSkill, SkillResult
from exec.skills.ee_pose import EEPoseSkill


@dataclass
class InitSkillConfig:
    arm_init_7 = [-0.77920043, 0.4921763, 0.74077785, 1.2483665, 2.8245757, 0.6868453, -0.050859887]

    interp_steps: int = 90

    do_reset: bool = False
    seed: int = 0

    render: bool = False
    verbose: bool = False


class InitSkill(BaseSkill):
    HAND_INIT_12 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self, env, *, cfg: Optional[InitSkillConfig] = None, trace_maxlen: int = 10):
        super().__init__(env, trace_maxlen=trace_maxlen)
        self.cfg = cfg or InitSkillConfig()

    # -------- helpers --------

    def _to_bool_any(self, x) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, np.ndarray):
            return bool(np.any(x))
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return False
            return bool(x.any().item())
        raise RuntimeError(f"terminated/truncated must be bool/np.ndarray/torch.Tensor, got {type(x)}")

    def _msg_xy(self) -> str:
        tcp = self.env.unwrapped.agent.tcp
        p = tcp.pose.p
        if not isinstance(p, torch.Tensor):
            p = torch.as_tensor(p, device=self.env.unwrapped.device, dtype=torch.float32)
        p = p.reshape(-1)[:3]
        return f"x={float(p[0].item()):.4f}, y={float(p[1].item()):.4f}"

    def _build_qpos_target_from_arm7(self, arm7_1x7: torch.Tensor) -> torch.Tensor:
        u = self.env.unwrapped
        dev = u.device

        if not hasattr(u, "ARM_DOF") or not hasattr(u, "DOF_TOTAL"):
            raise RuntimeError("Env must expose ARM_DOF and DOF_TOTAL")

        arm = arm7_1x7
        if not isinstance(arm, torch.Tensor):
            raise RuntimeError(f"arm7 must be torch.Tensor, got {type(arm)}")
        if arm.ndim == 1:
            arm = arm.unsqueeze(0)
        if arm.ndim != 2 or arm.shape[0] != 1:
            raise RuntimeError(f"arm7 must be shape (1,7) or (7,), got {tuple(arm.shape)}")

        arm = arm.to(dev, dtype=torch.float32)
        if arm.shape[1] != int(u.ARM_DOF):
            raise RuntimeError(f"arm7 dim mismatch: got {arm.shape[1]} expected {int(u.ARM_DOF)}")

        hand = torch.tensor(self.HAND_INIT_12, device=dev, dtype=torch.float32).reshape(1, -1)
        if hand.shape[1] != int(u.DOF_TOTAL - u.ARM_DOF):
            raise RuntimeError(f"hand init must be {int(u.DOF_TOTAL - u.ARM_DOF)} dims, got {hand.shape[1]}")

        robot = u.agent.robot
        dof = robot.dof
        if isinstance(dof, torch.Tensor):
            dof = int(dof[0].item())
        else:
            dof = int(dof)
        if dof < int(u.DOF_TOTAL):
            raise RuntimeError(f"robot dof < {int(u.DOF_TOTAL)} not supported, got dof={dof}")

        qpos = torch.zeros((1, dof), device=dev, dtype=torch.float32)
        qpos[0, 0:int(u.ARM_DOF)] = arm[0]
        qpos[0, int(u.ARM_DOF):int(u.DOF_TOTAL)] = hand[0]
        return qpos

    def _interp_to_target_with_env_step(self, qpos_target_1_dof: torch.Tensor):
        u = self.env.unwrapped
        dev = u.device

        qpos_start = u.agent.robot.get_qpos()
        if not isinstance(qpos_start, torch.Tensor):
            qpos_start = torch.as_tensor(qpos_start)
        if qpos_start.ndim == 1:
            qpos_start = qpos_start.unsqueeze(0)
        if qpos_start.ndim != 2 or qpos_start.shape[0] != 1:
            raise RuntimeError(f"plan qpos must be (1,dof), got {tuple(qpos_start.shape)}")

        qpos_start = qpos_start.to(dev, dtype=torch.float32)
        qpos_target = qpos_target_1_dof.to(dev, dtype=torch.float32)

        if qpos_start.shape != qpos_target.shape:
            raise RuntimeError(f"qpos shape mismatch: start={tuple(qpos_start.shape)} target={tuple(qpos_target.shape)}")

        action_dim = int(self.env.action_space.shape[0])
        zero_action = np.zeros((action_dim,), dtype=np.float32)

        steps = int(self.cfg.interp_steps)
        if steps <= 0:
            u.agent.reset(qpos_target)
            obs, reward, terminated, truncated, info = self.env.step(zero_action)
            if self._to_bool_any(terminated) or self._to_bool_any(truncated):
                raise RuntimeError("Episode ended after immediate reset to target qpos.")
            return obs

        for k in range(steps + 1):
            t = float(k) / float(steps)
            q = qpos_start * (1.0 - t) + qpos_target * t
            u.agent.reset(q)

            obs, reward, terminated, truncated, info = self.env.step(zero_action)

            if self.cfg.render:
                self.env.render()


        return obs

    # -------- public --------

    def initarm(self, name: str = "", *, render: Optional[bool] = None, verbose: Optional[bool] = None) -> SkillResult:
        self.reset_trace()
        self.cfg.render = self.cfg.render if render is None else bool(render)
        self.cfg.verbose = self.cfg.verbose if verbose is None else bool(verbose)


        u = self.env.unwrapped
        dev = u.device
        arm_init = torch.tensor(self.cfg.arm_init_7, dtype=torch.float32, device=dev).unsqueeze(0)  # (1,7)
        qpos_target = self._build_qpos_target_from_arm7(arm_init)

        if self.cfg.verbose:
            print("[InitSkill.initarm] target qpos computed:", tuple(qpos_target.shape))

        _ = self._interp_to_target_with_env_step(qpos_target)

        return self._result(
            ok=True,
            error_code="none",
            message=self._msg_xy(),
            advice="",
        )

    def inithand(self, *, render: Optional[bool] = None, verbose: Optional[bool] = None) -> SkillResult:
        self.reset_trace()
        do_render = self.cfg.render if render is None else bool(render)
        do_verbose = self.cfg.verbose if verbose is None else bool(verbose)

        pose_skill = EEPoseSkill(self.env)
        _ = pose_skill.set_pose("flat", render=do_render, verbose=do_verbose)

        return self._result(
            ok=True,
            error_code="none",
            message=self._msg_xy(),
            advice="",
        )
