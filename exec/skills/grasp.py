# exec/skills/grasp.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from exec.skills.base import BaseSkill, SkillResult


@dataclass
class GraspConfig:
    # Model
    model_zip: str = "data/models/ppo/PickClutterYCB-XArm7-v1/ppo_grasp.zip"
    device: str = "cpu"
    deterministic: bool = True

    # Rollout
    fps: int = 60
    sleep: bool = True
    max_steps: int = 300

    # Plan->grasp qpos init
    hand_init_12: Tuple[float, ...] = (
        1.655,
        0.102,
        0.0,
        0.0,
        0.0,
        -0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    # Robot-only interpolation reset
    interp_enable: bool = True
    interp_steps: int = 90
    interp_sleep: bool = True


def _obs_to_np_batch1(obs) -> np.ndarray:
    if isinstance(obs, torch.Tensor):
        x = obs.detach().cpu().numpy()
    elif isinstance(obs, np.ndarray):
        x = obs
    else:
        raise RuntimeError(f"obs must be torch.Tensor or np.ndarray, got {type(obs)}")

    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[0] != 1:
        raise RuntimeError(f"obs must be shape (1, obs_dim), got {x.shape}")

    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _action_to_np_1d(action, expected_dim: int) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        a = action.detach().cpu().numpy()
    elif isinstance(action, np.ndarray):
        a = action
    else:
        raise RuntimeError(f"action must be torch.Tensor or np.ndarray, got {type(action)}")

    if a.ndim == 2:
        if a.shape[0] != 1:
            raise RuntimeError(f"action batch must be 1, got {a.shape}")
        a = a[0]
    if a.ndim != 1:
        raise RuntimeError(f"action must be 1D, got shape={a.shape}")

    if a.shape[0] != expected_dim:
        raise RuntimeError(f"action_dim mismatch: got {a.shape[0]} expected {expected_dim}")

    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return a


def _to_bool_any(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, np.ndarray):
        return bool(np.any(x))
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return False
        return bool(x.any().item())
    raise RuntimeError(f"terminated/truncated must be bool/np.ndarray/torch.Tensor, got {type(x)}")


def _is_success(info) -> bool:
    if not isinstance(info, dict) or ("success" not in info):
        return False
    s = info["success"]
    if isinstance(s, (bool, np.bool_)):
        return bool(s)
    if isinstance(s, torch.Tensor):
        return bool(s.any().item()) if s.numel() > 0 else False
    if isinstance(s, np.ndarray):
        return bool(np.any(s))
    return False


def action_shaping(action_1d: np.ndarray, ext_step_count: int) -> np.ndarray:
    a = action_1d.copy()

    if ext_step_count <= 100:
        a[:6] *= 0.1
    else:
        a[:6] = 0.0

    a[2] = 0.1 if ext_step_count > 100 else 0.0
    return a


def compute_grasp_pose(env, hand_init_12: Tuple[float, ...]) -> Optional[torch.Tensor]:
    """
    Returns:
      - torch.Tensor qpos (1, dof) if IK solved
      - None if IK has no solution
    """
    u = env.unwrapped
    dev = u.device

    if not hasattr(u, "_precompute_arm_init_qpos_one"):
        raise RuntimeError("Env must provide _precompute_arm_init_qpos_one()")
    if not hasattr(u, "ARM_DOF") or not hasattr(u, "DOF_TOTAL"):
        raise RuntimeError("Env must expose ARM_DOF and DOF_TOTAL")

    u._arm_init_qpos_one = None
    # IK 无解时：不抛异常，返回 None
    try:
        u._precompute_arm_init_qpos_one()
    except RuntimeError as e:
        s = str(e)
        if ("IK" in s) or ("ik" in s):
            return None
        raise

    arm = u._arm_init_qpos_one
    if arm is None:
        # IK 无解：环境未写明错误文本，但结果为空
        return None

    if not isinstance(arm, torch.Tensor):
        raise RuntimeError(f"_arm_init_qpos_one must be torch.Tensor, got {type(arm)}")
    arm = arm.to(dev, dtype=torch.float32).reshape(-1)
    if arm.numel() != int(u.ARM_DOF):
        raise RuntimeError(f"arm IK must be {int(u.ARM_DOF)} dims, got {arm.numel()}")

    hand = torch.tensor(list(hand_init_12), device=dev, dtype=torch.float32).reshape(-1)
    if hand.numel() != int(u.DOF_TOTAL - u.ARM_DOF):
        raise RuntimeError(f"hand init must be {int(u.DOF_TOTAL - u.ARM_DOF)} dims, got {hand.numel()}")

    robot = u.agent.robot
    dof = robot.dof
    if isinstance(dof, torch.Tensor):
        dof = int(dof[0].item())
    else:
        dof = int(dof)

    if dof < int(u.DOF_TOTAL):
        raise RuntimeError(f"robot dof < {int(u.DOF_TOTAL)} not supported, got dof={dof}")

    qpos = torch.zeros((1, dof), device=dev, dtype=torch.float32)
    qpos[0, 0 : int(u.ARM_DOF)] = arm
    qpos[0, int(u.ARM_DOF) : int(u.DOF_TOTAL)] = hand
    return qpos


def interp_reset_robot_qpos(
    env,
    qpos_target_1_dof: torch.Tensor,
    *,
    steps: int,
    dt: float,
    render: bool,
    sleep_enable: bool,
) -> np.ndarray:
    u = env.unwrapped
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

    action_dim = int(env.action_space.shape[0])
    zero_action = np.zeros((action_dim,), dtype=np.float32)

    if steps <= 0:
        u.agent.reset(qpos_target)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if _to_bool_any(terminated) or _to_bool_any(truncated):
            raise RuntimeError("Episode ended after immediate robot reset to target qpos.")
        return _obs_to_np_batch1(obs)

    obs = None
    for k in range(int(steps) + 1):
        t = float(k) / float(steps)
        q = qpos_start * (1.0 - t) + qpos_target * t
        u.agent.reset(q)

        obs, reward, terminated, truncated, info = env.step(zero_action)

        if render:
            env.render()
        if sleep_enable:
            time.sleep(dt)

    return _obs_to_np_batch1(obs)


class GraspSkill(BaseSkill):
    """
    STRICT behavior:
      - does NOT call env.reset()
      - does NOT touch scene_cfg
      - assumes env already reset/initialized by caller
      - allowed: robot-only reset via u.agent.reset(qpos) + env.step(zero_action)
      - rollout PPO with action shaping, success from info['success']
    """

    def __init__(self, env, cfg: Optional[GraspConfig] = None, *, trace_maxlen: int = 10):
        super().__init__(env, trace_maxlen=trace_maxlen, eps_stuck=1e-6)
        self.cfg = cfg or GraspConfig()

        if not (os.path.exists(self.cfg.model_zip) and self.cfg.model_zip.endswith(".zip")):
            raise FileNotFoundError(f"model_zip not found or not .zip: {self.cfg.model_zip}")
        self.model = PPO.load(self.cfg.model_zip, device=self.cfg.device)

    def _get_current_obs_np(self) -> np.ndarray:
        u = self.env.unwrapped
        if not hasattr(u, "get_obs"):
            raise RuntimeError(
                "Env must provide unwrapped.get_obs() because grasp skill is not allowed to call env.reset()."
            )
        obs = u.get_obs()
        return _obs_to_np_batch1(obs)

    def grasp(self, *, render: bool = False) -> SkillResult:
        self.reset_trace()

        obs_np = self._get_current_obs_np()

        dt = 1.0 / float(self.cfg.fps)
        action_dim = int(self.env.action_space.shape[0])

        grasp_qpos = compute_grasp_pose(self.env, self.cfg.hand_init_12)
        if grasp_qpos is None:
            return self._result(
                ok=False,
                error_code="stuck",
                message="IK failed: the target pose cannot be solved to a feasible initial joint configuration for the robotic arm.",
                advice="The object cannot be grasped in its current state; try changing its position or orientation.",

            )

        if bool(self.cfg.interp_enable):
            obs_np = interp_reset_robot_qpos(
                self.env,
                grasp_qpos,
                steps=int(self.cfg.interp_steps),
                dt=dt,
                render=bool(render),
                sleep_enable=bool(self.cfg.interp_sleep) and bool(self.cfg.sleep),
            )
        else:
            u = self.env.unwrapped
            dev = u.device
            u.agent.reset(grasp_qpos.to(dev, dtype=torch.float32))

            zero_action = np.zeros((action_dim,), dtype=np.float32)
            obs, reward, terminated, truncated, info = self.env.step(zero_action)
            obs_np = _obs_to_np_batch1(obs)
            if render:
                self.env.render()
            if self.cfg.sleep:
                time.sleep(dt)

        ext_step_count = 0
        ep_success = False

        for _ in range(int(self.cfg.max_steps)):
            action, _ = self.model.predict(obs_np, deterministic=bool(self.cfg.deterministic))
            action_1d = _action_to_np_1d(action, expected_dim=action_dim)

            shaped_action = action_shaping(action_1d, ext_step_count)
            obs, reward, terminated, truncated, info = self.env.step(shaped_action)
            obs_np = _obs_to_np_batch1(obs)

            if render:
                self.env.render()
            if self.cfg.sleep:
                time.sleep(dt)

            ep_success = ep_success or _is_success(info)
            if ep_success:
                print("success!")
                break

            ext_step_count += 1

        return self._result(
            ok=True,
            error_code="none" if ep_success else "stuck",
            message="success" if ep_success else "failed",
            advice="",
        )
