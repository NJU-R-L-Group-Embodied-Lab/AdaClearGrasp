import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

import env.sim.pick_clutter_xarm  # noqa: F401

from mani_skill.utils.wrappers.record import RecordEpisode
from core.paths import ensure_data_dirs, get_data_paths


# =========================
# Global Params
# =========================
ENV_ID = "PickClutterYCB-XArm7-v1"
SCENE_JSON = "data/scenes/cube.json"
CONTROL_MODE = "pd_ee_delta_pose"
SEED = 0

MODE = "fast"

MODEL_ZIP = "data/models/ppo/PickClutterYCB-XArm7-v1/ppo_grasp.zip"
DEVICE = "cpu"

FPS = 60
MAX_STEPS = 300

INTERP_ENABLE = True
INTERP_STEPS = 90
INTERP_SLEEP = True  

HAND_INIT_12 = [1.655, 0.102, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _scene_name_from_json(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    if not name:
        raise RuntimeError(f"Bad SCENE_JSON: {path}")
    return name


def _is_fast_mode() -> bool:
    return MODE == "fast"


def _is_video_mode() -> bool:
    return MODE == "video"


def _is_window_mode() -> bool:
    return MODE == "window"


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


def action_shaping(action_1d: np.ndarray, ext_step_count: int) -> np.ndarray:
    a = action_1d.copy()

    if ext_step_count <= 100:
        a[:6] *= 0.1
    else:
        a[:6] = 0.0

    a[2] = 0.1 if ext_step_count > 100 else 0.0
    return a


def make_env(mode: str):
    if MODE not in ("video", "window", "fast"):
        raise RuntimeError(f"MODE must be one of 'video'|'window'|'fast', got: {MODE}")

    if _is_fast_mode():
        render_mode = None
    elif _is_video_mode():
        render_mode = "rgb_array"
    else:
        render_mode = "human"

    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        num_envs=1,
        control_mode=CONTROL_MODE,
        scene_json=SCENE_JSON,
        mode=mode,
        reconfiguration_freq=0,
        only_target_object=True
    )

    if _is_video_mode():
        ensure_data_dirs()
        data_paths = get_data_paths()

        scene_name = _scene_name_from_json(SCENE_JSON)
        out_dir = os.path.join(data_paths.videos, "grasp", scene_name)
        os.makedirs(out_dir, exist_ok=True)

        env = RecordEpisode(
            env,
            output_dir=out_dir,
            save_video=True,
            info_on_video=False,
        )
        print(f"[INFO] Recording to: {out_dir}")

    return env


def compute_grasp_pose(env) -> torch.Tensor:
    u = env.unwrapped
    dev = u.device

    if not hasattr(u, "_precompute_arm_init_qpos_one"):
        raise RuntimeError("Env must provide _precompute_arm_init_qpos_one()")
    if not hasattr(u, "ARM_DOF") or not hasattr(u, "DOF_TOTAL"):
        raise RuntimeError("Env must expose ARM_DOF and DOF_TOTAL")

    u._arm_init_qpos_one = None
    u._precompute_arm_init_qpos_one()

    arm = u._arm_init_qpos_one
    if arm is None:
        raise RuntimeError("_arm_init_qpos_one is None after _precompute_arm_init_qpos_one()")
    if not isinstance(arm, torch.Tensor):
        raise RuntimeError(f"_arm_init_qpos_one must be torch.Tensor, got {type(arm)}")
    arm = arm.to(dev, dtype=torch.float32).reshape(-1)
    if arm.numel() != int(u.ARM_DOF):
        raise RuntimeError(f"arm IK must be {int(u.ARM_DOF)} dims, got {arm.numel()}")

    hand = torch.tensor(HAND_INIT_12, device=dev, dtype=torch.float32).reshape(-1)
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
    qpos[0, 0:int(u.ARM_DOF)] = arm
    qpos[0, int(u.ARM_DOF):int(u.DOF_TOTAL)] = hand
    return qpos


def interp_reset_robot_qpos(env, qpos_target_1_dof: torch.Tensor, dt: float):
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

    steps = int(INTERP_STEPS)
    if steps <= 0:
        u.agent.reset(qpos_target)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if _to_bool_any(terminated) or _to_bool_any(truncated):
            raise RuntimeError("Episode ended after immediate reset to target qpos.")
        return _obs_to_np_batch1(obs)

    for k in range(steps + 1):
        t = float(k) / float(steps)
        q = qpos_start * (1.0 - t) + qpos_target * t
        u.agent.reset(q)

        obs, reward, terminated, truncated, info = env.step(zero_action)

        if _is_window_mode():
            env.render()
        if (not _is_fast_mode()) and INTERP_SLEEP:
            time.sleep(dt)

        if _to_bool_any(terminated) or _to_bool_any(truncated):
            raise RuntimeError("Episode ended during qpos interpolation.")

    return _obs_to_np_batch1(obs)


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


def main():
    if not (os.path.exists(MODEL_ZIP) and MODEL_ZIP.endswith(".zip")):
        raise FileNotFoundError(f"MODEL_ZIP not found or not .zip: {MODEL_ZIP}")
    model = PPO.load(MODEL_ZIP, device=DEVICE)

    dt = 1.0 / float(FPS)

    env = make_env(mode="plan")
    u = env.unwrapped
    X_MIN, X_MAX = -0.10, 0.10
    Y_MIN, Y_MAX = -0.10, 0.10

    xs = np.linspace(X_MIN, X_MAX, 6, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, 6, dtype=np.float32)
    xy_list = np.array([[float(x), float(y)] for x in xs for y in ys], dtype=np.float32)  # (36, 2)

    yaw_list = np.linspace(0.0, 360.0, 36, endpoint=False, dtype=np.float32)  # (36,)

    total_eps = 0
    succ_eps = 0

    action_dim = int(env.action_space.shape[0])

    if _is_fast_mode():
        print("[INFO] MODE=fast: no render, no sleep, terminal output only.")
    elif _is_window_mode():
        print("[INFO] MODE=window: human window rendering + sleep.")
    else:
        scene_name = _scene_name_from_json(SCENE_JSON)
        print(f"[INFO] MODE=video: saving videos to data/videos/grasp/{scene_name}")

    for i in range(36):
        xy = xy_list[i]
        yaw = float(yaw_list[i])

        u.scene_cfg["target"]["xy"] = [float(xy[0]), float(xy[1])]
        u.scene_cfg["target"]["yaw"] = yaw

        obs, info = env.reset(seed=SEED)
        obs_np = _obs_to_np_batch1(obs)

        grasp_qpos = compute_grasp_pose(env)

        if INTERP_ENABLE:
            obs_np = interp_reset_robot_qpos(env, grasp_qpos, dt=dt)
        else:
            dev = u.device
            u.agent.reset(grasp_qpos.to(dev, dtype=torch.float32))
            zero_action = np.zeros((action_dim,), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(zero_action)
            if _to_bool_any(terminated) or _to_bool_any(truncated):
                raise RuntimeError("Episode ended after direct reset to target qpos.")
            obs_np = _obs_to_np_batch1(obs)

        ext_step_count = 0
        ep_success = False

        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs_np, deterministic=True)
            action_1d = _action_to_np_1d(action, expected_dim=action_dim)

            shaped_action = action_shaping(action_1d, ext_step_count)

            obs, reward, terminated, truncated, info = env.step(shaped_action)
            obs_np = _obs_to_np_batch1(obs)

            if _is_window_mode():
                env.render()
            if not _is_fast_mode():
                time.sleep(dt)

            ep_success = ep_success or _is_success(info)

            ext_step_count += 1
            if _to_bool_any(terminated) or _to_bool_any(truncated):
                break

        total_eps += 1
        succ_eps += int(ep_success)
        print(f"[TEST] idx={i:02d} target_xy={[float(xy[0]), float(xy[1])]} yaw_deg={yaw:.1f} success={ep_success}")

    print(f"[RESULT] success_rate = {succ_eps}/{total_eps} = {succ_eps/float(total_eps):.3f}")
    env.close()


if __name__ == "__main__":
    main()
