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

# ===== video record wrapper =====
from mani_skill.utils.wrappers.record import RecordEpisode
from core.paths import ensure_data_dirs, get_data_paths


# =========================
# Global Params 
# =========================
ENV_ID = "PickClutterYCB-XArm7-v1"
SCENE_JSON = "data/scenes/apple.json"
CONTROL_MODE = "pd_ee_delta_pose"
SEED = 0

RECORD_VIDEO = False

VIDEO_TAG = "ppo_viz"

MODEL_ZIP = "data/models/ppo/PickClutterYCB-XArm7-v1/ppo_grasp.zip"
DEVICE = "cpu"

FPS = 60
MAX_STEPS = 300


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


def make_env():
    render_mode = "rgb_array" if RECORD_VIDEO else "human"

    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        num_envs=1,
        control_mode=CONTROL_MODE,
        scene_json=SCENE_JSON,
        mode="train",
        only_target_object=True,
        reconfiguration_freq=0,
    )

    if RECORD_VIDEO:
        ensure_data_dirs()
        data_paths = get_data_paths()
        out_dir = os.path.join(data_paths.videos, ENV_ID, VIDEO_TAG)
        os.makedirs(out_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=out_dir, save_video=True, info_on_video=False)
        print(f"[INFO] Recording to: {out_dir}")

    return env


def main():
    env = make_env()

    obs, info = env.reset(seed=SEED)
    obs_np = _obs_to_np_batch1(obs)

    if not (os.path.exists(MODEL_ZIP) and MODEL_ZIP.endswith(".zip")):
        raise FileNotFoundError(f"MODEL_ZIP not found or not .zip: {MODEL_ZIP}")
    model = PPO.load(MODEL_ZIP, device=DEVICE)

    action_dim = int(env.action_space.shape[0])

    dt = 1.0 / float(FPS)

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs_np, deterministic=True)
        action_1d = _action_to_np_1d(action, expected_dim=action_dim)

        obs, reward, terminated, truncated, info = env.step(action_1d)
        obs_np = _obs_to_np_batch1(obs)

        if not RECORD_VIDEO:
            env.render()

        time.sleep(dt)

        if _to_bool_any(terminated) or _to_bool_any(truncated):
            obs, info = env.reset()
            obs_np = _obs_to_np_batch1(obs)

    env.close()


if __name__ == "__main__":
    main()
