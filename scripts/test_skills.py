# test_skills.py
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

import env.sim.pick_clutter_xarm  # noqa: F401
from core.paths import ensure_data_dirs, get_data_paths

from exec.skills.ee_move import EEMoveSkill
from exec.skills.env_reset import EnvResetSkill
from exec.skills.grasp import GraspSkill
from exec.skills.clear import PullSkill, PushSkill
from exec.skills.init import InitSkill


RUN_MODE = "window"  # "window" | "video"
VIDEO_TAG = "test_skills"

ENV_ID = "PickClutterYCB-XArm7-v1"
CONTROL_MODE = "pd_ee_delta_pose"
MODE = "plan"
SETTLE_STEPS = 0


def make_env():
    if RUN_MODE not in ("window", "video"):
        raise RuntimeError(f"RUN_MODE must be 'window'|'video', got: {RUN_MODE}")

    render_mode = "rgb_array" if RUN_MODE == "video" else "human"

    env = gym.make(
        ENV_ID,
        num_envs=1,
        render_mode=render_mode,
        control_mode=CONTROL_MODE,
        mode=MODE,
        scene_json="data/scenes/apple/2/1.json",
        reconfiguration_freq=0,
        only_target_object=False,
    )

    if RUN_MODE == "video":
        ensure_data_dirs()
        data_paths = get_data_paths()

        # put videos under: data/videos/manual/...
        out_dir = os.path.join(data_paths.videos, "manual", ENV_ID, VIDEO_TAG)
        os.makedirs(out_dir, exist_ok=True)

        env = RecordEpisode(
            env,
            output_dir=out_dir,
            save_video=True,
            info_on_video=False,
        )
        print(f"[INFO] Recording to: {out_dir}")

    return env


def settle(env, steps=200):
    zero = np.zeros(18, dtype=np.float32)
    for _ in range(int(steps)):
        env.step(zero)
        # for window: show; for video: drive render so wrapper can capture frames
        env.render()


def MANUAL_CALLS(env):
    mv = EEMoveSkill(env)
    push = PushSkill(env)
    pull = PullSkill(env)
    init = InitSkill(env)

    rs = EnvResetSkill(env)
    grasp = GraspSkill(env)

    init.initarm(render=True)

    grasp.grasp(render=True)


def main():
    env = make_env()
    env.reset()

    settle(env, steps=SETTLE_STEPS)
    MANUAL_CALLS(env)

    env.close()
    print("[OK] Done.")


if __name__ == "__main__":
    main()
