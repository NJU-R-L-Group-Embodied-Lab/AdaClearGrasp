import os
import sys
import numpy as np
import gymnasium as gym

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import env.sim.pick_clutter_xarm  # noqa: F401


ENV_ID = "PickClutterYCB-XArm7-v1"
CONTROL_MODE = "pd_ee_delta_pose"


def debug_reset_motion():
    env = gym.make(
        ENV_ID,
        num_envs=1,
        render_mode="human",
        control_mode=CONTROL_MODE,
        scene_json="data/scenes/apple/6/3.json",
        mode="plan",
        only_target_object=False,
    )
    
    obs, info = env.reset()

    while True:
        action = np.zeros(18, dtype=np.float32)
        obs, rew, terminated, truncated, info = env.step(action)
        env.render()



if __name__ == "__main__":
    debug_reset_motion()
