# plan/replay_from_log.py
from __future__ import annotations

import os
import sys
import json
import time
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import gymnasium as gym

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import env.sim.pick_clutter_xarm  # noqa: F401

# Skills (same as mcp_server.py)
from exec.skills.ee_move import EEMoveSkill
from exec.skills.ee_pose import EEPoseSkill
from exec.skills.clear import PullSkill, PushSkill
from exec.skills.init import InitSkill
from exec.skills.grasp import GraspSkill

from mani_skill.utils.wrappers.record import RecordEpisode
from core.paths import ensure_data_dirs, get_data_paths

LOG_DIR = "data/logs/vlm_plan/apple/4/7"   # contains steps.jsonl
MODE = "window"                           # "window" | "video"

ENV_ID = "PickClutterYCB-XArm7-v1"
CONTROL_MODE = "pd_ee_delta_pose"
OBS_MODE = "rgb"

VIDEO_FPS = 30                           # timeline + (attempt) video fps
SLEEP_IN_WINDOW = True                   # window: sleep to match VIDEO_FPS

VIDEO_TAG = "replay"                     # output folder under data/videos/
SCENE_JSON_OVERRIDE = ""                 # set if you don't want auto-infer from LOG_DIR

TIMELINE_FILENAME = "replay_timeline.jsonl"  # saved in same folder as the video output (video mode)


# =========================================================
# Helpers
# =========================================================
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _infer_scene_json_from_log_dir(log_dir: str) -> str:
    # expected: data/logs/vlm_plan/<scene>/<clutter>/<scene_id>
    parts = os.path.normpath(log_dir).split(os.sep)
    if len(parts) < 4:
        raise RuntimeError(f"LOG_DIR too short to infer scene: {log_dir}")

    scene_name = parts[-3]
    clutter = parts[-2]
    scene_id = parts[-1]
    return os.path.join("data", "scenes", scene_name, clutter, f"{scene_id}.json")


def _ensure_mode() -> None:
    if MODE not in ("window", "video"):
        raise RuntimeError(f"MODE must be 'window' or 'video', got: {MODE}")


def _scene_triplet_from_scene_json(scene_json: str) -> Dict[str, str]:
    # expected: data/scenes/<scene>/<clutter>/<id>.json
    p = os.path.normpath(scene_json).split(os.sep)
    if len(p) < 4:
        raise RuntimeError(f"Bad scene_json path: {scene_json}")
    return {
        "scene_name": p[-3],
        "clutter": p[-2],
        "scene_id": os.path.splitext(p[-1])[0],
    }


def _to_int_scalar(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise RuntimeError(f"Expected scalar tensor, got shape={tuple(x.shape)}")
        return int(x.item())
    if isinstance(x, np.ndarray):
        if x.size != 1:
            raise RuntimeError(f"Expected scalar ndarray, got shape={x.shape}")
        return int(x.reshape(()).item())
    raise RuntimeError(f"Cannot convert to int scalar: {type(x)}")


def _get_env_step_count(env) -> int:
    u = env.unwrapped
    if hasattr(u, "_elapsed_steps"):
        return _to_int_scalar(getattr(u, "_elapsed_steps"))
    if hasattr(u, "elapsed_steps"):
        return _to_int_scalar(getattr(u, "elapsed_steps"))
    raise RuntimeError("Env does not expose _elapsed_steps/elapsed_steps; cannot compute video timeline.")


def _make_env(scene_json: str) -> tuple[Any, Optional[str]]:
    _ensure_mode()

    render_mode = "human" if MODE == "window" else "rgb_array"

    env = gym.make(
        ENV_ID,
        num_envs=1,
        render_mode=render_mode,
        control_mode=CONTROL_MODE,
        mode="plan",
        reconfiguration_freq=0,
        obs_mode=OBS_MODE,
        scene_json=scene_json,
        only_target_object=False,
        use_external_arm_init=True,
    )

    # Try to align video fps (RecordEpisode usually uses env.metadata["render_fps"])
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("Env has no metadata dict; cannot set render_fps.")
    env.metadata["render_fps"] = int(VIDEO_FPS)

    out_dir: Optional[str] = None
    if MODE == "video":
        ensure_data_dirs()
        data_paths = get_data_paths()

        trip = _scene_triplet_from_scene_json(scene_json)
        out_dir = os.path.join(
            data_paths.videos,
            VIDEO_TAG,
            trip["scene_name"],
            trip["clutter"],
            trip["scene_id"],
        )
        os.makedirs(out_dir, exist_ok=True)

        env = RecordEpisode(
            env,
            output_dir=out_dir,
            save_video=True,
            info_on_video=False,
        )
        print(f"[INFO] Recording to: {out_dir}")

    return env, out_dir


def _render_tick(env, dt: float) -> None:
    if MODE == "window":
        env.render()
        if SLEEP_IN_WINDOW:
            time.sleep(dt)


def _py_repr(v: Any) -> str:
    return repr(v)


def _call_string(action: str, args: Dict[str, Any]) -> str:
    if action in ("lower", "lift", "initarm", "inithand", "grasp", "env_reset", "done"):
        return f"{action}()"

    if action == "move_to":
        return f'move_to({_py_repr(args["name"])})'

    if action == "set_pose":
        return f'set_pose({_py_repr(args["pose"])})'

    if action in ("pull", "push"):
        return f'{action}({_py_repr(args["side"])}, {args["dist_m"]})'

    raise RuntimeError(f"Unknown action in log: {action!r}")


def _execute_action(
    action: str,
    args: Dict[str, Any],
    move: EEMoveSkill,
    pose: EEPoseSkill,
    pull: PullSkill,
    push: PushSkill,
    init: InitSkill,
    grasp: GraspSkill,
):
    if action == "move_to":
        return move.move_to(str(args["name"]), render=True, verbose=False)

    if action == "lower":
        return move.lower(render=True, verbose=False)

    if action == "lift":
        return move.lift(render=True, verbose=False)

    if action == "set_pose":
        return pose.set_pose(str(args["pose"]), render=True)

    if action == "pull":
        return pull.pull(str(args["side"]), float(args["dist_m"]), render=True, verbose=False)

    if action == "push":
        return push.push(str(args["side"]), float(args["dist_m"]), render=True, verbose=False)

    if action == "initarm":
        return init.initarm("", render=True, verbose=False)

    if action == "inithand":
        return init.inithand(render=True, verbose=False)

    if action == "grasp":
        return grasp.grasp(render=True)

    if action in ("env_reset", "done"):
        return None

    raise RuntimeError(f"Unknown action in log: {action!r}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    steps_path = os.path.join(LOG_DIR, "steps.jsonl")
    if not os.path.exists(steps_path):
        raise RuntimeError(f"steps.jsonl not found: {steps_path}")

    scene_json = SCENE_JSON_OVERRIDE or _infer_scene_json_from_log_dir(LOG_DIR)
    if not os.path.exists(scene_json):
        raise RuntimeError(f"scene_json not found: {scene_json}")

    steps = _read_jsonl(steps_path)
    if not steps:
        raise RuntimeError(f"steps.jsonl is empty: {steps_path}")

    dt = 1.0 / float(VIDEO_FPS)

    env, video_out_dir = _make_env(scene_json)
    env.reset()

    timeline_dir = video_out_dir if (MODE == "video" and video_out_dir is not None) else LOG_DIR
    os.makedirs(timeline_dir, exist_ok=True)
    timeline_path = os.path.join(timeline_dir, TIMELINE_FILENAME)
    f_tl = open(timeline_path, "w", encoding="utf-8")

    move = EEMoveSkill(env)
    pose = EEPoseSkill(env)
    pull = PullSkill(env)
    push = PushSkill(env)
    init = InitSkill(env)
    grasp = GraspSkill(env)

    step0 = _get_env_step_count(env)
    wall0 = perf_counter()

    _render_tick(env, dt)

    for rec in steps:
        action = str(rec["action"])
        args = rec.get("args", {}) or {}
        step_id = int(rec.get("step_id", -1))
        reason = str(rec.get("reason", ""))

        cur_steps = _get_env_step_count(env)
        t_video_s = float(cur_steps - step0) / float(VIDEO_FPS)
        t_video_ms = int(round(t_video_s * 1000.0))
        t_wall_s = float(perf_counter() - wall0)

        if action == "env_reset":
            call = _call_string(action, args)

            env.reset()
            _render_tick(env, dt)

            step0 = _get_env_step_count(env)
            wall0 = perf_counter()

            f_tl.write(json.dumps({
                "t_video_s": float(t_video_s),
                "t_video_ms": int(t_video_ms),
                "t_wall_s": float(t_wall_s),
                "step_id": step_id,
                "call": call,
                "reason": reason,
            }, ensure_ascii=False) + "\n")
            f_tl.flush()
            continue

        if action == "done":
            call = _call_string(action, args)

            f_tl.write(json.dumps({
                "t_video_s": float(t_video_s),
                "t_video_ms": int(t_video_ms),
                "t_wall_s": float(t_wall_s),
                "step_id": step_id,
                "call": call,
                "reason": reason,
            }, ensure_ascii=False) + "\n")
            f_tl.flush()

            print(f"[INFO] done at step_id={step_id}")
            break

        call = _call_string(action, args)
        res = _execute_action(action, args, move, pose, pull, push, init, grasp)

        ok = getattr(res, "ok", None)
        msg = getattr(res, "message", None)
        err = getattr(res, "error_code", None)

        f_tl.write(json.dumps({
            "t_video_s": float(t_video_s),
            "t_video_ms": int(t_video_ms),
            "t_wall_s": float(t_wall_s),
            "step_id": step_id,
            "call": call,
            "reason": reason,
            "ok": ok,
            "error_code": err,
            "message": msg,
        }, ensure_ascii=False) + "\n")
        f_tl.flush()

        print(f"[REPLAY] step_id={step_id:04d} call={call} ok={ok} err={err} msg={msg}")

        _render_tick(env, dt)

    f_tl.close()
    env.close()

    print("[OK] Replay finished.")
    print(f"[OK] Timeline saved: {timeline_path}")
    if MODE == "video" and video_out_dir is not None:
        print(f"[OK] Video folder: {video_out_dir}")


if __name__ == "__main__":
    main()
