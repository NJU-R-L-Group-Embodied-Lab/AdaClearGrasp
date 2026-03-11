# plan/batch_render_replays.py
from __future__ import annotations

import os
import sys
import json
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ManiSkill env registration
import env.sim.pick_clutter_xarm  # noqa: F401

# Skills
from exec.skills.ee_move import EEMoveSkill
from exec.skills.ee_pose import EEPoseSkill
from exec.skills.clear import PullSkill, PushSkill
from exec.skills.init import InitSkill
from exec.skills.grasp import GraspSkill

from mani_skill.utils.wrappers.record import RecordEpisode
from core.paths import ensure_data_dirs, get_data_paths


# =========================================================
# Global Params (edit here)
# =========================================================
SCENE_LIST = [
    "lego",
    "mug",
    "pear"
]

CLUTTER_LIST = [2, 4, 6]
SCENE_ID_LIST = list(range(1, 11))  # 1..10

LOG_ROOT = os.path.join("data", "logs", "vlm_plan")
SCENE_ROOT = os.path.join("data", "scenes")

VIDEO_TAG = "replay"
TIMELINE_FILENAME = "replay_timeline.jsonl"

ENV_ID = "PickClutterYCB-XArm7-v1"
CONTROL_MODE = "pd_ee_delta_pose"
OBS_MODE = "rgb"
VIDEO_FPS = 30

OVERWRITE_OUTPUT = True   
FAIL_FAST = False

# =========================================================
# Utils
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


def _scene_triplet(scene_json: str) -> Dict[str, str]:
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
        return move.move_to(str(args["name"]), render=False, verbose=False)
    if action == "lower":
        return move.lower(render=False, verbose=False)
    if action == "lift":
        return move.lift(render=False, verbose=False)
    if action == "set_pose":
        return pose.set_pose(str(args["pose"]), render=False)
    if action == "pull":
        return pull.pull(str(args["side"]), float(args["dist_m"]), render=False, verbose=False)
    if action == "push":
        return push.push(str(args["side"]), float(args["dist_m"]), render=False, verbose=False)
    if action == "initarm":
        return init.initarm("", render=False, verbose=False)
    if action == "inithand":
        return init.inithand(render=False, verbose=False)
    if action == "grasp":
        return grasp.grasp(render=False)
    if action in ("env_reset", "done"):
        return None
    raise RuntimeError(f"Unknown action in log: {action!r}")


def _maybe_clear_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        return
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        try:
            if os.path.isfile(full) or os.path.islink(full):
                os.unlink(full)
            elif os.path.isdir(full):
                import shutil
                shutil.rmtree(full)
        except Exception:
            # don't crash on cleanup
            pass


def _make_env_with_record(scene_json: str, out_dir: str):
    env = gym.make(
        ENV_ID,
        num_envs=1,
        render_mode="rgb_array",
        control_mode=CONTROL_MODE,
        mode="plan",
        reconfiguration_freq=0,
        obs_mode=OBS_MODE,
        scene_json=scene_json,
        only_target_object=False,
        use_external_arm_init=True,
    )

    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        raise RuntimeError("Env has no metadata dict; cannot set render_fps.")
    env.metadata["render_fps"] = int(VIDEO_FPS)

    env = RecordEpisode(
        env,
        output_dir=out_dir,
        save_video=True,
        info_on_video=False,
    )
    return env


def render_one(log_dir: str, scene_json: str, out_dir: str) -> None:
    steps_path = os.path.join(log_dir, "steps.jsonl")
    if not os.path.exists(steps_path):
        raise FileNotFoundError(f"steps.jsonl not found: {steps_path}")
    if not os.path.exists(scene_json):
        raise FileNotFoundError(f"scene_json not found: {scene_json}")

    steps = _read_jsonl(steps_path)
    if not steps:
        raise RuntimeError(f"steps.jsonl empty: {steps_path}")

    os.makedirs(out_dir, exist_ok=True)
    if OVERWRITE_OUTPUT:
        _maybe_clear_dir(out_dir)

    env = _make_env_with_record(scene_json, out_dir)
    env.reset()

    timeline_path = os.path.join(out_dir, TIMELINE_FILENAME)
    f_tl = open(timeline_path, "w", encoding="utf-8")

    move = EEMoveSkill(env)
    pose = EEPoseSkill(env)
    pull = PullSkill(env)
    push = PushSkill(env)
    init = InitSkill(env)
    grasp = GraspSkill(env)

    step0 = _get_env_step_count(env)
    wall0 = perf_counter()

    for rec in steps:
        action = str(rec["action"])
        args = rec.get("args", {}) or {}
        step_id = int(rec.get("step_id", -1))
        reason = str(rec.get("reason", ""))

        # time BEFORE executing the action (video timeline)
        cur_steps = _get_env_step_count(env)
        t_video_s = float(cur_steps - step0) / float(VIDEO_FPS)
        t_video_ms = int(round(t_video_s * 1000.0))
        t_wall_s = float(perf_counter() - wall0)

        if action == "env_reset":
            call = _call_string(action, args)

            env.reset()

            # rebase time origin after reset
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

        if action == "grasp":
            grasp_success = (ok is True) and (err == "none") and (msg == "success")
            if grasp_success:
                # timestamp for done() (nice-to-have; does not change other logic)
                cur_steps2 = _get_env_step_count(env)
                t_video_s2 = float(cur_steps2 - step0) / float(VIDEO_FPS)
                t_video_ms2 = int(round(t_video_s2 * 1000.0))
                t_wall_s2 = float(perf_counter() - wall0)

                f_tl.write(json.dumps({
                    "t_video_s": float(t_video_s2),
                    "t_video_ms": int(t_video_ms2),
                    "t_wall_s": float(t_wall_s2),
                    "step_id": step_id,
                    "call": "done()",
                    "reason": "auto_done_after_grasp_success",
                }, ensure_ascii=False) + "\n")
                f_tl.flush()
                break

    f_tl.close()
    env.close()


def main() -> None:
    ensure_data_dirs()
    data_paths = get_data_paths()

    total = 0
    done = 0
    skipped = 0
    failed = 0

    for scene_name in SCENE_LIST:
        for clutter in CLUTTER_LIST:
            for sid in SCENE_ID_LIST:
                total += 1

                log_dir = os.path.join(LOG_ROOT, scene_name, str(int(clutter)), str(int(sid)))
                steps_path = os.path.join(log_dir, "steps.jsonl")
                scene_json = os.path.join(SCENE_ROOT, scene_name, str(int(clutter)), f"{int(sid)}.json")

                out_dir = os.path.join(
                    data_paths.videos,
                    VIDEO_TAG,
                    scene_name,
                    str(int(clutter)),
                    str(int(sid)),
                )

                if not os.path.exists(steps_path):
                    print(f"[SKIP] missing steps.jsonl: {steps_path}")
                    skipped += 1
                    continue
                if not os.path.exists(scene_json):
                    print(f"[SKIP] missing scene_json: {scene_json}")
                    skipped += 1
                    continue

                try:
                    print(f"[RUN] scene={scene_name} clutter={clutter} id={sid}")
                    render_one(log_dir=log_dir, scene_json=scene_json, out_dir=out_dir)
                    print(f"[OK ] saved: {out_dir}")
                    done += 1
                except Exception as e:
                    failed += 1
                    print(
                        f"[FAIL] scene={scene_name} clutter={clutter} id={sid} "
                        f"err={type(e).__name__}: {e}"
                    )
                    if FAIL_FAST:
                        raise

    print("\n========== SUMMARY ==========")
    print(f"scenes: {SCENE_LIST}")
    print(f"total combos: {total}")
    print(f"rendered:     {done}")
    print(f"skipped:      {skipped}")
    print(f"failed:       {failed}")
    print("================================")


if __name__ == "__main__":
    main()
