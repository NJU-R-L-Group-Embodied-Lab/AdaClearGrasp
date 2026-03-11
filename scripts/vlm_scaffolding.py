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
import sapien  # noqa: F401

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import env.sim.pick_clutter_xarm  # noqa: F401

from exec.skills.ee_move import EEMoveSkill
from exec.skills.init import InitSkill
from exec.skills.grasp import GraspSkill

from mani_skill.utils.wrappers.record import RecordEpisode
from core.paths import ensure_data_dirs, get_data_paths

from mani_skill.utils.structs import Pose
from mani_skill.utils.building import actors

# =========================================================
# Global params (edit here)
# =========================================================
LOG_DIR = "data/logs/vlm_scaffolding/apple/4/1"  # baseline run_dir
MODE = "window"  # "window" | "video"

ENV_ID = "PickClutterYCB-XArm7-v1"
CONTROL_MODE = "pd_ee_delta_pose"
OBS_MODE = "rgb"

VIDEO_FPS = 30
SLEEP_IN_WINDOW = True

VIDEO_ROOT_TAG = "vlm_scaffolding"

SCENE_JSON_OVERRIDE = ""  

TIMELINE_FILENAME = "replay_timeline.jsonl"

DO_LIFT_FIRST = False
DO_GRASP_AT_END = False

MARK_TARGET_BALL = True
MARKER_RADIUS = 0.04  # meters
MARKER_Z = 0.03  # meters above tabletop
MARKER_RGBA = [1.0, 1.0, 1.0, 1.0]  # red semi-transparent


# =========================================================
# Helpers
# =========================================================
def _ensure_mode() -> None:
    if MODE not in ("window", "video"):
        raise RuntimeError(f"MODE must be 'window' or 'video', got: {MODE}")


def _infer_scene_json_from_log_dir(log_dir: str) -> str:
    # expected: data/logs/vlm_scaffolding/<scene>/<clutter>/<scene_id>
    parts = os.path.normpath(log_dir).split(os.sep)
    if len(parts) < 4:
        raise RuntimeError(f"LOG_DIR too short to infer scene: {log_dir}")
    scene_name = parts[-3]
    clutter = parts[-2]
    scene_id = parts[-1]
    return os.path.join("data", "scenes", scene_name, clutter, f"{scene_id}.json")


def _scene_triplet_from_scene_json(scene_json: str) -> Dict[str, str]:
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
            VIDEO_ROOT_TAG,
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


def _read_xyz(log_dir: str) -> List[List[float]]:
    traj_path = os.path.join(log_dir, "outputs", "traj_xyz.json")
    if not os.path.exists(traj_path):
        raise RuntimeError(f"traj_xyz.json not found: {traj_path}")
    with open(traj_path, "r", encoding="utf-8") as f:
        xyz = json.load(f)
    if not isinstance(xyz, list) or not xyz:
        raise RuntimeError(f"traj_xyz.json must be non-empty list, got: {type(xyz)}")
    for i, p in enumerate(xyz):
        if not isinstance(p, list) or len(p) != 3:
            raise RuntimeError(f"xyz[{i}] must be [x,y,z], got: {p!r}")
    return xyz


def _read_objects_xy_from_log_dir(log_dir: str) -> List[Dict[str, Any]]:
    p = os.path.join(log_dir, "inputs", "objects_xy.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"objects_xy.json must be a list, got: {type(obj)}")
    return obj


def _read_target_xy(scene_json: str, log_dir: str) -> tuple[float, float]:
    # Prefer logged inputs (exactly what you fed to the model)
    objs = _read_objects_xy_from_log_dir(log_dir)
    for it in objs:
        if isinstance(it, dict) and it.get("name") == "target":
            xy = it.get("xy")
            if isinstance(xy, list) and len(xy) == 2:
                return float(xy[0]), float(xy[1])

    # Fallback: read from scene json
    with open(scene_json, "r", encoding="utf-8") as f:
        scene = json.load(f)
    tgt = scene.get("target")
    if not isinstance(tgt, dict) or tgt.get("name") != "target":
        raise RuntimeError("scene_json target missing or target.name != 'target'")
    xy = tgt.get("xy")
    if not isinstance(xy, list) or len(xy) != 2:
        raise RuntimeError(f"scene_json target.xy must be [x,y], got {xy!r}")
    return float(xy[0]), float(xy[1])


def _make_sphere(scene: Any, p_xyz, radius: float, rgba, name: str):
    a = actors.build_sphere(
        scene,
        radius=float(radius),
        color=rgba,
        name=name,
        body_type="kinematic",
        add_collision=False,
        initial_pose=Pose.create_from_pq(
            [float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])],
            [1, 0, 0, 0],
        ),
    )
    return a


def _set_actor_pose(actor: Any, p_xyz) -> None:
    actor.set_pose(
        Pose.create_from_pq(
            [float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])],
            [1, 0, 0, 0],
        )
    )

def main() -> None:
    xyz = _read_xyz(LOG_DIR)

    scene_json = SCENE_JSON_OVERRIDE or _infer_scene_json_from_log_dir(LOG_DIR)
    if not os.path.exists(scene_json):
        raise RuntimeError(f"scene_json not found: {scene_json}")

    dt = 1.0 / float(VIDEO_FPS)
    env, video_out_dir = _make_env(scene_json)
    env.reset()

    ms_env = env.unwrapped
    scene = ms_env.scene

    timeline_dir = video_out_dir if (MODE == "video" and video_out_dir is not None) else LOG_DIR
    os.makedirs(timeline_dir, exist_ok=True)
    timeline_path = os.path.join(timeline_dir, TIMELINE_FILENAME)
    f_tl = open(timeline_path, "w", encoding="utf-8")

    move = EEMoveSkill(env)
    init = InitSkill(env)
    grasp = GraspSkill(env)

    step0 = _get_env_step_count(env)
    wall0 = perf_counter()

    def log_event(tag: str, extra: Dict[str, Any]) -> None:
        cur_steps = _get_env_step_count(env)
        t_video_s = float(cur_steps - step0) / float(VIDEO_FPS)
        t_video_ms = int(round(t_video_s * 1000.0))
        t_wall_s = float(perf_counter() - wall0)
        obj = {
            "t_video_s": float(t_video_s),
            "t_video_ms": int(t_video_ms),
            "t_wall_s": float(t_wall_s),
            "tag": str(tag),
        }
        obj.update(extra)
        f_tl.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f_tl.flush()

    # baseline parity: reset once
    env.reset()
    _render_tick(env, dt)
    step0 = _get_env_step_count(env)
    wall0 = perf_counter()
    log_event("env_reset", {})

    # Marker (target position ball)
    marker_actor = None
    if MARK_TARGET_BALL:
        tx, ty = _read_target_xy(scene_json, LOG_DIR)
        marker_actor = _make_sphere(scene, [tx, ty, float(MARKER_Z)], MARKER_RADIUS, MARKER_RGBA, "__target_marker__")
        log_event("marker_spawn", {"target_xy": [tx, ty], "marker_z": float(MARKER_Z)})
        _render_tick(env, dt)

    if DO_LIFT_FIRST:
        res = move.lift(render=True, verbose=False)
        log_event("lift", {"ok": res.ok, "error_code": res.error_code, "message": res.message})

    log_event("traj_begin", {"n_waypoints": len(xyz)})

    # Follow in REPLAY (not in skill): per-waypoint timeline
    for wi, p in enumerate(xyz):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        log_event("waypoint_start", {"waypoint_i": wi, "target_xyz": [x, y, z]})

        res = move.move_to_xyz(x, y, z, render=True, verbose=False)

        log_event(
            "waypoint_end",
            {
                "waypoint_i": wi,
                "target_xyz": [x, y, z],
                "ok": res.ok,
                "error_code": res.error_code,
                "message": res.message,
            },
        )

        print(f"[REPLAY] waypoint={wi:02d} target=({x:+.4f},{y:+.4f},{z:+.4f}) ok={res.ok} err={res.error_code} msg={res.message}")

        _render_tick(env, dt)

        if not res.ok:
            log_event("waypoint_failed", {"at_waypoint_i": wi})
            # DO NOT break; continue to next waypoint
            continue

    log_event("traj_end", {})

    if DO_GRASP_AT_END:
        resg = grasp.grasp(render=True)
        log_event("grasp", {"ok": resg.ok, "error_code": resg.error_code, "message": resg.message})
        _render_tick(env, dt)

    f_tl.close()
    env.close()

    print("[OK] Replay finished.")
    print(f"[OK] Timeline saved: {timeline_path}")
    if MODE == "video" and video_out_dir is not None:
        print(f"[OK] Video folder: {video_out_dir}")


if __name__ == "__main__":
    main()
