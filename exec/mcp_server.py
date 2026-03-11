# exec/mcp_server.py
from __future__ import annotations

import os
import sys
import base64
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Callable

import argparse
import numpy as np
import gymnasium as gym
from PIL import Image
from mcp.server.fastmcp import FastMCP

# =========================================================
# Project path
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# ManiSkill env registration
# =========================================================
import env.sim.pick_clutter_xarm  # noqa: F401

# =========================================================
# Skills
# =========================================================
from exec.skills.ee_move import EEMoveSkill
from exec.skills.ee_pose import EEPoseSkill
from exec.skills.clear import PullSkill, PushSkill
from exec.skills.init import InitSkill
from exec.skills.grasp import GraspSkill
from exec.skills.base import SkillResult

# =========================================================
# CLI args
# =========================================================
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_json", type=str, required=True)
    p.add_argument("--env_id", type=str, default="PickClutterYCB-XArm7-v1")
    p.add_argument("--control_mode", type=str, default="pd_ee_delta_pose")
    p.add_argument("--mode", type=str, default="plan")
    p.add_argument("--obs_mode", type=str, default="rgb")
    p.add_argument("--render_mode", type=str, default="rgb_array")
    p.add_argument("--only_target_object", action="store_true", default=False)
    p.add_argument("--use_external_arm_init", action="store_true", default=True)
    p.add_argument("--reconfiguration_freq", type=int, default=0)
    return p.parse_args()


_ARGS = _parse_args()

# =========================================================
# Runtime config
# =========================================================
@dataclass(frozen=True)
class ExecRuntimeConfig:
    env_id: str = "PickClutterYCB-XArm7-v1"
    control_mode: str = "pd_ee_delta_pose"
    mode: str = "plan"
    obs_mode: str = "rgb"
    render_mode: str = "rgb_array"
    scene_json: str = ""
    only_target_object: bool = False
    use_external_arm_init: bool = True
    reconfiguration_freq: int = 0


CFG = ExecRuntimeConfig(
    env_id=_ARGS.env_id,
    control_mode=_ARGS.control_mode,
    mode=_ARGS.mode,
    obs_mode=_ARGS.obs_mode,
    render_mode=_ARGS.render_mode,
    scene_json=_ARGS.scene_json,
    only_target_object=bool(_ARGS.only_target_object),
    use_external_arm_init=bool(_ARGS.use_external_arm_init),
    reconfiguration_freq=int(_ARGS.reconfiguration_freq),
)

# =========================================================
# Globals
# =========================================================
_ENV = None
_MOVE: Optional[EEMoveSkill] = None
_POSE: Optional[EEPoseSkill] = None
_PULL: Optional[PullSkill] = None
_PUSH: Optional[PushSkill] = None
_INIT: Optional[InitSkill] = None
_GRASP: Optional[GraspSkill] = None

# =========================================================
# Utils
# =========================================================
def _rgb_to_png_base64(rgb_u8: np.ndarray) -> str:
    img = Image.fromarray(rgb_u8, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ensure_runtime() -> None:
    global _ENV, _MOVE, _POSE, _PULL, _PUSH, _INIT, _GRASP

    if _ENV is not None:
        return

    if not os.path.exists(CFG.scene_json):
        raise RuntimeError(f"scene_json not found: {CFG.scene_json}")

    _ENV = gym.make(
        CFG.env_id,
        num_envs=1,
        render_mode=CFG.render_mode,
        control_mode=CFG.control_mode,
        mode=CFG.mode,
        reconfiguration_freq=CFG.reconfiguration_freq,
        obs_mode=CFG.obs_mode,
        scene_json=CFG.scene_json,
        only_target_object=CFG.only_target_object,
        use_external_arm_init=CFG.use_external_arm_init,
    )
    _ENV.reset()

    _MOVE = EEMoveSkill(_ENV)
    _POSE = EEPoseSkill(_ENV)
    _PULL = PullSkill(_ENV)
    _PUSH = PushSkill(_ENV)
    _INIT = InitSkill(_ENV)
    _GRASP = GraspSkill(_ENV)


def _get_scene_actor_names() -> List[str]:
    _ensure_runtime()
    scene = _ENV.unwrapped.scene
    actors = getattr(scene, "actors")
    if not isinstance(actors, dict):
        raise RuntimeError(f"scene.actors expected dict, got {type(actors)}")
    return sorted(list(actors.keys()))


def _skill_payload(res: SkillResult) -> Dict[str, Any]:
    return {
        "ok": bool(res.ok),
        "error_code": str(res.error_code),
        "message": str(res.message),
        "advice": str(res.advice),
    }


def _strict_call(skill_fn: Callable[[], Any], *, skill_name: str) -> SkillResult:
    out = skill_fn()
    if not isinstance(out, SkillResult):
        raise TypeError(f"{skill_name} must return SkillResult, got {type(out)}: {out!r}")
    return out


# =========================================================
# MCP server
# =========================================================
mcp = FastMCP("xarm_exec_server")


# ----------------------------
# Env tools
# ----------------------------
@mcp.tool()
def env_reset() -> Dict[str, Any]:
    _ensure_runtime()
    _ENV.reset()
    return {"ok": True}


@mcp.tool()
def list_objects() -> Dict[str, Any]:
    names = _get_scene_actor_names()
    return {"ok": True, "objects": names}


@mcp.tool()
def render_rgb() -> Dict[str, Any]:
    _ensure_runtime()
    frame = _ENV.render()

    if frame is None:
        raise RuntimeError(
            "env.render() returned None. "
            "Set configs/exec_runtime.yaml render_mode: rgb_array to enable images."
        )

    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()

    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f"Unexpected frame shape: {frame.shape}")

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    png_b64 = _rgb_to_png_base64(frame)
    h, w = frame.shape[:2]
    return {"png_base64": png_b64, "width": int(w), "height": int(h)}


# ----------------------------
# Skill tools
# ----------------------------
@mcp.tool()
def move_to(name: str, render: bool = False, verbose: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _MOVE.move_to(str(name), render=bool(render), verbose=bool(verbose)),
        skill_name="move_to",
    )
    return _skill_payload(res)


@mcp.tool()
def lower(render: bool = False, verbose: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _MOVE.lower(render=bool(render), verbose=bool(verbose)),
        skill_name="lower",
    )
    return _skill_payload(res)


@mcp.tool()
def lift(render: bool = False, verbose: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _MOVE.lift(render=bool(render), verbose=bool(verbose)),
        skill_name="lift",
    )
    return _skill_payload(res)


@mcp.tool()
def set_pose(pose: str, render: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _POSE.set_pose(str(pose), render=bool(render)),
        skill_name="set_pose",
    )
    return _skill_payload(res)


@mcp.tool()
def pull(side: str, dist_m: float, render: bool = False, verbose: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _PULL.pull(str(side), float(dist_m), render=bool(render), verbose=bool(verbose)),
        skill_name="pull",
    )
    return _skill_payload(res)


@mcp.tool()
def push(side: str, dist_m: float, render: bool = False, verbose: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _PUSH.push(str(side), float(dist_m), render=bool(render), verbose=bool(verbose)),
        skill_name="push",
    )
    return _skill_payload(res)


@mcp.tool()
def initarm(render: bool = False, verbose: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _INIT.initarm("", render=bool(render), verbose=bool(verbose)),
        skill_name="initarm",
    )
    return _skill_payload(res)


@mcp.tool()
def inithand(render: bool = False, verbose: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _INIT.inithand(render=bool(render), verbose=bool(verbose)),
        skill_name="inithand",
    )
    return _skill_payload(res)


@mcp.tool()
def grasp(render: bool = False) -> Dict[str, Any]:
    _ensure_runtime()
    res = _strict_call(
        lambda: _GRASP.grasp(render=bool(render)),
        skill_name="grasp",
    )
    return _skill_payload(res)


@mcp.tool()
def close() -> Dict[str, Any]:
    global _ENV, _MOVE, _POSE, _PULL, _PUSH, _INIT, _GRASP

    if _ENV is not None:
        _ENV.close()

    _ENV = None
    _MOVE = None
    _POSE = None
    _PULL = None
    _PUSH = None
    _INIT = None
    _GRASP = None

    return {"ok": True}


if __name__ == "__main__":
    mcp.run()
