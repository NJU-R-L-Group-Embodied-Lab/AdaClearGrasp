from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from PIL import Image

import env.sim.pick_clutter_xarm  # noqa: F401
from exec.skills.base import SkillResult
from exec.skills.clear import PullSkill, PushSkill
from exec.skills.ee_move import EEMoveSkill
from exec.skills.ee_pose import EEPoseSkill
from exec.skills.grasp import GraspSkill
from exec.skills.init import InitSkill

from .catalog import ENVIRONMENT_INSTRUCTIONS, public_action_catalog
from .config import ServiceConfig


def _model_label(model_id: str) -> str:
    text = str(model_id or "").strip()
    return text.split("_", 1)[1] if "_" in text and text.split("_", 1)[0].isdigit() else text


def _normalize_rgb(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    if isinstance(frame, (list, tuple)) and len(frame) == 1:
        frame = frame[0]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
    array = np.asarray(frame)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise RuntimeError(f"unsupported RGB frame shape: {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise RuntimeError(f"unsupported RGB channel count: {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _png_payload(frame: Any) -> dict[str, Any]:
    rgb = _normalize_rgb(frame)
    buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    raw = buffer.getvalue()
    return {
        "mime_type": "image/png",
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "byte_size": len(raw),
        "png_base64": base64.b64encode(raw).decode("ascii"),
    }


def _skill_payload(result: SkillResult) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "error_code": str(result.error_code),
        "message": str(result.message),
        "advice": str(result.advice),
    }


class ClutterBenchRuntime:
    """Own exactly one concrete simulator scene for the process lifetime."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._lock = RLock()
        self._env: Any | None = None
        self._move: EEMoveSkill | None = None
        self._pose: EEPoseSkill | None = None
        self._pull: PullSkill | None = None
        self._push: PushSkill | None = None
        self._init: InitSkill | None = None
        self._grasp: GraspSkill | None = None
        self._scene = self._load_scene(config.scene.scene_json)

    @staticmethod
    def _load_scene(path: Path) -> dict[str, Any]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("scene JSON root must be an object")
        target = raw.get("target")
        clutter = raw.get("clutter")
        if not isinstance(target, dict) or not isinstance(clutter, list):
            raise ValueError("scene JSON must contain target object and clutter list")
        return raw

    def start(self) -> None:
        with self._lock:
            if self._env is not None:
                return
            cfg = self.config.environment
            env = gym.make(
                cfg.env_id,
                num_envs=1,
                render_mode=cfg.render_mode,
                control_mode=cfg.control_mode,
                mode=cfg.mode,
                reconfiguration_freq=cfg.reconfiguration_freq,
                obs_mode=cfg.obs_mode,
                scene_json=str(self.config.scene.scene_json),
                only_target_object=cfg.only_target_object,
                use_external_arm_init=cfg.use_external_arm_init,
            )
            env.reset()
            self._env = env
            self._move = EEMoveSkill(env)
            self._pose = EEPoseSkill(env)
            self._pull = PullSkill(env)
            self._push = PushSkill(env)
            self._init = InitSkill(env)
            self._grasp = GraspSkill(env)

    def close(self) -> None:
        with self._lock:
            if self._env is not None:
                self._env.close()
            self._env = None
            self._move = None
            self._pose = None
            self._pull = None
            self._push = None
            self._init = None
            self._grasp = None

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("clutter bench runtime is not started")
        return self._env

    def _objects(self) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        target = dict(self._scene["target"])
        items.append(
            {
                "actor_name": str(target.get("name") or "target"),
                "label": _model_label(str(target.get("model_id") or "target")),
                "model_id": str(target.get("model_id") or ""),
                "role": "target",
            }
        )
        for item in self._scene["clutter"]:
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "actor_name": str(item.get("name") or item.get("model_id") or "clutter"),
                    "label": _model_label(str(item.get("model_id") or item.get("name") or "clutter")),
                    "model_id": str(item.get("model_id") or ""),
                    "role": "clutter",
                }
            )
        return items

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            self._require_env()
            return {
                "ok": True,
                "service": "clutter_bench_mcp",
                "ready": True,
                "scene_id": self.config.scene.scene_id,
                "fixed_scene": True,
                "session_state": False,
            }

    def environment_manifest(self) -> dict[str, Any]:
        with self._lock:
            self._require_env()
            objects = self._objects()
            return {
                "ok": True,
                "service": "clutter_bench_mcp",
                "display_name": "Clutter Bench Fixed Scene",
                "scene_id": self.config.scene.scene_id,
                "fixed_scene": True,
                "session_state": False,
                "automatic_latest_frame": True,
                "target": next(item for item in objects if item["role"] == "target"),
                "objects": objects,
                "instructions": ENVIRONMENT_INSTRUCTIONS,
                "actions": public_action_catalog(),
                "safety": {
                    "owner": "mcp",
                    "review_before_model": False,
                    "review_after_model_tool_call": True,
                    "frozen_tool_name_and_arguments": True,
                    "one_approval_per_tool_call": True,
                    "action_review_required": True,
                },
            }

    def scene_info(self) -> dict[str, Any]:
        with self._lock:
            self._require_env()
            objects = self._objects()
            return {
                "ok": True,
                "scene_id": self.config.scene.scene_id,
                "target": next(item for item in objects if item["role"] == "target"),
                "objects": objects,
            }

    def list_objects(self) -> dict[str, Any]:
        with self._lock:
            self._require_env()
            objects = self._objects()
            return {
                "ok": True,
                "scene_id": self.config.scene.scene_id,
                "objects": [item["actor_name"] for item in objects],
                "object_descriptors": objects,
            }

    def render_rgb(self) -> dict[str, Any]:
        with self._lock:
            frame = self._require_env().render()
            if frame is None:
                raise RuntimeError("environment renderer returned no RGB frame")
            return _png_payload(frame)

    def observe(self, include_rgb: bool = True) -> dict[str, Any]:
        with self._lock:
            objects = self._objects()
            result: dict[str, Any] = {
                "ok": True,
                "scene_id": self.config.scene.scene_id,
                "objects": [item["actor_name"] for item in objects],
                "object_descriptors": objects,
            }
            if include_rgb:
                frame = self._require_env().render()
                if frame is None:
                    raise RuntimeError("environment renderer returned no RGB frame")
                result["rgb"] = _png_payload(frame)
            return result

    def _call_skill(self, name: str, operation: Callable[[], SkillResult]) -> dict[str, Any]:
        with self._lock:
            self._require_env()
            result = operation()
            if not isinstance(result, SkillResult):
                raise TypeError(f"{name} returned {type(result).__name__}, expected SkillResult")
            return {
                "action": name,
                "scene_id": self.config.scene.scene_id,
                **_skill_payload(result),
            }

    def move_to(self, name: str) -> dict[str, Any]:
        return self._call_skill("move_to", lambda: self._move.move_to(str(name)))  # type: ignore[union-attr]

    def lift(self) -> dict[str, Any]:
        return self._call_skill("lift", lambda: self._move.lift())  # type: ignore[union-attr]

    def lower(self) -> dict[str, Any]:
        return self._call_skill("lower", lambda: self._move.lower())  # type: ignore[union-attr]

    def set_pose(self, pose: str) -> dict[str, Any]:
        normalized = str(pose).strip().lower()
        if normalized not in {"flat", "work"}:
            raise ValueError("pose must be 'flat' or 'work'")
        return self._call_skill("set_pose", lambda: self._pose.set_pose(normalized))  # type: ignore[arg-type,union-attr]

    def push(self, side: str, dist_m: float) -> dict[str, Any]:
        return self._call_skill(
            "push",
            lambda: self._push.push(str(side), float(dist_m)),  # type: ignore[arg-type,union-attr]
        )

    def pull(self, side: str, dist_m: float) -> dict[str, Any]:
        return self._call_skill(
            "pull",
            lambda: self._pull.pull(str(side), float(dist_m)),  # type: ignore[arg-type,union-attr]
        )

    def initarm(self) -> dict[str, Any]:
        return self._call_skill("initarm", lambda: self._init.initarm())  # type: ignore[union-attr]

    def inithand(self) -> dict[str, Any]:
        return self._call_skill("inithand", lambda: self._init.inithand())  # type: ignore[union-attr]

    def grasp(self) -> dict[str, Any]:
        return self._call_skill("grasp", lambda: self._grasp.grasp())  # type: ignore[union-attr]

    def reset(self) -> dict[str, Any]:
        with self._lock:
            env = self._require_env()
            env.reset()
            return {
                "ok": True,
                "action": "reset",
                "scene_id": self.config.scene.scene_id,
                "message": "固定场景已恢复到初始状态。",
                "error_code": "none",
                "advice": "",
            }


__all__ = ["ClutterBenchRuntime", "_normalize_rgb", "_png_payload"]
