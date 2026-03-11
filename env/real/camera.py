# env/real/camera.py
from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

@dataclass
class PoseServerConfig:
    server_url: str = os.environ.get("POSE_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
    timeout_s: float = float(os.environ.get("POSE_SERVER_TIMEOUT", "1.5"))

    z_margin: float = float(os.environ.get("POSE_Z_MARGIN", "0.003"))
    pos_scale: float = float(os.environ.get("POSE_POS_SCALE", "1000.0"))

    # /poses query flags
    include_camera_pose: bool = os.environ.get("POSE_INCLUDE_CAMERA_POSE", "0") == "1"
    include_target_pose: bool = os.environ.get("POSE_INCLUDE_TARGET_POSE", "1") != "0"
    include_objects: bool = os.environ.get("POSE_INCLUDE_OBJECTS", "1") != "0"

    # small retry to hide occasional 503 / transient issues
    retries: int = int(os.environ.get("POSE_SERVER_RETRIES", "1"))
    retry_backoff_s: float = float(os.environ.get("POSE_SERVER_RETRY_BACKOFF", "0.05"))

    # NEVER use env proxies (fix curl/proxy class of issues)
    disable_env_proxy: bool = os.environ.get("POSE_DISABLE_ENV_PROXY", "1") != "0"

    # yaw extraction behavior
    yaw_from: str = os.environ.get("POSE_YAW_FROM", "base")  # "base" or "cam"
    yaw_axis: str = os.environ.get("POSE_YAW_AXIS", "z")    # only "z" supported here

    # prefer server-mapped display names (object.txt mapping)
    prefer_server_mapped_names: bool = os.environ.get("POSE_PREFER_SERVER_NAMES", "1") != "0"

    # prefer which frame for xy / pose
    prefer_frame: str = os.environ.get("POSE_PREFER_FRAME", "base")  # "base" or "cam"


def _quat_to_yaw_z_xyzw(q: List[float]) -> float:
    """
    yaw around Z axis from quaternion (x,y,z,w).
    yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    """
    x, y, z, w = map(float, q)
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)

def _pick_pose_dict(obj: Dict[str, Any], frame: str) -> Optional[Dict[str, Any]]:
    """
    obj is one entry from /poses output:
      {
        "name": "...",
        "key": "target" or "object_1",
        "cam": {"xyz": [...], "quat_xyzw": [...], "T_4x4": [...]},
        "base": {...} or None,
        ...
      }
    """
    d = obj.get(frame, None)
    if not isinstance(d, dict):
        return None
    if "xyz" not in d or "quat_xyzw" not in d:
        return None
    return d

def _extract_xy_from_pose(p: Dict[str, Any]) -> Tuple[float, float]:
    xyz = p["xyz"]
    return float(xyz[0]), float(xyz[1])

def _extract_yaw_from_pose(p: Dict[str, Any]) -> float:
    q = p["quat_xyzw"]
    return _quat_to_yaw_z_xyzw(q)


def _scale_xyz_list(xyz: Any, scale: float) -> Any:
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
        return xyz
    return [float(xyz[0]) * scale, float(xyz[1]) * scale, float(xyz[2]) * scale]


def _scale_T_4x4(T: Any, scale: float) -> Any:
    if scale == 1.0:
        return T

    # nested 4x4
    if isinstance(T, list) and len(T) == 4 and all(isinstance(r, list) and len(r) == 4 for r in T):
        out = [[float(v) for v in row] for row in T]
        out[0][3] = out[0][3] * scale
        out[1][3] = out[1][3] * scale
        out[2][3] = out[2][3] * scale
        return out

    # flat 16
    if isinstance(T, list) and len(T) == 16:
        out = [float(v) for v in T]
        out[3] = out[3] * scale
        out[7] = out[7] * scale
        out[11] = out[11] * scale
        return out

    return T


def _scale_pose_dict(p: Any, scale: float) -> Any:
    if not isinstance(p, dict):
        return p
    out = dict(p)
    if "xyz" in out:
        out["xyz"] = _scale_xyz_list(out["xyz"], scale)
    if "T_4x4" in out and out["T_4x4"] is not None:
        out["T_4x4"] = _scale_T_4x4(out["T_4x4"], scale)
    return out


def _scale_obj_entry(obj: Any, scale: float) -> Any:
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "cam" in out:
        out["cam"] = _scale_pose_dict(out.get("cam"), scale)
    if "base" in out:
        out["base"] = _scale_pose_dict(out.get("base"), scale)
    return out


def _scale_raw_poses(raw: Any, scale: float) -> Any:
    if scale == 1.0 or not isinstance(raw, dict):
        return raw

    out = dict(raw)

    # target
    if "target" in out:
        out["target"] = _scale_obj_entry(out.get("target"), scale)

    # objects
    objs = out.get("objects", None)
    if isinstance(objs, list):
        out["objects"] = [_scale_obj_entry(o, scale) for o in objs]

    # camera pose fields（如果 server 带这些）
    if "camera_base" in out:
        out["camera_base"] = _scale_pose_dict(out.get("camera_base"), scale)
    if "camera_pose" in out:
        out["camera_pose"] = _scale_pose_dict(out.get("camera_pose"), scale)

    return out


# =========================
# Camera (network-backed), outward compatible with your FakeCamera-ish usage
# =========================

class Camera:

    def __init__(self, cfg: Optional[PoseServerConfig] = None):
        self.cfg = cfg or PoseServerConfig()
        self._sess = requests.Session()
        if self.cfg.disable_env_proxy:
            self._sess.trust_env = False

        # fallback key if server doesn't return target
        self._target_key_fallback = os.environ.get("POSE_TARGET_NAME", "target").strip() or "target"
        self._target_name_cached: Optional[str] = None

        self._static_clutter_csv = os.environ.get("POSE_CLUTTER_NAMES", "").strip()

        # scaled cache (mm)
        self._last_raw: Optional[Dict[str, Any]] = None
        self._last_fetch_ts: Optional[float] = None

        # optional: keep unscaled raw for debugging (meters)
        self._last_raw_unscaled: Optional[Dict[str, Any]] = None

    # ---------- basic ----------
    def z_margin(self) -> float:
        return float(self.cfg.z_margin) * float(self.cfg.pos_scale)

    def health(self) -> Dict[str, Any]:
        return self._request_json("/health")

    def fetch_poses(
        self,
        *,
        include_camera_pose: Optional[bool] = None,
        include_target_pose: Optional[bool] = None,
        include_objects: Optional[bool] = None,
    ) -> Dict[str, Any]:
        params = {
            "include_camera_pose": self.cfg.include_camera_pose if include_camera_pose is None else bool(include_camera_pose),
            "include_target_pose": self.cfg.include_target_pose if include_target_pose is None else bool(include_target_pose),
            "include_objects": self.cfg.include_objects if include_objects is None else bool(include_objects),
        }
        raw_unscaled = self._request_json("/poses", params=params)
        self._last_raw_unscaled = raw_unscaled

        raw = _scale_raw_poses(raw_unscaled, float(self.cfg.pos_scale))  # <- mm
        self._last_raw = raw
        self._last_fetch_ts = time.time()

        t = raw.get("target", None)
        if isinstance(t, dict) and t.get("name"):
            self._target_name_cached = str(t.get("name"))
        return raw

    def last_raw(self) -> Optional[Dict[str, Any]]:
        return self._last_raw

    def last_raw_unscaled(self) -> Optional[Dict[str, Any]]:
        return self._last_raw_unscaled

    # ---------- naming ----------
    def target_name(self) -> str:
        if not self.cfg.prefer_server_mapped_names:
            return self._target_key_fallback
        raw = self._ensure_latest()
        t = raw.get("target", None)
        if isinstance(t, dict) and t.get("name"):
            self._target_name_cached = str(t.get("name"))
            return self._target_name_cached
        return self._target_name_cached or self._target_key_fallback

    def list_clutter_names(self) -> List[str]:
        if self._static_clutter_csv:
            return [s.strip() for s in self._static_clutter_csv.split(",") if s.strip()]
        raw = self._ensure_latest()
        objs = raw.get("objects", []) or []
        out: List[str] = []
        for o in objs:
            nm = str(o.get("name", "")).strip()
            if nm:
                out.append(nm)
        return out

    # ---------- lookup ----------
    def has(self, name_or_key: str) -> bool:
        raw = self._ensure_latest()
        return self._find_obj(raw, name_or_key) is not None

    def get(self, name_or_key: str) -> Dict[str, Any]:
        raw = self._ensure_latest()
        obj = self._find_obj(raw, name_or_key)
        if obj is None:
            raise RuntimeError(
                f"Camera: unknown object '{name_or_key}' (not in latest /poses). "
                f"Tips: open /debug/frame overlay or print Camera().fetch_poses()."
            )
        return obj

    def get_target_pose(self, *, frame: str = "base") -> Dict[str, Any]:
        raw = self._ensure_latest()
        t = raw.get("target", None)
        if not isinstance(t, dict):
            raise RuntimeError("Camera: /poses has no 'target' (target not detected or include_target_pose=false)")

        prefer = (frame or "base").strip().lower()
        p = _pick_pose_dict(t, prefer)
        if p is None:
            other = "cam" if prefer == "base" else "base"
            p = _pick_pose_dict(t, other)

        if p is None:
            raise RuntimeError("Camera: target has no pose dict in /poses (cam/base both missing)")

        return {
            "xyz": list(map(float, p["xyz"])),              
            "quat_xyzw": list(map(float, p["quat_xyzw"])),
            "T_4x4": p.get("T_4x4", None),                 
            "frame": ("base" if _pick_pose_dict(t, "base") is p else "cam"),
            "name": t.get("name"),
            "key": t.get("key"),
            "type": t.get("type"),
            "score": t.get("score"),
            "iou": t.get("iou"),
            "color": t.get("color"),
            "match": t.get("match"),
            "timestamp": raw.get("timestamp"),
            "unit": "mm",
            "pos_scale": float(self.cfg.pos_scale),
        }

    def get_object_xy(self, name_or_key: str, *, frame: str = "base") -> Tuple[float, float]:
        obj = self.get(name_or_key)
        prefer = (frame or self.cfg.prefer_frame or "base").strip().lower()

        p = _pick_pose_dict(obj, prefer)
        if p is None:
            other = "cam" if prefer == "base" else "base"
            p = _pick_pose_dict(obj, other)
        if p is None:
            raise RuntimeError(f"Camera: object '{name_or_key}' has no pose in /poses (cam/base missing)")
        return _extract_xy_from_pose(p)  # mm

    def get_xy(self, name_or_key: str) -> Tuple[float, float]:
        return self.get_object_xy(name_or_key, frame=self.cfg.prefer_frame)

    def get_yaw(self, name_or_key: str) -> float:
        obj = self.get(name_or_key)
        prefer = (self.cfg.yaw_from or "base").strip().lower()
        p = _pick_pose_dict(obj, prefer)
        if p is None:
            other = "cam" if prefer == "base" else "base"
            p = _pick_pose_dict(obj, other)
        if p is None:
            return 0.0
        return _extract_yaw_from_pose(p)

    # ---------- network core ----------
    def _request_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.cfg.server_url}{path}"
        last_err: Optional[Exception] = None

        for i in range(max(1, self.cfg.retries + 1)):
            try:
                r = self._sess.get(url, params=params, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if i < self.cfg.retries:
                    time.sleep(self.cfg.retry_backoff_s * (2 ** i))

        raise RuntimeError(
            f"Pose server request failed.\n"
            f"  url: {url}\n"
            f"  params: {params}\n"
            f"  server_url: {self.cfg.server_url}\n"
            f"  disable_env_proxy: {self.cfg.disable_env_proxy}\n"
            f"  err: {repr(last_err)}\n"
            f"Tips:\n"
            f"  - same machine: POSE_SERVER_URL=http://127.0.0.1:8000\n"
            f"  - other machine: POSE_SERVER_URL=http://<server_lan_ip>:8000\n"
            f"  - if you insist on using env proxies, set POSE_DISABLE_ENV_PROXY=0\n"
        )

    def _ensure_latest(self) -> Dict[str, Any]:
        if self._last_raw is None:
            return self.fetch_poses()
        return self._last_raw

    # ---------- object finding (supports name OR key) ----------
    @staticmethod
    def _match_obj(obj: Dict[str, Any], name_or_key: str) -> bool:
        if not isinstance(obj, dict):
            return False
        if obj.get("name", None) == name_or_key:
            return True
        if obj.get("key", None) == name_or_key:
            return True
        return False

    def _find_obj(self, raw: Dict[str, Any], name_or_key: str) -> Optional[Dict[str, Any]]:
        name_or_key = str(name_or_key)

        t = raw.get("target", None)
        if isinstance(t, dict) and self._match_obj(t, name_or_key):
            return t

        for o in (raw.get("objects", []) or []):
            if isinstance(o, dict) and self._match_obj(o, name_or_key):
                return o

        # convenience: fallback target key
        if name_or_key == self._target_key_fallback and isinstance(t, dict):
            if t.get("key", None) == self._target_key_fallback:
                return t

        return None


def get_camera(env=None) -> Camera:
    if env is not None and hasattr(env, "camera") and env.camera is not None:
        return env.camera
    return Camera()
