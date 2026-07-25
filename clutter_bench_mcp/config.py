from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class SceneConfig:
    scene_id: str
    scene_json: Path


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    env_id: str = "PickClutterYCB-XArm7-v1"
    control_mode: str = "pd_ee_delta_pose"
    mode: str = "plan"
    obs_mode: str = "rgb"
    render_mode: str = "rgb_array"
    only_target_object: bool = False
    use_external_arm_init: bool = True
    reconfiguration_freq: int = 0


@dataclass(frozen=True, slots=True)
class ServerConfig:
    transport: str = "streamable-http"
    host: str = "127.0.0.1"
    port: int = 8877
    streamable_http_path: str = "/mcp"
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    audit_log: Path
    review_ttl_s: int = 900


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    scene: SceneConfig
    environment: EnvironmentConfig
    server: ServerConfig
    safety: SafetyConfig


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _resolve_project_path(value: Any, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> ServiceConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    scene_raw = _mapping(root.get("scene"), "scene")
    env_raw = _mapping(root.get("environment"), "environment")
    server_raw = _mapping(root.get("server"), "server")
    safety_raw = _mapping(root.get("safety"), "safety")

    scene_json = _resolve_project_path(scene_raw.get("scene_json"), "scene.scene_json")
    if not scene_json.is_file():
        raise ValueError(f"scene file does not exist: {scene_json}")
    scene_id = str(scene_raw.get("scene_id") or scene_json.stem).strip()
    if not scene_id:
        raise ValueError("scene.scene_id must not be empty")

    environment = EnvironmentConfig(
        env_id=str(env_raw.get("env_id", "PickClutterYCB-XArm7-v1")),
        control_mode=str(env_raw.get("control_mode", "pd_ee_delta_pose")),
        mode=str(env_raw.get("mode", "plan")),
        obs_mode=str(env_raw.get("obs_mode", "rgb")),
        render_mode=str(env_raw.get("render_mode", "rgb_array")),
        only_target_object=bool(env_raw.get("only_target_object", False)),
        use_external_arm_init=bool(env_raw.get("use_external_arm_init", True)),
        reconfiguration_freq=int(env_raw.get("reconfiguration_freq", 0)),
    )
    if environment.render_mode != "rgb_array":
        raise ValueError("environment.render_mode must be rgb_array")
    if environment.reconfiguration_freq < 0:
        raise ValueError("environment.reconfiguration_freq must be >= 0")

    transport = str(server_raw.get("transport", "streamable-http"))
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(f"unsupported server.transport: {transport}")
    path_value = str(server_raw.get("streamable_http_path", "/mcp"))
    if not path_value.startswith("/"):
        raise ValueError("server.streamable_http_path must start with '/'")
    server = ServerConfig(
        transport=transport,
        host=str(server_raw.get("host", "127.0.0.1")),
        port=int(server_raw.get("port", 8877)),
        streamable_http_path=path_value,
        log_level=str(server_raw.get("log_level", "INFO")).upper(),
    )
    if not 1 <= server.port <= 65535:
        raise ValueError("server.port must be between 1 and 65535")

    audit_log = _resolve_project_path(
        safety_raw.get("audit_log", "logs/clutter_bench_mcp_safety.jsonl"),
        "safety.audit_log",
    )
    review_ttl_s = int(safety_raw.get("review_ttl_s", 900))
    if review_ttl_s < 60:
        raise ValueError("safety.review_ttl_s must be >= 60")

    return ServiceConfig(
        scene=SceneConfig(scene_id=scene_id, scene_json=scene_json),
        environment=environment,
        server=server,
        safety=SafetyConfig(audit_log=audit_log, review_ttl_s=review_ttl_s),
    )
