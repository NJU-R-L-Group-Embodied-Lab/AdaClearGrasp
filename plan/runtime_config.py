# plan/runtime_config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 800
    image_detail: str = "low"


@dataclass(frozen=True)
class MCPConfig:
    command: str = "python"
    server_script: str = "exec/mcp_server.py"


@dataclass(frozen=True)
class PlanConfig:
    max_steps: int = 80


@dataclass(frozen=True)
class RuntimeConfig:
    openai: OpenAIConfig = OpenAIConfig()
    mcp: MCPConfig = MCPConfig()
    plan: PlanConfig = PlanConfig()


def load_runtime_config(path: str) -> RuntimeConfig:
    abs_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(abs_path):
        return RuntimeConfig()

    with open(abs_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    openai_raw = raw.get("openai", {}) or {}
    mcp_raw = raw.get("mcp", {}) or {}
    plan_raw = raw.get("plan", {}) or {}

    return RuntimeConfig(
        openai=OpenAIConfig(**openai_raw),
        mcp=MCPConfig(**mcp_raw),
        plan=PlanConfig(**plan_raw),
    )
