from __future__ import annotations

import argparse
import atexit
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .catalog import ACTION_CATALOG, ENVIRONMENT_INSTRUCTIONS, action_meta
from .config import PROJECT_ROOT, load_config
from .runtime import ClutterBenchRuntime
from .safety import SafetyReviewStore


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "clutter_bench_mcp.yaml"
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_AUDIT_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
_RESET = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)


def _normalize_execution_result(result: dict[str, Any]) -> dict[str, Any]:
    """Treat controller-reported ``stuck`` as a non-fatal execution condition.

    Physical contact and temporary joint/TCP immobility are common during embodied
    manipulation. Preserve the controller signal for the next visual decision, but do
    not collapse the whole action or agent loop into a failure solely because of it.
    """

    normalized = dict(result or {})
    if str(normalized.get("error_code") or "").strip().casefold() != "stuck":
        return normalized
    low_level_ok = bool(normalized.get("ok", False))
    normalized["ok"] = True
    normalized["non_fatal"] = True
    normalized["warning_code"] = "stuck"
    normalized["low_level_ok"] = low_level_ok
    normalized["advice"] = str(
        normalized.get("advice")
        or "The controller reported stuck; inspect the latest frame before choosing the next action."
    )
    return normalized


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one fixed AdaClearGrasp clutter scene over MCP.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def build_server(runtime: ClutterBenchRuntime, safety: SafetyReviewStore) -> FastMCP:
    server_cfg = runtime.config.server
    mcp = FastMCP(
        "clutter_bench_mcp",
        instructions=ENVIRONMENT_INSTRUCTIONS,
        host=server_cfg.host,
        port=server_cfg.port,
        streamable_http_path=server_cfg.streamable_http_path,
        json_response=True,
        stateless_http=True,
        log_level=server_cfg.log_level,
    )

    @mcp.tool(annotations=_READ_ONLY, meta={"clutter_bench": {"agent_action": False}})
    def healthcheck() -> dict[str, Any]:
        """报告 MCP、固定场景和具体工具调用审核模块的就绪状态。"""
        result = runtime.healthcheck()
        result["safety_review"] = "mcp_frozen_tool_call"
        return result

    @mcp.tool(annotations=_READ_ONLY, meta={"clutter_bench": {"agent_action": False}})
    def environment_manifest() -> dict[str, Any]:
        """返回唯一固定环境的场景信息、Agent 指令和可用原子动作目录。"""
        return runtime.environment_manifest()

    @mcp.tool(annotations=_READ_ONLY, meta={"clutter_bench": {"agent_action": False}})
    def scene_info() -> dict[str, Any]:
        """描述固定场景目标和杂物。"""
        return runtime.scene_info()

    @mcp.tool(annotations=_READ_ONLY, meta={"clutter_bench": {"agent_action": False}})
    def list_objects() -> dict[str, Any]:
        """列出固定场景中的 actor 名称和语义物体描述。"""
        return runtime.list_objects()

    @mcp.tool(annotations=_READ_ONLY, meta={"clutter_bench": {"agent_action": False}})
    def render_rgb() -> dict[str, Any]:
        """返回当前 RGB 画面的 base64 PNG。"""
        return runtime.render_rgb()

    @mcp.tool(annotations=_READ_ONLY, meta={"clutter_bench": {"agent_action": False}})
    def observe(include_rgb: bool = True) -> dict[str, Any]:
        """返回当前物体清单以及可选的最新 RGB 画面。"""
        return runtime.observe(include_rgb=include_rgb)

    def propose_action(action_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        review = safety.create_action_review(action_name, arguments)
        return {
            "ok": True,
            "status": "pending_user_review",
            "executed": False,
            "message": "具体工具调用已由 MCP 冻结，等待用户确认。",
            "review": review,
        }

    def execute_frozen(action_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if action_name == "move_to":
            return runtime.move_to(str(arguments["name"]))
        if action_name == "lift":
            return runtime.lift()
        if action_name == "lower":
            return runtime.lower()
        if action_name == "set_pose":
            return runtime.set_pose(str(arguments["pose"]))
        if action_name == "push":
            return runtime.push(str(arguments["side"]), float(arguments["dist_m"]))
        if action_name == "pull":
            return runtime.pull(str(arguments["side"]), float(arguments["dist_m"]))
        if action_name == "initarm":
            return runtime.initarm()
        if action_name == "inithand":
            return runtime.inithand()
        if action_name == "grasp":
            return runtime.grasp()
        if action_name == "reset":
            return runtime.reset()
        raise ValueError(f"unknown frozen action: {action_name}")

    @mcp.tool(annotations=_AUDIT_WRITE, meta={"clutter_bench": {"agent_action": False, "safety_api": True}})
    def resolve_action_review(review_id: str, approved: bool) -> dict[str, Any]:
        """记录用户决定；同意时仅执行 MCP 已冻结的原始工具名和参数。"""
        decision = safety.decide(review_id, approved)
        action_call = dict(decision["action_call"])
        if not approved:
            return {
                "ok": True,
                "status": "denied",
                "executed": False,
                "message": "用户拒绝执行该工具调用。",
                "review": decision,
                "result": None,
            }
        try:
            result = _normalize_execution_result(
                execute_frozen(
                    str(action_call["name"]),
                    dict(action_call.get("arguments") or {}),
                )
            )
        except Exception as exc:  # noqa: BLE001 - MCP execution boundary
            result = {
                "ok": False,
                "action": str(action_call["name"]),
                "message": f"{type(exc).__name__}: {exc}",
                "error_code": "execution_exception",
            }
            review = safety.finish_execution(review_id, ok=False, result=result)
            return {
                "ok": False,
                "status": "failed",
                "executed": True,
                "message": str(result["message"]),
                "review": review,
                "result": result,
            }
        ok = bool(result.get("ok", True))
        review = safety.finish_execution(review_id, ok=ok, result=result)
        return {
            "ok": ok,
            "status": "completed" if ok else "failed",
            "executed": True,
            "message": str(result.get("message") or ""),
            "review": review,
            "result": result,
        }

    @mcp.tool(annotations=_READ_ONLY, meta={"clutter_bench": {"agent_action": False, "safety_api": True}})
    def action_review_status(review_id: str) -> dict[str, Any]:
        """读取一个被冻结工具调用的审核和执行状态。"""
        return safety.get(review_id)

    @mcp.tool(
        title=ACTION_CATALOG["move_to"]["title"],
        description=ACTION_CATALOG["move_to"]["description"],
        annotations=_MUTATING,
        meta=action_meta("move_to"),
    )
    def move_to(name: str) -> dict[str, Any]:
        return propose_action("move_to", {"name": name})

    @mcp.tool(
        title=ACTION_CATALOG["lift"]["title"],
        description=ACTION_CATALOG["lift"]["description"],
        annotations=_MUTATING,
        meta=action_meta("lift"),
    )
    def lift() -> dict[str, Any]:
        return propose_action("lift", {})

    @mcp.tool(
        title=ACTION_CATALOG["lower"]["title"],
        description=ACTION_CATALOG["lower"]["description"],
        annotations=_MUTATING,
        meta=action_meta("lower"),
    )
    def lower() -> dict[str, Any]:
        return propose_action("lower", {})

    @mcp.tool(
        title=ACTION_CATALOG["set_pose"]["title"],
        description=ACTION_CATALOG["set_pose"]["description"],
        annotations=_MUTATING,
        meta=action_meta("set_pose"),
    )
    def set_pose(pose: str) -> dict[str, Any]:
        return propose_action("set_pose", {"pose": pose})

    @mcp.tool(
        title=ACTION_CATALOG["push"]["title"],
        description=ACTION_CATALOG["push"]["description"],
        annotations=_MUTATING,
        meta=action_meta("push"),
    )
    def push(side: str, dist_m: float) -> dict[str, Any]:
        return propose_action("push", {"side": side, "dist_m": dist_m})

    @mcp.tool(
        title=ACTION_CATALOG["pull"]["title"],
        description=ACTION_CATALOG["pull"]["description"],
        annotations=_MUTATING,
        meta=action_meta("pull"),
    )
    def pull(side: str, dist_m: float) -> dict[str, Any]:
        return propose_action("pull", {"side": side, "dist_m": dist_m})

    @mcp.tool(
        title=ACTION_CATALOG["initarm"]["title"],
        description=ACTION_CATALOG["initarm"]["description"],
        annotations=_MUTATING,
        meta=action_meta("initarm"),
    )
    def initarm() -> dict[str, Any]:
        return propose_action("initarm", {})

    @mcp.tool(
        title=ACTION_CATALOG["inithand"]["title"],
        description=ACTION_CATALOG["inithand"]["description"],
        annotations=_MUTATING,
        meta=action_meta("inithand"),
    )
    def inithand() -> dict[str, Any]:
        return propose_action("inithand", {})

    @mcp.tool(
        title=ACTION_CATALOG["grasp"]["title"],
        description=ACTION_CATALOG["grasp"]["description"],
        annotations=_MUTATING,
        meta=action_meta("grasp"),
    )
    def grasp() -> dict[str, Any]:
        return propose_action("grasp", {})

    @mcp.tool(
        title=ACTION_CATALOG["reset"]["title"],
        description=ACTION_CATALOG["reset"]["description"],
        annotations=_RESET,
        meta=action_meta("reset"),
    )
    def reset() -> dict[str, Any]:
        return propose_action("reset", {})

    return mcp


def main() -> None:
    args = _args()
    config = load_config(Path(args.config))
    runtime = ClutterBenchRuntime(config)
    safety = SafetyReviewStore(
        audit_log=config.safety.audit_log,
        review_ttl_s=config.safety.review_ttl_s,
    )
    runtime.start()
    atexit.register(runtime.close)
    server = build_server(runtime, safety)
    server.run(transport=config.server.transport)


if __name__ == "__main__":
    main()
