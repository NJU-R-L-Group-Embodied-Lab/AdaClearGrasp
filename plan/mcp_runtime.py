# plan/mcp_runtime.py
from __future__ import annotations

import json
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


_TOOL_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _to_safe_tool_name(mcp_tool_name: str) -> str:
    safe = _TOOL_NAME_SAFE_RE.sub("__", mcp_tool_name)
    if len(safe) > 64:
        safe = safe[:64]
    if not safe:
        safe = "tool"
    return safe


@dataclass
class ToolMapping:
    safe_to_mcp: Dict[str, str]
    mcp_to_safe: Dict[str, str]


@dataclass
class StepLog:
    step_id: int
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any


class MCPRuntime:
    """
    MCP stdio runtime wrapper.

    Key feature: supports passing extra server CLI args (server_args),
    so the MCP server can be started like:
        python exec/mcp_server.py --scene_json data/scenes/apple/6/1.json
    """

    def __init__(self, server_command: str, server_script: str, server_args: Optional[List[str]] = None):
        self.server_command = server_command
        self.server_script = server_script
        self.server_args: List[str] = [] if server_args is None else list(server_args)

        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

        self.tool_mapping: Optional[ToolMapping] = None
        self.tools_openai_schema: Optional[List[Dict[str, Any]]] = None

        self.history: List[StepLog] = []

    async def __aenter__(self) -> "MCPRuntime":
        await self.connect()
        await self.build_tool_schemas()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.exit_stack.aclose()

    async def connect(self) -> None:
        args = [self.server_script] + list(self.server_args)

        params = StdioServerParameters(
            command=self.server_command,
            args=args,
            env=None,
        )
        stdio = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio[0], stdio[1]))
        await self.session.initialize()

    async def build_tool_schemas(self) -> None:
        assert self.session is not None

        tools = await self.session.list_tools()

        safe_to_mcp: Dict[str, str] = {}
        mcp_to_safe: Dict[str, str] = {}
        openai_tools: List[Dict[str, Any]] = []

        for t in tools.tools:
            mcp_name = t.name
            safe_name = _to_safe_tool_name(mcp_name)

            if safe_name in safe_to_mcp and safe_to_mcp[safe_name] != mcp_name:
                k = 1
                base = safe_name
                while safe_name in safe_to_mcp:
                    safe_name = (base[:60] + f"__{k}")[:64]
                    k += 1

            safe_to_mcp[safe_name] = mcp_name
            mcp_to_safe[mcp_name] = safe_name

            params_schema = t.inputSchema or {"type": "object", "properties": {}, "additionalProperties": True}
            desc = t.description or ""

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": safe_name,
                        "description": desc,
                        "parameters": params_schema,
                    },
                }
            )

        self.tool_mapping = ToolMapping(safe_to_mcp=safe_to_mcp, mcp_to_safe=mcp_to_safe)
        self.tools_openai_schema = openai_tools

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        assert self.tools_openai_schema is not None
        return self.tools_openai_schema

    def format_history_text(self, n: int = 12) -> str:
        tail = self.history[-n:]
        if not tail:
            return "No previous steps."
        lines = []
        for r in tail:
            lines.append(f"- step#{r.step_id}: {r.tool_name} args={json.dumps(r.tool_args, ensure_ascii=False)}")
        return "\n".join(lines)

    async def call_mcp_tool(self, mcp_tool_name: str, args: Dict[str, Any]) -> Any:
        assert self.session is not None
        return await self.session.call_tool(mcp_tool_name, args)

    async def call_safe_tool(self, safe_tool_name: str, args: Dict[str, Any], step_id: int) -> Any:
        assert self.tool_mapping is not None
        mcp_name = self.tool_mapping.safe_to_mcp[safe_tool_name]
        result = await self.call_mcp_tool(mcp_name, args)
        self.history.append(StepLog(step_id=step_id, tool_name=mcp_name, tool_args=args, tool_result=result))
        return result
