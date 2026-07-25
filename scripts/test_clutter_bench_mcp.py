from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


URL = os.environ.get("CLUTTER_BENCH_MCP_URL", "http://127.0.0.1:8877/mcp")


def _payload(result: object) -> dict[str, object]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        value = structured.get("result", structured)
        if isinstance(value, dict):
            return value
    for item in list(getattr(result, "content", None) or []):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            value = json.loads(text)
            if isinstance(value, dict):
                nested = value.get("result", value)
                if isinstance(nested, dict):
                    return nested
    raise RuntimeError("tool returned no structured payload")


async def main() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    async with streamable_http_client(URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            health = _payload(await session.call_tool("healthcheck", {}))
            scene = _payload(await session.call_tool("scene_info", {}))
            image = _payload(await session.call_tool("render_rgb", {}))
            raw = base64.b64decode(str(image["png_base64"]), validate=True)
            output = Path("/tmp/clutter_bench_mcp_rgb.png")
            output.write_bytes(raw)
            print(json.dumps({
                "tools": names,
                "health": health,
                "scene": scene,
                "image": {
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "byte_size": len(raw),
                    "path": str(output),
                },
            }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
