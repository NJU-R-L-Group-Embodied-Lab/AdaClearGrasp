# plan/vlm_plan.py
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from json_repair import repair_json
from openai import OpenAI
import openai
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from plan.runtime_config import load_runtime_config
from plan.mcp_runtime import MCPRuntime
from plan.prompts import SYSTEM_PROMPT, USER_TASK_PROMPT

PLAN_CFG_PATH = "configs/runtime_config.yaml"

LOG_ROOT = os.path.join("data", "logs", "vlm_plan")
SCENE_JSON_TPL = os.path.join("data", "scenes", "{scene_name}", "{clutter_count}", "{scene_id}.json")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_name", default="apple", type=str, required=True)
    p.add_argument("--clutter_count", default=4, type=int, required=True)
    p.add_argument("--scene_id", default=1, type=int, required=True)

    p.add_argument("--max_plan_steps", type=int, default=80)
    p.add_argument("--max_model_retry", type=int, default=3)
    return p.parse_args()


# =========================================================
# MCP extractors (STRICT)
# =========================================================
def extract_payload(tool_result: Any) -> Dict[str, Any]:
    if getattr(tool_result, "isError", False):
        content = getattr(tool_result, "content", [])
        if content and getattr(content[0], "type", None) == "text":
            raise RuntimeError(content[0].text)
        raise RuntimeError(f"MCP tool error: {tool_result!r}")

    sc = getattr(tool_result, "structuredContent", None)
    if not isinstance(sc, dict):
        raise RuntimeError(f"Missing structuredContent dict. got={type(sc)} sc={sc!r}")

    if "result" not in sc:
        raise RuntimeError(f"structuredContent must have top-level 'result'. keys={list(sc.keys())}")

    payload = sc["result"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"structuredContent['result'] must be dict. got={type(payload)} payload={payload!r}")

    return payload


def extract_png_b64(tool_result: Any) -> str:
    payload = extract_payload(tool_result)
    if "png_base64" not in payload:
        raise RuntimeError(f"render_rgb payload missing png_base64: {payload!r}")
    return payload["png_base64"]


# =========================================================
# Scene paths + logging
# =========================================================
def _scene_json_path(scene_name: str, clutter_count: int, scene_id: int) -> str:
    return SCENE_JSON_TPL.format(
        scene_name=str(scene_name),
        clutter_count=str(int(clutter_count)),
        scene_id=str(int(scene_id)),
    )


def _run_dir(scene_name: str, clutter_count: int, scene_id: int) -> str:
    return os.path.join(LOG_ROOT, str(scene_name), str(int(clutter_count)), str(int(scene_id)))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _clear_dir(p: str) -> None:
    if not os.path.exists(p):
        return
    for name in os.listdir(p):
        full = os.path.join(p, name)
        if os.path.isfile(full) or os.path.islink(full):
            os.unlink(full)
        elif os.path.isdir(full):
            import shutil
            shutil.rmtree(full)


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _save_png_from_b64(path: str, png_b64: str) -> None:
    b = base64.b64decode(png_b64.encode("utf-8"))
    with open(path, "wb") as f:
        f.write(b)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# =========================================================
# Objects
# =========================================================
def _filter_objects(names: List[str]) -> List[str]:
    banned = {"goal_site", "table-workspace", "ground"}
    return [n for n in names if isinstance(n, str) and n not in banned]


def _format_object_hint(object_names: List[str]) -> str:
    return "Available object names (use these EXACT strings for move_to):\n" + "\n".join([str(x) for x in object_names])


# =========================================================
# Action validation (must match MCP server)
# =========================================================
_ALLOWED_ACTIONS = {
    "env_reset",
    "move_to",
    "lift",
    "lower",
    "pull",
    "push",
    "initarm",
    "inithand",
    "grasp",
    "done",
}


def validate_action(action: str, args: Dict[str, Any]) -> None:
    if action == "env_reset":
        if args:
            raise ValueError(f"env_reset: args must be empty, got {args!r}")
        return

    if action == "move_to":
        name = args.get("name")
        if not isinstance(name, str):
            raise ValueError(f"move_to: name must be string, got {args!r}")
        return

    if action in ("lift", "lower", "initarm", "inithand", "grasp"):
        if args:
            raise ValueError(f"{action}: args must be empty, got {args!r}")
        return

    if action in ("pull", "push"):
        side = args.get("side")
        dist_m = args.get("dist_m")
        if not isinstance(side, str):
            raise ValueError(f"{action}: side must be string, got {args!r}")
        if side not in ("left", "center", "right", "middle"):
            raise ValueError(f"{action}: invalid side {side!r}, expected left|center|right|middle")
        if not isinstance(dist_m, (int, float)):
            raise ValueError(f"{action}: dist_m must be number, got {args!r}")
        if float(dist_m) <= 0.0:
            raise ValueError(f"{action}: dist_m must be > 0, got {dist_m!r}")
        return

    if action == "done":
        if args:
            raise ValueError(f"done: args must be empty, got {args!r}")
        return

    raise ValueError(f"Unknown action: {action!r}")


# =========================================================
# Feedback shaping (aligned with human frontend)
# =========================================================
def _shorten(s: Any, max_len: int = 240) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _make_hint(action: str, ok: bool, err: str) -> str:
    parts: List[str] = []

    if not ok:
        if err == "stuck":
            parts.append(
                "This can be normal due to contact or object size differences. "
                "Use the image to judge whether the behavior is abnormal; "
                "do NOT blindly repeat the same action."
            )
        elif err == "max_steps":
            parts.append(
                "Maximum steps reached. Consider reducing the step size, "
                "changing the approach direction, or retrying from a safer height or pose."
            )
        elif err == "call_error":
            parts.append("Tool call error. Check MCP server logs or tool configuration.")
        elif err and err != "none":
            parts.append("Action failed. Try adjusting height/pose or repositioning before retrying.")

    return " ".join(parts)


def make_exec_feedback_brief(step_id: int, action: str, args: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    ok = bool(payload.get("ok"))
    err = str(payload.get("error_code", ""))

    return {
        "type": "exec_feedback_brief",
        "step_id": int(step_id),
        "action": str(action),
        "args": args,
        "ok": ok,
        "error_code": err,
        "message": _shorten(payload.get("message", ""), 240),
        "advice": _shorten(payload.get("advice", ""), 240),
        "hint": _shorten(_make_hint(action, ok, err), 360),
    }


def _format_feedback_for_model(brief: Optional[Dict[str, Any]]) -> str:
    if brief is None:
        return "No previous exec feedback."
    return "Latest exec feedback:\n" + json.dumps(brief, ensure_ascii=False)


# =========================================================
# Model JSON parsing (repair only; if still invalid -> ask re-output)
# =========================================================
def _parse_action_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        raise ValueError("empty model output")

    try:
        obj = json.loads(t)
    except Exception:
        fixed = repair_json(t)
        obj = json.loads(fixed)

    if not isinstance(obj, dict):
        raise ValueError(f"model output must be a JSON object, got {type(obj)}")

    required = {"action", "args", "reason"}
    missing = required - obj.keys()
    if missing:
        raise ValueError(f"missing required keys: {missing}")

    if not isinstance(obj["reason"], str) or not obj["reason"].strip():
        raise ValueError("reason must be a non-empty string")

    return obj


def _norm_msg(s: Any) -> str:
    s = "" if s is None else str(s)
    return " ".join(s.strip().lower().split())


def _is_grasp_success(brief: Dict[str, Any]) -> bool:
    msg = _norm_msg(brief.get("message", ""))
    success_msgs = {
        "success",
        "succeeded",
        "ok",
        "true",
        "1",
        "yes",
        "grasp success",
        "grasped",
    }
    return msg in success_msgs


# =========================================================
# Main
# =========================================================
async def main() -> None:
    args = _parse_args()
    cfg = load_runtime_config(PLAN_CFG_PATH)

    scene_name = args.scene_name
    clutter_count = int(args.clutter_count)
    scene_id = int(args.scene_id)

    max_plan_steps = int(args.max_plan_steps)
    max_model_retry = int(args.max_model_retry)

    scene_json = _scene_json_path(scene_name, clutter_count, scene_id)
    if not os.path.exists(scene_json):
        raise RuntimeError(f"scene_json not found: {scene_json}")

    run_dir = _run_dir(scene_name, clutter_count, scene_id)
    _clear_dir(run_dir)
    frames_dir = os.path.join(run_dir, "frames")
    _ensure_dir(frames_dir)

    steps_path = os.path.join(run_dir, "steps.jsonl")
    feedback_path = os.path.join(run_dir, "feedback.jsonl")
    objects_path = os.path.join(run_dir, "objects.json")

    llm = OpenAI(api_key=cfg.openai.api_key, base_url=cfg.openai.base_url)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_brief: Optional[Dict[str, Any]] = None

    async with MCPRuntime(
        cfg.mcp.command,
        cfg.mcp.server_script,
        server_args=["--scene_json", scene_json],
    ) as rt:
        # 0) list objects once (same as human)
        r0 = await rt.call_safe_tool("list_objects", {}, step_id=0)
        obj0 = extract_payload(r0)
        if "objects" not in obj0 or not isinstance(obj0["objects"], list) or not obj0["objects"]:
            raise RuntimeError(f"list_objects returned invalid objects: {obj0!r}")
        object_names = _filter_objects(obj0["objects"])

        with open(objects_path, "w", encoding="utf-8") as f:
            json.dump(object_names, f, ensure_ascii=False, indent=2)

        object_hint = _format_object_hint(object_names)

        # Give the model the same initial context (prompts + object list)
        messages.append(
            {
                "role": "user",
                "content": (
                    USER_TASK_PROMPT
                    + f"\n\nNOTE: The target object is a {scene_name}. But still use name 'target' to move."
                    + "\n\n"
                    + object_hint
                    + "\n\nFrom now on, output ONLY a single JSON object with keys: action, args and reason."
                ),
            }
        )

        # 1) reset once
        await rt.call_safe_tool("env_reset", {}, step_id=0)

        for step_id in range(1, max_plan_steps + 1):
            # A) render frame (same image source as human)
            r = await rt.call_safe_tool("render_rgb", {}, step_id)
            png_b64 = extract_png_b64(r)
            frame_path = os.path.join(frames_dir, f"{step_id:06d}.png")
            _save_png_from_b64(frame_path, png_b64)

            # B) ask model with: image + last feedback
            user_text = USER_TASK_PROMPT + "\n\n" + _format_feedback_for_model(last_brief)

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}", "detail": cfg.openai.image_detail},
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            )

            # C) model output + repair check; invalid -> re-ask (do NOT log invalid outputs)
            model_obj: Optional[Dict[str, Any]] = None
            model_raw: str = ""

            for _ in range(max_model_retry):
                while True:
                    try:
                        resp = llm.chat.completions.create(
                            model=cfg.openai.model,
                            messages=messages,
                            temperature=cfg.openai.temperature,
                            max_tokens=cfg.openai.max_tokens,
                        )
                        break
                    except openai.RateLimitError as e:
                        retry_after = None
                        resp_obj = getattr(e, "response", None)
                        headers = getattr(resp_obj, "headers", None)
                        if headers is not None:
                            retry_after = headers.get("Retry-After") or headers.get("retry-after")

                        if retry_after is None:
                            sleep_s = 15
                        else:
                            sleep_s = int(float(retry_after))

                        time.sleep(sleep_s)
                        continue 
                model_raw = resp.choices[0].message.content or ""

                try:
                    candidate = _parse_action_json(model_raw)
                    action = str(candidate.get("action"))
                    args2 = candidate.get("args")
                    if not isinstance(args2, dict):
                        raise ValueError("args must be a JSON object (dict)")
                    if action not in _ALLOWED_ACTIONS:
                        raise ValueError(f"Unknown action: {action!r}")
                    validate_action(action, args2)
                    model_obj = candidate
                    break
                except Exception as e:
                    messages.append({"role": "assistant", "content": model_raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous output is invalid or violates the action schema.\n"
                                f"Error: {str(e)}\n"
                                "Re-output ONLY a single valid JSON object with keys: action, args , reason.\n"
                                "Do not add any extra text."
                            ),
                        }
                    )

            if model_obj is None:
                raise RuntimeError(f"Model failed to produce a valid JSON action after {max_model_retry} retries.")

            action = str(model_obj["action"])
            args2 = model_obj["args"]
            reason = model_obj["reason"]

            _append_jsonl(
                steps_path,
                {
                    "ts": _now_iso(),
                    "step_id": int(step_id),
                    "action": action,
                    "args": args2,
                    "reason": reason,
                },
            )

            messages.append({"role": "assistant", "content": json.dumps(model_obj, ensure_ascii=False)})

            if action == "done":
                break
            
            tool_res = await rt.call_safe_tool(action, args2, step_id)
            payload = extract_payload(tool_res)

            brief = make_exec_feedback_brief(step_id, action, args2, payload)
            last_brief = brief

            _append_jsonl(
                feedback_path,
                {
                    "step_id": int(step_id),
                    "feedback": brief,
                },
            )


    print(f"[OK] Saved to: {run_dir}")
    print(" - steps.jsonl")
    print(" - feedback.jsonl")
    print(" - objects.json")
    print(" - frames/*.png")


if __name__ == "__main__":
    asyncio.run(main())
