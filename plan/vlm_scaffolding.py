# plan/vlm_scaffolding.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from plan.runtime_config import load_runtime_config
from plan.mcp_runtime import MCPRuntime
from plan.prompts import USER_TASK_PROMPT

# =========================================================
# Global params (edit here)
# =========================================================
PLAN_CFG_PATH = "configs/runtime_config.yaml"

SCENE_NAME = "apple"
CLUTTER_COUNT = 4
SCENE_ID = 1

# IMPORTANT: save to vlm_scaffolding baseline folder
LOG_ROOT = os.path.join("data", "logs", "vlm_scaffolding")
SCENE_JSON_TPL = os.path.join("data", "scenes", "{scene_name}", "{clutter_count}", "{scene_id}.json")


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
# Paths + logging helpers
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


def _save_png_from_b64(path: str, png_b64: str) -> None:
    b = base64.b64decode(png_b64.encode("utf-8"))
    with open(path, "wb") as f:
        f.write(b)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# =========================================================
# Inputs extraction (scene json)
# =========================================================
def _extract_objects_from_scene_xy(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expected scene schema:
    {
      "z_margin": float,
      "target": {"name":"target","xy":[x,y], ...},
      "clutter": [{"name":"orange","xy":[x,y], ...}, ...]
    }
    """
    if "z_margin" not in scene:
        raise RuntimeError("scene_json missing z_margin")
    z_margin = float(scene["z_margin"])

    target = scene.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("scene_json missing target dict")
    if target.get("name") != "target":
        raise RuntimeError(f"scene_json target.name must be 'target', got: {target.get('name')!r}")
    if "xy" not in target:
        raise RuntimeError(f"scene_json target missing xy: {target!r}")
    xy = target["xy"]
    if not isinstance(xy, list) or len(xy) != 2:
        raise RuntimeError(f"target.xy must be [x,y], got {xy!r}")

    objs: List[Dict[str, Any]] = []
    objs.append({"name": "target", "xy": [float(xy[0]), float(xy[1])], "z": z_margin})

    clutter = scene.get("clutter", [])
    if not isinstance(clutter, list):
        raise RuntimeError(f"clutter must be list, got {type(clutter)}")
    for it in clutter:
        if not isinstance(it, dict):
            raise RuntimeError(f"clutter item must be dict, got {type(it)}")
        if "name" not in it or "xy" not in it:
            raise RuntimeError(f"clutter item missing name/xy: {it!r}")
        xy2 = it["xy"]
        if not isinstance(xy2, list) or len(xy2) != 2:
            raise RuntimeError(f"{it.get('name','<unnamed>')}.xy must be [x,y], got {xy2!r}")
        objs.append({"name": str(it["name"]), "xy": [float(xy2[0]), float(xy2[1])], "z": z_margin})

    objs.sort(key=lambda d: d["name"])
    return objs


# =========================================================
# Object list helpers (same as your vlm_plan)
# =========================================================
def _filter_objects(names: List[str]) -> List[str]:
    banned = {"goal_site", "table-workspace", "ground"}
    return [n for n in names if isinstance(n, str) and n not in banned]


# =========================================================
# Prompt + model output parsing
# =========================================================
def _build_traj_prompt(
    scene_name: str,
    task_desc: str,
    object_names: List[str],
    objects_xy: List[Dict[str, Any]],
    z_margin: float,
) -> str:
    # One-shot trajectory generation prompt (NO closed-loop feedback, NO atomic skills)
    return (
        "You are a planner for a robot end-effector in a tabletop clutter scene.\n"
        "Coordinate system:\n"
        "- WORLD frame in meters.\n"
        "- Tabletop is at z = 0.\n"
        "- z is height above tabletop.\n\n"
        "You are given:\n"
        "1) A task description\n"
        "2) An RGB image of the scene\n"
        "3) A list of object names in the scene (EXACT strings)\n"
        "4) Approx object (x,y) positions from the scene JSON (z is unknown; use z_margin as safe near-table height)\n\n"
        "Your job:\n"
        "- Output ONLY a single JSON object with schema: {\"xyz\": [[x,y,z], ...]}\n"
        "- Generate a ONE-SHOT end-effector waypoint trajectory to complete the task.\n"
        "- The trajectory should include key waypoints to grasp the target:\n"
        "  (a) approach above target (z significantly above z_margin)\n"
        "  (b) descend to near-grasp (z >= z_margin)\n"
        "  (c) lift upward after grasp\n"
        "- If clutter blocks access, include a brief clearing motion in the same xyz trajectory before grasping.\n"
        "- Keep waypoints concise (typically 4-8 points).\n"
        f"- z_margin = {z_margin}\n"
        "- Do NOT include any extra keys or any extra text.\n\n"
        f"Task:\n{task_desc}\n\n"
        "Object names (EXACT strings):\n"
        + json.dumps(object_names, ensure_ascii=False)
        + "\n\n"
        "Approx object positions from scene JSON:\n"
        + json.dumps(objects_xy, ensure_ascii=False)
        + "\n\n"
        "Important:\n"
        f"- The target object is a {scene_name}.\n"
        "- The target actor name is exactly 'target'.\n"
    )


def _parse_xyz_json(text: str) -> List[List[float]]:
    t = (text or "").strip()
    obj = json.loads(t)
    if not isinstance(obj, dict) or "xyz" not in obj:
        raise ValueError(f"Model output must be JSON object with key 'xyz', got: {obj!r}")
    xyz = obj["xyz"]
    if not isinstance(xyz, list) or not xyz:
        raise ValueError(f"'xyz' must be a non-empty list, got: {xyz!r}")
    out: List[List[float]] = []
    for i, p in enumerate(xyz):
        if not isinstance(p, list) or len(p) != 3:
            raise ValueError(f"xyz[{i}] must be [x,y,z], got: {p!r}")
        out.append([float(p[0]), float(p[1]), float(p[2])])
    return out


# =========================================================
# Main
# =========================================================
async def main() -> None:
    cfg = load_runtime_config(PLAN_CFG_PATH)

    scene_json = _scene_json_path(SCENE_NAME, CLUTTER_COUNT, SCENE_ID)
    if not os.path.exists(scene_json):
        raise RuntimeError(f"scene_json not found: {scene_json}")

    with open(scene_json, "r", encoding="utf-8") as f:
        scene = json.load(f)

    z_margin = float(scene["z_margin"])
    objects_xy = _extract_objects_from_scene_xy(scene)

    run_dir = _run_dir(SCENE_NAME, CLUTTER_COUNT, SCENE_ID)
    _clear_dir(run_dir)
    _ensure_dir(run_dir)

    inputs_dir = os.path.join(run_dir, "inputs")
    outputs_dir = os.path.join(run_dir, "outputs")
    _ensure_dir(inputs_dir)
    _ensure_dir(outputs_dir)

    llm = OpenAI(api_key=cfg.openai.api_key, base_url=cfg.openai.base_url)

    # Render 1 frame + list objects (same source as your other pipeline)
    async with MCPRuntime(
        cfg.mcp.command,
        cfg.mcp.server_script,
        server_args=["--scene_json", scene_json],
    ) as rt:
        r0 = await rt.call_safe_tool("list_objects", {}, step_id=0)
        obj0 = extract_payload(r0)
        if "objects" not in obj0 or not isinstance(obj0["objects"], list) or not obj0["objects"]:
            raise RuntimeError(f"list_objects returned invalid objects: {obj0!r}")
        object_names = _filter_objects(obj0["objects"])

        await rt.call_safe_tool("env_reset", {}, step_id=0)

        r1 = await rt.call_safe_tool("render_rgb", {}, step_id=1)
        png_b64 = extract_png_b64(r1)

    # -------- Save ALL model inputs --------
    frame_path = os.path.join(inputs_dir, "frame.png")
    _save_png_from_b64(frame_path, png_b64)

    object_names_path = os.path.join(inputs_dir, "object_names.json")
    with open(object_names_path, "w", encoding="utf-8") as f:
        json.dump(object_names, f, ensure_ascii=False, indent=2)

    objects_xy_path = os.path.join(inputs_dir, "objects_xy.json")
    with open(objects_xy_path, "w", encoding="utf-8") as f:
        json.dump(objects_xy, f, ensure_ascii=False, indent=2)

    # One-shot task description (NO feedback loop text)
    task_desc = (
        "Goal: In the current scene, clear clutter as needed, then grasp the target.\n"
        "Plan and output a ONE-SHOT end-effector waypoint trajectory (xyz) to achieve this."
        f"\n\nNOTE: The target object is a {SCENE_NAME}."
    )
    task_desc_path = os.path.join(inputs_dir, "task_desc.txt")
    with open(task_desc_path, "w", encoding="utf-8") as f:
        f.write(task_desc)

    prompt = _build_traj_prompt(
        scene_name=SCENE_NAME,
        task_desc=task_desc,
        object_names=object_names,
        objects_xy=objects_xy,
        z_margin=z_margin,
    )
    prompt_path = os.path.join(inputs_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    # -------- Call model once: image + prompt --------
    resp = llm.chat.completions.create(
        model=cfg.openai.model,
        messages=[
            {"role": "system", "content": "You output strictly valid JSON and nothing else."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}", "detail": cfg.openai.image_detail},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=cfg.openai.temperature,
        max_tokens=cfg.openai.max_tokens,
    )

    raw = resp.choices[0].message.content or ""

    # -------- Save outputs --------
    model_raw_path = os.path.join(outputs_dir, "model_raw.txt")
    with open(model_raw_path, "w", encoding="utf-8") as f:
        f.write(raw)

    xyz = _parse_xyz_json(raw)

    traj_path = os.path.join(outputs_dir, "traj_xyz.json")
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(xyz, f, ensure_ascii=False, indent=2)

    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ts": _now_iso(),
                "scene_name": SCENE_NAME,
                "clutter_count": int(CLUTTER_COUNT),
                "scene_id": int(SCENE_ID),
                "scene_json": scene_json,
                "z_margin": float(z_margin),
                "model": cfg.openai.model,
                "saved_inputs": {
                    "frame_png": frame_path,
                    "prompt_txt": prompt_path,
                    "task_desc_txt": task_desc_path,
                    "object_names_json": object_names_path,
                    "objects_xy_json": objects_xy_path,
                },
                "saved_outputs": {
                    "traj_xyz_json": traj_path,
                    "model_raw_txt": model_raw_path,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[OK] Saved run dir:")
    print(f" - {run_dir}")
    print("[OK] Inputs:")
    print(f" - {frame_path}")
    print(f" - {prompt_path}")
    print(f" - {task_desc_path}")
    print(f" - {object_names_path}")
    print(f" - {objects_xy_path}")
    print("[OK] Outputs:")
    print(f" - {traj_path}")
    print(f" - {model_raw_path}")
    print(f"[OK] Meta: {meta_path}")


if __name__ == "__main__":
    asyncio.run(main())
