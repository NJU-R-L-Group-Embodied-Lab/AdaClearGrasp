# tools/summarize_vlm_plan_steps.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

LOG_ROOT = Path("data/logs/vlm_plan")
OUTPUT_CSV = Path("data/analysis/vlm_plan_tasks_summary.csv")


@dataclass
class TaskStats:
    scene_name: str
    clutter_count: int
    scene_id: int

    success: int
    grasp_count: int
    env_reset_called: int
    max_step_id: int


def iter_steps_files(root: Path):
    yield from root.glob("*/*/*/steps.jsonl")


def parse_task_keys_from_path(p: Path):
    parent = p.parent
    scene_id = int(parent.name)
    clutter_count = int(parent.parent.name)
    scene_name = parent.parent.parent.name
    return scene_name, clutter_count, scene_id


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_success_from_feedback(feedback_path: Path) -> int:
    for obj in load_jsonl(feedback_path):
        fb = obj["feedback"]
        if fb["action"] == "grasp" and fb["message"] == "success":
            return 1
    return 0


def summarize_one_steps_file(path: Path) -> TaskStats:
    scene_name, clutter_count, scene_id = parse_task_keys_from_path(path)

    feedback_path = path.with_name("feedback.jsonl")
    success = summarize_success_from_feedback(feedback_path)

    grasp_count = 0
    env_reset_called = 0
    max_step_id = 0

    for obj in load_jsonl(path):
        step_id = int(obj["step_id"])
        if step_id > max_step_id:
            max_step_id = step_id

        action = obj["action"]
        if action == "grasp":
            grasp_count += 1
        elif action == "env_reset":
            env_reset_called = 1

    return TaskStats(
        scene_name=scene_name,
        clutter_count=clutter_count,
        scene_id=scene_id,
        success=success,
        grasp_count=grasp_count,
        env_reset_called=env_reset_called,
        max_step_id=max_step_id,
    )


def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[TaskStats], out_path: Path):
    ensure_parent_dir(out_path)

    fieldnames = [
        "scene_name",
        "clutter_count",
        "scene_id",
        "success",
        "grasp_count",
        "env_reset_called",
        "max_step_id",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "scene_name": r.scene_name,
                    "clutter_count": r.clutter_count,
                    "scene_id": r.scene_id,
                    "success": r.success,
                    "grasp_count": r.grasp_count,
                    "env_reset_called": r.env_reset_called,
                    "max_step_id": r.max_step_id,
                }
            )


def main():
    steps_files = sorted(iter_steps_files(LOG_ROOT))
    rows = [summarize_one_steps_file(p) for p in steps_files]
    write_csv(rows, OUTPUT_CSV)

    print(f"[OK] Wrote: {OUTPUT_CSV}")
    print(f"[OK] Tasks summarized: {len(rows)}")


if __name__ == "__main__":
    main()
