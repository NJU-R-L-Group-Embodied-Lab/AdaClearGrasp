# plan/recorder.py
import os
import json
import base64
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from core.paths import ensure_data_dirs


@dataclass
class RunPaths:
    run_dir: str
    frames_dir: str
    steps_jsonl: str


def make_run_dir(tag: str = "vlm_run") -> RunPaths:
    p = ensure_data_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(p.logs, "plan_runs", f"{tag}_{ts}")
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    steps_jsonl = os.path.join(run_dir, "steps.jsonl")
    return RunPaths(run_dir=run_dir, frames_dir=frames_dir, steps_jsonl=steps_jsonl)


class EpisodeRecorder:
    def __init__(self, tag: str = "vlm_run"):
        self.paths = make_run_dir(tag=tag)
        self._f = open(self.paths.steps_jsonl, "w", encoding="utf-8")

    def close(self):
        self._f.close()

    def save_frame(self, step_id: int, png_b64: str) -> str:
        fn = f"{step_id:04d}.png"
        path = os.path.join(self.paths.frames_dir, fn)
        with open(path, "wb") as f:
            f.write(base64.b64decode(png_b64))
        return path

    def log_step(
        self,
        step_id: int,
        frame_path: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        model_text: Optional[str],
    ):
        rec = {
            "step_id": step_id,
            "frame_path": frame_path,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "model_text": model_text,
        }
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
