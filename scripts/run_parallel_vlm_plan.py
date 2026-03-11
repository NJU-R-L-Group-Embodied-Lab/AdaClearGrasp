# run_parallel_vlm_plan.py
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# Global params (edit here)
# =========================================================
SCENE_NAME_DEFAULT: str = "can"
CLUTTER_COUNT_DEFAULT: int = 4
SCENE_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Optional: override vlm_plan defaults
MAX_PLAN_STEPS: int = 40
MAX_MODEL_RETRY: int = 3

# Concurrency control
MAX_CONCURRENCY: int = 10

# Python executable
PYTHON_BIN: str = sys.executable 
VLM_PLAN_PATH = os.path.join(PROJECT_ROOT, "plan", "vlm_plan.py")


@dataclass(frozen=True)
class Config:
    scene_name: str
    clutter_count: int


@dataclass(frozen=True)
class RunResult:
    scene_id: int
    returncode: int


def _cmd_for(cfg: Config, scene_id: int) -> List[str]:
    return [
        PYTHON_BIN,
        "-u",
        VLM_PLAN_PATH,
        "--scene_name",
        str(cfg.scene_name),
        "--clutter_count",
        str(int(cfg.clutter_count)),
        "--scene_id",
        str(int(scene_id)),
        "--max_plan_steps",
        str(int(MAX_PLAN_STEPS)),
        "--max_model_retry",
        str(int(MAX_MODEL_RETRY)),
    ]


async def _run_one(cfg: Config, scene_id: int, sem: asyncio.Semaphore) -> RunResult:
    cmd = _cmd_for(cfg, scene_id)

    async with sem:
        p = await asyncio.create_subprocess_exec(*cmd)
        rc = await p.wait()

    return RunResult(scene_id=scene_id, returncode=int(rc))


async def _amain(cfg: Config) -> None:
    if not os.path.exists(VLM_PLAN_PATH):
        raise RuntimeError(f"vlm_plan.py not found: {VLM_PLAN_PATH}")

    sem = asyncio.Semaphore(int(MAX_CONCURRENCY))
    tasks = [asyncio.create_task(_run_one(cfg, scene_id, sem)) for scene_id in SCENE_IDS]

    results = await asyncio.gather(*tasks)

    ok_ids = [r.scene_id for r in results if r.returncode == 0]
    fail = [(r.scene_id, r.returncode) for r in results if r.returncode != 0]

    print("[DONE] Parallel runs finished.")
    print(f"scene_name={cfg.scene_name} clutter_count={cfg.clutter_count}")
    print(f"OK:   {ok_ids}")
    if fail:
        print("FAIL:")
        for scene_id, rc in fail:
            print(f" - scene_id={scene_id} returncode={rc}")
        raise RuntimeError("Some runs failed.")
    else:
        print("All OK.")


def _parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--scene_name", type=str, default=SCENE_NAME_DEFAULT)
    p.add_argument("--clutter_count", type=int, default=CLUTTER_COUNT_DEFAULT)
    args = p.parse_args()
    return Config(scene_name=args.scene_name, clutter_count=int(args.clutter_count))


def main() -> None:
    cfg = _parse_args()
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
