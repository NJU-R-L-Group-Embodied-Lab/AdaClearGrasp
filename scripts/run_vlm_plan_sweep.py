from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RUNS: List[Tuple[str, int]] = [
    # ("apple", 2),
    # ("apple", 4),
    # ("apple", 6),
    ("ball", 2),
    ("ball", 4),
    ("ball", 6),
    ("can", 2),
    ("can", 4),
    ("can", 6),
    ("cube", 2),
    ("cube", 4),
    ("cube", 6),
    ("lego", 2),
    ("lego", 4),
    ("lego", 6),
    ("mug", 2),
    ("mug", 4),
    ("mug", 6),
    ("pear", 2),
    ("pear", 4),
    ("pear", 6),
]

PYTHON_BIN: str = sys.executable

RUN_PARALLEL_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "tools","run_parallel_vlm_plan.py")


@dataclass(frozen=True)
class OneRunResult:
    scene_name: str
    clutter_count: int
    returncode: int


def _cmd_for(scene_name: str, clutter_count: int) -> List[str]:
    return [
        PYTHON_BIN,
        "-u",
        RUN_PARALLEL_SCRIPT_PATH,
        "--scene_name",
        str(scene_name),
        "--clutter_count",
        str(int(clutter_count)),
    ]


async def _run_one(scene_name: str, clutter_count: int) -> OneRunResult:
    cmd = _cmd_for(scene_name, clutter_count)
    p = await asyncio.create_subprocess_exec(*cmd)
    rc = await p.wait()
    return OneRunResult(scene_name=scene_name, clutter_count=int(clutter_count), returncode=int(rc))


async def main() -> None:
    if not os.path.exists(RUN_PARALLEL_SCRIPT_PATH):
        raise RuntimeError(f"run_parallel_vlm_plan.py not found: {RUN_PARALLEL_SCRIPT_PATH}")

    all_results: List[OneRunResult] = []

    for scene_name, clutter_count in RUNS:
        print("=" * 80)
        print(f"[RUN] scene_name={scene_name} clutter_count={clutter_count}")

        r = await _run_one(scene_name, clutter_count)
        all_results.append(r)

        if r.returncode == 0:
            print(f"[OK]  scene_name={scene_name} clutter_count={clutter_count}")
        else:
            print(f"[FAIL] scene_name={scene_name} clutter_count={clutter_count} returncode={r.returncode}")
            print("[NEXT] continue to next run (no retry).")

    print("=" * 80)
    ok = [(r.scene_name, r.clutter_count) for r in all_results if r.returncode == 0]
    fail = [(r.scene_name, r.clutter_count, r.returncode) for r in all_results if r.returncode != 0]

    print("[DONE] Sweep finished.")
    print(f"OK:   {ok}")
    if fail:
        print("FAIL:")
        for scene_name, clutter_count, rc in fail:
            print(f" - scene_name={scene_name} clutter_count={clutter_count} returncode={rc}")
        sys.exit(1)
    else:
        print("All OK.")


if __name__ == "__main__":
    asyncio.run(main())
