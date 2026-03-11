import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================
# Global Params
# =========================
MODE = "video"  # "fast" | "window" | "video"

# (scene_name, clutter_count)
TASKS = [
    ("apple", 2),
    ("apple", 4),
    ("apple", 6),
    # ("ball", 2),
    # ("ball", 4),
    # ("ball", 6),
    # ("can", 2),
    # ("can", 4),
    # ("can", 6),
    # ("cube", 2),
    # ("cube", 4),
    # ("cube", 6),
    # ("lego", 2),
    # ("lego", 4),
    # ("lego", 6),
    ("mug", 2),
    ("mug", 4),
    ("mug", 6),
    # ("pear", 2),
    # ("pear", 4),
    # ("pear", 6),
]

# 你的测试脚本路径（相对本文件）
TEST_SCRIPT = os.path.join(os.path.dirname(__file__), "eval_policy_clutter.py")


def main():
    if not os.path.isfile(TEST_SCRIPT):
        raise FileNotFoundError(f"TEST_SCRIPT not found: {TEST_SCRIPT}")

    for idx, (scene_name, clutter_count) in enumerate(TASKS):
        cmd = [
            sys.executable,
            TEST_SCRIPT,
            "--scene_name",
            str(scene_name),
            "--clutter_count",
            str(int(clutter_count)),
            "--mode",
            str(MODE),
        ]
        print(f"\n[RUN] {idx:02d} cmd = {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
