import os
import sys
import json
import time
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import gymnasium as gym

import env.sim.pick_clutter_xarm  # noqa: F401


# =========================
# Global Params
# =========================
ENV_ID = "PickClutterYCB-XArm7-v1"
BASE_SCENE_JSON = "data/scenes/lego.json"
CONTROL_MODE = "pd_ee_delta_pose"
SEED = 0

MODE = "fast"  # "fast" | "window"
FPS = 60

NUM_SAFE_PER_COUNT = 10
CLUTTER_COUNTS = (2, 4, 6)

# =========================
# Randomization Params
# =========================
RAND_X_MIN, RAND_X_MAX = -0.10, 0.10
RAND_Y_MIN, RAND_Y_MAX = -0.10, 0.10
RAND_YAW_MIN, RAND_YAW_MAX = -np.pi, np.pi

RAND_MIN_DIST = 0.06
RAND_MAX_RESAMPLE = 300

SIM_STEPS_SETTLE = 30
SIM_STEPS_MEASURE = 60

POSE_XY_EPS = 0.01 
CHECK_AT_T0 = True


def _scene_name_from_json(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    if not name:
        raise RuntimeError(f"Bad scene json path: {path}")
    return name


def _is_fast_mode() -> bool:
    return MODE == "fast"


def _is_window_mode() -> bool:
    return MODE == "window"


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _dump_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_bool_any(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, np.ndarray):
        return bool(np.any(x))
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return False
        return bool(x.any().item())
    raise RuntimeError(f"terminated/truncated must be bool/np.ndarray/torch.Tensor, got {type(x)}")


def _actor_pos_xy_np(actor) -> np.ndarray:
    p = actor.pose.p
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    else:
        p = np.asarray(p)
    p = p.reshape(-1)[:3].astype(np.float64)
    return p[:2].copy()


def _random_pose_xy_yaw(rng: np.random.Generator) -> tuple[float, float, float]:
    x = float(rng.uniform(RAND_X_MIN, RAND_X_MAX))
    y = float(rng.uniform(RAND_Y_MIN, RAND_Y_MAX))
    yaw = float(rng.uniform(RAND_YAW_MIN, RAND_YAW_MAX))
    return x, y, yaw


def _randomize_scene_cfg_inplace(scene_cfg: dict, rng: np.random.Generator):
    if not isinstance(scene_cfg, dict):
        raise RuntimeError("scene_cfg must be dict")
    if "target" not in scene_cfg or "clutter" not in scene_cfg:
        raise RuntimeError("scene_cfg must contain keys: target, clutter")
    if not isinstance(scene_cfg["clutter"], list) or len(scene_cfg["clutter"]) == 0:
        raise RuntimeError("scene_cfg['clutter'] must be non-empty list")

    placed_xy: list[tuple[float, float]] = []

    # target
    tgt = scene_cfg["target"]
    if not isinstance(tgt, dict):
        raise RuntimeError("scene_cfg['target'] must be dict")

    ok = False
    for _ in range(int(RAND_MAX_RESAMPLE)):
        x, y, yaw = _random_pose_xy_yaw(rng)
        if RAND_MIN_DIST > 0 and len(placed_xy) > 0:
            d2 = [(x - px) ** 2 + (y - py) ** 2 for (px, py) in placed_xy]
            if min(d2) < (float(RAND_MIN_DIST) ** 2):
                continue
        tgt["xy"] = [x, y]
        tgt["yaw"] = yaw
        placed_xy.append((x, y))
        ok = True
        break
    if not ok:
        raise RuntimeError("Failed to sample target pose. Relax RAND_MIN_DIST or increase range.")

    # clutter
    for obj in scene_cfg["clutter"]:
        if not isinstance(obj, dict):
            raise RuntimeError("each clutter item must be dict")
        if "xy" not in obj or "yaw" not in obj:
            raise RuntimeError("each clutter item must contain 'xy' and 'yaw'")

        ok = False
        for _ in range(int(RAND_MAX_RESAMPLE)):
            x, y, yaw = _random_pose_xy_yaw(rng)
            if RAND_MIN_DIST > 0 and len(placed_xy) > 0:
                d2 = [(x - px) ** 2 + (y - py) ** 2 for (px, py) in placed_xy]
                if min(d2) < (float(RAND_MIN_DIST) ** 2):
                    continue
            obj["xy"] = [x, y]
            obj["yaw"] = yaw
            placed_xy.append((x, y))
            ok = True
            break

        if not ok:
            raise RuntimeError("Failed to sample clutter poses. Relax RAND_MIN_DIST or increase range.")


def make_env(scene_json_path: str, mode: str):
    if MODE not in ("window", "fast"):
        raise RuntimeError(f"MODE must be 'window'|'fast', got: {MODE}")

    render_mode = None if _is_fast_mode() else "human"

    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        num_envs=1,
        control_mode=CONTROL_MODE,
        scene_json=scene_json_path,
        mode=mode,
        reconfiguration_freq=0,
        only_target_object=False,
    )
    return env


def _get_all_object_actors(u) -> list:
    if not hasattr(u, "target_actors") or len(u.target_actors) == 0:
        raise RuntimeError("env.unwrapped must expose target_actors")
    if not hasattr(u, "clutter_actors_per_item"):
        raise RuntimeError("env.unwrapped must expose clutter_actors_per_item")

    actors = [u.target_actors[0]]

    per_item = u.clutter_actors_per_item
    if not isinstance(per_item, list):
        raise RuntimeError("clutter_actors_per_item must be list")
    for j, per_env_list in enumerate(per_item):
        if not isinstance(per_env_list, list) or len(per_env_list) == 0:
            raise RuntimeError(f"clutter_actors_per_item[{j}] must be non-empty list")
        actors.append(per_env_list[0])

    return actors


def _expected_xy_list_from_scene_cfg(scene_cfg: dict) -> list[np.ndarray]:
    tgt_xy = scene_cfg["target"].get("xy", None)
    if tgt_xy is None:
        raise RuntimeError("target must contain 'xy'")
    exps = [np.asarray(tgt_xy, dtype=np.float64).reshape(2)]

    for obj in scene_cfg["clutter"]:
        xy = obj.get("xy", None)
        if xy is None:
            raise RuntimeError("clutter item must contain 'xy'")
        exps.append(np.asarray(xy, dtype=np.float64).reshape(2))
    return exps


def _step_n(env, n: int, dt: float):
    action_dim = int(env.action_space.shape[0])
    zero_action = np.zeros((action_dim,), dtype=np.float32)

    for _ in range(int(n)):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        if _is_window_mode():
            env.render()
        if not _is_fast_mode():
            time.sleep(dt)

        if _to_bool_any(terminated) or _to_bool_any(truncated):
            return False, {"reason": "episode_end"}
    return True, {}


def _max_xy_error_against_expected(env) -> tuple[float, dict]:
    u = env.unwrapped
    actors = _get_all_object_actors(u)
    exp_xys = _expected_xy_list_from_scene_cfg(u.scene_cfg)

    if len(actors) != len(exp_xys):
        raise RuntimeError(f"actors({len(actors)}) != expected({len(exp_xys)})")

    names = [a.name for a in actors]

    errs = []
    max_err = 0.0
    worst_i = 0
    for i, a in enumerate(actors):
        xy = _actor_pos_xy_np(a)
        e = exp_xys[i]
        err = float(np.linalg.norm(xy - e, ord=2))
        errs.append(err * 1000.0)  # mm
        if err > max_err:
            max_err = err
            worst_i = i

    metrics = {
        "names": names,
        "expected_xy": [e.tolist() for e in exp_xys],
        "err_xy_mm": errs,
        "max_err_xy_mm": max_err * 1000.0,
        "worst": {"idx": int(worst_i), "name": names[worst_i], "err_xy_mm": errs[worst_i]},
    }
    return max_err, metrics


def _check_pose_consistency(env, dt: float) -> tuple[bool, dict]:
    # t0 check (optional)
    if CHECK_AT_T0:
        max_err0, m0 = _max_xy_error_against_expected(env)
        if max_err0 > float(POSE_XY_EPS):
            m0["stage"] = "t0"
            return False, m0

    # settle
    ok, m = _step_n(env, SIM_STEPS_SETTLE, dt)
    if not ok:
        return False, {"reason": "episode_end_during_settle"}

    max_err1, m1 = _max_xy_error_against_expected(env)
    if max_err1 > float(POSE_XY_EPS):
        m1["stage"] = "after_settle"
        return False, m1

    # measure
    ok, m = _step_n(env, SIM_STEPS_MEASURE, dt)
    if not ok:
        return False, {"reason": "episode_end_during_measure"}

    max_err2, m2 = _max_xy_error_against_expected(env)
    if max_err2 > float(POSE_XY_EPS):
        m2["stage"] = "after_measure"
        return False, m2

    out = {"stage": "pass", "after_settle": m1, "after_measure": m2}
    if CHECK_AT_T0:
        out["t0"] = m0
    return True, out


def _write_scene_variant(base_cfg: dict, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    _dump_json(out_path, base_cfg)


def main():
    if not os.path.exists(BASE_SCENE_JSON):
        raise FileNotFoundError(f"BASE_SCENE_JSON not found: {BASE_SCENE_JSON}")

    with open(BASE_SCENE_JSON, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    if "target" not in base_cfg or "clutter" not in base_cfg:
        raise RuntimeError("base scene json must contain keys: target, clutter")
    if not isinstance(base_cfg["clutter"], list) or len(base_cfg["clutter"]) < 6:
        raise RuntimeError("base scene json must contain clutter list with at least 6 items")

    scene_name = _scene_name_from_json(BASE_SCENE_JSON)
    base_dir = os.path.dirname(BASE_SCENE_JSON)
    dt = 1.0 / float(FPS)
    rng = np.random.default_rng(SEED)

    print(f"[INFO] MODE={MODE} base={BASE_SCENE_JSON}", flush=True)
    print(f"[INFO] counts={CLUTTER_COUNTS} num_safe_per_count={NUM_SAFE_PER_COUNT}", flush=True)
    print(
        f"[INFO] settle={SIM_STEPS_SETTLE} measure={SIM_STEPS_MEASURE} pose_xy_eps={POSE_XY_EPS} "
        f"check_t0={CHECK_AT_T0} rand_min_dist={RAND_MIN_DIST} "
        f"rand_range=([{RAND_X_MIN},{RAND_X_MAX}],[{RAND_Y_MIN},{RAND_Y_MAX}])",
        flush=True,
    )

    for n_clutter in CLUTTER_COUNTS:
        if n_clutter > len(base_cfg["clutter"]):
            raise RuntimeError(f"n_clutter={n_clutter} exceeds base clutter size={len(base_cfg['clutter'])}")

        # 生成一个临时 scene json（只保留前 n_clutter 个 clutter）
        variant_cfg = deepcopy(base_cfg)
        variant_cfg["clutter"] = deepcopy(base_cfg["clutter"][: int(n_clutter)])

        # 让 env 读取一个实际路径（避免 env 内部只从文件读取）
        variant_dir = os.path.join(base_dir, scene_name, f"_variant_{n_clutter}")
        _ensure_dir(variant_dir)
        variant_json_path = os.path.join(variant_dir, "base.json")
        _write_scene_variant(variant_cfg, variant_json_path)

        out_dir = os.path.join(base_dir, scene_name, str(int(n_clutter)))
        _ensure_dir(out_dir)

        env = make_env(scene_json_path=variant_json_path, mode="plan")
        u = env.unwrapped

        try:
            safe_n = 0
            attempt = 0

            while safe_n < int(NUM_SAFE_PER_COUNT):
                attempt += 1
                _randomize_scene_cfg_inplace(u.scene_cfg, rng)

                obs, info = env.reset(seed=SEED + attempt)

                if _is_window_mode():
                    env.render()

                ok, m = _check_pose_consistency(env, dt=dt)

                target_state = {
                    "name": u.scene_cfg["target"].get("name", "target"),
                    "xy": u.scene_cfg["target"].get("xy", None),
                    "yaw": u.scene_cfg["target"].get("yaw", None),
                }
                clutter_state = [
                    {"name": obj.get("name", "?"), "xy": obj.get("xy", None), "yaw": obj.get("yaw", None)}
                    for obj in u.scene_cfg["clutter"]
                ]

                if ok:
                    safe_n += 1
                    out_path = os.path.join(out_dir, f"{safe_n}.json")
                    _dump_json(out_path, deepcopy(u.scene_cfg))
                    worst = m["after_measure"]["worst"]
                    print(
                        f"[SAFE] n={n_clutter} {safe_n:02d}/{NUM_SAFE_PER_COUNT} saved={out_path} "
                        f"max_err_xy={m['after_measure']['max_err_xy_mm']:.3f}mm worst={worst} "
                        f"target={target_state} clutter={clutter_state}",
                        flush=True,
                    )
                else:
                    if "max_err_xy_mm" in m:
                        print(
                            f"[REJECT] n={n_clutter} attempt={attempt} stage={m.get('stage','?')} "
                            f"max_err_xy={m['max_err_xy_mm']:.3f}mm worst={m.get('worst', None)} "
                            f"target={target_state} clutter={clutter_state}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[REJECT] n={n_clutter} attempt={attempt} reason={m.get('reason','unknown')} "
                            f"target={target_state} clutter={clutter_state}",
                            flush=True,
                        )

            print(f"[DONE] n={n_clutter} generated {NUM_SAFE_PER_COUNT} scenes in: {out_dir}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
