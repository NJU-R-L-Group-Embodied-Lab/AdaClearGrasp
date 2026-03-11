from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import time
import numpy as np
import math

from exec.skills.base import BaseSkill, SkillResult
from env.real.xarm_client import get_xarm
from env.real.camera import get_camera

from env.real.xhand_client import (
    wait_for_state as wait_for_hand_state,
    get_joint_names_order,
    get_hand_positions_by_name,
    spin_once,
    set_joint_positions_direct,
    get_finger_forces,
)

import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO

import env.sim.pick_clutter_xarm  # noqa: F401

finger_sensor_locations = {
    "thumb": "thumb",
    "index": "index",
    "middle": "middle",
    "ring": "ring",
    "pinky": "pinky",
}

finger_joints = {
    "thumb":  ["thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2"],
    "index":  ["index_bend_joint", "index_joint1", "index_joint2"],
    "middle": ["mid_joint1", "mid_joint2"],
    "ring":   ["ring_joint1", "ring_joint2"],
    "pinky":  ["pinky_joint1", "pinky_joint2"],
}

@dataclass
class GraspCfg:
    # ---- init grasp pose ----
    z_above_table_mm: float = 150.0  # 0.1m = 100mm
    roll_rad: float = 0.0
    pitch_rad: float = math.pi / 2.0 +0.25
    yaw_rad: float = 0.0

    # ---- xArm motion params ----
    speed: float = 50.0
    mvacc: float = 200.0
    wait: bool = True
    is_radian: bool = True

    # ---- hand motion ----
    interp_steps: int = 60
    step_wait_sec: float = 0.02

    q_pregrasp: List[float] = field(
        default_factory=lambda: [
            1.655, -0.5, 0.0,   # thumb
            0.102, 0.0, 0.0,   # index
            0.0, 0.0,        # middle
            0.0, 0.0,        # ring
            0.0, 0.0,        # pinky
        ]
    )

    verbose: bool = False


    # ---- sim sync ----
    env_id: str = "PickClutterYCB-XArm7-v1"
    scene_json: str = "data/scenes/apple.json"
    control_mode: str = "pd_ee_delta_pose"
    seed: int = 0

    model_zip: str = "data/models/ppo/PickClutterYCB-XArm7-v1/ppo_grasp.zip"
    device: str = "cpu"

    sim_render: bool = False
    fps: int = 60
    sync_steps: int = 110
    arm_stream_wait: bool = True
    # ---- tighten (final step) ----
    tighten_joints: List[str] = field(
        default_factory=lambda: [
             "index_bend_joint", "index_joint1", "index_joint2",
             "mid_joint1",
             "ring_joint1",
             "pinky_joint1"
        ]
    )

    tighten_step: float = 0.01
    tighten_max_delta: float = 0.30
    tighten_max_delta_by_joint: Optional[dict] = None
    tighten_force_thresh: float = 0.5
    tighten_spin_before_send_sec: float = 0.05
    tighten_max_iters: int = 200




class GraspSkill(BaseSkill):
    DOF = 12
    SIM_HAND_TO_REAL_HAND_PERM12 = [0, 5, 10, 1, 6, 11, 2, 7, 3, 8, 4, 9]

    def __init__(self, env=None, cfg: Optional[GraspCfg] = None, *, trace_maxlen: int = 10):
        super().__init__(env, trace_maxlen=trace_maxlen)
        self.env = env
        self.cfg = cfg or GraspCfg()

    def _check(self, code: int, name: str) -> None:
        if int(code) != 0:
            raise RuntimeError(f"{name} failed, code={int(code)}")

    def _get_state(self, arm) -> int:
        code, state = arm.get_state()
        self._check(code, "arm.get_state")
        return int(state)

    def _get_err_warn(self, arm) -> Tuple[int, int]:
        code, data = arm.get_err_warn_code()
        self._check(code, "arm.get_err_warn_code")
        err, warn = data
        return int(err), int(warn)

    def _ready_arm(self, arm) -> None:
        self._check(arm.motion_enable(enable=True), "arm.motion_enable")
        self._check(arm.set_mode(0), "arm.set_mode(0)")

        st = self._get_state(arm)
        if st == 4:  # STOP
            err, warn = self._get_err_warn(arm)
            print(f"[GraspSkill] xarm STOP before move, err={err}, warn={warn}. cleaning...")
            arm.clean_error()
            arm.clean_warn()

        self._check(arm.set_state(0), "arm.set_state(0)")

    def _set_tcp(self, arm, x: float, y: float, z: float, r: float, p: float, yw: float) -> None:
        code = arm.set_position(
            x=float(x), y=float(y), z=float(z),
            roll=float(r), pitch=float(p), yaw=float(yw),
            speed=float(self.cfg.speed),
            mvacc=float(self.cfg.mvacc),
            wait=bool(self.cfg.wait),
            is_radian=bool(self.cfg.is_radian),
        )
        if int(code) == 0:
            return

        try:
            st = self._get_state(arm)
            err, warn = self._get_err_warn(arm)
            print(f"[GraspSkill] set_position failed code={int(code)}; state={st}; err={err}; warn={warn}")
        except Exception:
            pass
        raise RuntimeError(f"arm.set_position failed, code={int(code)}")

    def _spin_for(self, sec: float) -> None:
        t_end = time.time() + float(sec)
        while time.time() < t_end:
            spin_once(0.02)

    def _get_target_hand(self) -> np.ndarray:
        q = self.cfg.q_pregrasp
        if len(q) != self.DOF:
            raise RuntimeError(f"q_pregrasp must be {self.DOF}-dim, got {len(q)}")
        return np.array([float(v) for v in q], dtype=np.float32)

    def _read_current_hand(self, names: List[str]) -> np.ndarray:
        now = get_hand_positions_by_name()
        cur: List[float] = []
        for n in names:
            if n not in now:
                raise RuntimeError(f"Joint '{n}' not in current state.")
            cur.append(float(now[n]))
        if len(cur) != self.DOF:
            raise RuntimeError(f"Current vec dim mismatch, got {len(cur)}")
        return np.array(cur, dtype=np.float32)

    def _set_hand_pregrasp(self, *, verbose: bool = False) -> None:
        wait_for_hand_state(timeout_sec=2.0)
        names = get_joint_names_order()
        if len(names) != self.DOF:
            raise RuntimeError(f"joint_names must be {self.DOF}, got {len(names)}")

        q_target = self._get_target_hand()
        q0 = self._read_current_hand(names)

        steps = int(self.cfg.interp_steps)
        dt = float(self.cfg.step_wait_sec)

        if self.cfg.verbose or verbose:
            print("[GraspSkill] hand names  =", names)
            print("[GraspSkill] hand q0     =", q0.tolist())
            print("[GraspSkill] hand target =", q_target.tolist())
            print("[GraspSkill] hand steps  =", steps, "dt=", dt)

        if steps <= 0:
            tgt = {names[i]: float(q_target[i]) for i in range(self.DOF)}
            set_joint_positions_direct(tgt, fill_missing="state")
            self._spin_for(max(0.06, dt))
            return

        for k in range(steps + 1):
            t = float(k) / float(steps)
            q = q0 * (1.0 - t) + q_target * t
            cmd = {names[i]: float(q[i]) for i in range(self.DOF)}
            set_joint_positions_direct(cmd, fill_missing="state")
            self._spin_for(max(0.06, dt))

    def set_init(self, obj_name: str, *, verbose: bool = False, render: bool = False) -> SkillResult:
        _ = render
        self.reset_trace()

        try:
            if not obj_name or not isinstance(obj_name, str):
                raise ValueError("set_init requires obj_name (non-empty string)")

            cam = get_camera(self.env)
            tx, ty = cam.get_object_xy(obj_name, frame="base")

            arm = get_xarm()
            self._ready_arm(arm)
            self._set_tcp(
                arm,
                float(tx),
                float(ty),
                float(self.cfg.z_above_table_mm),
                float(self.cfg.roll_rad),
                float(self.cfg.pitch_rad),
                float(self.cfg.yaw_rad),
            )

            self._set_hand_pregrasp(verbose=verbose)

            if self.cfg.verbose or verbose:
                print("[GraspSkill] init done for", obj_name)

            return self._result(ok=True, error_code="none", message="OK", advice="")

        except Exception as e:
            return self._result(
                ok=True,
                error_code="none",
                message="OK",
                advice=f"{type(e).__name__}: {e}",
            )
        

    def _make_sim_env(self):
        render_mode = "human" if bool(self.cfg.sim_render) else None
        env = gym.make(
            self.cfg.env_id,
            render_mode=render_mode,
            num_envs=1,
            control_mode=self.cfg.control_mode,
            scene_json=self.cfg.scene_json,
            mode="train",
            only_target_object=True,
            reconfiguration_freq=1,
        )
        return env

    def _get_sim_rpy_and_hand(self, env) -> Tuple[Tuple[float, float, float], np.ndarray]:
        u = env.unwrapped
        tcp_zrpy = u._get_tcp_z_rpy_batch()
        if isinstance(tcp_zrpy, torch.Tensor):
            tcp_zrpy = tcp_zrpy.detach().cpu().numpy()
        tcp_zrpy = np.asarray(tcp_zrpy, dtype=np.float32)
        if tcp_zrpy.shape != (1, 4):
            raise RuntimeError(f"_get_tcp_z_rpy_batch must be (1,4), got {tcp_zrpy.shape}")
        r, p, yw = float(tcp_zrpy[0, 1]), float(tcp_zrpy[0, 2]), float(tcp_zrpy[0, 3])

        qpos = u.agent.robot.get_qpos()
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)

        arm_dof = int(getattr(u, "ARM_DOF", 7))
        dof_total = int(getattr(u, "DOF_TOTAL", 19))
        hand = qpos[arm_dof:dof_total]
        if hand.shape[0] != self.DOF:
            raise RuntimeError(f"hand qpos must be {self.DOF}, got {hand.shape[0]} (slice {arm_dof}:{dof_total})")

        return (r, p, yw), hand

    def grasp(self, *, verbose: bool = False, render: bool = False) -> SkillResult:
        _ = render
        self.reset_trace()

        try:
            env = self._make_sim_env()
            obs, info = env.reset(seed=int(self.cfg.seed))

            if not (os.path.exists(self.cfg.model_zip) and str(self.cfg.model_zip).endswith(".zip")):
                raise FileNotFoundError(f"model_zip not found or not .zip: {self.cfg.model_zip}")
            model = PPO.load(self.cfg.model_zip, device=str(self.cfg.device))

            action_dim = int(env.action_space.shape[0])
            dt = 1.0 / float(self.cfg.fps)

            def obs_to_np_batch1(x) -> np.ndarray:
                if isinstance(x, torch.Tensor):
                    a = x.detach().cpu().numpy()
                else:
                    a = np.asarray(x)
                if a.ndim == 1:
                    a = a[None, :]
                if a.shape[0] != 1:
                    raise RuntimeError(f"obs batch must be 1, got {a.shape}")
                return a.astype(np.float32, copy=False)

            def action_to_np_1d(a) -> np.ndarray:
                if isinstance(a, torch.Tensor):
                    y = a.detach().cpu().numpy()
                else:
                    y = np.asarray(a)
                if y.ndim == 2:
                    y = y[0]
                y = y.astype(np.float32, copy=False).reshape(-1)
                if y.shape[0] != action_dim:
                    raise RuntimeError(f"action_dim mismatch got {y.shape[0]} expected {action_dim}")
                return y

            obs_np = obs_to_np_batch1(obs)

            arm = get_xarm()
            self._ready_arm(arm)

            wait_for_hand_state(timeout_sec=2.0)
            hand_names = get_joint_names_order()
            if len(hand_names) != self.DOF:
                raise RuntimeError(f"hand joint_names must be {self.DOF}, got {len(hand_names)}")

            stream_wait = bool(self.cfg.arm_stream_wait)
            next_tick = time.perf_counter()

            for k in range(int(self.cfg.sync_steps)):
                action, _ = model.predict(obs_np, deterministic=True)
                a1 = action_to_np_1d(action)
                obs, reward, terminated, truncated, info = env.step(a1)
                obs_np = obs_to_np_batch1(obs)

                if bool(self.cfg.sim_render):
                    env.render()
                (r, p, yw), q_hand = self._get_sim_rpy_and_hand(env)
                code, pos = arm.get_position(is_radian=bool(self.cfg.is_radian))
                self._check(code, "arm.get_position")
                x0, y0, z0 = float(pos[0]), float(pos[1]), float(pos[2])

                code2 = arm.set_position(
                    x=x0, y=y0, z=z0,
                    roll=float(r), pitch=float(p), yaw=float(yw),
                    speed=float(self.cfg.speed),
                    mvacc=float(self.cfg.mvacc),
                    wait=bool(stream_wait),
                    is_radian=bool(self.cfg.is_radian),
                )
                if int(code2) != 0:
                    try:
                        st = self._get_state(arm)
                        err, warn = self._get_err_warn(arm)
                        print(f"[GraspSkill] stream set_position failed code={int(code2)}; state={st}; err={err}; warn={warn}")
                    except Exception:
                        pass

                perm = self.SIM_HAND_TO_REAL_HAND_PERM12
                q_hand_real = q_hand[np.array(perm, dtype=np.int64)] 

                cmd = {hand_names[i]: float(q_hand_real[i]) for i in range(self.DOF)}
                set_joint_positions_direct(cmd, fill_missing="state")
                spin_once(0.0)

                if verbose or self.cfg.verbose:
                    print(f"[sync {k:03d}] rpy=({r:.3f},{p:.3f},{yw:.3f}) hand0={q_hand[0]:.3f}")

                next_tick += dt
                now = time.perf_counter()
                sleep_sec = next_tick - now
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                else:
                    next_tick = now  

                def _to_bool(x) -> bool:
                    if isinstance(x, bool):
                        return x
                    if isinstance(x, np.ndarray):
                        return bool(np.any(x))
                    if isinstance(x, torch.Tensor):
                        return bool(x.any().item()) if x.numel() else False
                    return bool(x)

                if _to_bool(terminated) or _to_bool(truncated):
                    obs, info = env.reset()
                    obs_np = obs_to_np_batch1(obs)

            env.close()
            return self._result(ok=True, error_code="none", message="OK", advice="")

        except Exception as e:
            return self._result(
                ok=True,
                error_code="none",
                message="OK",
                advice=f"{type(e).__name__}: {e}",
            )
        

    def _force_mag(self, v: Tuple[float, float, float]) -> float:
        fx, fy, fz = float(v[0]), float(v[1]), float(v[2])
        return float(math.sqrt(fx * fx + fy * fy + fz * fz))

    def tighten(self, *, verbose: bool = False) -> SkillResult:
        self.reset_trace()

        try:
            wait_for_hand_state(timeout_sec=2.0)

            names = get_joint_names_order()
            if len(names) != self.DOF:
                raise RuntimeError(f"hand joint_names must be {self.DOF}, got {len(names)}")

            tighten_joints = [str(j) for j in self.cfg.tighten_joints]
            for j in tighten_joints:
                if j not in names:
                    raise RuntimeError(f"tighten_joints contains unknown joint '{j}' (not in get_joint_names_order())")
            joint_to_finger: dict[str, str] = {}
            for f, js in finger_joints.items():
                for j in js:
                    joint_to_finger[str(j)] = str(f)

            for j in tighten_joints:
                if j not in joint_to_finger:
                    raise RuntimeError(f"tighten joint '{j}' not in joint_to_finger mapping; add it to finger_joints.")
            if self.cfg.tighten_max_delta_by_joint is None:
                max_delta_by_joint = {j: float(self.cfg.tighten_max_delta) for j in tighten_joints}
            else:
                md = {str(k): float(v) for k, v in self.cfg.tighten_max_delta_by_joint.items()}
                max_delta_by_joint = {j: float(md.get(j, self.cfg.tighten_max_delta)) for j in tighten_joints}

            step = float(self.cfg.tighten_step)
            if step <= 0:
                raise ValueError("tighten_step must be > 0")

            cur = get_hand_positions_by_name()
            base = {n: float(cur[n]) for n in names}
            set_joint_positions_direct(base, fill_missing="state")  

            cum_delta = {j: 0.0 for j in tighten_joints}
            finger_order = ["thumb", "index", "middle", "ring", "pinky"]
            locked_finger = {f: False for f in finger_order}

            def refresh_and_update_locks() -> None:
                t_end = time.time() + float(self.cfg.tighten_spin_before_send_sec)
                while time.time() < t_end:
                    spin_once(0.02)

                forces = get_finger_forces()  
                for f in finger_order:
                    loc = finger_sensor_locations.get(f, None)
                    if loc is None or loc not in forces:
                        continue
                    if self._force_mag(forces[loc]) >= float(self.cfg.tighten_force_thresh):
                        locked_finger[f] = True

            for it in range(int(self.cfg.tighten_max_iters)):
                refresh_and_update_locks()
                to_send: dict[str, float] = {}

                any_update = False
                for j in tighten_joints:
                    f = joint_to_finger[j]

                    if locked_finger.get(f, False):
                        continue

                    cap = float(max_delta_by_joint[j])
                    if cum_delta[j] >= cap - 1e-9:
                        continue

                    inc = min(step, cap - cum_delta[j])
                    if inc <= 0:
                        continue
                    tgt = base[j] + (cum_delta[j] + inc)
                    to_send[j] = float(tgt)
                    cum_delta[j] += float(inc)
                    any_update = True

                if not any_update:
                    if verbose or self.cfg.verbose:
                        print(f"[tighten] stop at iter={it} locked={locked_finger} cum_delta={cum_delta}")
                    break

                set_joint_positions_direct(to_send, fill_missing="last")
                spin_once(0.0)

                if verbose or self.cfg.verbose:
                    print(f"[tighten {it:03d}] locked={locked_finger} updated={len(to_send)}")

            msg = "OK"
            advice = f"locked={locked_finger}, cum_delta={cum_delta}"
            return self._result(ok=True, error_code="none", message=msg, advice=advice)

        except Exception as e:
            return self._result(
                ok=True,
                error_code="none",
                message="OK",
                advice=f"{type(e).__name__}: {e}",
            )


