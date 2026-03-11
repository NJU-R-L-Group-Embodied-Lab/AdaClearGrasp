# env/sim/pick_clutter_xarm.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
import sapien.physx as physx

from mani_skill.agents.robots import Fetch, Panda, XArm7Ability
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from core.perception.geometry.nearest_vectors import joints_to_pointcloud_nearest_vectors_batch
from core.perception.geometry.rigid_pointcloud import (
    world_to_local_points_wxyz,
    local_to_world_points_batch_wxyz,
)

from .gt_pointcloud import sample_world_points_on_actor_strict
from .robot.xarm_hand_robot import XArm7Xhand


@register_env("PickClutterYCB-XArm7-v1", max_episode_steps=300)
class PickClutterYCBXArm7Env(BaseEnv):
    ENV_ID = "PickClutterYCB-XArm7-v1"
    SUPPORTED_ROBOTS = ["panda", "fetch", "xarm7_xhand", "xarm7_ability"]
    agent: Union[Panda, Fetch, XArm7Xhand, XArm7Ability]

    # ========= hard constants =========
    ARM_DOF = 7
    HAND_DOF = 12
    DOF_TOTAL = 19

    HAND_LINK_COUNT = 18

    # obs: 54(nnvec=18*3) + 1(target_height) + 4(tcp_z + rpy) = 59
    NN_FLAT_DIM = HAND_LINK_COUNT * 3
    OBS_DIM = NN_FLAT_DIM + 1 + 4  # 59

    # ===== contact reward =====
    CONTACT_MIN_FORCE = 0.5
    CONTACT_REWARD_SCALE = 10.0

    # ===== nn delta reward scale (encourage distance decrease) =====
    NN_DELTA_REWARD_SCALE = 10.0

    # ===== extra tip2 squared penalty (currently disabled in shaped reward) =====
    TIP2_SQ_PENALTY_SCALE = 10.0
    TIP2_SQ_PENALTY_CAP = 0.02
    # ===== target friction params (tune here) =====
    TARGET_STATIC_FRICTION = 2.0
    TARGET_DYNAMIC_FRICTION = 2.0
    TARGET_RESTITUTION = 0.0

    # ===== clutter friction params (tune here) =====
    CLUTTER_STATIC_FRICTION = 1.0
    CLUTTER_DYNAMIC_FRICTION = 1.0
    CLUTTER_RESTITUTION = 0.0

    def __init__(
        self,
        *args,
        robot_uids: str = "xarm7_xhand",
        robot_init_qpos_noise: float = 0.0,
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        scene_json: str = "data/scenes/my_scene.json",
        use_external_arm_init: bool = True,
        only_target_object: bool = True,
        grasp_mode: str = "top",
        mode: str = "train",
        **kwargs,
    ):
        self.mode = str(mode).lower()
        if self.mode not in ("train", "plan"):
            raise ValueError(f"mode must be train/plan, got {self.mode}")
        self.grasp_mode = str(grasp_mode).lower()

        if self.mode == "plan":
            use_external_arm_init = False
            # only_target_object = True

        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.use_external_arm_init = bool(use_external_arm_init)
        self.only_target_object = bool(only_target_object)

        if scene_json is None:
            raise ValueError("scene_json must be provided.")
        if not os.path.exists(scene_json):
            raise FileNotFoundError(f"scene_json not found: {scene_json}")
        self.scene_cfg: Dict = load_json(scene_json)

        self.obs_dim = int(self.OBS_DIM)

        # caches
        self._arm_init_qpos_one: torch.Tensor | None = None  # (7,)
        self._target_xyz_local: torch.Tensor | None = None   # (N,3)

        self._table_top_z: float | None = None
        self._target_half_height: float | None = None
        self._clutter_half_heights: List[float] = []

        self.target_base_name: str | None = None
        self.clutter_base_names: List[str] = []

        # per clutter item: [num_clutter][num_envs]
        self.clutter_actors_per_item: List[List[Actor]] = []

        # ===== per-episode / per-step caches =====
        self._height0: torch.Tensor | None = None            # (B,)
        self._prev_nn_sumdist: torch.Tensor | None = None    # (B,)

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

        flat = self._flatten_raw_obs()
        if flat.numel() != self.obs_dim:
            raise RuntimeError(f"_flatten_raw_obs returns {flat.numel()} but obs_dim={self.obs_dim}")

    # ------------------- ManiSkill defaults -------------------

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19,
            ),
            scene_config=SceneConfig(
                gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.5, 0.8], target=[0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.8, 0.6, 0.7], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1,
            near=0.01,
            far=100,
        )

    # ------------------- Assets & agent -------------------

    def _load_model(self, model_id: str) -> ActorBuilder:
        return actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _actor_name(self, base_name: str, env_i: int) -> str:
        if self.num_envs == 1:
            return base_name
        return f"env{env_i}__{base_name}"

    # ------------------- Scene helpers -------------------

    def _actor_local_half_height(self, actor: Actor) -> float:
        mesh = actor.get_first_collision_mesh(to_world_frame=False)
        if mesh is None:
            raise RuntimeError(f"Actor {actor.name} has no collision mesh.")
        zmin = float(mesh.bounds[0, 2])
        zmax = float(mesh.bounds[1, 2])
        return 0.5 * (zmax - zmin)

    def _get_table_actor(self) -> Actor:
        if not hasattr(self.scene_builder, "table"):
            raise RuntimeError("TableSceneBuilder has no attribute `table`.")
        return self.scene_builder.table

    def _compute_table_top_z(self) -> float:
        table = self._get_table_actor()
        mesh_local = table.get_first_collision_mesh(to_world_frame=False)
        if mesh_local is None:
            raise RuntimeError("Table actor has no collision mesh.")
        if not hasattr(table, "_objs") or len(table._objs) == 0:
            raise RuntimeError("Table actor has no underlying sapien entities.")
        ent = table._objs[0]
        pz = float(ent.pose.p[2])
        top_local = float(mesh_local.bounds[1, 2])
        return pz + top_local
    
    def _set_target_friction(self, static_f: float, dynamic_f: float, restitution: float = 0.0):
        if not hasattr(self, "target_object"):
            raise RuntimeError("target_object not built yet.")
        bodies = self.target_object._bodies
        if bodies is None or len(bodies) == 0:
            raise RuntimeError(f"Actor {self.target_object.name} has no PhysX bodies (_bodies).")

        mat = physx.PhysxMaterial(float(static_f), float(dynamic_f), float(restitution))

        for body in bodies:
            for cs in body.get_collision_shapes():
                cs.set_physical_material(mat)

    def _set_clutter_friction(self, static_f: float, dynamic_f: float, restitution: float = 0.0):
        if not hasattr(self, "clutter_objects"):
            raise RuntimeError("clutter_objects not built yet.")
        if self.clutter_objects is None:
            raise RuntimeError("clutter_objects is None.")
        if len(self.clutter_objects) == 0:
            return  # 没有杂物就不用设

        mat = physx.PhysxMaterial(float(static_f), float(dynamic_f), float(restitution))

        for clutter in self.clutter_objects:
            bodies = clutter._bodies
            if bodies is None or len(bodies) == 0:
                raise RuntimeError(f"Actor {clutter.name} has no PhysX bodies (_bodies).")
            for body in bodies:
                for cs in body.get_collision_shapes():
                    cs.set_physical_material(mat)


    # ------------------- Target pose/pointcloud (batched) -------------------

    def _get_target_pose_pq_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ps: List[torch.Tensor] = []
        qs: List[torch.Tensor] = []
        for i in range(self.num_envs):
            pi = self.target_actors[i].pose.p
            qi = self.target_actors[i].pose.q

            pi_t = torch.as_tensor(pi) if not isinstance(pi, torch.Tensor) else pi
            qi_t = torch.as_tensor(qi) if not isinstance(qi, torch.Tensor) else qi

            ps.append(pi_t.reshape(-1)[:3].to(self.device, torch.float32))
            qs.append(qi_t.reshape(-1)[:4].to(self.device, torch.float32))

        p = torch.stack(ps, dim=0)
        q = torch.stack(qs, dim=0)
        return p, q

    def _init_target_pointcloud_local_once(self, num_points: int = 1024):
        if self._target_xyz_local is not None:
            return

        actor0 = self.target_actors[0]
        xyz_w_np = sample_world_points_on_actor_strict(actor0, num_points)  # (N,3) numpy
        xyz_w = torch.as_tensor(xyz_w_np, device=self.device, dtype=torch.float32)

        p0 = actor0.pose.p
        q0 = actor0.pose.q
        p0 = torch.as_tensor(p0, device=self.device, dtype=torch.float32) if not isinstance(p0, torch.Tensor) else p0.to(self.device, torch.float32)
        q0 = torch.as_tensor(q0, device=self.device, dtype=torch.float32) if not isinstance(q0, torch.Tensor) else q0.to(self.device, torch.float32)

        p0 = p0.reshape(-1)[:3]
        q0 = q0.reshape(-1)[:4]
        self._target_xyz_local = world_to_local_points_wxyz(xyz_w, p0, q0)  # (N,3)

    def _get_target_pointcloud_world_batch(self) -> torch.Tensor:
        if self._target_xyz_local is None:
            raise RuntimeError("_target_xyz_local is None; call _init_target_pointcloud_local_once() first.")
        p, q = self._get_target_pose_pq_batch()
        xyz = local_to_world_points_batch_wxyz(self._target_xyz_local, p, q)  # (B,N,3)
        if xyz.ndim != 3 or xyz.shape[0] != self.num_envs or xyz.shape[2] != 3:
            raise RuntimeError(f"target_xyz must be (B,N,3), got {tuple(xyz.shape)}")
        return xyz

    def _get_target_height_batch(self) -> torch.Tensor:
        heights = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        for i in range(self.num_envs):
            pi = self.target_actors[i].pose.p
            pi_t = torch.as_tensor(pi) if not isinstance(pi, torch.Tensor) else pi
            pi_t = pi_t.to(self.device, torch.float32).reshape(-1)[:3]
            heights[i, 0] = pi_t[2]
        return heights

    # ------------------- TCP pose (z + rpy) -------------------

    def _quat_wxyz_to_rpy(self, q_wxyz: torch.Tensor) -> torch.Tensor:
        if q_wxyz.shape[-1] != 4:
            raise RuntimeError(f"quat must be (...,4), got {tuple(q_wxyz.shape)}")
        w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=-1)

    def _get_tcp_z_rpy_batch(self) -> torch.Tensor:
        tcp = getattr(self.agent, "tcp", None)
        if tcp is None:
            raise RuntimeError("agent.tcp not found; please ensure your agent exposes tcp")

        pose = tcp.pose
        p = pose.p
        q = pose.q

        if not isinstance(p, torch.Tensor):
            p = torch.as_tensor(p, device=self.device, dtype=torch.float32)
        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q, device=self.device, dtype=torch.float32)

        p = p.to(self.device, torch.float32)[..., :3]
        q = q.to(self.device, torch.float32)[..., :4]

        if p.ndim == 1:
            p = p.unsqueeze(0)
        if q.ndim == 1:
            q = q.unsqueeze(0)

        if p.shape != (self.num_envs, 3):
            raise RuntimeError(f"tcp pose.p must be (B,3) with B={self.num_envs}, got {tuple(p.shape)}")
        if q.shape != (self.num_envs, 4):
            raise RuntimeError(f"tcp pose.q must be (B,4) with B={self.num_envs}, got {tuple(q.shape)}")

        rpy = self._quat_wxyz_to_rpy(q)   # (B,3)
        tcp_z = p[:, 2:3]                 # (B,1)
        return torch.cat([tcp_z, rpy], dim=-1)  # (B,4)


    # ------------------- NN vectors -------------------

    def _get_hand_link_poses_l_b_3(self) -> torch.Tensor:
        x = self.agent.hand_link_poses
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("agent.hand_link_poses must be torch.Tensor")

        x = x.to(self.device, torch.float32)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise RuntimeError(f"hand_link_poses must be (L,B,3), got {tuple(x.shape)}")
        if x.shape[0] != self.HAND_LINK_COUNT:
            raise RuntimeError(f"hand_link_poses L must be {self.HAND_LINK_COUNT}, got {x.shape[0]}")
        if x.shape[1] != self.num_envs:
            raise RuntimeError(f"hand_link_poses B must be num_envs={self.num_envs}, got {x.shape[1]}")
        return x  # (L,B,3)

    def _compute_nn_vectors(self, hand_links_l_b_3: torch.Tensor, target_xyz_b_n_3: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hand_links_l_b_3.shape != (self.HAND_LINK_COUNT, self.num_envs, 3):
            raise RuntimeError(f"hand_links must be (L,B,3), got {tuple(hand_links_l_b_3.shape)}")
        if target_xyz_b_n_3.ndim != 3 or target_xyz_b_n_3.shape[0] != self.num_envs or target_xyz_b_n_3.shape[2] != 3:
            raise RuntimeError(f"target_xyz must be (B,N,3), got {tuple(target_xyz_b_n_3.shape)}")

        es = joints_to_pointcloud_nearest_vectors_batch(hand_links_l_b_3, target_xyz_b_n_3, normalize=True)
        vec = es["vectors"]
        dist = es["distances"]

        if vec.shape != (self.num_envs, self.HAND_LINK_COUNT, 3):
            raise RuntimeError(f"vectors must be (B,L,3), got {tuple(vec.shape)}")
        if dist.shape != (self.num_envs, self.HAND_LINK_COUNT):
            raise RuntimeError(f"distances must be (B,L), got {tuple(dist.shape)}")

        return {"vectors": vec, "distances": dist}

    # ------------------- Obs build (object-centric) -------------------

    def _build_obs(self, nn_vectors_b_l_3: torch.Tensor, target_height_b_1: torch.Tensor, tcp_zrpy_b_4: torch.Tensor) -> torch.Tensor:
        if nn_vectors_b_l_3.shape != (self.num_envs, self.HAND_LINK_COUNT, 3):
            raise RuntimeError(f"nn_vectors must be (B,L,3), got {tuple(nn_vectors_b_l_3.shape)}")
        if target_height_b_1.shape != (self.num_envs, 1):
            raise RuntimeError(f"target_height must be (B,1), got {tuple(target_height_b_1.shape)}")
        if tcp_zrpy_b_4.shape != (self.num_envs, 4):
            raise RuntimeError(f"tcp_zrpy must be (B,4), got {tuple(tcp_zrpy_b_4.shape)}")

        nn_flat = nn_vectors_b_l_3.reshape(self.num_envs, self.NN_FLAT_DIM)  # (B,54)
        obs = torch.cat([nn_flat, target_height_b_1, tcp_zrpy_b_4], dim=-1)  # (B,59)
        if obs.shape != (self.num_envs, self.obs_dim):
            raise RuntimeError(f"obs must be (B,{self.obs_dim}), got {tuple(obs.shape)}")
        return obs

    # ------------------- Scene build -------------------

    def _build_target(self) -> Actor:
        cfg = self.scene_cfg.get("target", None)
        if cfg is None:
            raise RuntimeError("scene_json must contain `target`.")
        if str(cfg.get("type", "ycb")) != "ycb":
            raise RuntimeError("Only target.type=='ycb' is implemented.")

        model_id = str(cfg["model_id"])
        base_name = str(cfg.get("name", "target"))
        self.target_base_name = base_name

        target_actors: List[Actor] = []
        for i in range(self.num_envs):
            builder = self._load_model(model_id)
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            a = builder.build(name=self._actor_name(base_name, i))
            target_actors.append(a)

        self.target_object = Actor.merge(target_actors, name=base_name)
        self.target_actors = target_actors
        return self.target_object

    def _build_clutter(self) -> Actor:
        clutter_cfg = self.scene_cfg.get("clutter", [])
        if not isinstance(clutter_cfg, list):
            raise RuntimeError("scene_json: `clutter` must be a list.")

        all_clutter: List[Actor] = []
        self._clutter_half_heights = []
        self.clutter_base_names = []
        self.clutter_actors_per_item = []

        for j, item in enumerate(clutter_cfg):
            if str(item.get("type", "ycb")) != "ycb":
                raise RuntimeError("Only clutter.type=='ycb' is implemented.")

            model_id = str(item["model_id"])
            base_name = str(item.get("name", f"clutter{j}"))
            self.clutter_base_names.append(base_name)

            per_env: List[Actor] = []
            for i in range(self.num_envs):
                builder = self._load_model(model_id)
                builder.set_scene_idxs([i])
                builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
                a = builder.build(name=self._actor_name(base_name, i))
                per_env.append(a)

            self.clutter_actors_per_item.append(per_env)

            merged = Actor.merge(per_env, name=base_name)
            all_clutter.append(merged)
            self._clutter_half_heights.append(self._actor_local_half_height(per_env[0]))

        self.clutter_objects = all_clutter
        if len(all_clutter) == 0:
            self.all_objects = self.target_object
            return self.all_objects

        self.all_objects = Actor.merge([self.target_object] + all_clutter, name="all_objects")
        return self.all_objects

    def _remove_non_target_objects_collision(self):
        for clutter in getattr(self, "clutter_objects", []):
            for g in range(4):
                clutter.set_collision_group(g, 0)

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.scene_builder.build()

        self._build_target()

        self._set_target_friction(
            static_f=float(self.TARGET_STATIC_FRICTION),
            dynamic_f=float(self.TARGET_DYNAMIC_FRICTION),
            restitution=float(self.TARGET_RESTITUTION),
        )
        self._build_clutter()
        # self._set_clutter_friction(
        #     static_f=float(self.CLUTTER_STATIC_FRICTION),
        #     dynamic_f=float(self.CLUTTER_DYNAMIC_FRICTION),
        #     restitution=float(self.CLUTTER_RESTITUTION),
        # )

        self._table_top_z = self._compute_table_top_z()
        self._target_half_height = self._actor_local_half_height(self.target_actors[0])

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

        if self.only_target_object:
            self._remove_non_target_objects_collision()

    # ------------------- IK init (once, but only after target pose is correct) -------------------

    def _precompute_arm_init_qpos_one(self):
        c = self.target_actors[0].pose.p
        c = torch.as_tensor(c) if not isinstance(c, torch.Tensor) else c
        c = c.reshape(-1)[:3].to(self.device, torch.float32).unsqueeze(0)  # (1,3)

        from core.kinematics.xarm7_xhand_ik import (
            solve_ik_palm_down_rollfree,
            solve_ik_side_grasp,
            ARM_JOINT_COUNT,
        )

        if self.grasp_mode == "top":
            qpos = solve_ik_palm_down_rollfree(self, c, height_offset=0.10)
            arm = qpos[0, :ARM_JOINT_COUNT]
        elif self.grasp_mode == "side":
            qpos = solve_ik_side_grasp(self, c, yaw=0.0)
            arm = qpos[0, :ARM_JOINT_COUNT]
        else:
            raise ValueError(f"Unknown grasp_mode: {self.grasp_mode}")

        arm = arm.detach().to(self.device, torch.float32).reshape(-1)
        if arm.numel() != self.ARM_DOF:
            raise RuntimeError(f"arm init qpos dim mismatch: got {arm.numel()} expected {self.ARM_DOF}")
        self._arm_init_qpos_one = arm

    # ------------------- init episode/agent -------------------

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            if isinstance(env_idx, np.ndarray):
                env_idx = torch.from_numpy(env_idx).to(self.device)
            env_idx = env_idx.to(self.device, torch.long)
            b = int(env_idx.numel())

            dof = int(self.agent.robot.dof[0].item() if isinstance(self.agent.robot.dof, torch.Tensor) else self.agent.robot.dof)
            if dof < self.DOF_TOTAL:
                raise RuntimeError(f"robot dof < {self.DOF_TOTAL} not supported, got dof={dof}")

            init_qpos = torch.zeros((b, dof), dtype=torch.float32, device=self.device)

            if self.use_external_arm_init:
                if self._arm_init_qpos_one is None:
                    raise RuntimeError("use_external_arm_init=True but _arm_init_qpos_one is None.")
                arm_init = self._arm_init_qpos_one.unsqueeze(0).expand(b, -1)
            else:
                arm_init = torch.tensor(
                    [-0.77920043, 0.4921763, 0.74077785, 1.2483665, 2.8245757, 0.6868453, -0.050859887],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0).expand(b, -1)

            hand_init = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [1.655, 0.102, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).expand(b, -1)

            init_qpos[:, 0:self.ARM_DOF] = arm_init
            init_qpos[:, self.ARM_DOF:self.DOF_TOTAL] = hand_init
            self.agent.reset(init_qpos)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            if isinstance(env_idx, np.ndarray):
                env_idx = torch.from_numpy(env_idx).to(self.device)
            env_idx = env_idx.to(self.device, torch.long)

            self.scene_builder.initialize(env_idx)

            if self._table_top_z is None or self._target_half_height is None:
                raise RuntimeError("Scene caches not initialized.")

            z_margin = float(self.scene_cfg.get("z_margin", 0.003))

            # target per-env set_pose
            target_cfg = self.scene_cfg["target"]
            txy = target_cfg.get("xy", [0.0, 0.0])
            tx = float(txy[0])
            ty = float(txy[1])
            tyaw = float(target_cfg.get("yaw", 0.0))

            half = torch.tensor(tyaw * 0.5, device=self.device, dtype=torch.float32)
            qw = float(torch.cos(half))
            qz = float(torch.sin(half))
            pz = float(self._table_top_z + self._target_half_height + z_margin)

            for i in env_idx.tolist():
                self.target_actors[i].set_pose(Pose.create_from_pq(p=[tx, ty, pz], q=[qw, 0.0, 0.0, qz]))

            self._init_target_pointcloud_local_once(num_points=1024)

            # clutter per-env set_pose
            clutter_cfg = self.scene_cfg.get("clutter", [])
            if len(clutter_cfg) != len(self.clutter_actors_per_item):
                raise RuntimeError("clutter length mismatch.")

            for j, item in enumerate(clutter_cfg):
                xy = item["xy"]
                x = float(xy[0])
                y = float(xy[1])
                yaw = float(item.get("yaw", 0.0))

                halfj = torch.tensor(yaw * 0.5, device=self.device, dtype=torch.float32)
                qwj = float(torch.cos(halfj))
                qzj = float(torch.sin(halfj))
                pjz = float(self._table_top_z + self._clutter_half_heights[j] + z_margin)

                per_env_actors = self.clutter_actors_per_item[j]
                for i in env_idx.tolist():
                    per_env_actors[i].set_pose(Pose.create_from_pq(p=[x, y, pjz], q=[qwj, 0.0, 0.0, qzj]))

            if self.use_external_arm_init:
                if bool(options.get("reconfigure", False)):
                    self._arm_init_qpos_one = None
                if self._arm_init_qpos_one is None:
                    self._precompute_arm_init_qpos_one()

            self._initialize_agent(env_idx)

    # ------------------- Step/reset -------------------

    def reset_step(self):
        self._step_count = 0

    def step(self, action, grasp: bool = False):
        a = action.copy() if isinstance(action, np.ndarray) else action.clone()

        if grasp or self.mode == "train":
            if a.ndim == 1:
                if self._step_count <= 100:
                    a[:6] *= 0.1
                else:
                    a[:6] = 0.0
                a[2] = 0.1 if self._step_count > 100 else 0.0
            else:
                if self._step_count <= 100:
                    a[..., :6] *= 0.1
                else:
                    a[..., :6] = 0.0
                a[..., 2] = 0.1 if self._step_count > 100 else 0.0

        self._step_count += 1
        _, reward, terminated, truncated, info = super().step(a)

        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        if reward.ndim != 1 or reward.shape[0] != self.num_envs:
            raise RuntimeError(f"unexpected reward shape from super().step: {tuple(reward.shape)}")

        target_xyz = self._get_target_pointcloud_world_batch()
        hand_links = self._get_hand_link_poses_l_b_3()
        es = self._compute_nn_vectors(hand_links, target_xyz)

        distances = es["distances"]  # (B,L)
        nn_vectors = es["vectors"]   # (B,L,3)

        # ===== nn delta reward: prev_sumdist - curr_sumdist =====
        curr_sumdist = distances.sum(dim=1)  # (B,)
        if self._prev_nn_sumdist is None:
            nn_delta_reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        else:
            nn_delta_reward = (self._prev_nn_sumdist - curr_sumdist) * float(self.NN_DELTA_REWARD_SCALE)
        self._prev_nn_sumdist = curr_sumdist.detach()

        # ===== tip2 squared penalty disabled for now =====
        tip2_sq_penalty = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        target_h = self._get_target_height_batch()  # (B,1)
        tcp_zrpy = self._get_tcp_z_rpy_batch()      # (B,4)
        obs = self._build_obs(nn_vectors, target_h, tcp_zrpy)

        shaped = reward #+ nn_delta_reward - curr_sumdist
        shaped = torch.clamp(shaped, min=-100.0, max=100.0)

        info["dense_reward"] = reward
        info["nn_penalty"] = nn_delta_reward
        info["tip2_sq_penalty"] = tip2_sq_penalty
        info["shaped_reward"] = shaped

        ev = self.evaluate()
        info["is_success"] = ev["success"]
        info["success"] = ev["success"]

        return obs, shaped, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self._step_count = 0
        _, info = super().reset(seed, options)

        if options is None:
            env_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ei = options.get("env_idx", None)
            if ei is None:
                env_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            else:
                env_idx = torch.as_tensor(ei, device=self.device, dtype=torch.long)

        if self._target_xyz_local is None:
            self._init_target_pointcloud_local_once(num_points=1024)

        ev0 = self.evaluate()
        h = ev0["height"].detach().to(self.device, torch.float32)

        if self._height0 is None:
            self._height0 = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        if self._prev_nn_sumdist is None:
            self._prev_nn_sumdist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        self._height0[env_idx] = h[env_idx]

        target_xyz = self._get_target_pointcloud_world_batch()
        hand_links = self._get_hand_link_poses_l_b_3()
        es = self._compute_nn_vectors(hand_links, target_xyz)
        self._prev_nn_sumdist[env_idx] = es["distances"].sum(dim=1).detach()[env_idx]

        target_h = self._get_target_height_batch()
        tcp_zrpy = self._get_tcp_z_rpy_batch()
        obs = self._build_obs(es["vectors"], target_h, tcp_zrpy)

        ev = self.evaluate()
        info["is_success"] = ev["success"]
        info["success"] = ev["success"]
        return obs, info

    def _flatten_raw_obs(self, info=None):
        return torch.zeros(self.obs_dim, dtype=torch.float32)

    @property
    def get_ik(self, **kwargs):
        return []

    # ------------------- Reward/eval -------------------

    def evaluate(self):
        heights = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        for i in range(self.num_envs):
            obj = self.target_actors[i]
            pi = obj.pose.p
            pi_t = torch.as_tensor(pi) if not isinstance(pi, torch.Tensor) else pi

            if pi_t.ndim == 2:
                if pi_t.shape != (1, 3):
                    raise RuntimeError(
                        f"[EVALUATE SHAPE ERROR] env={i} pose.p is {tuple(pi_t.shape)}, expected (1,3)"
                    )
                heights[i] = pi_t[0, 2]
            elif pi_t.ndim == 1:
                if pi_t.shape != (3,):
                    raise RuntimeError(
                        f"[EVALUATE SHAPE ERROR] env={i} pose.p is {tuple(pi_t.shape)}, expected (3,)"
                    )
                heights[i] = pi_t[2]
            else:
                raise RuntimeError(
                    f"[EVALUATE SHAPE ERROR] env={i} unexpected pose.p ndim={pi_t.ndim}"
                )

        success = heights > 0.15
        return {
            "success": success,
            "fail": torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool),
            "height": heights,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        rewards = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        if not hasattr(self, "target_actors"):
            return rewards

        evaluation = self.evaluate()
        height = evaluation["height"]                # (B,)
        success = evaluation["success"].float()      # (B,)

        if self._height0 is None:
            raise RuntimeError("_height0 is None; reset() must set it before compute_dense_reward().")

        # ===== height reward uses delta from episode start =====
        height_delta = height - self._height0
        lift_reward = height_delta * 50.0

        success_reward = success * 200.0
        action_penalty = -torch.norm(action, dim=-1) * 0.03

        contact_n = self.agent.count_finger_contacts(
            self.target_object, min_force=self.CONTACT_MIN_FORCE
        )  # (B,) int
        contact_reward = contact_n.to(self.device, torch.float32) * float(self.CONTACT_REWARD_SCALE)

        return lift_reward + success_reward + action_penalty + contact_reward

    def get_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
