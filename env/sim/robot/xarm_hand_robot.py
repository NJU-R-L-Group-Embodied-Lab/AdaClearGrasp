# env/sim/robot/xarm_hand_robot.py
from __future__ import annotations

import os
from copy import deepcopy
from typing import List

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

from core.paths import project_root


@register_agent()
class XArm7Xhand(BaseAgent):
    uid = "xarm7_xhand"

    urdf_path = os.path.join(project_root(), "assets", "xhand_right", "urdf", "xarm7_xhand.urdf")

    urdf_config = dict(
        _materials=dict(
            front_finger=dict(static_friction=4.0, dynamic_friction=1.5, restitution=0.0)
        ),
        link=dict(
            thumnb_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            index_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            middle_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            ring_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            pinky_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
        ),
    )

    disable_self_collisions = True

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0, 0, 0, 1, 0.0, 0.01, 0,
                    4.6845e-02, -4.7024e-03, -3.0234e-03,
                    -5.3306e-03, 1.4591e-02, 1.1627e-01,
                    -4.6615e-04, 2.7319e-02, 5.4885e-02,
                    5.5824e-06, -2.0346e-03, -4.6357e-04
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    def __init__(self, *args, disable_arm_collision: bool = False, **kwargs):
        self.disable_arm_collision = bool(disable_arm_collision)

        self.arm_joint_names: List[str] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.hand_joint_names: List[str] = [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]

        self.hand_link_names: List[str] = [
            "right_hand_link",
            "right_hand_thumb_bend_link",
            "right_hand_thumb_rota_link1",
            "right_hand_thumb_rota_link2",
            "right_hand_thumb_rota_tip",
            "right_hand_index_bend_link",
            "right_hand_index_rota_link1",
            "right_hand_index_rota_link2",
            "right_hand_index_rota_tip",
            "right_hand_mid_link1",
            "right_hand_mid_link2",
            "right_hand_mid_tip",
            "right_hand_ring_link1",
            "right_hand_ring_link2",
            "right_hand_ring_tip",
            "right_hand_pinky_link1",
            "right_hand_pinky_link2",
            "right_hand_pinky_tip",
        ]

        self.ee_link_name = "right_hand_ee_link"

        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 500

        self.hand_stiffness = 3e2
        self.hand_damping = 4e1
        self.hand_force_limit = 1000

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None, None,
            self.arm_stiffness, self.arm_damping, self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1, 0.1,
            self.arm_stiffness, self.arm_damping, self.arm_force_limit,
            use_delta=True,
        )

        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.03, pos_upper=0.03,
            rot_lower=-0.05, rot_upper=0.05,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        hand_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.2, 0.2,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            use_delta=True,
        )
        hand_pd_joint_delta_pos.use_target = False

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=hand_pd_joint_delta_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_pd_joint_delta_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=hand_pd_joint_delta_pos),
        )
        return deepcopy_dict(deepcopy(controller_configs))

    @property
    def hand_link_poses(self) -> torch.Tensor:
        poses = []
        for link in self.hand_links:
            p = link.pose.p
            if not isinstance(p, torch.Tensor):
                p = torch.as_tensor(p, device=self.robot.device, dtype=torch.float32)
            p = p.to(self.robot.device, torch.float32)[..., :3]
            if p.ndim == 1:
                p = p.unsqueeze(0)
            poses.append(p)
        return torch.stack(poses, dim=0)  # (L,B,3)

    def _after_init(self):
        for link in self.robot.get_links():
            if hasattr(link, "set_disable_gravity"):
                link.set_disable_gravity(True)
            elif hasattr(link, "set_enable_gravity"):
                link.set_enable_gravity(False)
            elif hasattr(link, "actor") and hasattr(link.actor, "set_disable_gravity"):
                link.actor.set_disable_gravity(True)

        self.hand_links = sapien_utils.get_objs_by_names(self.robot.get_links(), self.hand_link_names)

        finger_tip_link_names = [
            "right_hand_thumb_rota_tip",
            "right_hand_index_rota_tip",
            "right_hand_mid_tip",
            "right_hand_ring_tip",
            "right_hand_pinky_tip",
        ]
        self.finger_tip_links = sapien_utils.get_objs_by_names(self.robot.get_links(), finger_tip_link_names)

        self.finger1_link = self.finger_tip_links[0]
        self.finger2_link = self.finger_tip_links[1]
        self.finger3_link = self.finger_tip_links[2]
        self.finger4_link = self.finger_tip_links[3]
        self.finger5_link = self.finger_tip_links[4]

        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

        if self.disable_arm_collision:
            hand_link_name_set = set(self.hand_link_names)
            for link in self.robot.get_links():
                if link.get_name() in hand_link_name_set:
                    continue
                link.set_collision_group(0, 0)


    def count_finger_contacts(self, object: Actor, min_force: float = 0.5) -> torch.Tensor:
        forces = []
        for tip in self.finger_tip_links:
            f = self.scene.get_pairwise_contact_forces(tip, object)  
            forces.append(f)

        # (F,B,3), F=5
        forces = torch.stack(forces, dim=0)

        # (F,B)
        force_norm = torch.linalg.norm(forces, dim=-1)

        # (F,B) bool
        contact = force_norm >= float(min_force)

        # (B,) int64
        return contact.sum(dim=0)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        first_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        second_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        third_contact_forces = self.scene.get_pairwise_contact_forces(self.finger3_link, object)
        four_contact_forces = self.scene.get_pairwise_contact_forces(self.finger4_link, object)
        five_contact_forces = self.scene.get_pairwise_contact_forces(self.finger5_link, object)

        first_force = torch.linalg.norm(first_contact_forces, axis=1)
        second_force = torch.linalg.norm(second_contact_forces, axis=1)
        third_force = torch.linalg.norm(third_contact_forces, axis=1)
        four_force = torch.linalg.norm(four_contact_forces, axis=1)
        five_force = torch.linalg.norm(five_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]

        first_angle = common.compute_angle_between(ldirection, first_contact_forces)
        second_angle = common.compute_angle_between(rdirection, second_contact_forces)
        third_angle = common.compute_angle_between(ldirection, third_contact_forces)
        four_angle = common.compute_angle_between(rdirection, four_contact_forces)
        five_angle = common.compute_angle_between(ldirection, five_contact_forces)

        first_flag = torch.logical_and(first_force >= min_force, torch.rad2deg(first_angle) <= max_angle)
        second_flag = torch.logical_and(second_force >= min_force, torch.rad2deg(second_angle) <= max_angle)
        third_flag = torch.logical_and(third_force >= min_force, torch.rad2deg(third_angle) <= max_angle)
        four_flag = torch.logical_and(four_force >= min_force, torch.rad2deg(four_angle) <= max_angle)
        five_flag = torch.logical_and(five_force >= min_force, torch.rad2deg(five_angle) <= max_angle)

        flags = torch.stack([first_flag, second_flag, third_flag, four_flag, five_flag], dim=0)
        return flags.all(dim=0)
