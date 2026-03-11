# kinematics/xarm7_xhand_ik.py
import os
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from core.paths import project_root

ARM_JOINT_COUNT = 7

urdf_path = os.path.join(project_root(), "assets", "xhand_right", "urdf", "xarm7_xhand.urdf")
URDF_PATH = os.getenv("XARM7_XHAND_URDF_PATH", urdf_path)
END_LINK_NAME = os.getenv("XARM7_XHAND_END_LINK", "right_hand_ee_link")

TOP_ROLL = float(os.getenv("XARM7_XHAND_TOP_ROLL", "0.2"))
TOP_TILT_AXIS = os.getenv("XARM7_XHAND_TOP_TILT_AXIS", "y")
TOP_TILT_ANGLE = float(os.getenv("XARM7_XHAND_TOP_TILT_ANGLE", "0.25"))

SIDE_RADIUS = float(os.getenv("XARM7_XHAND_SIDE_RADIUS", "0"))
SIDE_YAW = float(os.getenv("XARM7_XHAND_SIDE_YAW", "0.3"))


def _require_urdf(path: str):
    if not path:
        raise RuntimeError("URDF_PATH is empty: pass urdf_path or set env var XARM7_XHAND_URDF_PATH")
    if not os.path.exists(path):
        raise FileNotFoundError(f"URDF_PATH does not exist: {path}")

def _roll_frame_x_torch(device: torch.device) -> torch.Tensor:
    """
    Reference frame for top grasp (palm facing down).
    Column vectors = [Xw, Yw, Zw]:
      local +X (palm normal)  -> world -Z
      local +Y (thumb)        -> world +Y
      local +Z (four fingers) -> world +X
    """
    return torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _side_frame_torch(device: torch.device) -> torch.Tensor:
    """
    Reference frame for side grasp (side approach).
    Column vectors = [Xw, Yw, Zw]:
      local +X (palm normal)  -> world +X
      local +Y (thumb)        -> world +Z
      local +Z (four fingers) -> world +Y
    """
    return torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _side_down_frame_torch(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _axis_angle_rot_torch(axis: str, angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    dev = angle.device

    if axis == "x":
        return torch.tensor([[1.0, 0.0, 0.0],
                             [0.0, c, -s],
                             [0.0, s,  c]], dtype=torch.float32, device=dev)
    if axis == "y":
        return torch.tensor([[ c, 0.0, s],
                             [0.0, 1.0, 0.0],
                             [-s, 0.0, c]], dtype=torch.float32, device=dev)
    if axis == "z":
        return torch.tensor([[c, -s, 0.0],
                             [s,  c, 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32, device=dev)
    raise ValueError(f"axis must be one of x/y/z, got: {axis}")


def _build_kin(env, urdf_path: str, end_link_name: str, arm_joint_count: int):
    robot = env.unwrapped.agent.robot
    active = torch.arange(arm_joint_count, device=robot.device, dtype=torch.long)
    kin = Kinematics(
        articulation=robot,
        urdf_path=urdf_path,
        end_link_name=end_link_name,
        active_joint_indices=active,
    )
    return kin


@torch.no_grad()
def solve_ik_palm_down_rollfree(
    env,
    target_center_w: torch.Tensor,
    height_offset: float = 0.12,
    urdf_path: str = None,
    end_link_name: str = None,
    arm_joint_count: int = ARM_JOINT_COUNT,
    theta: float = 0.0,
    tilt_axis: str = None,
    tilt_angle: float = None,
) -> torch.Tensor:
    robot = env.unwrapped.agent.robot
    dev = robot.device
    dtype = robot.get_qpos().dtype

    urdf_path = URDF_PATH if urdf_path is None else urdf_path
    end_link_name = END_LINK_NAME if end_link_name is None else end_link_name
    _require_urdf(urdf_path)

    theta = TOP_ROLL if theta is None else float(theta)
    tilt_axis = TOP_TILT_AXIS if tilt_axis is None else tilt_axis
    tilt_angle = TOP_TILT_ANGLE if tilt_angle is None else float(tilt_angle)

    target_center_w = target_center_w.to(device=dev, dtype=torch.float32)
    pos_w = target_center_w + torch.tensor([0.0, 0.0, float(height_offset)], device=dev)

    q0 = robot.get_qpos()
    kin = _build_kin(env, urdf_path, end_link_name, arm_joint_count)

    base_pose = robot.get_root_pose()
    base_inv = base_pose.inv()

    R_w = _roll_frame_x_torch(dev)

    if tilt_angle != 0.0:
        R_w = R_w @ _axis_angle_rot_torch(tilt_axis, torch.tensor(tilt_angle, device=dev))
    if theta != 0.0:
        R_w = R_w @ _axis_angle_rot_torch("x", torch.tensor(theta, device=dev))

    quat_wxyz = matrix_to_quaternion(R_w.unsqueeze(0))[0] 
    target_pose_w = Pose.create_from_pq(pos_w, quat_wxyz)
    target_pose_base = base_inv * target_pose_w

    # sol = kin.compute_ik(target_pose=target_pose_base, q0=q0, pos_only=False)
    sol = kin.compute_ik(pose=target_pose_base, q0=q0)
    if sol is None or (not torch.isfinite(sol).all()):
        raise RuntimeError(f"top IK failed: height_offset={height_offset}")

    qtar = q0.clone()
    qtar[:, :arm_joint_count] = sol[:, :arm_joint_count]
    return qtar.to(dtype=dtype, device=dev)


@torch.no_grad()
def solve_ik_side_grasp(
    env,
    target_center_w: torch.Tensor,
    radius: float = None,
    yaw: float = None,
    height_offset: float = 0.01,
    urdf_path: str = None,
    end_link_name: str = None,
    arm_joint_count: int = ARM_JOINT_COUNT,
) -> torch.Tensor:
    robot = env.unwrapped.agent.robot
    dev = robot.device
    dtype = robot.get_qpos().dtype

    urdf_path = URDF_PATH if urdf_path is None else urdf_path
    end_link_name = END_LINK_NAME if end_link_name is None else end_link_name
    _require_urdf(urdf_path)

    radius = SIDE_RADIUS if radius is None else float(radius)
    yaw = SIDE_YAW if yaw is None else float(yaw)

    target_center_w = target_center_w.to(device=dev, dtype=torch.float32)
    pos_w = target_center_w + torch.tensor([0.0, 0.0, float(height_offset)], device=dev)

    q0 = robot.get_qpos()
    kin = _build_kin(env, urdf_path, end_link_name, arm_joint_count)

    base_pose = robot.get_root_pose()
    base_inv = base_pose.inv()

    R_w = _side_frame_torch(dev)
    if yaw != 0.0:
        R_w = _axis_angle_rot_torch("z", torch.tensor(yaw, device=dev)) @ R_w

    n_w = R_w[:, 0]
    pos_w = pos_w - radius * n_w

    quat_wxyz = matrix_to_quaternion(R_w.unsqueeze(0))[0] 
    target_pose_w = Pose.create_from_pq(pos_w, quat_wxyz)
    target_pose_base = base_inv * target_pose_w

    sol = kin.compute_ik(target_pose=target_pose_base, q0=q0, pos_only=False)
    if sol is None or (not torch.isfinite(sol).all()):
        raise RuntimeError(f"side IK failed: radius={radius}, yaw={yaw}")

    qtar = q0.clone()
    qtar[:, :arm_joint_count] = sol[:, :arm_joint_count]
    return qtar.to(dtype=dtype, device=dev)

