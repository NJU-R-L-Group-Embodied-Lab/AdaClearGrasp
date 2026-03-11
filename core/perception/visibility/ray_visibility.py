# perception/visibility/ray_visibility.py
import numpy as np


def calculate_visible_points_from_camera(
    env,
    object_actor,
    object_points_xyz: np.ndarray,
    camera_pos_world: np.ndarray,
):
    if object_points_xyz.ndim != 2 or object_points_xyz.shape[1] != 3:
        raise ValueError("object_points_xyz must be (N,3)")
    N = object_points_xyz.shape[0]
    if N == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    camera_pos = np.asarray(camera_pos_world).reshape(1, 3)
    view_point_world = np.repeat(camera_pos, N, axis=0)

    obj_pose = object_actor.pose
    T = obj_pose.to_transformation_matrix().squeeze(0)
    obj_rot = T[:3, :3].cpu().numpy()
    obj_pos = T[:3, 3].cpu().numpy()

    view_point_obj = (view_point_world - obj_pos) @ obj_rot.T
    points_obj = (object_points_xyz - obj_pos) @ obj_rot.T

    directions = points_obj - view_point_obj
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    directions = directions / norms

    mesh = object_actor.get_first_collision_mesh()
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=view_point_obj,
        ray_directions=directions,
        multiple_hits=False,
    )

    if len(locations) != N:
        valid = np.zeros((N, 3), dtype=np.float32)
        valid[: len(locations)] = locations
        locations = valid

    visible_points_world = locations @ obj_rot + obj_pos
    return visible_points_world, locations
