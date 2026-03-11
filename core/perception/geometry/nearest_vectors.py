# perception/geometry/nearest_vectors.py
import numpy as np
import torch
from scipy.spatial import cKDTree


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def joints_to_pointcloud_nearest_vectors(joint_positions, target_object_xyz, normalize=False):
    """
    joint_positions: (M,3)
    target_object_xyz: (N,3)
    """
    target_object_xyz = _to_numpy(target_object_xyz)
    joints = _to_numpy(joint_positions)

    if target_object_xyz.ndim != 2 or target_object_xyz.shape[1] != 3:
        raise ValueError("target_object_xyz must be an array of shape (N,3).")
    if target_object_xyz.shape[0] == 0:
        raise ValueError("target point cloud is empty.")

    if joints.ndim == 1 and joints.size == 3:
        joints = joints.reshape(1, 3)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("joint_positions must be (M,3) or an iterable of M elements of shape (3,).")   

    tree = cKDTree(target_object_xyz)
    dists, idxs = tree.query(joints, k=1)

    nearest_points = target_object_xyz[idxs]
    vectors = nearest_points - joints
    distances = dists

    directions = None
    if normalize:
        directions = np.zeros_like(vectors)
        nonzero = distances > 1e-12
        directions[nonzero] = vectors[nonzero] / distances[nonzero][:, None]

    return {
        "nearest_points": nearest_points,
        "vectors": vectors,
        "distances": distances,
        "directions": directions,
        "indices": idxs,
    }


def joints_to_pointcloud_nearest_vectors_batch(joint_positions, target_object_xyz, normalize=False):
    """
    joint_positions:
      - (num_joints, num_envs, 3) or (num_joints, 3)
    target_object_xyz:
      - (num_envs, num_points, 3) or (num_points, 3)
    """
    # ---------- torch path ----------
    if isinstance(joint_positions, torch.Tensor) or isinstance(target_object_xyz, torch.Tensor):
        jp = joint_positions if isinstance(joint_positions, torch.Tensor) else torch.as_tensor(joint_positions)
        pc = target_object_xyz if isinstance(target_object_xyz, torch.Tensor) else torch.as_tensor(target_object_xyz)

        original_target_dims = pc.ndim
        original_joint_dims = jp.ndim

        if pc.ndim == 2:
            if pc.shape[1] != 3:
                raise ValueError(f"target_object_xyz must be (N,3), got {tuple(pc.shape)}")
            pc = pc.unsqueeze(0)  # (1,N,3)
        elif pc.ndim == 3:
            if pc.shape[2] != 3:
                raise ValueError(f"target_object_xyz must be (...,3), got {tuple(pc.shape)}")
        else:
            raise ValueError(f"target_object_xyz must be 2D or 3D, got ndim={pc.ndim}")

        if jp.ndim == 2:
            if jp.shape[1] != 3:
                raise ValueError(f"joint_positions must be (M,3), got {tuple(jp.shape)}")
            jp = jp.unsqueeze(1)  # (M,1,3)
        elif jp.ndim == 3:
            if jp.shape[2] != 3:
                raise ValueError(f"joint_positions must be (...,3), got {tuple(jp.shape)}")
        else:
            raise ValueError(f"joint_positions must be 2D or 3D, got ndim={jp.ndim}")

        num_envs = pc.shape[0]
        num_joints = jp.shape[0]

        if jp.shape[1] != num_envs:
            raise ValueError(
                f"joint_positions second dim must equal num_envs. "
                f"Got joint_positions {tuple(jp.shape)} vs target_object_xyz {tuple(pc.shape)}"
            )
        if pc.shape[1] == 0:
            nearest_points = torch.zeros((num_envs, num_joints, 3), device=pc.device, dtype=torch.float32)
            vectors = torch.zeros((num_envs, num_joints, 3), device=pc.device, dtype=torch.float32)
            distances = torch.zeros((num_envs, num_joints), device=pc.device, dtype=torch.float32)
            indices = torch.zeros((num_envs, num_joints), device=pc.device, dtype=torch.int64)
            directions = torch.zeros((num_envs, num_joints, 3), device=pc.device, dtype=torch.float32) if normalize else None
        else:
            jp_b = jp.permute(1, 0, 2).contiguous()
            pc_b = pc.contiguous()

            # diff: (B,M,N,3)
            diff = pc_b[:, None, :, :] - jp_b[:, :, None, :]
            dist2 = (diff * diff).sum(dim=-1)  # (B,M,N)

            idx = torch.argmin(dist2, dim=-1)  # (B,M)
            min_dist2 = torch.gather(dist2, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # (B,M)

            # gather nearest points: pc_b (B,N,3) with idx (B,M)
            idx3 = idx.unsqueeze(-1).expand(-1, -1, 3)  # (B,M,3)
            nearest = torch.gather(pc_b, dim=1, index=idx3)  # (B,M,3)

            vec = nearest - jp_b  # (B,M,3)
            dist = torch.sqrt(torch.clamp(min_dist2, min=0.0))  # (B,M)

            if normalize:
                denom = dist.unsqueeze(-1)  # (B,M,1)
                directions_b = torch.zeros_like(vec, dtype=torch.float32)
                nonzero = dist > 1e-12
                directions_b[nonzero] = vec[nonzero] / denom[nonzero]
            else:
                directions_b = None

            nearest_points = nearest.to(dtype=torch.float32)
            vectors = vec.to(dtype=torch.float32)
            distances = dist.to(dtype=torch.float32)
            indices = idx.to(dtype=torch.int64)
            directions = directions_b.to(dtype=torch.float32) if normalize else None

        if original_target_dims == 2 and original_joint_dims == 2:
            out = {
                "nearest_points": nearest_points[0],
                "vectors": vectors[0],
                "distances": distances[0],
                "indices": indices[0],
            }
            if normalize:
                out["directions"] = directions[0]
            return out

        out = {
            "nearest_points": nearest_points,
            "vectors": vectors,
            "distances": distances,
            "indices": indices,
        }
        if normalize:
            out["directions"] = directions
        return out

    # ---------- numpy path ----------
    target_object_xyz = np.asarray(target_object_xyz)
    joint_positions = np.asarray(joint_positions)

    original_target_dims = target_object_xyz.ndim
    original_joint_dims = joint_positions.ndim

    if target_object_xyz.ndim == 2:
        if target_object_xyz.shape[1] != 3:
            raise ValueError(f"target_object_xyz must be (N,3), got {target_object_xyz.shape}")
        target_object_xyz = target_object_xyz[np.newaxis, :, :]  # (1,N,3)
    elif target_object_xyz.ndim == 3:
        if target_object_xyz.shape[2] != 3:
            raise ValueError(f"target_object_xyz must be (...,3), got {target_object_xyz.shape}")
    else:
        raise ValueError(f"target_object_xyz must be 2D or 3D, got ndim={target_object_xyz.ndim}")

    if joint_positions.ndim == 2:
        if joint_positions.shape[1] != 3:
            raise ValueError(f"joint_positions must be (M,3), got {joint_positions.shape}")
        joint_positions = joint_positions[:, np.newaxis, :]  # (M,1,3)
    elif joint_positions.ndim == 3:
        if joint_positions.shape[2] != 3:
            raise ValueError(f"joint_positions must be (...,3), got {joint_positions.shape}")
    else:
        raise ValueError(f"joint_positions must be 2D or 3D, got ndim={joint_positions.ndim}")

    num_envs = target_object_xyz.shape[0]
    num_joints = joint_positions.shape[0]

    if joint_positions.shape[1] != num_envs:
        raise ValueError(
            f"joint_positions second dim must equal num_envs. "
            f"Got joint_positions {joint_positions.shape} vs target_object_xyz {target_object_xyz.shape}"
        )

    nearest_points = np.zeros((num_envs, num_joints, 3), dtype=np.float32)
    vectors = np.zeros((num_envs, num_joints, 3), dtype=np.float32)
    distances = np.zeros((num_envs, num_joints), dtype=np.float32)
    indices = np.zeros((num_envs, num_joints), dtype=np.int64)
    directions = np.zeros((num_envs, num_joints, 3), dtype=np.float32) if normalize else None

    if target_object_xyz.shape[1] != 0:
        # batch-first: jp (M,B,3) -> (B,M,3)
        jp_b = np.transpose(joint_positions, (1, 0, 2))  # (B,M,3)
        pc_b = target_object_xyz  # (B,N,3)

        # diff: (B,M,N,3)
        diff = pc_b[:, None, :, :] - jp_b[:, :, None, :]
        dist2 = np.sum(diff * diff, axis=-1)  # (B,M,N)

        idx = np.argmin(dist2, axis=-1)  # (B,M)
        min_dist2 = np.take_along_axis(dist2, idx[..., None], axis=-1)[..., 0]  # (B,M)

        # gather nearest points
        # build (B,M,3) indices for take_along_axis on axis=1
        idx3 = np.repeat(idx[..., None], 3, axis=-1)  # (B,M,3)
        nearest = np.take_along_axis(pc_b, idx3[:, :, None, :], axis=1)[:, :, 0, :]  # (B,M,3)

        vec = nearest - jp_b  # (B,M,3)
        dist = np.sqrt(np.maximum(min_dist2, 0.0)).astype(np.float32)  # (B,M)

        nearest_points = nearest.astype(np.float32)
        vectors = vec.astype(np.float32)
        distances = dist.astype(np.float32)
        indices = idx.astype(np.int64)

        if normalize:
            env_dir = np.zeros_like(vec, dtype=np.float32)
            nonzero = dist > 1e-12
            env_dir[nonzero] = vec[nonzero] / dist[nonzero][..., None]
            directions = env_dir.astype(np.float32)

    if original_target_dims == 2 and original_joint_dims == 2:
        out = {
            "nearest_points": nearest_points[0],
            "vectors": vectors[0],
            "distances": distances[0],
            "indices": indices[0],
        }
        if normalize:
            out["directions"] = directions[0]
        return out

    out = {
        "nearest_points": nearest_points,
        "vectors": vectors,
        "distances": distances,
        "indices": indices,
    }
    if normalize:
        out["directions"] = directions
    return out
