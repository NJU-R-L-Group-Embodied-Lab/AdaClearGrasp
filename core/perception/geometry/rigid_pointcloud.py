# core/perception/geometry/rigid_pointcloud.py
from __future__ import annotations

import torch


def quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4) in wxyz
    return: (..., 4) conjugate
    """
    qc = q.clone()
    qc[..., 1:4] = -qc[..., 1:4]
    return qc


def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors v by quaternion q.

    q: (..., 4) wxyz
    v: (..., 3)
    return: (..., 3)
    """
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def world_to_local_points_wxyz(
    xyz_w: torch.Tensor,
    p_w: torch.Tensor,
    q_wxyz: torch.Tensor,
) -> torch.Tensor:
    """
    Convert world points to local points for a rigid body pose (p,q).
    local = R(q)^T (world - p)

    xyz_w: (N,3)
    p_w:   (3,)
    q_wxyz:(4,)
    return: (N,3)
    """
    if xyz_w.ndim != 2 or xyz_w.shape[1] != 3:
        raise RuntimeError(f"xyz_w must be (N,3), got {tuple(xyz_w.shape)}")
    if p_w.shape != (3,):
        raise RuntimeError(f"p_w must be (3,), got {tuple(p_w.shape)}")
    if q_wxyz.shape != (4,):
        raise RuntimeError(f"q_wxyz must be (4,), got {tuple(q_wxyz.shape)}")

    q_conj = quat_conj_wxyz(q_wxyz)
    return quat_apply_wxyz(q_conj.unsqueeze(0).expand(xyz_w.shape[0], -1), xyz_w - p_w.unsqueeze(0))


def local_to_world_points_batch_wxyz(
    xyz_local: torch.Tensor,
    p_w: torch.Tensor,
    q_wxyz: torch.Tensor,
) -> torch.Tensor:
    """
    Convert local points to world points for a batch of rigid body poses.

    xyz_local: (N,3)
    p_w:       (B,3)
    q_wxyz:    (B,4)
    return:    (B,N,3)
    """
    if xyz_local.ndim != 2 or xyz_local.shape[1] != 3:
        raise RuntimeError(f"xyz_local must be (N,3), got {tuple(xyz_local.shape)}")
    if p_w.ndim != 2 or p_w.shape[1] != 3:
        raise RuntimeError(f"p_w must be (B,3), got {tuple(p_w.shape)}")
    if q_wxyz.ndim != 2 or q_wxyz.shape[1] != 4:
        raise RuntimeError(f"q_wxyz must be (B,4), got {tuple(q_wxyz.shape)}")

    b = p_w.shape[0]
    n = xyz_local.shape[0]

    local = xyz_local.unsqueeze(0).expand(b, n, 3)                 # (B,N,3)
    q = q_wxyz.unsqueeze(1).expand(b, n, 4)                        # (B,N,4)
    p = p_w.unsqueeze(1)                                           # (B,1,3)
    return quat_apply_wxyz(q, local) + p
