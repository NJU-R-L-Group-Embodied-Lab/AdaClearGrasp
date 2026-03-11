# env/sim/gt_pointcloud.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import trimesh

def set_global_seed(seed: int):
    import os, random
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        
def _to_np(x):
    import torch
    return x.detach().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

def build_collision_mesh_world(actor) -> trimesh.Trimesh:
    get_fcm = getattr(actor, "get_first_collision_mesh", None)
    if get_fcm is None:
        raise TypeError(
            "actor does not support get_first_collision_mesh(); current Actor type: "
            f"{type(actor).__name__}. Please ensure you pass a ManiSkill Actor / merged Actor."
        )

    mesh = get_fcm()
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(
            "get_first_collision_mesh() did not return trimesh.Trimesh; "
            f"got: {type(mesh).__name__}"
        )
    if mesh.is_empty:
        raise RuntimeError("first_collision_mesh is empty (is_empty=True)")
    return mesh.copy()

def _sample_on_triangles(
    vertices: np.ndarray, faces: np.ndarray, n_points: int
) -> np.ndarray:
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.size == 0:
        raise RuntimeError("mesh has no triangles (faces is empty)")

    tri = V[F]  # (M, 3, 3)
    v01 = tri[:, 1] - tri[:, 0]
    v02 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(v01, v02), axis=1) * 0.5  # (M,)
    if not np.all(np.isfinite(areas)):
        raise RuntimeError("non-finite triangle areas detected (NaN/Inf)")

    total_area = float(areas.sum())
    if total_area <= 0:
        raise RuntimeError("total area is 0; cannot sample")


    probs = areas / total_area

    face_idx = np.random.choice(len(F), size=n_points, replace=True, p=probs)
    tri_sel = tri[face_idx]  # (n, 3, 3)

    u = np.random.rand(n_points).astype(np.float64)
    v = np.random.rand(n_points).astype(np.float64)
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    w = 1.0 - u - v  # (n,)

    P = (
        (tri_sel[:, 0] * w[:, None])
        + (tri_sel[:, 1] * u[:, None])
        + (tri_sel[:, 2] * v[:, None])
    )
    return P.astype(np.float32)


def sample_world_points_on_actor_strict(
    actor,
    n_points: int = 4096,
) -> np.ndarray:
    mw = build_collision_mesh_world(actor)
    if (mw.vertices is None) or (mw.faces is None) or (mw.faces.size == 0):
        mw = mw.copy()
        mw = mw.as_triangles()
        if (mw.faces is None) or (mw.faces.size == 0):
            raise RuntimeError("mesh triangulation failed or still has no triangle faces")

    return _sample_on_triangles(mw.vertices, mw.faces, n_points)


def sample_world_points_on_actor(actor, n_points: int = 4096) -> np.ndarray:
    return sample_world_points_on_actor_strict(actor, n_points=n_points)