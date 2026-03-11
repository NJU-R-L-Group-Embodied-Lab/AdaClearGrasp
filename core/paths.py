# core/paths.py
import os
from dataclasses import dataclass

def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_root() -> str:
    return os.getenv("VLM_RL_DATA", os.path.join(project_root(), "data"))

@dataclass(frozen=True)
class DataPaths:
    root: str
    ik_init: str
    models: str
    logs: str
    videos: str

def get_data_paths() -> DataPaths:
    root = data_root()
    return DataPaths(
        root=root,
        ik_init=os.path.join(root, "ik_init"),
        models=os.path.join(root, "models"),
        logs=os.path.join(root, "logs"),
        videos=os.path.join(root, "videos"),
    )

def ensure_data_dirs():
    p = get_data_paths()
    os.makedirs(p.ik_init, exist_ok=True)
    os.makedirs(p.models, exist_ok=True)
    os.makedirs(p.logs, exist_ok=True)
    os.makedirs(p.videos, exist_ok=True)
    return p
