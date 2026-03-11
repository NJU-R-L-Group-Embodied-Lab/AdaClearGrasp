# exec/skills/env_reset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class EnvResetConfig:
    settle_steps: int = 200    
    action_dim: int = 18 
    render: bool = False


class EnvResetSkill:
    def __init__(self, env, cfg: Optional[EnvResetConfig] = None):
        self.env = env
        self.cfg = EnvResetConfig() if cfg is None else cfg

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
              settle_steps: Optional[int] = None, render: Optional[bool] = None) -> Tuple[Any, Dict[str, Any]]:
        if settle_steps is None:
            settle_steps = self.cfg.settle_steps
        if render is None:
            render = self.cfg.render

        if options is None:
            options = {}

        obs, info = self.env.reset(seed=seed, options=options)

        if int(settle_steps) > 0:
            zero = np.zeros(self.cfg.action_dim, dtype=np.float32)
            for _ in range(int(settle_steps)):
                self.env.step(zero)
                if render:
                    self.env.render()

        return obs, info
