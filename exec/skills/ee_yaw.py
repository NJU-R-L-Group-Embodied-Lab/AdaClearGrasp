# exec/skills/ee_yaw.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class EEYawConfig:
    deg_per_unit_step: float = -1.098821

    eps_deg: float = 0.5        
    max_steps: int = 5000      

    action_dim: int = 18
    yaw_index: int = 5          
    hand_dof: int = 12         


class EEYawSkill:
    def __init__(self, env, cfg: Optional[EEYawConfig] = None):
        self.env = env
        self.cfg = EEYawConfig() if cfg is None else cfg
        if self.cfg.deg_per_unit_step == 0:
            raise ValueError("deg_per_unit_step cannot be 0")

    def rotate_deg(self, delta_deg: float, *, render: bool = False, verbose: bool = False) -> Tuple[bool, int]:
        remaining = float(delta_deg)
        k = float(self.cfg.deg_per_unit_step)  
        steps = 0

        while abs(remaining) > self.cfg.eps_deg and steps < self.cfg.max_steps:
            u = remaining / k

            if u > 1.0:
                u = 1.0
            elif u < -1.0:
                u = -1.0

            action = np.zeros(self.cfg.action_dim, dtype=np.float32)
            action[self.cfg.yaw_index] = float(u)

            self.env.step(action)
            if render:
                self.env.render()

            remaining -= u * k
            steps += 1

            if verbose and steps % 50 == 0:
                print(f"[EEYaw] step={steps} u={u:+.3f} remaining_deg={remaining:+.3f}")

        ok = abs(remaining) <= self.cfg.eps_deg
        if verbose:
            print(f"[EEYaw] done={ok} steps={steps} final_remaining_deg={remaining:+.3f}")
        return ok, steps
