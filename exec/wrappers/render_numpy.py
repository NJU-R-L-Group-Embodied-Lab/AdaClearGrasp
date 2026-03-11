# exec/wrappers/render_numpy.py
from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym


class RenderNumpyWrapper(gym.Wrapper):
    def render(self, *args, **kwargs):
        frame = self.env.render(*args, **kwargs)
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if isinstance(frame, (list, tuple)) and len(frame) == 1:
            frame = frame[0]
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
        if frame is None:
            return None
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame
