# exec/skills/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple

import numpy as np


ErrorCode = Literal[
    "none",        # Finished successfully
    "max_step",    # Reached max steps but not finished
    "stuck",       # Stuck / not moving
    "call_error",  # Failed before execution due to an exception
]


@dataclass
class StepEvent:
    i: int
    moved: float
    event: Optional[str] = None


@dataclass
class Trace:
    maxlen: int = 10
    eps_stuck: float = 1e-4
    events: List[StepEvent] = field(default_factory=list)

    def add(self, e: StepEvent) -> None:
        self.events.append(e)
        if len(self.events) > self.maxlen:
            self.events = self.events[-self.maxlen:]

    def summary(self) -> Dict[str, Any]:
        if not self.events:
            return {"has_steps": False}

        moved_vals = [float(e.moved) for e in self.events]
        stuck = max(moved_vals) < float(self.eps_stuck)

        last_event = None
        for e in reversed(self.events):
            if e.event is not None:
                last_event = {"i": int(e.i), "event": str(e.event)}
                break

        return {
            "has_steps": True,
            "n": int(len(self.events)),
            "stuck": bool(stuck),
            "last_event": last_event,
        }


@dataclass
class SkillResult:
    ok: bool
    error_code: ErrorCode = "none"
    message: str = ""
    advice: str = ""


class BaseSkill:
    def __init__(
        self,
        env,
        *,
        trace_maxlen: int = 10,
        eps_stuck: float = 1e-4,
    ):
        self.env = env
        self._trace = Trace(maxlen=int(trace_maxlen), eps_stuck=float(eps_stuck))

    # -------------------------
    # Trace helpers
    # -------------------------
    def reset_trace(self) -> None:
        self._trace = Trace(maxlen=self._trace.maxlen, eps_stuck=self._trace.eps_stuck)

    def trace_summary(self) -> Dict[str, Any]:
        return self._trace.summary()

    # -------------------------
    # Env step wrapper
    # -------------------------
    def _step(self, action, *, i: int, render: bool) -> Tuple[Any, Any, bool, bool, Any]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if render:
            self.env.render()
        return obs, reward, bool(terminated), bool(truncated), info

    # -------------------------
    # Minimal result constructor
    # -------------------------
    def _result(
        self,
        *,
        ok: bool,
        error_code: ErrorCode = "none",
        message: str = "",
        advice: str = "",
    ) -> SkillResult:
        return SkillResult(
            ok=bool(ok),
            error_code=error_code,
            message=str(message),
            advice=str(advice),
        )
