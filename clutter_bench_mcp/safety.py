from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from .catalog import ACTION_CATALOG


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


@dataclass(slots=True)
class ActionReviewRecord:
    review_id: str
    action_name: str
    arguments: dict[str, Any]
    title: str
    risk_level: str
    status: str
    created_at: str
    created_monotonic: float = field(repr=False)
    decided_at: str | None = None
    completed_at: str | None = None
    approved: bool | None = None
    outcome: str | None = None
    execution_result: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "status": self.status,
            "approval_required": self.status == "pending",
            "approved": self.approved,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "action_call": {
                "name": self.action_name,
                "title": self.title,
                "arguments": _json_copy(self.arguments),
                "risk_level": self.risk_level,
            },
            "execution_result": _json_copy(self.execution_result) if self.execution_result is not None else None,
        }


class SafetyReviewStore:
    """MCP-owned gate for concrete, frozen model tool calls.

    The model proposes a fully specified action first. MCP then freezes its name and
    arguments, waits for a user decision, and permits exactly that call to execute once.
    No conversation or agent session is stored here.
    """

    def __init__(self, *, audit_log: Path, review_ttl_s: int = 900) -> None:
        self.audit_log = Path(audit_log).expanduser().resolve()
        self.review_ttl_s = max(60, int(review_ttl_s))
        self._lock = RLock()
        self._records: dict[str, ActionReviewRecord] = {}
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)

    def create_action_review(self, action_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(action_name or "").strip()
        if action not in ACTION_CATALOG:
            raise ValueError(f"unknown agent action: {action}")
        frozen_arguments = _json_copy(dict(arguments or {}))
        item = ACTION_CATALOG[action]
        record = ActionReviewRecord(
            review_id=f"review_{uuid4().hex[:16]}",
            action_name=action,
            arguments=frozen_arguments,
            title=str(item["title"]),
            risk_level=str(item["risk_level"]),
            status="pending",
            created_at=_now(),
            created_monotonic=monotonic(),
        )
        with self._lock:
            self._records[record.review_id] = record
            self._audit("action_review_created", record, {})
        payload = record.public()
        payload["reason"] = "模型已提出一个会改变环境状态的具体工具调用；MCP 已冻结工具名和参数，等待用户确认。"
        return payload

    def decide(self, review_id: str, approved: bool) -> dict[str, Any]:
        with self._lock:
            record = self._require(review_id)
            self._check_expired(record)
            if record.status != "pending":
                raise ValueError(f"review is not pending: {record.status}")
            record.approved = bool(approved)
            record.decided_at = _now()
            if approved:
                record.status = "executing"
                record.outcome = "user_approved"
            else:
                record.status = "denied"
                record.completed_at = record.decided_at
                record.outcome = "user_denied"
            self._audit("action_review_resolved", record, {"approved": bool(approved)})
            return record.public()

    def finish_execution(
        self,
        review_id: str,
        *,
        ok: bool,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            record = self._require(review_id)
            if record.status != "executing" or record.approved is not True:
                raise ValueError(f"review is not executing: {record.status}")
            record.execution_result = _json_copy(dict(result or {}))
            record.status = "completed" if ok else "failed"
            record.completed_at = _now()
            record.outcome = "executed" if ok else "execution_failed"
            self._audit(
                "action_execution_finished",
                record,
                {
                    "ok": bool(ok),
                    "message": str(result.get("message") or "")[:500],
                },
            )
            return record.public()

    def get(self, review_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require(review_id)
            if record.status == "pending":
                self._check_expired(record)
            return record.public()

    def _require(self, review_id: str) -> ActionReviewRecord:
        key = str(review_id or "").strip()
        record = self._records.get(key)
        if record is None:
            raise ValueError("unknown review_id")
        return record

    def _check_expired(self, record: ActionReviewRecord) -> None:
        if monotonic() - record.created_monotonic <= self.review_ttl_s:
            return
        if record.status == "pending":
            record.status = "expired"
            record.completed_at = _now()
            record.outcome = "review_ttl_expired"
            self._audit("action_review_expired", record, {})
        raise PermissionError("action review has expired")

    def _audit(self, event: str, record: ActionReviewRecord, extra: dict[str, Any]) -> None:
        payload = {
            "timestamp": _now(),
            "event": event,
            "review": record.public(),
            **extra,
        }
        with self.audit_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["SafetyReviewStore"]
