from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clutter_bench_mcp.catalog import ACTION_CATALOG
from clutter_bench_mcp.config import load_config
from clutter_bench_mcp.runtime import ClutterBenchRuntime, _normalize_rgb, _png_payload
from clutter_bench_mcp.safety import SafetyReviewStore
from clutter_bench_mcp.server import _normalize_execution_result


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_scene_config_loads() -> None:
    config = load_config(ROOT / "configs" / "clutter_bench_mcp.yaml")
    assert config.scene.scene_id == "apple_2_1"
    assert config.scene.scene_json == (ROOT / "data/scenes/apple/2/1.json").resolve()
    assert config.environment.render_mode == "rgb_array"
    assert config.server.streamable_http_path == "/mcp"
    assert config.safety.review_ttl_s == 900
    assert config.safety.audit_log == (ROOT / "logs/clutter_bench_mcp_safety.jsonl").resolve()


def test_scene_info_is_concrete_and_session_free() -> None:
    runtime = ClutterBenchRuntime(load_config(ROOT / "configs" / "clutter_bench_mcp.yaml"))
    objects = runtime._objects()
    assert objects == [
        {"actor_name": "target", "label": "apple", "model_id": "013_apple", "role": "target"},
        {"actor_name": "orange", "label": "orange", "model_id": "017_orange", "role": "clutter"},
        {"actor_name": "brick", "label": "foam_brick", "model_id": "061_foam_brick", "role": "clutter"},
    ]
    assert not hasattr(runtime, "session_id")
    assert "reset" in ACTION_CATALOG


def test_rgb_normalization_and_png_encoding() -> None:
    frame = np.ones((1, 4, 5, 4), dtype=np.float32)
    rgb = _normalize_rgb(frame)
    assert rgb.shape == (4, 5, 3)
    assert rgb.dtype == np.uint8
    payload = _png_payload(rgb)
    assert payload["mime_type"] == "image/png"
    assert payload["width"] == 5
    assert payload["height"] == 4
    assert payload["byte_size"] > 0
    assert isinstance(payload["png_base64"], str)


def test_safety_freezes_concrete_tool_call_and_executes_once(tmp_path: Path) -> None:
    audit = tmp_path / "safety.jsonl"
    store = SafetyReviewStore(audit_log=audit, review_ttl_s=900)

    pending = store.create_action_review("push", {"side": "left", "dist_m": 0.05})
    assert pending["status"] == "pending"
    assert pending["action_call"] == {
        "name": "push",
        "title": "推开障碍物",
        "arguments": {"side": "left", "dist_m": 0.05},
        "risk_level": "high",
    }

    approved = store.decide(pending["review_id"], True)
    assert approved["status"] == "executing"
    assert approved["action_call"]["arguments"] == {"side": "left", "dist_m": 0.05}
    completed = store.finish_execution(
        pending["review_id"],
        ok=True,
        result={"ok": True, "action": "push", "message": "done"},
    )
    assert completed["status"] == "completed"
    assert completed["execution_result"]["message"] == "done"
    with pytest.raises(ValueError):
        store.decide(pending["review_id"], True)

    text = audit.read_text(encoding="utf-8")
    assert "action_review_created" in text
    assert "action_review_resolved" in text
    assert "action_execution_finished" in text


def test_denied_frozen_call_cannot_be_executed(tmp_path: Path) -> None:
    store = SafetyReviewStore(audit_log=tmp_path / "safety.jsonl", review_ttl_s=900)
    pending = store.create_action_review("reset", {})
    denied = store.decide(pending["review_id"], False)
    assert denied["status"] == "denied"
    assert denied["approved"] is False
    assert denied["action_call"]["name"] == "reset"
    with pytest.raises(ValueError):
        store.finish_execution(pending["review_id"], ok=True, result={"ok": True})


def test_stuck_is_normalized_to_non_fatal_success() -> None:
    normalized = _normalize_execution_result(
        {
            "ok": False,
            "action": "pull",
            "error_code": "stuck",
            "message": "x=-0.08, y=-0.05",
        }
    )

    assert normalized["ok"] is True
    assert normalized["non_fatal"] is True
    assert normalized["warning_code"] == "stuck"
    assert normalized["low_level_ok"] is False
    assert normalized["error_code"] == "stuck"
    assert normalized["message"] == "x=-0.08, y=-0.05"


def test_non_stuck_error_remains_failure() -> None:
    normalized = _normalize_execution_result(
        {
            "ok": False,
            "action": "move_to",
            "error_code": "ik_failed",
            "message": "unreachable",
        }
    )

    assert normalized["ok"] is False
    assert "non_fatal" not in normalized
