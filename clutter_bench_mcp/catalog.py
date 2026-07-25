from __future__ import annotations

from typing import Any


ENVIRONMENT_INSTRUCTIONS = (
    "你连接的是一个已经创建并持续运行的固定桌面杂乱场景。"
    "每次模型调用前，宿主会自动获取唯一最新 RGB 帧；不要主动请求观测。"
    "只使用 MCP 暴露的原子动作，并且每次只调用一个动作工具。"
    "动作工具会先在 MCP 中冻结具体工具名和参数，等待用户确认后再实际执行；"
    "不要自行向用户索要确认，也不要重复提交同一个待审核调用。"
    "控制器返回 error_code=stuck 时按非致命状态处理，不代表动作整体失败；"
    "应结合动作结果和自动刷新的最新画面决定是否继续、调整或结束。"
    "动作完成后根据最新画面简短回答。"
)


ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "move_to": {
        "title": "移动到物体上方",
        "description": "将末端执行器移动到指定场景物体或命名区域上方。",
        "risk_level": "medium",
        "keywords": ["移动到", "移到", "前往", "靠近", "move to", "goto"],
    },
    "lift": {
        "title": "抬升机械臂",
        "description": "将末端执行器抬升到安全移动高度。",
        "risk_level": "medium",
        "keywords": ["抬升", "抬高", "升高", "lift", "raise"],
    },
    "lower": {
        "title": "降低机械臂",
        "description": "将末端执行器下降到桌面工作高度。",
        "risk_level": "medium",
        "keywords": ["下降", "降低", "放低", "lower", "move down"],
    },
    "set_pose": {
        "title": "设置手部姿态",
        "description": "将手部姿态设置为 flat 或 work。",
        "risk_level": "medium",
        "keywords": ["设置姿态", "手部姿态", "平展开", "工作姿态", "set pose"],
    },
    "push": {
        "title": "推开障碍物",
        "description": "沿 left、center 或 right 方向推开障碍物指定距离。",
        "risk_level": "high",
        "keywords": ["推开", "推动", "推一下", "push"],
    },
    "pull": {
        "title": "拉开障碍物",
        "description": "沿 left、center 或 right 方向拉开障碍物指定距离。",
        "risk_level": "high",
        "keywords": ["拉开", "拉动", "拉一下", "pull"],
    },
    "initarm": {
        "title": "初始化机械臂",
        "description": "将机械臂恢复到配置的初始姿态，不重置场景物体。",
        "risk_level": "medium",
        "keywords": ["初始化手臂", "初始化机械臂", "手臂复位", "机械臂复位", "initarm", "init arm"],
    },
    "inithand": {
        "title": "初始化手部",
        "description": "将手部恢复到平展初始姿态，不重置场景物体。",
        "risk_level": "medium",
        "keywords": ["初始化手掌", "初始化手部", "初始化夹爪", "手掌复位", "手部复位", "打开夹爪", "inithand", "init hand"],
    },
    "grasp": {
        "title": "抓取目标",
        "description": "运行固定场景目标抓取策略。",
        "risk_level": "high",
        "keywords": ["抓取", "抓住", "拿起", "拾取", "夹住", "grasp", "pick up", "pick"],
    },
    "reset": {
        "title": "重置环境",
        "description": "将当前固定场景恢复到初始状态，并保留同一个 MCP 服务连接。",
        "risk_level": "high",
        "keywords": ["重置环境", "重置场景", "恢复场景", "场景复位", "环境复位", "reset environment", "reset scene"],
    },
}


def action_meta(name: str) -> dict[str, Any]:
    item = ACTION_CATALOG[name]
    return {
        "clutter_bench": {
            "agent_action": True,
            "requires_user_review": True,
            "review_mode": "mcp_frozen_tool_call",
            "risk_level": item["risk_level"],
            "title": item["title"],
        }
    }


def public_action_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "title": item["title"],
            "description": item["description"],
            "risk_level": item["risk_level"],
            "requires_user_review": True,
        }
        for name, item in ACTION_CATALOG.items()
    ]


__all__ = ["ACTION_CATALOG", "ENVIRONMENT_INSTRUCTIONS", "action_meta", "public_action_catalog"]
