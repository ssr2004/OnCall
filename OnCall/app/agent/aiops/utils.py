"""
AIOps Agent 通用工具函数
"""

import re
from typing import List


PLAN_STEP_PATTERN = re.compile(
    r"(?m)(?:^|(?<=\s))"
    r"(?:[-*]\s+)?"
    r"(?:\d+\s*[.)、]\s*)?"
    r"(?:\*\*)?\s*步骤\s*(\d+)\s*"
    r"(?:[:：]\s*(?:\*\*)?|(?:\*\*)\s*[:：])\s*"
)


def format_tools_description(tools: List) -> str:
    """格式化工具列表为描述文本"""
    tool_descriptions = []
    for tool in tools:
        if hasattr(tool, 'name') and hasattr(tool, 'description'):
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)


def normalize_plan_steps(steps: List[str]) -> List[str]:
    """将模型塞进单个字符串的普通或 Markdown 步骤拆回独立步骤。"""
    normalized: List[str] = []
    for raw_step in steps:
        text = str(raw_step or "").strip()
        matches = list(PLAN_STEP_PATTERN.finditer(text))
        if not matches:
            if text:
                normalized.append(text)
            continue
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[match.end():end].strip()
            if content:
                normalized.append(f"步骤{match.group(1)}: {content}")
    return normalized


def plan_covers_active_alert_diagnosis(steps: List[str]) -> bool:
    """检查活动告警计划是否具备形成证据链所需的最小覆盖。"""
    if len(steps) < 5:
        return False

    combined = "\n".join(str(step).lower() for step in steps)
    coverage_groups = (
        (
            "query_grid_service_status",
            "query_grid_data_sync_metrics",
            "query_grid_telemetry_metrics",
            "业务指标",
            "服务状态",
        ),
        ("search_topic_by_service_name", "日志主题"),
        ("search_log", "业务日志", "异常日志"),
        ("retrieve_knowledge", "知识库", "处置手册"),
        ("诊断报告", "分析报告", "生成报告"),
    )
    return all(any(keyword in combined for keyword in group) for group in coverage_groups)
