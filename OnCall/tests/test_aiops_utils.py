"""AIOps 计划规范化测试。"""

from app.agent.aiops.utils import (
    normalize_plan_steps,
    plan_covers_active_alert_diagnosis,
)


def test_normalize_plan_steps_splits_numbered_steps_from_one_string() -> None:
    raw = [
        "已有告警上下文。\n\n"
        "步骤1: 查询业务指标\n- service_name=grid-data-sync-service\n\n"
        "步骤2：查询日志主题\n- 使用 search_topic_by_service_name\n\n"
        "步骤3: 查询业务日志"
    ]

    result = normalize_plan_steps(raw)

    assert len(result) == 3
    assert result[0].startswith("步骤1: 查询业务指标")
    assert result[1].startswith("步骤2: 查询日志主题")
    assert result[2] == "步骤3: 查询业务日志"


def test_normalize_plan_steps_keeps_already_independent_steps() -> None:
    result = normalize_plan_steps(["查询告警", "查询指标", "生成报告"])

    assert result == ["查询告警", "查询指标", "生成报告"]


def test_normalize_plan_steps_splits_inline_markdown_steps() -> None:
    raw = [
        "根据预取告警信息开始诊断。 "
        "**步骤1**: 确认活动告警。 "
        "**步骤2**：调用 `query_grid_service_status` 查询业务指标。 "
        "**步骤3:** 使用 `search_log` 查询业务日志。"
    ]

    result = normalize_plan_steps(raw)

    assert result == [
        "步骤1: 确认活动告警。",
        "步骤2: 调用 `query_grid_service_status` 查询业务指标。",
        "步骤3: 使用 `search_log` 查询业务日志。",
    ]


def test_active_alert_plan_requires_separate_complete_evidence_steps() -> None:
    valid = [
        "查询业务指标",
        "查询日志主题",
        "查询异常日志",
        "检索知识库处置手册",
        "生成诊断报告",
    ]
    one_oversized_step = [
        "查询业务指标、日志主题、异常日志和知识库处置手册，然后生成诊断报告"
    ]

    assert plan_covers_active_alert_diagnosis(valid) is True
    assert plan_covers_active_alert_diagnosis(one_oversized_step) is False
