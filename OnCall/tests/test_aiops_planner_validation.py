"""Planner 活动告警计划完整性校验测试。"""

from app.agent.aiops.planner import _ensure_plan_coverage


ACTIVE_ALERT_INPUT = """诊断电网业务告警。
PREFETCHED_ACTIVE_ALERTS_TOTAL=1
PREFETCHED_ACTIVE_ALERTS_JSON:
{"total": 1}
"""


def test_single_oversized_active_alert_step_falls_back_to_six_steps() -> None:
    oversized = [
        "查询业务指标、日志主题、业务日志和知识库处置手册，然后生成诊断报告"
    ]

    result = _ensure_plan_coverage(ACTIVE_ALERT_INPUT, oversized)

    assert len(result) == 6
    assert "query_grid_service_status" in result[1]
    assert "search_topic_by_service_name" in result[2]
    assert "search_log" in result[3]
    assert "retrieve_knowledge" in result[4]
    assert "诊断报告" in result[5]


def test_complete_active_alert_plan_is_preserved() -> None:
    complete = [
        "查询业务指标",
        "查询日志主题",
        "查询业务日志",
        "检索知识库处置手册",
        "生成诊断报告",
    ]

    assert _ensure_plan_coverage(ACTIVE_ALERT_INPUT, complete) == complete
