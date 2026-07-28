"""AIOps 流式进度事件测试。"""

from types import SimpleNamespace

import pytest

from app.services.aiops_service import AIOpsService


class FakeGraph:
    """返回与 LangGraph ``stream_mode=updates`` 相同形状的局部更新。"""

    async def astream(self, **_kwargs):
        yield {"planner": {"plan": ["确认活动告警", "生成诊断报告"]}}
        yield {
            "executor": {
                "plan": ["生成诊断报告"],
                "past_steps": [("确认活动告警", "当前无活动告警")],
            }
        }
        yield {"replanner": {"response": "# 告警分析报告\n\n当前无活动告警。"}}

    def get_state(self, _config):
        return SimpleNamespace(
            values={"response": "# 告警分析报告\n\n当前无活动告警。"}
        )


class EarlyFinishGraph:
    """模拟 Replanner 在证据充分后收敛原计划。"""

    async def astream(self, **_kwargs):
        yield {"planner": {"plan": ["查询告警", "查询指标", "查询日志", "生成报告"]}}
        yield {
            "executor": {
                "plan": ["查询指标", "查询日志", "生成报告"],
                "past_steps": [("查询告警", "没有活动告警")],
            }
        }
        yield {"replanner": {"response": "证据充分，直接生成报告"}}

    def get_state(self, _config):
        return SimpleNamespace(values={"response": "证据充分，直接生成报告"})


async def collect_events(graph) -> list[dict]:
    service = object.__new__(AIOpsService)
    service.graph = graph
    return [event async for event in service.execute("诊断电网告警", "test-session")]


@pytest.mark.asyncio
async def test_final_report_completes_two_step_progress() -> None:
    events = await collect_events(FakeGraph())

    plan = next(event for event in events if event["type"] == "plan")
    step = next(event for event in events if event["type"] == "step_complete")
    report = next(event for event in events if event["type"] == "report")
    complete = next(event for event in events if event["type"] == "complete")

    assert plan["total_steps"] == 2
    assert step["message"] == "步骤执行完成 (1/2)"
    assert report["message"] == "诊断流程完成 (2/2)"
    assert report["remaining_steps"] == 0
    assert complete["message"] == "诊断流程完成 (2/2)"


@pytest.mark.asyncio
async def test_replanner_reports_skipped_steps_as_a_converged_plan() -> None:
    events = await collect_events(EarlyFinishGraph())

    report = next(event for event in events if event["type"] == "report")

    assert report["completed_steps"] == 2
    assert report["total_steps"] == 2
    assert report["skipped_steps"] == 2
    assert report["message"] == "诊断流程完成 (2/2)，重规划后省略 2 个不再需要的步骤"
