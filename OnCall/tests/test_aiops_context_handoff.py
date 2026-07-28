"""AIOps 最终报告与普通聊天会话的上下文交接测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.api import aiops as aiops_api
from app.models.aiops import AIOpsRequest
from app.services.rag_agent_service import RagAgentService


class FakeAgent:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.updated_values = []

    async def aget_state(self, _config):
        return SimpleNamespace(values={"messages": self.messages})

    async def aupdate_state(self, _config, values):
        self.updated_values.append(values)
        self.messages.extend(values["messages"])


def make_rag_service(agent):
    service = object.__new__(RagAgentService)
    service.agent = agent
    service._agent_initialized = True
    return service


@pytest.mark.asyncio
async def test_record_aiops_report_adds_hidden_handoff_and_report():
    agent = FakeAgent()
    service = make_rag_service(agent)

    added = await service.record_aiops_report(
        "session-aiops",
        "  # 告警分析报告\n\nGridDataSyncFailureRateHigh  ",
    )

    assert added is True
    assert len(agent.updated_values) == 1
    messages = agent.updated_values[0]["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].additional_kwargs["hidden_from_history"] is True
    assert "当前无告警否定历史" in messages[0].content
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content.startswith("# 告警分析报告")
    assert messages[1].additional_kwargs["source"] == "aiops_report"


@pytest.mark.asyncio
async def test_record_aiops_report_is_idempotent_for_same_report():
    report = "# 告警分析报告\n\n通信中断"
    existing = AIMessage(
        content=report,
        additional_kwargs={"source": "aiops_report"},
    )
    agent = FakeAgent([existing])
    service = make_rag_service(agent)

    added = await service.record_aiops_report("session-aiops", report)

    assert added is False
    assert agent.updated_values == []


def test_session_history_hides_internal_handoff_but_keeps_report():
    service = make_rag_service(FakeAgent())
    service.checkpointer = SimpleNamespace(
        get=lambda _config: {
            "channel_values": {
                "messages": [
                    HumanMessage(
                        content="内部交接说明",
                        additional_kwargs={"hidden_from_history": True},
                    ),
                    AIMessage(
                        content="# 告警分析报告",
                        additional_kwargs={"source": "aiops_report"},
                    ),
                ]
            }
        }
    )

    history = service.get_session_history("session-aiops")

    assert history == [
        {
            "role": "assistant",
            "content": "# 告警分析报告",
            "timestamp": history[0]["timestamp"],
        }
    ]


@pytest.mark.asyncio
async def test_aiops_stream_records_final_report_once(monkeypatch):
    async def fake_diagnose(session_id):
        assert session_id == "session-aiops"
        yield {
            "type": "report",
            "stage": "final_report",
            "report": "# 告警分析报告\n\n同步失败率达到45%",
        }
        yield {
            "type": "complete",
            "stage": "diagnosis_complete",
            "diagnosis": {
                "status": "completed",
                "report": "# 告警分析报告\n\n同步失败率达到45%",
            },
        }

    record_report = AsyncMock(return_value=True)
    monkeypatch.setattr(aiops_api.aiops_service, "diagnose", fake_diagnose)
    monkeypatch.setattr(
        aiops_api.rag_agent_service,
        "record_aiops_report",
        record_report,
    )

    response = await aiops_api.diagnose_stream(
        AIOpsRequest(session_id="session-aiops")
    )
    events = [event async for event in response.body_iterator]

    assert len(events) == 2
    record_report.assert_awaited_once_with(
        "session-aiops",
        "# 告警分析报告\n\n同步失败率达到45%",
    )
