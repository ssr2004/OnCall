"""生成模型路由和降级行为测试。"""

import importlib

import pytest

from app.config import Settings

replanner_module = importlib.import_module("app.agent.aiops.replanner")


def test_default_model_routing_uses_fixed_snapshots():
    settings = Settings(_env_file=None)

    assert settings.chat_model == "qwen3.7-plus-2026-05-26"
    assert settings.chat_summary_model == "qwen3.7-flash-2026-07-15"
    assert settings.aiops_planner_model == "qwen3.7-plus-2026-05-26"
    assert settings.aiops_executor_model == "qwen3.7-plus-2026-05-26"
    assert settings.aiops_replanner_model == "qwen3.7-plus-2026-05-26"
    assert settings.aiops_report_model == "qwen3.7-max-2026-06-08"
    assert settings.rag_query_rewrite_model == "qwen3.7-flash-2026-07-15"
    assert settings.rag_eval_judge_model == "qwen3.7-max-2026-06-08"


@pytest.mark.asyncio
async def test_report_generation_falls_back_from_max_to_plus(monkeypatch):
    called_models = []

    class FakeModel:
        def __init__(self, model: str):
            self.model = model

        def with_structured_output(self, _schema):
            return self

    class FakeChain:
        def __init__(self, model):
            self.model = model

        async def ainvoke(self, _input):
            called_models.append(self.model.model)
            if self.model.model == replanner_module.config.aiops_report_model:
                raise RuntimeError("Max unavailable")
            return replanner_module.Response(response="Plus 降级报告")

    class FakePrompt:
        def __or__(self, model):
            return FakeChain(model)

    def fake_chat_qwen(**kwargs):
        return FakeModel(kwargs["model"])

    monkeypatch.setattr(replanner_module, "ChatQwen", fake_chat_qwen)
    monkeypatch.setattr(replanner_module, "response_prompt", FakePrompt())

    result = await replanner_module._generate_response(
        {
            "input": "诊断当前告警",
            "past_steps": [("查询告警", "发现通信中断")],
        }
    )

    assert called_models == [
        replanner_module.config.aiops_report_model,
        replanner_module.config.aiops_replanner_model,
    ]
    assert result == {"response": "Plus 降级报告"}
