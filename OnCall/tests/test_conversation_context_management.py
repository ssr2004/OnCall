"""本地 Transcript、Compaction 和重启恢复测试。"""

from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.services.conversation_context_service import (
    CONVERSATION_SUMMARY_PROMPT,
    ContextWindowExceededError,
    OldToolOutputPruningMiddleware,
    QwenChatTokenCounter,
    SafeSummarizationMiddleware,
    TwoTierContextMiddleware,
    build_conversation_context_budget,
    calculate_dynamic_summary_output_tokens,
)
from app.services.conversation_transcript_service import (
    ConversationTranscriptStore,
)
from app.services.rag_agent_service import RagAgentService


class FakeAgent:
    def __init__(self):
        self.messages = []

    async def aget_state(self, _config):
        return SimpleNamespace(values={"messages": self.messages})

    async def aupdate_state(self, _config, values):
        self.messages.extend(values["messages"])


def make_local_chat_service(tmp_path, response: str) -> RagAgentService:
    service = object.__new__(RagAgentService)
    service.model = FakeListChatModel(responses=[response])
    service.system_prompt = "你是测试助手"
    service.checkpointer = MemorySaver()
    service.transcript_store = ConversationTranscriptStore(tmp_path)
    service._session_locks = {}
    service._restored_sessions = set()
    service.agent = create_agent(
        service.model,
        tools=[],
        system_prompt=service.system_prompt,
        checkpointer=service.checkpointer,
    )
    service._agent_initialized = True
    return service


def test_transcript_keeps_full_visible_history_and_filters_runtime_messages(tmp_path):
    store = ConversationTranscriptStore(tmp_path)
    session_id = "session-history"
    store.append_messages(
        session_id,
        [
            HumanMessage(content="第一次提问"),
            AIMessage(content="第一次回答"),
            HumanMessage(
                content="内部交接",
                additional_kwargs={"hidden_from_history": True},
            ),
            ToolMessage(content="原始工具结果", tool_call_id="tool-1"),
            HumanMessage(
                content="历史压缩摘要",
                additional_kwargs={"lc_source": "summarization"},
            ),
            AIMessage(
                content="# 告警分析报告",
                additional_kwargs={"source": "aiops_report"},
            ),
        ],
    )

    history = store.get_history(session_id)

    assert [(item["role"], item["content"]) for item in history] == [
        ("user", "第一次提问"),
        ("assistant", "第一次回答"),
        ("assistant", "# 告警分析报告"),
    ]
    assert store.contains_aiops_report(session_id, "# 告警分析报告") is True


def test_latest_compaction_snapshot_plus_tail_restores_working_context(tmp_path):
    store = ConversationTranscriptStore(tmp_path)
    session_id = "session-recovery"
    first_sequence = store.append_messages(
        session_id,
        [HumanMessage(content="很早的问题"), AIMessage(content="很早的回答")],
    )
    summary = HumanMessage(
        id="summary-1",
        content="这里是结构化历史摘要",
        additional_kwargs={"lc_source": "summarization"},
    )
    recent = HumanMessage(content="最近的问题")
    assert store.write_compaction_snapshot_if_changed(
        session_id,
        [summary, recent],
        first_sequence,
    )
    assert not store.write_compaction_snapshot_if_changed(
        session_id,
        [summary, recent],
        first_sequence,
    )
    store.append_messages(session_id, [AIMessage(content="最近的回答")])

    restored = store.load_recovery_messages(session_id)

    assert [message.content for message in restored] == [
        "这里是结构化历史摘要",
        "最近的问题",
        "最近的回答",
    ]


def test_corrupt_transcript_tail_is_ignored(tmp_path):
    store = ConversationTranscriptStore(tmp_path)
    session_id = "session-corrupt-tail"
    store.append_messages(session_id, [HumanMessage(content="完整消息")])
    with store.transcript_path(session_id).open("a", encoding="utf-8") as file:
        file.write('{"incomplete":')

    restored = store.load_recovery_messages(session_id)

    assert [message.content for message in restored] == ["完整消息"]


def test_clear_event_hides_old_history_without_deleting_transcript(tmp_path):
    store = ConversationTranscriptStore(tmp_path)
    session_id = "session-clear"
    store.append_messages(session_id, [HumanMessage(content="清空前")])
    original_path = store.transcript_path(session_id)
    store.append_clear(session_id)
    store.append_messages(session_id, [HumanMessage(content="清空后")])

    assert original_path.exists()
    assert [item["content"] for item in store.get_history(session_id)] == [
        "清空后"
    ]
    assert [message.content for message in store.load_recovery_messages(session_id)] == [
        "清空后"
    ]


def test_session_id_is_not_used_as_a_filesystem_path(tmp_path):
    store = ConversationTranscriptStore(tmp_path)

    path = store.transcript_path("../../outside", create=True)

    assert path.parent.parent == tmp_path.resolve()
    assert "outside" not in str(path.parent)


@pytest.mark.asyncio
async def test_new_service_restores_memorysaver_state_from_transcript(tmp_path):
    store = ConversationTranscriptStore(tmp_path)
    store.append_messages(
        "session-restart",
        [HumanMessage(content="重启前问题"), AIMessage(content="重启前回答")],
    )
    service = object.__new__(RagAgentService)
    service.agent = FakeAgent()
    service.transcript_store = store
    service._restored_sessions = set()

    await service._restore_session_unlocked("session-restart")

    assert [message.content for message in service.agent.messages] == [
        "重启前问题",
        "重启前回答",
    ]


@pytest.mark.asyncio
async def test_aiops_report_is_persisted_and_remains_idempotent_after_restart(tmp_path):
    store = ConversationTranscriptStore(tmp_path)
    first_service = object.__new__(RagAgentService)
    first_service.agent = FakeAgent()
    first_service._agent_initialized = True
    first_service.transcript_store = store
    first_service._restored_sessions = set()

    added = await first_service.record_aiops_report(
        "session-aiops-restart",
        "# 告警分析报告\n\n通信中断",
    )

    assert added is True
    assert [item["content"] for item in store.get_history("session-aiops-restart")] == [
        "# 告警分析报告\n\n通信中断"
    ]

    restarted_service = object.__new__(RagAgentService)
    restarted_service.agent = FakeAgent()
    restarted_service._agent_initialized = True
    restarted_service.transcript_store = store
    restarted_service._restored_sessions = set()

    duplicate = await restarted_service.record_aiops_report(
        "session-aiops-restart",
        "# 告警分析报告\n\n通信中断",
    )

    assert duplicate is False
    assert len(store.get_history("session-aiops-restart")) == 1


@pytest.mark.asyncio
async def test_query_restores_previous_turns_after_service_restart(tmp_path):
    first_service = make_local_chat_service(tmp_path, "第一次回答")

    first_answer = await first_service.query("第一次问题", "session-query-restart")

    assert first_answer == "第一次回答"

    restarted_service = make_local_chat_service(tmp_path, "重启后的回答")
    second_answer = await restarted_service.query(
        "继续刚才的问题",
        "session-query-restart",
    )

    assert second_answer == "重启后的回答"
    assert [
        (item["role"], item["content"])
        for item in restarted_service.get_session_history("session-query-restart")
    ] == [
        ("user", "第一次问题"),
        ("assistant", "第一次回答"),
        ("user", "继续刚才的问题"),
        ("assistant", "重启后的回答"),
    ]


def test_old_tool_output_is_pruned_before_full_compaction():
    middleware = OldToolOutputPruningMiddleware(
        trigger_tokens=1,
        keep_recent_messages=1,
        max_tool_output_chars=20,
    )
    old_tool = ToolMessage(
        id="tool-result-1",
        content="x" * 200,
        tool_call_id="tool-call-1",
    )

    result = middleware.before_model(
        {"messages": [old_tool, HumanMessage(content="最近问题")]},
        None,
    )

    assert result is not None
    replacement = result["messages"][0]
    assert replacement.id == old_tool.id
    assert replacement.additional_kwargs["context_pruned"] is True
    assert "Transcript" in replacement.content


def test_context_budget_is_derived_from_model_limits_and_runtime_overhead():
    budget = build_conversation_context_budget(
        context_window_tokens=1_000_000,
        max_input_tokens=991_808,
        max_output_tokens=4096,
        operating_input_tokens=256_000,
        fixed_prompt_tokens=4000,
        safety_ratio=0.8,
        keep_recent_turns=8,
    )

    assert budget.hard_input_tokens == 991_808
    assert budget.effective_input_tokens == 256_000
    assert budget.message_trigger_tokens == 252_000
    assert budget.keep_recent_tokens == 8192
    assert budget.cost_trigger_tokens == 256_000
    assert budget.safety_trigger_tokens == 793_446
    assert budget.keep_recent_turns == 8


def test_token_retention_moves_cutoff_to_complete_conversation_turn():
    middleware = OldToolOutputPruningMiddleware(
        trigger_tokens=1,
        keep_recent_tokens=1,
        max_tool_output_chars=20,
    )
    old_tool = ToolMessage(
        id="old-tool-result",
        content="x" * 200,
        tool_call_id="old-tool-call",
    )
    messages = [
        old_tool,
        HumanMessage(content="最近一轮问题"),
        AIMessage(content="最近一轮回答"),
    ]

    assert middleware._recent_cutoff(messages) == 1


@pytest.mark.asyncio
async def test_summarization_preserves_recent_messages_and_uses_custom_prompt():
    model = FakeListChatModel(responses=["## 会话目标\n继续诊断当前告警"])
    middleware = SafeSummarizationMiddleware(
        model=model,
        trigger=("tokens", 1),
        keep=("messages", 2),
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
        trim_tokens_to_summarize=None,
    )
    messages = [
        HumanMessage(content="早期问题"),
        AIMessage(content="早期回答"),
        HumanMessage(content="最近问题"),
        AIMessage(content="最近回答"),
    ]

    result = await middleware.abefore_model({"messages": messages}, None)

    assert result is not None
    compacted_messages = result["messages"]
    summary_messages = [
        message
        for message in compacted_messages
        if getattr(message, "additional_kwargs", {}).get("lc_source")
        == "summarization"
    ]
    assert len(summary_messages) == 1
    assert "继续诊断当前告警" in summary_messages[0].content
    assert [message.content for message in compacted_messages[-2:]] == [
        "最近问题",
        "最近回答",
    ]


def _content_length_tokens(messages):
    return sum(len(str(message.content)) for message in messages)


@pytest.mark.asyncio
async def test_two_tier_cost_compaction_keeps_latest_eight_complete_turns():
    middleware = TwoTierContextMiddleware(
        model=FakeListChatModel(responses=["结构化历史摘要"]),
        cost_trigger_tokens=100,
        safety_trigger_tokens=1000,
        hard_input_tokens=2000,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=_content_length_tokens,
        summary_text_token_counter=len,
        summary_min_output_tokens=8,
        summary_max_output_tokens=32,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=5000,
        summary_model_max_input_tokens=4000,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )
    messages = []
    for index in range(10):
        messages.extend(
            [
                HumanMessage(content=f"第{index}轮问题" + "问" * 5),
                AIMessage(content=f"第{index}轮回答" + "答" * 5),
            ]
        )

    result = await middleware.abefore_model({"messages": messages}, None)

    assert result is not None
    compacted = result["messages"][1:]
    assert compacted[0].additional_kwargs["lc_source"] == "summarization"
    assert [message.content for message in compacted[1:3]] == [
        "第2轮问题" + "问" * 5,
        "第2轮回答" + "答" * 5,
    ]
    assert len(compacted[1:]) == 16


@pytest.mark.asyncio
async def test_existing_summary_waits_until_safety_tier_before_recompaction():
    middleware = TwoTierContextMiddleware(
        model=FakeListChatModel(responses=["不应调用"]),
        cost_trigger_tokens=20,
        safety_trigger_tokens=200,
        hard_input_tokens=400,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=_content_length_tokens,
        summary_text_token_counter=len,
        summary_min_output_tokens=8,
        summary_max_output_tokens=32,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=5000,
        summary_model_max_input_tokens=4000,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )
    messages = [
        HumanMessage(
            content="已有摘要",
            additional_kwargs={"lc_source": "summarization"},
        ),
        HumanMessage(content="新增问题"),
        AIMessage(content="新增回答"),
    ]

    result = await middleware.abefore_model({"messages": messages}, None)

    assert result is None


@pytest.mark.asyncio
async def test_safety_tier_merges_previous_summary_and_accumulated_turns():
    middleware = TwoTierContextMiddleware(
        model=FakeListChatModel(responses=["第二档合并摘要"]),
        cost_trigger_tokens=20,
        safety_trigger_tokens=120,
        hard_input_tokens=500,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=_content_length_tokens,
        summary_text_token_counter=len,
        summary_min_output_tokens=8,
        summary_max_output_tokens=32,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=5000,
        summary_model_max_input_tokens=4000,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )
    messages = [
        HumanMessage(
            content="第一次压缩形成的历史摘要" + "摘" * 20,
            additional_kwargs={"lc_source": "summarization"},
        )
    ]
    for index in range(10):
        messages.extend(
            [
                HumanMessage(content=f"新增第{index}轮问题" + "问" * 4),
                AIMessage(content=f"新增第{index}轮回答" + "答" * 4),
            ]
        )

    result = await middleware.abefore_model({"messages": messages}, None)

    assert result is not None
    compacted = result["messages"][1:]
    assert "第二档合并摘要" in compacted[0].content
    assert compacted[1].content.startswith("新增第2轮问题")
    assert len(compacted[1:]) == 16


def test_safety_tool_pruning_never_prunes_unconsumed_current_result():
    middleware = TwoTierContextMiddleware(
        model=FakeListChatModel(responses=["摘要"]),
        cost_trigger_tokens=20,
        safety_trigger_tokens=50,
        hard_input_tokens=1000,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=_content_length_tokens,
        summary_text_token_counter=len,
        summary_min_output_tokens=8,
        summary_max_output_tokens=32,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=5000,
        summary_model_max_input_tokens=4000,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )
    consumed = ToolMessage(
        id="consumed-tool",
        content="a" * 100,
        tool_call_id="call-consumed",
    )
    current = ToolMessage(
        id="current-tool",
        content="b" * 100,
        tool_call_id="call-current",
    )
    messages = [
        HumanMessage(
            content="历史摘要",
            additional_kwargs={"lc_source": "summarization"},
        ),
        HumanMessage(content="较早问题"),
        consumed,
        AIMessage(content="已使用工具结果形成的回答"),
        HumanMessage(content="当前问题"),
        current,
    ]

    pruned, changed = middleware._prune_consumed_tools(messages, safety_mode=True)

    assert changed is True
    assert pruned[2].additional_kwargs["context_pruned"] is True
    assert pruned[5].content == "b" * 100


def test_hard_limit_blocks_request_that_cannot_be_compacted():
    middleware = TwoTierContextMiddleware(
        model=FakeListChatModel(responses=["摘要"]),
        cost_trigger_tokens=10,
        safety_trigger_tokens=20,
        hard_input_tokens=30,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=_content_length_tokens,
        summary_text_token_counter=len,
        summary_min_output_tokens=8,
        summary_max_output_tokens=32,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=5000,
        summary_model_max_input_tokens=4000,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )

    with pytest.raises(ContextWindowExceededError):
        middleware.before_model(
            {"messages": [HumanMessage(content="当前问题" * 20)]},
            None,
        )


def test_qwen_tokenizer_counts_system_messages_and_tools():
    def sample_tool(query: str) -> str:
        """查询示例数据。"""
        return query

    counter_without_tools = QwenChatTokenCounter(
        model_name="qwen3.7-plus-2026-05-26",
        system_prompt="你是运维助手",
        tools=[],
    )
    counter_with_tools = QwenChatTokenCounter(
        model_name="qwen3.7-plus-2026-05-26",
        system_prompt="你是运维助手",
        tools=[sample_tool],
    )
    messages = [HumanMessage(content="查询当前活动告警")]

    assert counter_without_tools(messages) > counter_without_tools([])
    assert counter_with_tools(messages) > counter_without_tools(messages)


@pytest.mark.parametrize(
    ("history_tokens", "expected_summary_limit"),
    [
        (100_000, 8_192),
        (200_000, 10_000),
        (256_000, 12_800),
        (500_000, 25_000),
        (700_000, 32_768),
    ],
)
def test_dynamic_summary_budget_scales_between_configured_bounds(
    history_tokens,
    expected_summary_limit,
):
    assert calculate_dynamic_summary_output_tokens(
        history_tokens,
        minimum_tokens=8_192,
        maximum_tokens=32_768,
        output_ratio=0.05,
    ) == expected_summary_limit


def test_summary_output_budget_is_capped_by_remaining_model_context():
    middleware = TwoTierContextMiddleware(
        model=FakeListChatModel(responses=["摘要"]),
        cost_trigger_tokens=10,
        safety_trigger_tokens=20,
        hard_input_tokens=100,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=lambda _messages: 500,
        summary_text_token_counter=lambda _prompt: 950,
        summary_min_output_tokens=80,
        summary_max_output_tokens=320,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=1000,
        summary_model_max_input_tokens=980,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )

    _, max_tokens = middleware._summary_request(
        [HumanMessage(content="需要压缩的历史")]
    )

    assert max_tokens == 50


def test_summary_request_rejects_input_above_summary_model_limit():
    middleware = TwoTierContextMiddleware(
        model=FakeListChatModel(responses=["摘要"]),
        cost_trigger_tokens=10,
        safety_trigger_tokens=20,
        hard_input_tokens=100,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=lambda _messages: 500,
        summary_text_token_counter=lambda _prompt: 981,
        summary_min_output_tokens=80,
        summary_max_output_tokens=320,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=1000,
        summary_model_max_input_tokens=980,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )

    with pytest.raises(ContextWindowExceededError, match="超过摘要模型最大输入"):
        middleware._summary_request([HumanMessage(content="超长历史")])


@pytest.mark.asyncio
async def test_dynamic_summary_limit_is_bound_to_each_model_call():
    class RecordingSummaryModel:
        def __init__(self):
            self.bound_max_tokens = []

        def bind(self, *, max_tokens):
            self.bound_max_tokens.append(max_tokens)
            return self

        async def ainvoke(self, _prompt, config=None):
            assert config["metadata"]["summary_max_tokens"] == 250
            return AIMessage(content="动态摘要")

    model = RecordingSummaryModel()
    middleware = TwoTierContextMiddleware(
        model=model,
        cost_trigger_tokens=10,
        safety_trigger_tokens=20,
        hard_input_tokens=100,
        keep_recent_turns=8,
        max_tool_output_chars=20,
        token_counter=_content_length_tokens,
        summary_input_token_counter=lambda _messages: 5000,
        summary_text_token_counter=lambda _prompt: 100,
        summary_min_output_tokens=80,
        summary_max_output_tokens=320,
        summary_output_ratio=0.05,
        summary_model_context_window_tokens=1000,
        summary_model_max_input_tokens=980,
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )

    summary = await middleware._acreate_summary(
        [HumanMessage(content="需要动态确定长度的历史")]
    )

    assert summary == "动态摘要"
    assert model.bound_max_tokens == [250]
