"""对话 Agent 的 Token 预算、旧工具输出裁剪和上下文压缩。"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from dashscope.tokenizers import get_tokenizer
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
    get_buffer_string,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

TokenCounter = Callable[[Iterable[Any]], int]
TextTokenCounter = Callable[[str], int]


def count_qwen_tokens_conservatively(messages: Iterable[Any]) -> int:
    """保守估算中英文混合 Qwen 消息 Token，避免按英文 chars/4 低估中文。"""
    return count_tokens_approximately(messages, chars_per_token=2.0)


class QwenChatTokenCounter:
    """使用 DashScope 随包提供的 Qwen Tokenizer 计算完整聊天请求。

    计数覆盖 System Prompt、工具 Schema、历史消息、工具调用和工具结果。
    ChatML 标记与 OpenAI 兼容工具协议也会进入计数，因此可以直接使用
    完整请求阈值，而不需要再从阈值中人工扣除固定提示词开销。
    """

    def __init__(
        self,
        *,
        model_name: str,
        system_prompt: str,
        tools: Sequence[Any],
    ) -> None:
        self._tokenizer = get_tokenizer(model_name)
        self._system_prompt = system_prompt
        self._tool_schemas = [convert_to_openai_tool(tool) for tool in tools]

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    def _count_text(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def count_text(self, text: str) -> int:
        """计算普通文本的 Qwen Token 数。"""
        return self._count_text(text)

    def _count_chat_message(self, message: dict[str, Any]) -> int:
        role = str(message.get("role", "user"))
        body = {key: value for key, value in message.items() if key != "role"}
        # Qwen 对话采用 ChatML 边界；复杂内容和 tool_calls 按实际 JSON 形态计数。
        return (
            self._count_text("<|im_start|>")
            + self._count_text(role)
            + self._count_text("\n")
            + self._count_text(self._json(body))
            + self._count_text("<|im_end|>\n")
        )

    def __call__(self, messages: Iterable[Any]) -> int:
        materialized = list(messages)
        openai_messages: list[dict[str, Any]] = []
        if self._system_prompt:
            openai_messages.append(
                {"role": "system", "content": self._system_prompt}
            )
        if materialized:
            converted = convert_to_openai_messages(materialized)
            if isinstance(converted, dict):
                openai_messages.append(converted)
            else:
                openai_messages.extend(converted)

        total = sum(self._count_chat_message(message) for message in openai_messages)
        if self._tool_schemas:
            total += self._count_text(self._json({"tools": self._tool_schemas}))
        # 为即将生成的 assistant 消息预留 ChatML 起始标记。
        total += self._count_text("<|im_start|>assistant\n")
        return total

    def count_messages_only(self, messages: Iterable[Any]) -> int:
        """仅计算给定历史消息，不包含 System Prompt 和工具 Schema。"""
        materialized = list(messages)
        if not materialized:
            return 0
        converted = convert_to_openai_messages(materialized)
        openai_messages = [converted] if isinstance(converted, dict) else converted
        return sum(self._count_chat_message(message) for message in openai_messages)


def calculate_dynamic_summary_output_tokens(
    tokens_to_summarize: int,
    *,
    minimum_tokens: int,
    maximum_tokens: int,
    output_ratio: float,
) -> int:
    """根据实际待摘要历史长度计算本次摘要输出上限。"""
    if tokens_to_summarize < 0:
        raise ValueError("待摘要 Token 数不能为负数")
    if minimum_tokens <= 0 or maximum_tokens < minimum_tokens:
        raise ValueError("摘要输出范围配置无效")
    if not 0 < output_ratio <= 1:
        raise ValueError("摘要输出比例必须在 0 和 1 之间")
    proportional_tokens = math.ceil(tokens_to_summarize * output_ratio)
    return min(maximum_tokens, max(minimum_tokens, proportional_tokens))


def estimate_fixed_prompt_tokens(
    system_prompt: str,
    tools: Sequence[Any],
) -> int:
    """估算每次模型调用固定携带的 System Prompt 与工具 Schema 开销。"""
    return count_tokens_approximately(
        [SystemMessage(content=system_prompt)],
        chars_per_token=2.0,
        tools=list(tools),
    )


@dataclass(frozen=True)
class ConversationContextBudget:
    """由模型真实规格和运行时固定开销推导出的对话工作窗口预算。"""

    hard_input_tokens: int
    effective_input_tokens: int
    fixed_prompt_tokens: int
    message_trigger_tokens: int
    keep_recent_tokens: int
    cost_trigger_tokens: int
    safety_trigger_tokens: int
    keep_recent_turns: int


def build_conversation_context_budget(
    *,
    context_window_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    operating_input_tokens: int,
    fixed_prompt_tokens: int,
    safety_ratio: float = 0.8,
    keep_recent_turns: int = 8,
) -> ConversationContextBudget:
    """计算 256K 成本档、80% 安全档和模型硬上限。"""
    if min(context_window_tokens, max_input_tokens, max_output_tokens) <= 0:
        raise ValueError("模型上下文、最大输入和最大输出必须为正整数")
    if max_output_tokens >= context_window_tokens:
        raise ValueError("模型最大输出必须小于上下文窗口")
    if not 0 < safety_ratio < 1:
        raise ValueError("安全档比例必须在 0 和 1 之间")
    if keep_recent_turns <= 0:
        raise ValueError("最近完整轮次数必须为正整数")

    hard_input_tokens = min(
        max_input_tokens,
        context_window_tokens - max_output_tokens,
    )
    effective_input_tokens = hard_input_tokens
    if operating_input_tokens > 0:
        effective_input_tokens = min(
            effective_input_tokens,
            operating_input_tokens,
        )
    safety_trigger_tokens = int(hard_input_tokens * safety_ratio)
    if safety_trigger_tokens <= effective_input_tokens:
        raise ValueError("安全档必须高于成本档")

    message_trigger_tokens = effective_input_tokens - fixed_prompt_tokens
    if message_trigger_tokens <= 0:
        raise ValueError("System Prompt 与工具 Schema 已超过模型输入预算")

    # 保留约两个最大回答长度的最近完整对话，兼顾连续追问和压缩收益。
    keep_recent_tokens = min(
        max_output_tokens * 2,
        message_trigger_tokens,
    )
    return ConversationContextBudget(
        hard_input_tokens=hard_input_tokens,
        effective_input_tokens=effective_input_tokens,
        fixed_prompt_tokens=fixed_prompt_tokens,
        message_trigger_tokens=message_trigger_tokens,
        keep_recent_tokens=keep_recent_tokens,
        cost_trigger_tokens=effective_input_tokens,
        safety_trigger_tokens=safety_trigger_tokens,
        keep_recent_turns=keep_recent_turns,
    )

CONVERSATION_SUMMARY_PROMPT = """
<role>
你是智能运维对话的上下文压缩器。
</role>

<objective>
将即将退出模型工作窗口的历史消息压缩为可继续对话的可靠交接摘要。
摘要会替代原始消息，因此必须保留后续回答所需的事实，不能补充或猜测信息。
</objective>

<requirements>
请严格使用以下章节；没有内容时写“无”：

## 会话目标
用户当前希望解决的问题，以及本次会话的主要目标。

## 已确认事实
用户明确提供或确认的事实、精确服务名、告警名、时间、数值和状态。

## AIOps 诊断上下文
保留历史诊断报告中的告警、根因、关键证据、处置建议、风险等级、事件标识和引用来源。
明确区分“诊断时刻的历史状态”和“当前实时状态”，不得用当前状态覆盖历史故障。

## 用户反馈与约束
保留用户对诊断的确认、否定、纠正、偏好和尚未改变的要求。

## 已完成事项
已经执行的查询、工具调用和得到的有效结论。工具原始输出只保留关键证据与来源标识。

## 未解决问题与下一步
仍需回答的问题、待验证假设和建议的下一步动作。

## 可追溯引用
保留所有仍被结论引用的知识来源编号、事件 ID、证据 ID 或报告标识。
</requirements>

<messages>
{messages}
</messages>

只输出压缩摘要，不要添加开场白或解释。
""".strip()


class OldToolOutputPruningMiddleware(AgentMiddleware):
    """在整体压缩前先缩短较早且过长的工具输出。"""

    def __init__(
        self,
        *,
        trigger_tokens: int,
        keep_recent_messages: int | None = None,
        keep_recent_tokens: int | None = None,
        max_tool_output_chars: int,
        token_counter: TokenCounter = count_qwen_tokens_conservatively,
    ) -> None:
        super().__init__()
        if keep_recent_messages is None and keep_recent_tokens is None:
            raise ValueError("必须配置最近消息数量或最近消息 Token 预算")
        self.trigger_tokens = trigger_tokens
        self.keep_recent_messages = keep_recent_messages
        self.keep_recent_tokens = keep_recent_tokens
        self.max_tool_output_chars = max_tool_output_chars
        self.token_counter = token_counter

    def _recent_cutoff(self, messages: Sequence[BaseMessage]) -> int:
        if self.keep_recent_tokens is None:
            return max(0, len(messages) - int(self.keep_recent_messages or 0))

        cutoff = len(messages)
        recent_tokens = 0
        for index in range(len(messages) - 1, -1, -1):
            message_tokens = self.token_counter([messages[index]])
            if cutoff < len(messages) and (
                recent_tokens + message_tokens > self.keep_recent_tokens
            ):
                break
            recent_tokens += message_tokens
            cutoff = index

        # Token 边界落在一轮对话中间时，回退到该轮 HumanMessage。
        while cutoff > 0 and not isinstance(messages[cutoff], HumanMessage):
            cutoff -= 1
        return cutoff

    def _prune(self, state: dict[str, Any]) -> dict[str, Any] | None:
        messages: Sequence[BaseMessage] = state.get("messages", [])
        if self.token_counter(messages) < self.trigger_tokens:
            return None

        cutoff = self._recent_cutoff(messages)
        replacements: list[ToolMessage] = []
        for message in messages[:cutoff]:
            if not isinstance(message, ToolMessage):
                continue
            content = message.content
            rendered = content if isinstance(content, str) else str(content)
            if len(rendered) <= self.max_tool_output_chars:
                continue
            metadata = dict(message.additional_kwargs or {})
            metadata["context_pruned"] = True
            replacement = message.model_copy(
                update={
                    "content": (
                        "[较早的工具原始输出已从模型工作上下文裁剪；"
                        "完整内容仍保存在本地会话 Transcript 中。]"
                    ),
                    "additional_kwargs": metadata,
                }
            )
            replacements.append(replacement)

        if not replacements:
            return None
        return {"messages": replacements}

    def before_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        return self._prune(state)

    async def abefore_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        return self._prune(state)


class SafeSummarizationMiddleware(SummarizationMiddleware):
    """摘要模型异常时保持原上下文，避免错误摘要替换完整历史。"""

    @staticmethod
    def _find_safe_cutoff_point(
        messages: list[BaseMessage],
        cutoff_index: int,
    ) -> int:
        """在框架工具调用安全边界基础上，再保持完整用户对话轮次。"""
        safe_cutoff = SummarizationMiddleware._find_safe_cutoff_point(
            messages,
            cutoff_index,
        )
        while safe_cutoff > 0 and not isinstance(
            messages[safe_cutoff],
            HumanMessage,
        ):
            safe_cutoff -= 1
        return safe_cutoff

    @staticmethod
    def _summary_failed(result: dict[str, Any] | None) -> bool:
        if not result:
            return False
        for message in result.get("messages", []):
            content = getattr(message, "content", "")
            if isinstance(content, str) and "Error generating summary:" in content:
                return True
        return False

    def before_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        result = super().before_model(state, runtime)
        return None if self._summary_failed(result) else result

    async def abefore_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        result = await super().abefore_model(state, runtime)
        return None if self._summary_failed(result) else result


class ContextWindowExceededError(RuntimeError):
    """压缩后仍无法安全装入模型输入窗口。"""


class TwoTierContextMiddleware(SafeSummarizationMiddleware):
    """对话上下文的 256K 成本档与 80% 安全档压缩策略。

    会话中尚无历史摘要时使用成本档；生成首份摘要后，不会在每个新增轮次
    重复摘要，而是允许原始轮次继续积累到安全档。每次压缩都重新选择当时
    最新的完整对话轮次，旧摘要会与滑出窗口的原始轮次一起合并。
    """

    def __init__(
        self,
        *,
        model: Any,
        cost_trigger_tokens: int,
        safety_trigger_tokens: int,
        hard_input_tokens: int,
        keep_recent_turns: int,
        max_tool_output_chars: int,
        token_counter: TokenCounter,
        summary_input_token_counter: TokenCounter,
        summary_text_token_counter: TextTokenCounter,
        summary_min_output_tokens: int,
        summary_max_output_tokens: int,
        summary_output_ratio: float,
        summary_model_context_window_tokens: int,
        summary_model_max_input_tokens: int,
        summary_prompt: str,
    ) -> None:
        if not 0 < cost_trigger_tokens < safety_trigger_tokens < hard_input_tokens:
            raise ValueError("上下文阈值必须满足 成本档 < 安全档 < 硬上限")
        if keep_recent_turns <= 0:
            raise ValueError("最近完整轮次数必须为正整数")
        if max_tool_output_chars <= 0:
            raise ValueError("工具结果裁剪长度必须为正整数")
        calculate_dynamic_summary_output_tokens(
            0,
            minimum_tokens=summary_min_output_tokens,
            maximum_tokens=summary_max_output_tokens,
            output_ratio=summary_output_ratio,
        )
        if min(
            summary_model_context_window_tokens,
            summary_model_max_input_tokens,
        ) <= 0:
            raise ValueError("摘要模型上下文和最大输入必须为正整数")

        super().__init__(
            model=model,
            trigger=None,
            keep=("messages", 1),
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=None,
        )
        self.cost_trigger_tokens = cost_trigger_tokens
        self.safety_trigger_tokens = safety_trigger_tokens
        self.hard_input_tokens = hard_input_tokens
        self.keep_recent_turns = keep_recent_turns
        self.max_tool_output_chars = max_tool_output_chars
        self.summary_input_token_counter = summary_input_token_counter
        self.summary_text_token_counter = summary_text_token_counter
        self.summary_min_output_tokens = summary_min_output_tokens
        self.summary_max_output_tokens = summary_max_output_tokens
        self.summary_output_ratio = summary_output_ratio
        self.summary_model_context_window_tokens = summary_model_context_window_tokens
        self.summary_model_max_input_tokens = summary_model_max_input_tokens

    def _summary_request(
        self,
        messages_to_summarize: list[BaseMessage],
    ) -> tuple[str, int]:
        """构造摘要请求，并按真实待摘要长度计算本次 max_tokens。"""
        formatted_messages = get_buffer_string(messages_to_summarize)
        prompt = self.summary_prompt.format(messages=formatted_messages).rstrip()
        tokens_to_summarize = self.summary_input_token_counter(
            messages_to_summarize
        )
        desired_output_tokens = calculate_dynamic_summary_output_tokens(
            tokens_to_summarize,
            minimum_tokens=self.summary_min_output_tokens,
            maximum_tokens=self.summary_max_output_tokens,
            output_ratio=self.summary_output_ratio,
        )
        summary_input_tokens = self.summary_text_token_counter(prompt)
        if summary_input_tokens > self.summary_model_max_input_tokens:
            raise ContextWindowExceededError(
                "待摘要历史本身已有 "
                f"{summary_input_tokens} Token，超过摘要模型最大输入 "
                f"{self.summary_model_max_input_tokens}。"
            )

        available_output_tokens = (
            self.summary_model_context_window_tokens - summary_input_tokens
        )
        if available_output_tokens <= 0:
            raise ContextWindowExceededError(
                "摘要请求输入已占满摘要模型上下文窗口，无法生成摘要。"
            )
        return prompt, min(desired_output_tokens, available_output_tokens)

    def _create_summary(self, messages_to_summarize: list[BaseMessage]) -> str:
        if not messages_to_summarize:
            return "No previous conversation history."
        try:
            prompt, max_tokens = self._summary_request(messages_to_summarize)
            response = self.model.bind(max_tokens=max_tokens).invoke(
                prompt,
                config={
                    "metadata": {
                        "lc_source": "summarization",
                        "summary_max_tokens": max_tokens,
                    }
                },
            )
            return response.text.strip()
        except ContextWindowExceededError:
            raise
        except Exception as exc:
            return f"Error generating summary: {exc!s}"

    async def _acreate_summary(
        self,
        messages_to_summarize: list[BaseMessage],
    ) -> str:
        if not messages_to_summarize:
            return "No previous conversation history."
        try:
            prompt, max_tokens = self._summary_request(messages_to_summarize)
            response = await self.model.bind(max_tokens=max_tokens).ainvoke(
                prompt,
                config={
                    "metadata": {
                        "lc_source": "summarization",
                        "summary_max_tokens": max_tokens,
                    }
                },
            )
            return response.text.strip()
        except ContextWindowExceededError:
            raise
        except Exception as exc:
            return f"Error generating summary: {exc!s}"

    @staticmethod
    def _is_summary(message: BaseMessage) -> bool:
        metadata = getattr(message, "additional_kwargs", {}) or {}
        return metadata.get("lc_source") == "summarization"

    def _has_summary(self, messages: Sequence[BaseMessage]) -> bool:
        return any(self._is_summary(message) for message in messages)

    def _recent_turn_cutoff(self, messages: Sequence[BaseMessage]) -> int:
        """返回最新 N 个完整 Human 轮次的起点，不把历史摘要算作新轮次。"""
        turn_starts = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage) and not self._is_summary(message)
        ]
        if len(turn_starts) <= self.keep_recent_turns:
            return 0
        return turn_starts[-self.keep_recent_turns]

    @staticmethod
    def _current_turn_start(messages: Sequence[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            metadata = getattr(message, "additional_kwargs", {}) or {}
            if (
                isinstance(message, HumanMessage)
                and metadata.get("lc_source") != "summarization"
            ):
                return index
        return len(messages)

    @staticmethod
    def _tool_result_was_consumed(
        messages: Sequence[BaseMessage],
        tool_index: int,
    ) -> bool:
        """只有结果之后已经出现 AIMessage，才说明模型消费过该工具结果。"""
        return any(
            isinstance(message, AIMessage)
            for message in messages[tool_index + 1 :]
        )

    def _pruned_tool_message(self, message: ToolMessage, rendered: str) -> ToolMessage:
        metadata = dict(message.additional_kwargs or {})
        metadata.update(
            {
                "context_pruned": True,
                "original_content_chars": len(rendered),
            }
        )
        preview_budget = max(0, self.max_tool_output_chars - 320)
        preview = rendered[:preview_budget].rstrip()
        details = [
            "[已消费的工具原始输出已压缩]",
            f"tool_name={message.name or 'unknown'}",
            f"tool_call_id={message.tool_call_id}",
            f"original_chars={len(rendered)}",
        ]
        if preview:
            details.append(f"evidence_preview={preview}")
        details.append("完整结果仍保存在本地会话 Transcript 中。")
        return message.model_copy(
            update={
                "content": "\n".join(details),
                "additional_kwargs": metadata,
            }
        )

    def _prune_consumed_tools(
        self,
        messages: Sequence[BaseMessage],
        *,
        safety_mode: bool,
    ) -> tuple[list[BaseMessage], bool]:
        """成本档只裁剪保留窗口以前的结果，安全档扩展到当前轮以前。"""
        retained_cutoff = self._recent_turn_cutoff(messages)
        prune_before = (
            self._current_turn_start(messages) if safety_mode else retained_cutoff
        )
        updated = list(messages)
        changed = False
        for index, message in enumerate(messages[:prune_before]):
            if not isinstance(message, ToolMessage):
                continue
            if (message.additional_kwargs or {}).get("context_pruned"):
                continue
            if not self._tool_result_was_consumed(messages, index):
                continue
            content = message.content
            rendered = content if isinstance(content, str) else self._json_content(content)
            if len(rendered) <= self.max_tool_output_chars:
                continue
            updated[index] = self._pruned_tool_message(message, rendered)
            changed = True
        return updated, changed

    @staticmethod
    def _json_content(content: Any) -> str:
        return json.dumps(content, ensure_ascii=False, default=str)

    def _assert_below_hard_limit(self, messages: Sequence[BaseMessage]) -> None:
        total_tokens = self.token_counter(messages)
        if total_tokens > self.hard_input_tokens:
            raise ContextWindowExceededError(
                "对话上下文在工具裁剪和摘要后仍有 "
                f"{total_tokens} Token，超过模型输入硬上限 "
                f"{self.hard_input_tokens}；请缩小当前输入或对超大工具结果分段处理。"
            )

    @staticmethod
    def _replace_all(messages: Sequence[BaseMessage]) -> dict[str, Any]:
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *messages,
            ]
        }

    def _summarized_result(
        self,
        messages: list[BaseMessage],
        summary: str,
        *,
        tools_changed: bool,
    ) -> dict[str, Any] | None:
        if summary.startswith("Error generating summary:"):
            self._assert_below_hard_limit(messages)
            return self._replace_all(messages) if tools_changed else None

        cutoff = self._recent_turn_cutoff(messages)
        if cutoff <= 0:
            self._assert_below_hard_limit(messages)
            return self._replace_all(messages) if tools_changed else None

        compacted = [
            *self._build_new_messages(summary),
            *messages[cutoff:],
        ]
        self._assert_below_hard_limit(compacted)
        return self._replace_all(compacted)

    def _prepare(
        self,
        state: dict[str, Any],
    ) -> tuple[list[BaseMessage], bool, int] | None:
        messages = list(state.get("messages", []))
        has_summary = self._has_summary(messages)
        trigger = self.safety_trigger_tokens if has_summary else self.cost_trigger_tokens
        if self.token_counter(messages) < trigger:
            return None

        pruned, tools_changed = self._prune_consumed_tools(
            messages,
            safety_mode=has_summary,
        )
        return pruned, tools_changed, self._recent_turn_cutoff(pruned)

    def before_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        prepared = self._prepare(state)
        if prepared is None:
            return None
        messages, tools_changed, cutoff = prepared
        if cutoff <= 0:
            self._assert_below_hard_limit(messages)
            return self._replace_all(messages) if tools_changed else None
        summary = self._create_summary(messages[:cutoff])
        return self._summarized_result(
            messages,
            summary,
            tools_changed=tools_changed,
        )

    async def abefore_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        prepared = self._prepare(state)
        if prepared is None:
            return None
        messages, tools_changed, cutoff = prepared
        if cutoff <= 0:
            self._assert_below_hard_limit(messages)
            return self._replace_all(messages) if tools_changed else None
        summary = await self._acreate_summary(messages[:cutoff])
        return self._summarized_result(
            messages,
            summary,
            tools_changed=tools_changed,
        )
