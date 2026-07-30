"""RAGAS 0.4.3 与百炼 OpenAI 兼容接口的适配器。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import instructor
from openai import AsyncOpenAI

from app.config import config


class RagasMetricSuite:
    """复用模型客户端、Embedding 与磁盘缓存执行五项 RAGAS 指标。"""

    def __init__(self, cache_dir: str | Path) -> None:
        if not config.dashscope_api_key:
            raise RuntimeError("运行 RAGAS 在线评测前必须配置 DASHSCOPE_API_KEY")

        from ragas.cache import DiskCacheBackend
        from ragas.embeddings.base import embedding_factory
        from ragas.llms.adapters.instructor import InstructorLLM
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        cache = DiskCacheBackend(str(cache_dir))
        client = AsyncOpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )
        # 百炼的 OpenAI 兼容端点在 Instructor JSON 模式下会偶发生成残缺对象；
        # 工具调用模式能让 qwen-max 稳定遵循 RAGAS 的嵌套 Pydantic Schema。
        tool_client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
        tool_judge_llm = InstructorLLM(
            client=tool_client,
            model=config.rag_eval_judge_model,
            provider="openai",
            cache=cache,
            temperature=0,
            max_tokens=4096,
            extra_body={"enable_thinking": False},
        )
        json_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=config.dashscope_api_key,
                base_url=config.dashscope_api_base,
            ),
            mode=instructor.Mode.JSON,
        )
        json_judge_llm = InstructorLLM(
            client=json_client,
            model=config.rag_eval_judge_model,
            provider="openai",
            cache=cache,
            temperature=0,
            max_tokens=4096,
            extra_body={"enable_thinking": False},
        )
        embeddings = embedding_factory(
            "openai",
            model=config.dashscope_embedding_model,
            client=client,
            interface="modern",
            cache=cache,
        )
        self.metrics = {
            # ContextPrecision 的逐片段 Schema 在百炼 Tool Calling 下偶发把 verdict
            # 拼入 reason；JSON 模式对此结构更稳定，其余嵌套列表指标继续用 Tools。
            "context_precision": ContextPrecision(llm=json_judge_llm),
            "context_recall": ContextRecall(llm=tool_judge_llm),
            "faithfulness": Faithfulness(llm=tool_judge_llm),
            "answer_relevancy": AnswerRelevancy(llm=tool_judge_llm, embeddings=embeddings),
            "factual_correctness": FactualCorrectness(llm=tool_judge_llm),
        }

    async def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
    ) -> tuple[dict[str, float | None], dict[str, str | None]]:
        calls = {
            "context_precision": self.metrics["context_precision"].ascore(
                user_input=question,
                reference=reference_answer,
                retrieved_contexts=contexts,
            ),
            "context_recall": self.metrics["context_recall"].ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=reference_answer,
            ),
            "faithfulness": self.metrics["faithfulness"].ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            ),
            "answer_relevancy": self.metrics["answer_relevancy"].ascore(
                user_input=question,
                response=answer,
            ),
            "factual_correctness": self.metrics["factual_correctness"].ascore(
                response=answer,
                reference=reference_answer,
            ),
        }
        names = list(calls)
        raw_results = await asyncio.gather(*(calls[name] for name in names), return_exceptions=True)
        scores: dict[str, float | None] = {}
        reasons: dict[str, str | None] = {}
        for name, result in zip(names, raw_results, strict=True):
            if isinstance(result, Exception):
                scores[name] = None
                detail = " ".join(str(result).split())[:1000]
                reasons[name] = f"ERROR: {type(result).__name__}: {detail}"
                continue
            value: Any = getattr(result, "value", result)
            try:
                scores[name] = float(value)
            except (TypeError, ValueError):
                scores[name] = None
            reasons[name] = getattr(result, "reason", None)
        return scores, reasons
