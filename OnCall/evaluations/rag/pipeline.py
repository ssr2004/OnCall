"""基于真实生产检索链路的 RAG 评测流水线。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from openai import AsyncOpenAI

from app.config import config
from app.services.hybrid_knowledge_service import HybridKnowledgeService, hybrid_knowledge_service
from evaluations.rag.metrics import answer_metrics, retrieval_metrics
from evaluations.rag.ragas_metrics import RagasMetricSuite
from evaluations.rag.schema import RagEvalSample, RagEvaluationRecord

DEFAULT_VARIANTS = ("dense", "bm25", "rrf", "production")


class AnswerGenerator(Protocol):
    async def generate(self, sample: RagEvalSample, contexts: list[dict[str, Any]]) -> str: ...


class GroundedAnswerGenerator:
    """仅依据本次检索 Top3 生成答案，以隔离评测 RAG 本身。"""

    def __init__(self) -> None:
        if not config.dashscope_api_key:
            raise RuntimeError("生成评测答案前必须配置 DASHSCOPE_API_KEY")
        self.client = AsyncOpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )

    @staticmethod
    def _format_contexts(contexts: list[dict[str, Any]]) -> str:
        parts = []
        for index, context in enumerate(contexts, 1):
            parts.append(
                f"[来源{index}]\n"
                f"文件：{context.get('file_name') or context.get('source') or '未知'}\n"
                f"片段ID：{context.get('chunk_id') or 'unknown'}\n"
                f"内容：\n{context.get('content') or ''}"
            )
        return "\n\n".join(parts)

    async def generate(self, sample: RagEvalSample, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "当前知识库没有找到足够资料，无法根据现有资料回答。"
        response = await self.client.chat.completions.create(
            model=config.rag_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是电网运维知识问答评测助手。只能依据给定资料回答，不得补充资料外事实。"
                        "每项知识性结论必须在句末用 [来源N] 标注；末尾列出实际引用文件。"
                        "如果资料不足，应明确说无法根据现有资料回答。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{sample.question}\n\n检索资料：\n{self._format_contexts(contexts)}"
                    ),
                },
            ],
        )
        return str(response.choices[0].message.content or "").strip()


class AgentAnswerGenerator:
    """调用现有对话 Agent，供小规模端到端集成评测使用。"""

    async def generate(self, sample: RagEvalSample, contexts: list[dict[str, Any]]) -> str:
        del contexts
        from app.services.rag_agent_service import rag_agent_service

        return await rag_agent_service.query(
            sample.question,
            session_id=f"rag-eval-{sample.id}-{uuid4().hex}",
        )


@dataclass(slots=True)
class PipelineOptions:
    variants: tuple[str, ...] = DEFAULT_VARIANTS
    final_k: int = 3
    generation_mode: str = "grounded"
    generate_answers: bool = True
    run_ragas: bool = False
    cache_dir: Path = field(default_factory=lambda: Path("evaluations/rag/.cache"))

    def __post_init__(self) -> None:
        unknown = set(self.variants) - set(DEFAULT_VARIANTS)
        if unknown:
            raise ValueError(f"未知消融版本: {', '.join(sorted(unknown))}")
        if self.final_k < 1:
            raise ValueError("final_k 必须大于 0")
        if self.generation_mode not in {"grounded", "agent"}:
            raise ValueError("generation_mode 只能是 grounded 或 agent")
        if self.run_ragas and not self.generate_answers:
            raise ValueError("RAGAS 评测需要先生成答案")


class RagEvaluationPipeline:
    def __init__(
        self,
        *,
        options: PipelineOptions | None = None,
        search_service: HybridKnowledgeService | None = None,
        answer_generator: AnswerGenerator | None = None,
        ragas_suite: RagasMetricSuite | None = None,
    ) -> None:
        self.options = options or PipelineOptions()
        self.search_service = search_service or hybrid_knowledge_service
        if answer_generator is not None:
            self.answer_generator = answer_generator
        elif self.options.generate_answers:
            self.answer_generator = (
                GroundedAnswerGenerator()
                if self.options.generation_mode == "grounded"
                else AgentAnswerGenerator()
            )
        else:
            self.answer_generator = None
        self.ragas_suite = ragas_suite
        if self.options.run_ragas and self.ragas_suite is None:
            self.ragas_suite = RagasMetricSuite(self.options.cache_dir)

    @staticmethod
    def _context_snapshot(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": item.get("chunk_id"),
            "source": item.get("source"),
            "file_name": item.get("file_name"),
            "content": str(item.get("content") or ""),
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "rrf_score": item.get("rrf_score"),
            "rerank_rank": item.get("rerank_rank"),
            "rerank_score": item.get("rerank_score"),
            "rerank_fallback": item.get("rerank_fallback", False),
        }

    async def evaluate_sample(self, sample: RagEvalSample) -> RagEvaluationRecord:
        total_started = perf_counter()
        trace = await self.search_service.search_with_trace(sample.question)
        retrieval: dict[str, dict[str, Any]] = {}
        for variant in self.options.variants:
            items = trace.results_for(variant, self.options.final_k)
            retrieval[variant] = retrieval_metrics(
                items,
                sample.expected_sources,
                k=self.options.final_k,
            )

        production_items = trace.results_for("production", self.options.final_k)
        contexts = [self._context_snapshot(item) for item in production_items]
        timings = dict(trace.timings_ms)
        answer: str | None = None
        deterministic_answer_metrics: dict[str, float | int | None] = {}
        ragas_scores: dict[str, float | None] = {}
        ragas_reasons: dict[str, str | None] = {}
        errors = [trace.error] if trace.error else []

        if self.answer_generator is not None:
            started = perf_counter()
            try:
                answer = await self.answer_generator.generate(sample, contexts)
                deterministic_answer_metrics = answer_metrics(
                    answer,
                    contexts,
                    sample.expected_sources,
                    sample.required_terms,
                    sample.answerable,
                )
            except Exception as exc:
                errors.append(f"答案生成失败: {exc}")
            timings["generation"] = (perf_counter() - started) * 1000

        if self.ragas_suite is not None and answer:
            started = perf_counter()
            try:
                ragas_scores, ragas_reasons = await self.ragas_suite.score(
                    question=sample.question,
                    answer=answer,
                    contexts=[context["content"] for context in contexts],
                    reference_answer=sample.reference_answer,
                )
                failed_metrics = [name for name, value in ragas_scores.items() if value is None]
                if failed_metrics:
                    errors.append(f"RAGAS 指标失败: {', '.join(failed_metrics)}")
            except Exception as exc:
                errors.append(f"RAGAS 评分失败: {exc}")
            timings["ragas"] = (perf_counter() - started) * 1000

        timings["evaluation_total"] = (perf_counter() - total_started) * 1000
        return RagEvaluationRecord(
            sample_id=sample.id,
            question=sample.question,
            reference_answer=sample.reference_answer,
            expected_sources=sample.expected_sources,
            answerable=sample.answerable,
            tags=sample.tags,
            rewritten_query=trace.rewritten.model_dump(),
            retrieval=retrieval,
            production_contexts=contexts,
            answer=answer,
            answer_metrics=deterministic_answer_metrics,
            ragas_metrics=ragas_scores,
            ragas_reasons=ragas_reasons,
            timings_ms=timings,
            error="; ".join(error for error in errors if error) or None,
        )

    async def evaluate(
        self,
        samples: list[RagEvalSample],
        progress: Callable[[int, int, RagEvaluationRecord], Awaitable[None] | None] | None = None,
    ) -> list[RagEvaluationRecord]:
        records = []
        total = len(samples)
        for index, sample in enumerate(samples, 1):
            record = await self.evaluate_sample(sample)
            records.append(record)
            if progress is not None:
                callback_result = progress(index, total, record)
                if callback_result is not None:
                    await callback_result
        return records
