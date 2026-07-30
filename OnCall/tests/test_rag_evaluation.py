"""RAG 评测集、确定性指标、流水线和报告测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.hybrid_knowledge_service import HybridSearchTrace
from app.services.rag_query_rewrite_service import RewrittenQuery
from evaluations.rag.metrics import answer_metrics, retrieval_metrics
from evaluations.rag.pipeline import PipelineOptions, RagEvaluationPipeline
from evaluations.rag.report import aggregate_records, load_report_records, write_reports
from evaluations.rag.schema import RagEvalSample, load_dataset


def _item(chunk_id: str, file_name: str, content: str = "处置内容"):
    return {
        "chunk_id": chunk_id,
        "source": f"D:/aiops-docs/{file_name}",
        "file_name": file_name,
        "content": content,
    }


def test_grid_dataset_has_at_least_40_unique_human_verifiable_samples():
    path = Path(__file__).parents[1] / "evaluations" / "rag" / "datasets" / "grid_rag_v1.jsonl"
    samples = load_dataset(path)

    assert len(samples) >= 40
    assert len({sample.id for sample in samples}) == len(samples)
    assert any(not sample.answerable for sample in samples)
    assert any(len(sample.expected_sources) > 1 for sample in samples)
    assert {"direct", "paraphrase", "multi_document", "unanswerable", "citation"} <= {
        tag for sample in samples for tag in sample.tags
    }


def test_retrieval_metrics_use_rank_and_unique_expected_sources():
    metrics = retrieval_metrics(
        [
            _item("1", "wrong.md"),
            _item("2", "grid_sync_failure.md"),
            _item("3", "grid_sync_failure.md"),
        ],
        ["grid_sync_failure.md", "grid_queue_backlog.md"],
        k=3,
    )

    assert metrics["source_hit_at_k"] == 1.0
    assert metrics["source_precision_at_k"] == pytest.approx(2 / 3)
    assert metrics["source_recall_at_k"] == 0.5
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg_at_k"] < 1


def test_answer_metrics_validate_citations_facts_and_refusal():
    contexts = [
        _item("1", "grid_sync_failure.md"),
        _item("2", "wrong.md"),
    ]
    metrics = answer_metrics(
        "失败率应低于 20%，队列不再增长。[来源1][来源9]",
        contexts,
        ["grid_sync_failure.md"],
        [["20%"], ["队列不再增长"]],
        True,
    )
    refusal = answer_metrics(
        "当前知识库没有提供该电压阈值，无法确定。",
        [],
        [],
        [["无法确定"]],
        False,
    )

    assert metrics["required_fact_coverage"] == 1.0
    assert metrics["citation_validity"] == 0.5
    assert metrics["citation_source_precision"] == 1.0
    assert refusal["unanswerable_refusal_accuracy"] == 1.0


class FakeSearchService:
    async def search_with_trace(self, query):
        dense = [_item("d", "wrong.md")]
        bm25 = [_item("b", "grid_sync_failure.md")]
        rrf = [_item("r", "grid_sync_failure.md")]
        production = [
            dict(
                _item("p", "grid_sync_failure.md", "失败率连续15秒超过20%，恢复后队列不再增长。"),
                rerank_rank=1,
                rerank_score=0.9,
            )
        ]
        return HybridSearchTrace(
            query=query,
            rewritten=RewrittenQuery(semantic_query=query, keywords=["同步失败"]),
            dense_results=dense,
            bm25_results=bm25,
            rrf_results=rrf,
            reranked_results=production,
            timings_ms={"rewrite": 1.0, "total": 5.0},
        )


class FakeGenerator:
    async def generate(self, sample, contexts):
        assert contexts[0]["file_name"] == "grid_sync_failure.md"
        return "失败率连续15秒超过20%，恢复后队列不再增长。[来源1]"


class FakeRagasSuite:
    async def score(self, **kwargs):
        assert kwargs["contexts"]
        return {"faithfulness": 0.9}, {"faithfulness": "有上下文支持"}


class PartiallyFailingRagasSuite:
    async def score(self, **_kwargs):
        return {"faithfulness": None}, {"faithfulness": "ERROR: invalid schema"}


@pytest.mark.asyncio
async def test_pipeline_compares_variants_and_scores_production_answer():
    sample = RagEvalSample(
        id="sample-1",
        question="同步失败阈值？",
        reference_answer="连续15秒超过20%。",
        expected_sources=["grid_sync_failure.md"],
        required_terms=[["15秒"], ["20%"], ["队列不再增长"]],
        tags=["sync_failure"],
    )
    pipeline = RagEvaluationPipeline(
        options=PipelineOptions(run_ragas=True),
        search_service=FakeSearchService(),
        answer_generator=FakeGenerator(),
        ragas_suite=FakeRagasSuite(),
    )

    record = await pipeline.evaluate_sample(sample)

    assert record.retrieval["dense"]["source_hit_at_k"] == 0.0
    assert record.retrieval["production"]["source_hit_at_k"] == 1.0
    assert record.answer_metrics["required_fact_coverage"] == 1.0
    assert record.ragas_metrics["faithfulness"] == 0.9
    assert record.error is None


@pytest.mark.asyncio
async def test_pipeline_marks_partial_ragas_failure_as_record_error():
    sample = RagEvalSample(
        id="sample-ragas-error",
        question="同步失败阈值？",
        reference_answer="连续15秒超过20%。",
        expected_sources=["grid_sync_failure.md"],
    )
    pipeline = RagEvaluationPipeline(
        options=PipelineOptions(run_ragas=True),
        search_service=FakeSearchService(),
        answer_generator=FakeGenerator(),
        ragas_suite=PartiallyFailingRagasSuite(),
    )

    record = await pipeline.evaluate_sample(sample)

    assert record.ragas_metrics["faithfulness"] is None
    assert record.error == "RAGAS 指标失败: faithfulness"


@pytest.mark.asyncio
async def test_pipeline_and_report_support_retrieval_only(tmp_path):
    sample = RagEvalSample(
        id="sample-2",
        question="同步失败阈值？",
        reference_answer="连续15秒超过20%。",
        expected_sources=["grid_sync_failure.md"],
    )
    pipeline = RagEvaluationPipeline(
        options=PipelineOptions(generate_answers=False),
        search_service=FakeSearchService(),
    )
    records = await pipeline.evaluate([sample])
    summary = aggregate_records(records)
    paths = write_reports(records, tmp_path)

    assert summary["sample_count"] == 1
    assert summary["groups"]["answerable"]["sample_count"] == 1
    assert summary["groups"]["unanswerable"]["sample_count"] == 0
    assert summary["retrieval"]["production"]["source_hit_at_k"] == 1.0
    assert set(paths) == {"json", "csv", "markdown"}
    assert all(path.exists() for path in paths.values())
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["records"][0]["sample_id"] == "sample-2"
    assert load_report_records(paths["json"])[0].answerable is True
    assert "检索消融" in paths["markdown"].read_text(encoding="utf-8")
