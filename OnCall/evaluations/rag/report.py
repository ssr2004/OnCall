"""RAG 评测结果的 JSON、CSV 与 Markdown 报告。"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any

from app.config import config
from evaluations.rag.schema import RagEvaluationRecord


def _mean(values: Iterable[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return mean(valid) if valid else None


def _aggregate_group(records: list[RagEvaluationRecord]) -> dict[str, Any]:
    variants = sorted({variant for record in records for variant in record.retrieval})
    retrieval_summary: dict[str, dict[str, float | None]] = {}
    retrieval_names = (
        "source_hit_at_k",
        "source_precision_at_k",
        "source_recall_at_k",
        "mrr",
        "ndcg_at_k",
    )
    for variant in variants:
        retrieval_summary[variant] = {
            name: _mean(record.retrieval.get(variant, {}).get(name) for record in records)
            for name in retrieval_names
        }

    answer_names = sorted({name for record in records for name in record.answer_metrics})
    ragas_names = sorted({name for record in records for name in record.ragas_metrics})
    timing_names = sorted({name for record in records for name in record.timings_ms})
    return {
        "sample_count": len(records),
        "success_count": sum(record.error is None for record in records),
        "error_count": sum(record.error is not None for record in records),
        "retrieval": retrieval_summary,
        "answer": {
            name: _mean(record.answer_metrics.get(name) for record in records)
            for name in answer_names
        },
        "ragas": {
            name: _mean(record.ragas_metrics.get(name) for record in records)
            for name in ragas_names
        },
        "timings_ms": {
            name: _mean(record.timings_ms.get(name) for record in records) for name in timing_names
        },
    }


def aggregate_records(records: list[RagEvaluationRecord]) -> dict[str, Any]:
    """汇总总体结果，并分别保留可回答与无答案样本口径。"""
    summary = _aggregate_group(records)
    summary["groups"] = {
        "answerable": _aggregate_group([record for record in records if record.answerable]),
        "unanswerable": _aggregate_group([record for record in records if not record.answerable]),
    }
    return summary


def _format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def evaluation_configuration() -> dict[str, Any]:
    return {
        "answer_model": config.rag_model,
        "judge_model": config.rag_eval_judge_model,
        "embedding_model": config.dashscope_embedding_model,
        "rerank_model": config.rag_rerank_model,
        "ragas_version": version("ragas"),
    }


def render_markdown(
    records: list[RagEvaluationRecord],
    summary: dict[str, Any],
    configuration: dict[str, Any] | None = None,
) -> str:
    configuration = configuration or evaluation_configuration()
    lines = [
        "# RAG 评测报告",
        "",
        f"- 样本数：{summary['sample_count']}",
        f"- 成功：{summary['success_count']}",
        f"- 异常：{summary['error_count']}",
        f"- 回答模型：{configuration['answer_model']}",
        f"- RAGAS 评审模型：{configuration['judge_model']}",
        f"- RAGAS 版本：{configuration['ragas_version']}",
        "",
        "## 检索消融",
        "",
        "| 版本 | Hit@3 | Precision@3 | Recall@3 | MRR | nDCG@3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in summary["retrieval"].items():
        lines.append(
            "| {variant} | {hit} | {precision} | {recall} | {mrr} | {ndcg} |".format(
                variant=variant,
                hit=_format_score(metrics.get("source_hit_at_k")),
                precision=_format_score(metrics.get("source_precision_at_k")),
                recall=_format_score(metrics.get("source_recall_at_k")),
                mrr=_format_score(metrics.get("mrr")),
                ndcg=_format_score(metrics.get("ndcg_at_k")),
            )
        )

    if summary["answer"]:
        lines.extend(["", "## 答案与引用", "", "| 指标 | 均值 |", "|---|---:|"])
        lines.extend(
            f"| {name} | {_format_score(value)} |" for name, value in summary["answer"].items()
        )
    if summary["ragas"]:
        lines.extend(["", "## RAGAS", "", "| 指标 | 均值 |", "|---|---:|"])
        lines.extend(
            f"| {name} | {_format_score(value)} |" for name, value in summary["ragas"].items()
        )
        ragas_names = (
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "factual_correctness",
        )
        lines.extend(
            [
                "",
                "### 分组结果",
                "",
                "| 分组 | 样本数 | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Factual Correctness |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        group_labels = {"answerable": "可回答", "unanswerable": "无答案"}
        for group_name, group_summary in summary["groups"].items():
            values = group_summary["ragas"]
            lines.append(
                "| {label} | {count} | {scores} |".format(
                    label=group_labels[group_name],
                    count=group_summary["sample_count"],
                    scores=" | ".join(_format_score(values.get(name)) for name in ragas_names),
                )
            )

    lines.extend(
        [
            "",
            "## 样本明细",
            "",
            "| 样本 | Production Hit@3 | 必要事实覆盖 | 引用有效率 | 状态 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for record in records:
        production = record.retrieval.get("production", {})
        lines.append(
            "| {sample} | {hit} | {facts} | {citations} | {status} |".format(
                sample=record.sample_id,
                hit=_format_score(production.get("source_hit_at_k")),
                facts=_format_score(record.answer_metrics.get("required_fact_coverage")),
                citations=_format_score(record.answer_metrics.get("citation_validity")),
                status=record.error or "OK",
            )
        )
    return "\n".join(lines) + "\n"


def write_reports(
    records: list[RagEvaluationRecord],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate_records(records)
    configuration = evaluation_configuration()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "summary": summary,
        "records": [record.model_dump(mode="json") for record in records],
    }

    json_path = output / "rag_evaluation.json"
    csv_path = output / "rag_evaluation.csv"
    markdown_path = output / "rag_evaluation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for record in records:
        variants = record.retrieval or {"production": {}}
        for variant, metrics in variants.items():
            row: dict[str, Any] = {
                "sample_id": record.sample_id,
                "answerable": record.answerable,
                "variant": variant,
                "question": record.question,
                "expected_sources": "|".join(record.expected_sources),
                "retrieved_sources": "|".join(metrics.get("retrieved_sources", [])),
                "error": record.error or "",
            }
            row.update(
                {name: value for name, value in metrics.items() if not isinstance(value, list)}
            )
            row.update({f"answer_{name}": value for name, value in record.answer_metrics.items()})
            row.update({f"ragas_{name}": value for name, value in record.ragas_metrics.items()})
            rows.append(row)
    fieldnames = sorted({name for row in rows for name in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path.write_text(
        render_markdown(records, summary, configuration),
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def load_report_records(path: str | Path) -> list[RagEvaluationRecord]:
    """读取检查点报告，供长时间在线评测断点续跑。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RagEvaluationRecord.model_validate(item) for item in payload.get("records", [])]
