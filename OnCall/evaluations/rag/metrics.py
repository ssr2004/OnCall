"""无需模型调用的检索、事实覆盖与引用指标。"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

_CITATION_PATTERN = re.compile(r"\[来源\s*(\d+)\]")
_REFUSAL_PATTERNS = (
    "没有找到",
    "未找到",
    "无法从",
    "无法根据",
    "资料不足",
    "信息不足",
    "无法确定",
    "未提供",
    "没有相关",
    "超出当前知识库",
)


def normalize_source(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value).replace("\\", "/")).name.casefold()


def item_source(item: dict[str, Any]) -> str:
    return normalize_source(str(item.get("file_name") or item.get("source") or ""))


def retrieval_metrics(
    items: list[dict[str, Any]],
    expected_sources: list[str],
    *,
    k: int = 3,
) -> dict[str, Any]:
    """按来源文件计算 Hit、Precision、Recall、MRR 与 nDCG。"""
    top_items = items[:k]
    retrieved = [item_source(item) for item in top_items]
    expected = {normalize_source(source) for source in expected_sources if source}
    result: dict[str, Any] = {
        "k": k,
        "retrieved_sources": retrieved,
        "retrieved_chunk_ids": [str(item.get("chunk_id") or "") for item in top_items],
        "result_count": len(top_items),
    }
    if not expected:
        result.update(
            {
                "source_hit_at_k": None,
                "source_precision_at_k": None,
                "source_recall_at_k": None,
                "mrr": None,
                "ndcg_at_k": None,
            }
        )
        return result

    relevance = [1 if source in expected else 0 for source in retrieved]
    relevant_sources = {source for source in retrieved if source in expected}
    first_rank = next((rank for rank, rel in enumerate(relevance, 1) if rel), None)
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, 1))
    ideal_relevant = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    result.update(
        {
            "source_hit_at_k": float(bool(relevant_sources)),
            "source_precision_at_k": sum(relevance) / k,
            "source_recall_at_k": len(relevant_sources) / len(expected),
            "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
            "ndcg_at_k": 0.0 if idcg == 0 else dcg / idcg,
        }
    )
    return result


def required_fact_coverage(answer: str, required_terms: list[list[str]]) -> float | None:
    """每个词组命中任意一个同义表达即视为该必要事实被覆盖。"""
    if not required_terms:
        return None
    normalized = answer.casefold()
    hits = sum(
        any(term.casefold() in normalized for term in alternatives)
        for alternatives in required_terms
    )
    return hits / len(required_terms)


def is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.split()).casefold()
    return any(pattern.casefold() in normalized for pattern in _REFUSAL_PATTERNS)


def answer_metrics(
    answer: str,
    contexts: list[dict[str, Any]],
    expected_sources: list[str],
    required_terms: list[list[str]],
    answerable: bool,
) -> dict[str, float | int | None]:
    """计算必要事实、引用合法性与无答案拒答指标。"""
    citation_numbers = [int(value) for value in _CITATION_PATTERN.findall(answer)]
    unique_numbers = list(dict.fromkeys(citation_numbers))
    valid_numbers = [number for number in unique_numbers if 1 <= number <= len(contexts)]
    cited_sources = {
        item_source(contexts[number - 1])
        for number in valid_numbers
        if item_source(contexts[number - 1])
    }
    expected = {normalize_source(source) for source in expected_sources if source}
    cited_relevant = cited_sources & expected
    citation_validity = (
        len(valid_numbers) / len(unique_numbers) if unique_numbers else (0.0 if answerable else 1.0)
    )
    citation_precision = (
        len(cited_relevant) / len(cited_sources) if cited_sources else (0.0 if expected else None)
    )
    citation_recall = len(cited_relevant) / len(expected) if expected else None
    return {
        "required_fact_coverage": required_fact_coverage(answer, required_terms),
        "citation_count": len(unique_numbers),
        "valid_citation_count": len(valid_numbers),
        "citation_validity": citation_validity,
        "citation_source_precision": citation_precision,
        "citation_source_recall": citation_recall,
        "unanswerable_refusal_accuracy": float(is_refusal(answer)) if not answerable else None,
    }
