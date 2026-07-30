"""RAG 人工基准集与评测结果的数据结构。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RagEvalSample(BaseModel):
    """一条可人工核验的 RAG 基准样本。"""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    expected_sources: list[str] = Field(default_factory=list)
    required_terms: list[list[str]] = Field(default_factory=list)
    answerable: bool = True
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("expected_sources")
    @classmethod
    def normalize_sources(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(Path(item.replace("\\", "/")).name for item in value if item))

    @field_validator("required_terms")
    @classmethod
    def validate_term_groups(cls, value: list[list[str]]) -> list[list[str]]:
        groups = [
            list(dict.fromkeys(term.strip() for term in group if term.strip())) for group in value
        ]
        if any(not group for group in groups):
            raise ValueError("required_terms 中的每个同义词组都必须至少包含一个词")
        return groups


class RagEvaluationRecord(BaseModel):
    """一条样本的完整检索、生成与评分记录。"""

    sample_id: str
    question: str
    reference_answer: str
    expected_sources: list[str]
    answerable: bool = True
    tags: list[str] = Field(default_factory=list)
    rewritten_query: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, dict[str, Any]] = Field(default_factory=dict)
    production_contexts: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    answer_metrics: dict[str, float | int | None] = Field(default_factory=dict)
    ragas_metrics: dict[str, float | None] = Field(default_factory=dict)
    ragas_reasons: dict[str, str | None] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


def load_dataset(path: str | Path) -> list[RagEvalSample]:
    """读取 JSONL 基准集，并拒绝重复样本 ID。"""
    dataset_path = Path(path)
    samples: list[RagEvalSample] = []
    seen: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = RagEvalSample.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"评测集第 {line_number} 行无效: {exc}") from exc
            if sample.id in seen:
                raise ValueError(f"评测集存在重复 ID: {sample.id}")
            seen.add(sample.id)
            samples.append(sample)
    if not samples:
        raise ValueError(f"评测集为空: {dataset_path}")
    return samples
