"""RAG 评测命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from evaluations.rag.pipeline import DEFAULT_VARIANTS, PipelineOptions, RagEvaluationPipeline
from evaluations.rag.report import aggregate_records, load_report_records, write_reports
from evaluations.rag.schema import load_dataset

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PACKAGE_DIR / "datasets" / "grid_rag_v1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行电网运维知识库 RAG 评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="只评测前 N 条，用于在线冒烟")
    parser.add_argument("--tag", help="只评测包含指定 tag 的样本")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--mode", choices=("grounded", "agent"), default="grounded")
    parser.add_argument("--skip-generation", action="store_true", help="只跑检索和消融指标")
    parser.add_argument(
        "--ragas", action="store_true", help="对 production 答案运行五项 RAGAS 指标"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 output-dir 中的 JSON 检查点继续，跳过已完成样本",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须大于 0")
    if args.ragas and args.skip_generation:
        raise ValueError("--ragas 不能与 --skip-generation 同时使用")

    samples = load_dataset(args.dataset)
    if args.tag:
        samples = [sample for sample in samples if args.tag in sample.tags]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("筛选后没有可评测样本")

    output_dir = args.output_dir or (
        PACKAGE_DIR / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    options = PipelineOptions(
        variants=tuple(args.variants),
        generation_mode=args.mode,
        generate_answers=not args.skip_generation,
        run_ragas=args.ragas,
        cache_dir=PACKAGE_DIR / ".cache",
    )
    pipeline = RagEvaluationPipeline(options=options)

    checkpoint_path = output_dir / "rag_evaluation.json"
    existing_records = (
        load_report_records(checkpoint_path) if args.resume and checkpoint_path.exists() else []
    )
    selected_ids = {sample.id for sample in samples}
    existing_records = [
        record
        for record in existing_records
        if record.sample_id in selected_ids and record.error is None
    ]
    completed_ids = {record.sample_id for record in existing_records}
    pending_samples = [sample for sample in samples if sample.id not in completed_ids]
    checkpoint_records = list(existing_records)

    def progress(index, total, record):
        state = "ERROR" if record.error else "OK"
        checkpoint_records.append(record)
        write_reports(checkpoint_records, output_dir)
        completed = len(existing_records) + index
        overall_total = len(existing_records) + total
        print(f"[{completed}/{overall_total}] {record.sample_id}: {state}", flush=True)

    new_records = await pipeline.evaluate(pending_samples, progress=progress)
    records_by_id = {record.sample_id: record for record in [*existing_records, *new_records]}
    records = [records_by_id[sample.id] for sample in samples if sample.id in records_by_id]
    paths = write_reports(records, output_dir)
    summary = aggregate_records(records)
    print(f"评测完成：{summary['success_count']}/{summary['sample_count']} 条无运行错误")
    for kind, path in paths.items():
        print(f"{kind}: {path.resolve()}")
    return 0 if summary["error_count"] == 0 else 1


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
