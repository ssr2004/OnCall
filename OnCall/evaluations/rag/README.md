# 电网运维 RAG 评测

本目录直接采集生产检索链路的一次 Trace：查询改写、Dense Top20、BM25 Top20、RRF Top10、百炼精排 Top3。四种消融版本复用同一次改写和召回，避免重复调用模型与 Embedding。

## 指标

- 检索：Source Hit@3、Precision@3、Recall@3、MRR、nDCG@3。
- 答案：必要事实覆盖率、引用编号有效率、引用来源精确率/召回率、无答案拒答正确率。
- RAGAS 0.4.3：Context Precision、Context Recall、Faithfulness、Answer Relevancy、Factual Correctness。
- 性能：问题改写、Embedding、Dense、BM25、RRF、Rerank、生成与 RAGAS 评审耗时。

RAGAS 使用项目已有的百炼 OpenAI 兼容接口。评审模型固定为 `qwen3.7-max-2026-06-08`，避免模型别名升级造成基线漂移，并关闭与强制 Tool Choice 冲突的思考模式。Context Precision 使用适合其逐片段对象结构的 JSON 模式，其余四项嵌套指标使用 Tool Calling。Embedding 仍使用 RAGAS 的现代 OpenAI 工厂。重复评测结果缓存到 `evaluations/rag/.cache/`，缓存和结果目录不会提交到 Git。任一 RAGAS 单项失败都会在样本状态中明确标为异常；断点续跑会自动重试异常样本。

## 运行

先确认 Milvus 已启动且 `biz_hybrid` 已持久化当前处置手册。

只运行全部 42 条检索消融（不生成答案、不调用 RAGAS；仍会调用生产链路中的查询改写和百炼精排）：

```powershell
uv run --extra eval python -m evaluations.rag.run --skip-generation
```

生成 grounded 答案，但不调用 RAGAS 评审：

```powershell
uv run --extra eval python -m evaluations.rag.run
```

先用 2 条样本做包含 RAGAS 的在线冒烟：

```powershell
uv run --extra eval python -m evaluations.rag.run --ragas --limit 2
```

确认费用和接口正常后再运行完整 production RAGAS：

```powershell
uv run --extra eval python -m evaluations.rag.run --ragas
```

42 条都会执行并在报告中分为“38 条可回答”和“4 条无答案”两组。在线评测每完成一条都会刷新结果文件；如果进程中断，可使用同一个输出目录继续：

```powershell
uv run --extra eval python -m evaluations.rag.run --ragas --output-dir evaluations/rag/results/ragas-full-v1 --resume
```

端到端 Agent 模式用于小规模集成检查；由于 Agent 会自行决定是否再次调用工具，它不适合作为四种检索消融的默认生成方式：

```powershell
uv run --extra eval python -m evaluations.rag.run --mode agent --limit 3
```

每次运行输出 `rag_evaluation.json`、`rag_evaluation.csv` 和 `rag_evaluation.md`。JSON 保留逐样本 Trace，CSV 便于做统计，Markdown 用于人工评审和项目汇报。
