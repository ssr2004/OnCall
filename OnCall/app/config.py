"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen3.7-plus-2026-05-26"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # 生成式模型路由。线上任务使用固定快照，避免别名升级导致演示和评测漂移。
    chat_model: str = "qwen3.7-plus-2026-05-26"
    chat_summary_model: str = "qwen3.7-flash-2026-07-15"
    aiops_planner_model: str = "qwen3.7-plus-2026-05-26"
    aiops_executor_model: str = "qwen3.7-plus-2026-05-26"
    aiops_replanner_model: str = "qwen3.7-plus-2026-05-26"
    aiops_report_model: str = "qwen3.7-max-2026-06-08"

    chat_model_max_output_tokens: int = 4096
    chat_summary_max_output_tokens: int = 2048
    aiops_planner_max_output_tokens: int = 2048
    aiops_executor_max_output_tokens: int = 2048
    aiops_replanner_max_output_tokens: int = 1024
    aiops_report_max_output_tokens: int = 4096

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # 故障情景记忆检索配置。Dense 与 BM25 分别召回候选，再由应用层 RRF 融合。
    incident_dense_recall_k: int = 20
    incident_bm25_recall_k: int = 20
    incident_rrf_final_k: int = 3
    incident_rrf_rank_constant: int = 60
    incident_pending_ttl_seconds: int = 1800
    incident_pending_max_items: int = 100

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen3.7-plus-2026-05-26"  # 兼容离线评测和旧调用入口
    rag_eval_judge_model: str = "qwen3.7-max-2026-06-08"
    rag_query_rewrite_model: str = "qwen3.7-flash-2026-07-15"
    rag_query_rewrite_max_output_tokens: int = 512
    rag_dense_recall_k: int = 20
    rag_bm25_recall_k: int = 20
    rag_rrf_candidate_k: int = 10
    rag_rerank_final_k: int = 3
    rag_rrf_rank_constant: int = 60
    rag_rerank_model: str = "gte-rerank-v2"
    rag_rerank_enabled: bool = True

    # 对话 Agent 上下文管理。会话原文保存在当前操作系统用户的本地 JSONL 中；
    # 留空时默认使用 %LOCALAPPDATA%/OnCall/sessions（非 Windows 使用用户数据目录）。
    chat_session_data_dir: str = ""
    # qwen3.7-plus 固定快照的百炼模型规格。
    chat_model_context_window_tokens: int = 1_000_000
    chat_model_max_input_tokens: int = 991_808
    # 256K 是该模型当前最低输入价格档的上界，用作成本与延迟软上限。
    chat_context_operating_input_tokens: int = 256_000
    # 以下固定阈值仅为旧环境变量兼容项；运行时使用模型规格动态计算。
    chat_tool_prune_trigger_tokens: int = 18000
    chat_context_compaction_trigger_tokens: int = 24000
    chat_context_keep_messages: int = 16
    chat_tool_output_max_chars: int = 1200

    # 已确认故障情景记忆与知识库使用相同的召回/精排漏斗。
    incident_rrf_candidate_k: int = 10

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置（transport: stdio | sse | streamable-http）
    # 腾讯云托管 MCP 的 URL 通常含 /sse/，需使用 sse；本地 FastMCP 使用 streamable-http
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # Prometheus
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    @property
    def mcp_servers(self) -> dict[str, dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
        }


# 全局配置实例
config = Settings()
