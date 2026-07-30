"""知识检索工具 - 查询改写、混合召回、RRF 与模型精排。"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.services.hybrid_knowledge_service import hybrid_knowledge_service


@tool(response_format="content_and_artifact")
async def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库检索资料；回答必须使用工具返回的 [来源N] 标注依据。
    
    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    
    Args:
        query: 用户的问题或查询
        
    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")
        
        rewritten, items = await hybrid_knowledge_service.search(query)
        docs = hybrid_knowledge_service.to_documents(items)
        
        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []
        
        # 格式化文档为上下文
        context = format_docs(docs)
        context = (
            "以下资料经过查询改写、Dense/BM25 混合召回、RRF 融合和百炼精排。\n"
            f"语义检索问题：{rewritten.semantic_query}\n"
            f"BM25关键词：{', '.join(rewritten.keywords)}\n\n"
            f"{context}\n\n"
            "引用要求：只能依据上述资料陈述知识性结论，并在相关句末使用"
            "[来源1]、[来源2]格式标注；回答末尾列出实际引用的来源。"
        )
        
        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs
        
    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        source_id = metadata.get("source_id", f"来源{i}")
        formatted = f"[{source_id}]"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n文件: {source}"
        formatted += f"\n片段ID: {metadata.get('chunk_id', 'unknown')}"
        rerank_score = metadata.get("rerank_score")
        if rerank_score is not None:
            formatted += f"\n精排分数: {float(rerank_score):.6f}"
        elif metadata.get("rerank_fallback"):
            formatted += "\n精排状态: 百炼不可用，使用RRF降级排序"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)
