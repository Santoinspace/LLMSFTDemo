# RAG 集成模块（脚手架）

本目录提供检索增强生成 (RAG) 的基础组件，为后续将 RAG 集成到微调后模型做准备。

## 模块说明

| 文件 | 功能 | 状态 |
|------|------|------|
| `embeddings.py` | 使用 BAAI/bge-m3 生成文本嵌入 | 就绪 |
| `vector_store.py` | ChromaDB 向量存储封装 | 就绪 |
| `document_processor.py` | 文档加载、分块处理 | 就绪 |
| `rag_pipeline.py` | 完整 RAG 流程（检索 + 生成） | 就绪 |

## 核心参数

- Embedding 模型: `BAAI/bge-m3`（多语言，1024 维）
- 文档分块: `chunk_size=512, chunk_overlap=64`
- 向量数据库: ChromaDB（本地持久化）
- 默认检索数量: top_k=5

## 快速使用

```bash
# 1. 生成文档嵌入
python rag/embeddings.py --input docs/ --output_dir rag_db

# 2. 构建向量存储
python rag/vector_store.py --add_dir rag_db/ --persist_dir chroma_db

# 3. 运行 RAG 推理
python rag/rag_pipeline.py \
    --model_path outputs/merged_model \
    --query "你的问题" \
    --persist_dir chroma_db
```

## 集成到推理服务

在 `inference/api_server.py` 中，设置 `use_rag=True` 即可启用 RAG：

```python
# POST /generate 请求
{
    "messages": [...],
    "use_rag": true  # 启用 RAG
}
```

## 后续规划

- [ ] 集成 API 服务的 `use_rag` 参数
- [ ] 支持混合检索（关键词 + 向量）
- [ ] 支持重排序 (reranker)
- [ ] 添加检索结果缓存
