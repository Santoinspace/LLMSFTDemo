# RAG 检索增强生成模块

本目录提供 RAG 全栈组件：文档处理、多检索引擎、检索增强生成 pipeline。

## 模块说明

| 文件 | 功能 | 检索方式 | 硬件 |
|------|------|---------|------|
| `bm25_store.py` | Whoosh BM25 倒排索引，纯 CPU | BM25 关键词匹配 | CPU |
| `vector_store.py` | ChromaDB 向量存储封装 | BGE-M3 密集向量 + Cosine ANN | GPU |
| `embeddings.py` | 使用 BAAI/bge-m3 生成文本嵌入 | — | GPU |
| `document_processor.py` | 文档加载、分块处理 | — | CPU |
| `rag_pipeline.py` | 完整 RAG 流程（检索 + 生成），支持多检索引擎 | — | GPU |
| `ingest_swebench.py` | SWE-bench_bm25_27K 知识库导入（支持 bm25/chromadb） | — | — |

## 两种检索器

| 检索器 | 适用场景 | 索引速度 | 检索速度 | 语义理解 |
|--------|---------|---------|---------|---------|
| BM25 (`bm25_store.py`) | 代码搜索、关键词匹配、已知数据集 | 毫秒级 | 毫秒级 | 精确匹配 |
| ChromaDB (`vector_store.py`) | 通用文档 QA、概念检索、跨语言 | 分钟级 (GPU) | 秒级 (GPU) | 语义近似 |

两个检索器具有相同 Python 接口（`add_documents` / `similarity_search` / `count` / `delete_collection`），`rag_pipeline.py` 通过 `retriever=` 参数接受任意检索器。

## 快速使用

### 通用文档 QA (ChromaDB + BGE-M3)

```bash
# 1. 处理文档
python rag/document_processor.py --input docs/ --output data/processed_docs.jsonl

# 2. 构建向量存储
python rag/vector_store.py --action add --input data/processed_docs.jsonl --persist_dir chroma_db

# 3. RAG 推理
python rag/rag_pipeline.py \
    --model_path outputs/outputs_codealpacas/merged_model \
    --query "你的问题" \
    --mode compare
```

### SWE-bench 代码知识库 (Whoosh BM25)

```bash
# 1. 导入知识库（纯 CPU，秒级）
python rag/ingest_swebench.py --shuffle --retriever bm25

# 2. BM25 检索
python rag/bm25_store.py \
    --persist_dir ./bm25_index_swebench \
    --collection swebench_instances \
    --action search \
    --query "database connection pooling" --k 5

# 3. RAG 推理（自动使用 BM25）
python rag/rag_pipeline.py \
    --retriever bm25 \
    --persist_dir ./bm25_index_swebench \
    --collection swebench_instances \
    --query "How to handle database transactions?" \
    --mode compare
```

## 后续规划

- [ ] 集成 API 服务的 `use_rag` 参数
- [ ] 支持混合检索（BM25 初筛 + 向量重排）
- [ ] 支持重排序 (reranker)
- [ ] 添加检索结果缓存
