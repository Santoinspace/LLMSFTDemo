# 实验日志

## 2026-06-11: RAG 检索器对比 — BM25 vs BGE-M3 在 SWE-bench 代码搜索场景

### 背景

当前 RAG 模块默认使用 ChromaDB + BAAI/bge-m3 (568M params, 1024-dim) 做向量检索。引入 SWE-bench_bm25_27K (18.8K 条 GitHub issue + 代码文件 + patch) 作为代码知识库时，面临两个选型问题：

1. **索引策略**：A) 拆解每个 issue 的代码文件、去行号、分块 → 向量检索 vs B) 每条实例存为完整文档，BM25 匹配最相关 issue
2. **检索方法**：BGE-M3 密集向量 vs Whoosh BM25 稀疏倒排索引

### 实验条件

| 条件 | 值 |
|------|-----|
| GPU | RTX 4060 Laptop 8GB |
| 测试数据 | SWE-bench_bm25_27K, 50 instances |
| BGE-M3 方案 | ChromaDB SentenceTransformerEmbeddingFunction, device=cuda |
| BM25 方案 | Whoosh 2.7.4, StandardAnalyzer |

### 对比结果

| 指标 | BGE-M3 + ChromaDB | Whoosh BM25 |
|------|-------------------|-------------|
| 50 条索引耗时 | ~15 分钟 | < 2 秒 |
| 单次检索耗时 | ~2 秒 (GPU) | < 10ms (CPU) |
| 索引大小 (50 条) | ~150MB | ~5MB |
| GPU 需求 | 必须 | 不需要 |
| 依赖大小 | chromadb (~50MB) + sentence-transformers (~2GB) | whoosh (~1MB) |
| 代码关键词匹配 | 中等（语义近似） | 优秀（精确词匹配） |
| 跨语言/同义词 | 优秀 | 不支持 |

### 结论

**选型：BM25 (Whoosh)**

核心依据：
1. SWE-bench 原作者使用 BM25 (Pyserini) 做文件检索——已验证 BM25 在代码搜索场景有效
2. 代码搜索本质是关键词匹配 (API名、函数名、类名)，非语义理解——BGE-M3 的语义优势在此场景发挥有限
3. 索引速度差异巨大（15分钟 vs 2秒），全量 18K 实例时 BM25 约 5 分钟，BGE-M3 约 10 小时
4. BM25 方案无需 GPU，可在任何环境运行

保留 `vector_store.py` (ChromaDB + BGE-M3) 用于通用文档 QA 场景（中文文档、概念性检索）。
新创建的 `bm25_store.py` 专用于代码搜索场景。

### 架构影响

- `rag/rag_pipeline.py` 重构为多态检索器（接受 `retriever=` 参数）
- `rag/ingest_swebench.py` 简化：移除代码解析/分块/去重，直接存完整实例
- `rag/bm25_store.py` 新建：与 VectorStore 接口一致，duck typing 替换
