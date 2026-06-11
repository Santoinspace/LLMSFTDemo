# Qwen3-1.7B QLoRA 微调项目

面向特定领域知识问答的 Qwen3-1.7B 模型 QLoRA 微调方案，支持后续 RAG 流程集成。

## 环境要求

- Python >= 3.10
- CUDA >= 12.1
- GPU 显存 >= 8GB（单卡）
- conda 环境（推荐）

## 快速开始

### 1. 创建环境

```bash
conda create -n LLMSFTDemo python=3.10 -y
conda activate LLMSFTDemo
```

### 2. 安装依赖

**方式一：一键安装（推荐）**
```bash
chmod +x setup.sh
./setup.sh
```

**方式二：手动安装**
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. 准备数据

```bash
# 下载数据集（以 BELLE 为例）
python data/download_dataset.py --dataset belle --output_dir data/raw

# 预处理为 ChatML 格式
python data/preprocess.py \
    --input data/raw/train.jsonl \
    --output_dir data/processed \
    --format alpaca \
    --system_prompt "你是一个专业的领域知识助手。" \
    --max_length 512

# 验证数据质量
python data/validate_data.py --input data/processed/train.jsonl
```

### 4. 开始训练

```bash
python train/train.py \
    --qlora_config configs/qlora_config.yaml \
    --training_config configs/training_config.yaml \
    --train_data data/processed/train.jsonl \
    --val_data data/processed/val.jsonl \
    --output_dir outputs
```

### 5. 合并 LoRA 权重

```bash
python train/merge_lora.py \
    --base_model Qwen/Qwen3-1.7B-Instruct \
    --adapter_path outputs/checkpoint-final \
    --output_dir outputs/outputs_codealpacas/merged_model
```

### 6. 评估模型

```bash
python eval/evaluate.py \
    --base_model Qwen/Qwen3-1.7B-Instruct \
    --finetuned_model outputs/outputs_codealpacas/merged_model \
    --test_data data/processed/val.jsonl \
    --output_dir eval_outputs
```

### 7. 启动推理服务

```bash
python inference/api_server.py \
    --model_path outputs/outputs_codealpacas/merged_model \
    --host 0.0.0.0 \
    --port 8000
```

### 8. 构建 SWE-bench 代码知识库 (RAG + BM25)

```bash
# 查看数据集统计
python rag/ingest_swebench.py --stats --max_instances 200

# 导入知识库（纯 CPU，秒级索引）
python rag/ingest_swebench.py --shuffle --retriever bm25

# BM25 检索
python rag/bm25_store.py \
    --persist_dir ./bm25_index_swebench \
    --collection swebench_instances \
    --action search \
    --query "database connection pooling" \
    --k 5

# RAG 问答
python rag/rag_pipeline.py \
    --retriever bm25 \
    --persist_dir ./bm25_index_swebench \
    --collection swebench_instances \
    --query "How does Django handle database migrations?" \
    --mode rag
```

---

## PROJECT STATE

| 模块 | 状态 | 备注 |
|------|------|------|
| 训练 (train/) | 完成 | CodeAlpaca-20k QLoRA 微调，3 epoch，loss 4.08→0.51 |
| 评估 (eval/) | 完成 | PPL 34.3→2.0, ROUGE-L 0.09→0.45 |
| 推理 (inference/) | 完成 | FastAPI + 批量推理可用 |
| RAG - 向量检索 | 完成 | ChromaDB + BGE-M3，通用文档 QA |
| RAG - BM25 检索 | **新增** | Whoosh BM25，SWE-bench 代码知识库，纯 CPU |
| RAG - API 集成 | 预留 | `use_rag` 参数已定义，待接入 RAG pipeline |

### 当前 RAG 检索器能力

| 检索器 | 适用场景 | 检索方式 | 硬件 |
|--------|---------|---------|------|
| `bm25_store.py` | 代码搜索 / SWE-bench | BM25 倒排索引 | CPU |
| `vector_store.py` | 通用文档 QA | BGE-M3 + Cosine ANN | GPU |

---

## RECENT DECISIONS

### 2026-06-11: SWE-bench 知识库采用 Whoosh BM25 替代 ChromaDB + BGE-M3

**决策**：将 SWE-bench_bm25_27K 导入方式从"代码文件分块 + BGE-M3 embedding + ChromaDB 向量检索"改为"完整实例文档 + Whoosh BM25 倒排索引"。

**原因**：
1. SWE-bench 数据集本身已将 BM25 作为文件检索方式，验证了 BM25 在代码搜索场景的有效性
2. 代码富含 API 名、函数名、关键词——这些正是 BM25 擅长的精确匹配场景
3. BGE-M3 (568M 参数) 在 RTX 4060 Laptop 上索引 100 实例需 ~15 分钟；BM25 索引同等数据 < 2 秒
4. 用户提问时匹配最相关 issue，Princeton NLP 已做好的 BM25 检索结果（`text` 字段）可以直接作为 LLM 上下文，不需自建检索

**实现**：创建 `rag/bm25_store.py`，接口与 `VectorStore` 完全一致（duck typing），`RAGPipeline` 通过 `retriever=` 参数接受任意检索器。

---

## 完整参数说明

### QLoRA 配置（configs/qlora_config.yaml）

| 参数 | 值 | 说明 |
|------|-----|------|
| model_name | Qwen/Qwen3-1.7B-Instruct | 基座模型 |
| load_in_4bit | True | 4-bit 量化加载 |
| bnb_4bit_quant_type | nf4 | NormalFloat4 量化类型 |
| bnb_4bit_use_double_quant | True | 双重量化节省显存 |
| bnb_4bit_compute_dtype | bfloat16 | 计算精度 |
| lora_r | 16 | LoRA 秩 |
| lora_alpha | 32 | LoRA 缩放因子 |
| lora_dropout | 0.05 | LoRA dropout |
| bias | none | 偏置训练方式 |
| target_modules | q/k/v/o_proj + gate/up/down_proj | 注意力与 FFN 层 |
| batch_size | 1 | 训练批次大小 |
| gradient_accumulation_steps | 16 | 梯度累积步数 |
| max_seq_length | 512 | 最大序列长度 |
| num_epochs | 3 | 训练轮数 |
| learning_rate | 2e-4 | 学习率 |
| optimizer | paged_adamw_8bit | 8-bit 分页优化器 |
| warmup_ratio | 0.03 | 预热比例 |
| lr_scheduler | cosine | 学习率调度 |
| save_steps | 100 | 保存步数 |
| logging_steps | 10 | 日志步数 |
| eval_steps | 100 | 评估步数 |

### 训练配置（configs/training_config.yaml）

| 参数 | 说明 |
|------|------|
| seed | 随机种子，默认 42 |
| fp16 | 是否使用混合精度 |
| bf16 | 是否使用 bfloat16 |
| gradient_checkpointing | 梯度检查点（节省显存） |
| max_grad_norm | 梯度裁剪 |
| wandb_project | WandB 项目名称 |
| wandb_run_name | WandB 运行名称 |

---

## 项目结构

```
qwen3-finetune/
├── configs/          # 配置文件
├── data/             # 数据处理
│   ├── download_dataset.py   # 数据集下载
│   ├── preprocess.py         # 数据预处理
│   ├── validate_data.py      # 数据校验
│   └── sample_data.jsonl     # 示例数据
├── train/            # 训练脚本
│   ├── train.py              # 主训练脚本
│   ├── merge_lora.py         # LoRA 权重合并
│   └── resume_train.py       # 断点续训
├── eval/             # 评估脚本
│   ├── evaluate.py           # 评估主脚本
│   ├── metrics.py            # 指标计算
│   ├── generate_test_cases.py # 测试用例生成
│   └── eval_report.py        # HTML 报告生成
├── inference/        # 推理接口
│   ├── inference.py          # 单次推理
│   ├── batch_inference.py    # 批量推理
│   └── api_server.py         # FastAPI 服务
└── rag/              # RAG 检索增强生成
    ├── bm25_store.py         # Whoosh BM25 检索存储
    ├── vector_store.py       # ChromaDB 向量存储封装
    ├── embeddings.py         # BGE-M3 embedding 生成
    ├── document_processor.py # 文档加载与分块
    ├── rag_pipeline.py       # RAG 流程（检索 + 生成）
    ├── ingest_swebench.py    # SWE-bench 知识库导入
    └── README.md             # RAG 模块说明
```

---

## 常见问题

### CUDA Out of Memory (OOM)

1. **减小 max_seq_length**：将 512 改为 256 或 128
2. **减小 batch_size**：确保为 1
3. **增加 gradient_accumulation_steps**：用更多步数补偿小 batch
4. **启用梯度检查点**：`gradient_checkpointing=True`（默认已开启）
5. **减小 LoRA rank**：将 r=16 改为 r=8
6. **检查后台进程**：`nvidia-smi` 确认无其他占用显存的进程

### 数据格式

支持的输入格式：
- **alpaca**：`{"instruction": "...", "input": "...", "output": "..."}`
- **sharegpt**：`{"conversations": [{"from": "human", "value": "..."}, ...]}`
- **raw**：`{"question": "...", "answer": "..."}`

### 模型加载失败

- 检查 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 检查本地缓存：`ls ~/.cache/huggingface/`

---

## License

本项目仅供学习研究使用。
