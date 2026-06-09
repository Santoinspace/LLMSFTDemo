你是一名大模型微调专家。请帮我构建一个完整的 Qwen3-1.7B QLoRA 微调项目，项目结构清晰，代码生产可用。
## 背景要求
There’s a file modification bug in Claude Code. The workaround is: always use complete absolute Windows paths with drive letters and backslashes for ALL file operations. Apply this rule going forward, not just for this file.

## 项目目标

对 Qwen3-1.7B 模型进行 QLoRA 微调，面向特定领域知识问答，后续需集成 RAG 流程。

## 技术要求

### 硬件约束

- GPU 显存：8GB（单卡）
- 需要 4-bit 量化 + 梯度检查点确保显存可用

---

## 请构建以下完整项目结构

qwen3-finetune/
├── README.md                    # 详细使用说明
├── requirements.txt             # 精确版本依赖
├── setup.sh                     # 一键环境安装脚本
│
├── configs/
│   ├── qlora_config.yaml        # QLoRA 超参数配置
│   └── training_config.yaml     # 训练流程配置
│
├── data/
│   ├── download_dataset.py      # 数据集下载脚本（HuggingFace datasets）
│   ├── preprocess.py            # 数据预处理（转换为 ChatML 格式）
│   ├── validate_data.py         # 数据质量校验
│   └── sample_data.jsonl        # 10条示例数据（ChatML格式）
│
├── train/
│   ├── train.py                 # 主训练脚本（TRL SFTTrainer）
│   ├── merge_lora.py            # LoRA 权重合并脚本
│   └── resume_train.py          # 断点续训脚本
│
├── eval/
│   ├── evaluate.py              # 微调前后对比评估主脚本
│   ├── metrics.py               # PPL、ROUGE、BLEU、领域准确率计算
│   ├── generate_test_cases.py   # 测试用例生成
│   └── eval_report.py           # 生成对比评估报告（HTML格式）
│
├── inference/
│   ├── inference.py             # 单次推理接口
│   ├── batch_inference.py       # 批量推理
│   └── api_server.py            # FastAPI 推理服务（为后续 RAG 集成预留）
│
└── rag/  （脚手架，预留接口）
├── README.md                # RAG 集成说明
├── embeddings.py            # 使用 BAAI/bge-m3 生成 embedding
├── vector_store.py          # ChromaDB 向量存储封装
├── document_processor.py   # 文档加载、分块（chunk_size=512, overlap=64）
└── rag_pipeline.py          # 完整 RAG 流程（检索 + 生成）

## 各文件详细要求

### configs/qlora_config.yaml

包含以下参数：

- 模型：Qwen/Qwen3-1.7B-Instruct（从 HuggingFace 加载）
- 量化：load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16
- LoRA：r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
- target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- 训练：batch_size=1, gradient_accumulation_steps=16, max_seq_length=512, num_epochs=3
- 优化器：paged_adamw_8bit, lr=2e-4, warmup_ratio=0.03, lr_scheduler=cosine
- 保存：save_steps=100, logging_steps=10, eval_steps=100

### data/preprocess.py

- 支持输入格式：alpaca（instruction/input/output）、sharegpt、raw问答对
- 统一输出为 Qwen3 ChatML 格式
- 支持 system prompt 注入（领域专用 system prompt 通过配置文件指定）
- 数据过滤：去除长度超限、空内容、乱码条目
- 输出训练集/验证集划分（90%/10%）
- 打印数据统计信息（条数、平均长度、token分布）

### train/train.py

- 使用 trl.SFTTrainer
- 集成 WandB 日志（可选，通过环境变量控制）
- 支持梯度检查点（gradient_checkpointing=True）
- 训练时打印显存使用量
- 保存 adapter_model（不保存完整模型节省空间）
- 每个 checkpoint 保存验证集 loss

### train/merge_lora.py

- 加载 base model（4-bit）
- 合并 LoRA adapter
- 以 float16 保存完整模型
- 验证合并后模型可正常推理

### eval/evaluate.py

要求：

- 接受两个模型路径：base_model 和 finetuned_model
- 在同一测试集上分别推理
- 计算并对比以下指标：
  1. Perplexity (PPL) - 使用验证集
  2. ROUGE-1/2/L - 对比生成答案与参考答案
  3. BLEU-4
  4. 领域准确率 - 基于关键词匹配和语义相似度
  5. 平均生成长度
  6. 推理速度（tokens/sec）
- 输出对比表格到终端
- 保存详细结果到 eval_results.json
- 调用 eval_report.py 生成可视化 HTML 报告

### eval/eval_report.py

生成包含以下内容的 HTML 报告：

- 指标对比柱状图（base vs finetuned）
- 逐条案例对比（问题 | 参考答案 | base输出 | 微调输出）
- 指标变化百分比汇总表
- 使用 plotly 生成交互式图表（不依赖外部服务器）

### inference/api_server.py

- 使用 FastAPI + uvicorn
- POST /generate 接口：接收 {messages: [...], max_tokens: int}
- 支持流式输出（SSE）
- 接口设计考虑后续 RAG 集成（预留 use_rag 参数）
- 包含健康检查接口 GET /health

### rag/vector_store.py

- 封装 ChromaDB（persist_directory 可配置）
- 方法：add_documents(), similarity_search(query, k=5), delete_collection()
- 支持 metadata 过滤
- 使用 BAAI/bge-m3 作为 embedding 函数

### rag/rag_pipeline.py

- 实现标准 RAG 流程：文档检索 → prompt 构建 → 模型生成
- 支持两种模式：with_rag / without_rag（便于对比）
- 可配置 top_k 检索数量
- 检索结果注入 system prompt（而非 user prompt）

---

## 代码规范

- 所有脚本支持命令行参数（使用 argparse 或 hydra）
- 关键步骤添加中文注释
- 使用 logging 而非 print（日志级别可配置）
- 异常处理完整，显存不足时给出明确提示和调整建议
- 每个模块包含 if __name__ == "__main__" 的使用示例

## 额外要求

1. README.md 包含：环境安装、快速开始、完整参数说明、常见问题（OOM解决方案）
2. 提供 setup.sh 一键安装所有依赖
3. 在 data/download_dataset.py 中实现下载以下数据集的选项：
   - BELLE (BelleGroup/train_0.5M_CN)
   - Alpaca-GPT4-zh
   - 用户自定义 jsonl 文件路径
4. 所有路径使用 pathlib.Path，兼容 Windows/Linux

开始构建项目，先创建目录结构和所有文件。
