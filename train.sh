# 解决GBK报错
chcp 65001
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
# 快速切换目录
cd E:/Apostgraduate/projection/LLMSFTDemo/qwen3-finetune


# 工单数据集上SFT
python train/train.py ^
      --qlora_config configs/qlora_qwen3-1.7b_wl-cs_r16_len512_ep3.yaml ^
      --training_config configs/training_qwen3-1.7b_wl-cs.yaml ^
      --train_data data/wl_customer_support_split/sft_train.jsonl ^
      --val_data data/wl_customer_support_split/sft_val.jsonl ^
      --output_dir outputs/outputs_wl_customer_support

# 工单数据集上断点续训
python train/resume_train.py ^
      --qlora_config configs/qlora_qwen3-1.7b_wl-cs_r16_len512_ep3.yaml ^
      --training_config configs/training_qwen3-1.7b_wl-cs.yaml ^
      --train_data data/wl_customer_support_split/sft_train.jsonl ^
      --val_data data/wl_customer_support_split/sft_val.jsonl ^
      --output_dir outputs/outputs_wl_customer_support

# 工单数据集上测试
python eval/compare_validate.py ^
  --base_model Qwen/Qwen3-1.7B ^
  --finetuned_model outputs/outputs_wl_customer_support/merged_model ^
  --test_cases data/wl_customer_support_split/rag_eval.jsonl ^
  --retriever bm25 ^
  --retriever_path bm25_index_wl_customer_support ^
  --retriever_collection wl_customer_support ^
  --modes both ^
  --batch_size 4 ^
  --max_samples 0 ^
  --max_new_tokens 160 ^
  --top_k 2 ^
  --temperature 0 ^
  --output eval_outputs/wl_customer_support/compare_validate_1415.json

# 85%工单数据集，15%通用数据集，replay
python train/train.py ^
python train/resume_train.py ^
    --qlora_config configs/qlora_qwen3-1.7b_wl-cs_r16_len512_ep3.yaml ^
    --training_config configs/training_qwen3-1.7b_wl-cs.yaml ^
    --train_data data/preprocessed/wl_customer_support_replay15/mixed_train.jsonl ^
    --val_data data/wl_customer_support_split/sft_val.jsonl ^
    --output_dir outputs/outputs_wl_customer_support_replay15


# base + no_rag
python rag/rag_pipeline.py ^
    --model_path Qwen/Qwen3-1.7B ^
    --mode no_rag ^
    --query "Write a Python function to safely parse JSON from untrusted input."

# base + rag
python rag/rag_pipeline.py ^
    --model_path Qwen/Qwen3-1.7B ^
    --retriever bm25 ^
    --persist_dir chroma_db_swebench ^
    --collection swebench_instances ^
    --mode rag ^
    --query "Write a Python function to safely parse JSON from untrusted input."

# fine-tuned + no_rag
python rag/rag_pipeline.py ^
    --model_path outputs/outputs_codealpacas/merged_model ^
    --mode no_rag ^
    --query "Write a Python function to safely parse JSON from untrusted input."

# fine-tuned + rag
python rag/rag_pipeline.py ^
    --model_path outputs/outputs_codealpacas/merged_model ^
    --retriever bm25 ^
    --persist_dir chroma_db_swebench ^
    --collection swebench_instances ^
    --mode rag ^
    --query "Write a Python function to safely parse JSON from untrusted input."

或者使用 --mode compare 一次出两个答案（模型只加载一次）:
# base 组: RAG vs no-RAG
python rag/rag_pipeline.py ^
    --model_path Qwen/Qwen3-1.7B ^
    --retriever bm25 ^
    --persist_dir chroma_db_swebench ^
    --collection swebench_instances ^
    --mode compare ^
    --query "Write a Python function to safely parse JSON."

# fine-tuned 组: RAG vs no-RAG
python rag/rag_pipeline.py ^
    --model_path outputs_codealpacas/merged_model ^
    --retriever bm25 ^
    --persist_dir chroma_db_swebench ^
    --collection swebench_instances ^
    --mode compare ^
    --query "Write a Python function to safely parse JSON."


# 启动服务
python inference/api_server.py --model_path outputs/outputs_codealpacas/merged_model --enable_rag
# 不带RAG
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query": "How to handle database connection errors?", "use_rag": false}'

# 训练
python train/train.py --train_data data/processed/train.jsonl --val_data data/processed/val.jsonl --output_dir outputs_codealpacas
# 继续训练
python train/resume_train.py --train_data data/processed/train.jsonl --val_data data/processed/val.jsonl --output_dir outputs_codealpacas

# 合并lora
python train/merge_lora.py --base_model Qwen/Qwen3-1.7B --adapter_path outputs_codealpacas/checkpoint-final  --output_dir outputs_codealpacas/merged_model

# 快速推理测试
python inference/inference.py --model_path outputs_codealpacas/merged_model --prompt "Write a Python function to check if a string is a palindrome"
# 快速推理测试，没有sft，没有rag
python rag/rag_pipeline.py --model_path Qwen/Qwen3-1.7B --mode no_rag --query "Write a Python function to safely parse JSON from untrusted input. Only code, do not explain"

# 生成测试用例（从验证集抽取）
python eval/generate_test_cases.py --input data/processed/val.jsonl --output eval/test_cases.jsonl --num_samples 30

# 对比评估（逐条加载两个模型，8GB 够用）
python eval/evaluate.py --base_model Qwen/Qwen3-1.7B --finetuned_model outputs_codealpacas/merged_model --test_data eval/test_cases.jsonl --output_dir eval_outputs
# 四个对比试验
python rag/compare_experiments.py --output eval_outputs/experiment_4group_codealpaca.json

# 如何进一步测试微调效果
1. 人工测试几道编程题，确认生成代码质量：
python inference/inference.py \
    --model_path outputs_codealpacas/merged_model \
    --prompt "Write a Python function to merge two sorted lists"
2. 把 merged_model 挂到 API 服务，用 /chat 接口交互式测试：
python inference/api_server.py --model_path outputs_codealpacas/merged_model
# 然后浏览器打开 http://localhost:8000/docs
3. 如果觉得过拟合明显，下次可以考虑：只用 2 个 epoch、加更多数据（如 Magicoder）、增大 LoRA dropout。

# RAG使用方式
# 先看数据统计
python rag/ingest_swebench.py --stats --max_instances 200
# 小规模导入测试
python rag/ingest_swebench.py --max_instances 100 --shuffle --output_jsonl data/swebench_chunks.jsonl
# 全量导入
python rag/ingest_swebench.py --shuffle
# 只索引特定仓库
python rag/ingest_swebench.py --filter_repo django/django

# 然后用 RAG pipeline 检索
python rag/rag_pipeline.py ^
    --persist_dir ./chroma_db_swebench ^
    --collection swebench_code ^
    --query "How to handle Django model save transactions?" ^
    --mode rag
