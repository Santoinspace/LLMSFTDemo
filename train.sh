# 解决GBK报错
set PYTHONUTF8=1

# 训练
python train/train.py --train_data data/processed/train.jsonl --val_data data/processed/val.jsonl --output_dir outputs_codealpacas

# 继续训练
python train/resume_train.py --train_data data/processed/train.jsonl --val_data data/processed/val.jsonl --output_dir outputs_codealpacas

# 合并lora
python train/merge_lora.py --base_model Qwen/Qwen3-1.7B --adapter_path outputs_codealpacas/checkpoint-final  --output_dir outputs_codealpacas/merged_model

# 快速推理测试
python inference/inference.py --model_path outputs_codealpacas/merged_model --prompt "Write a Python function to check if a string is a palindrome"

# 生成测试用例（从验证集抽取）
python eval/generate_test_cases.py --input data/processed/val.jsonl --output eval/test_cases.jsonl --num_samples 30

# 对比评估（逐条加载两个模型，8GB 够用）
python eval/evaluate.py --base_model Qwen/Qwen3-1.7B --finetuned_model outputs_codealpacas/merged_model --test_data eval/test_cases.jsonl --output_dir eval_outputs

# 如何进一步测试微调效果
1. 人工测试几道编程题，确认生成代码质量：
python inference/inference.py \
    --model_path outputs_codealpacas/merged_model \
    --prompt "Write a Python function to merge two sorted lists"
2. 把 merged_model 挂到 API 服务，用 /chat 接口交互式测试：
python inference/api_server.py --model_path outputs_codealpacas/merged_model
# 然后浏览器打开 http://localhost:8000/docs
3. 如果觉得过拟合明显，下次可以考虑：只用 2 个 epoch、加更多数据（如 Magicoder）、增大 LoRA dropout。