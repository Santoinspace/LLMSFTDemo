"""
断点续训脚本

从已保存的 checkpoint 恢复训练，支持：
- 自动检测最新 checkpoint
- 手动指定 checkpoint 路径
- 保持训练参数和 LoRA 配置一致
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: Path) -> Path:
    """自动查找最新的 checkpoint"""
    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"在 {output_dir} 中未找到任何 checkpoint")

    # 按 step 数排序，取最新的
    def get_step(p: Path) -> int:
        match = re.search(r"checkpoint-(\d+)", p.name)
        return int(match.group(1)) if match else 0

    latest = max(checkpoint_dirs, key=get_step)
    logger.info(f"找到最新 checkpoint: {latest}")
    return latest


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B QLoRA 断点续训")
    parser.add_argument(
        "--qlora_config",
        type=str,
        default="configs/qlora_qwen3-1.7b_codealpaca_r16_len1024_ep3.yaml",
        help="QLoRA 配置文件路径",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_qwen3-1.7b_codealpaca.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/processed/train.jsonl",
        help="训练数据文件路径",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/processed/val.jsonl",
        help="验证数据文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="指定 checkpoint 路径（不指定则自动选择最新的）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 确定 checkpoint 路径
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    logger.info("=" * 60)
    logger.info(" Qwen3-1.7B QLoRA 断点续训")
    logger.info("=" * 60)
    logger.info(f"从 checkpoint 恢复: {checkpoint_path}")

    # 检查 GPU
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU 环境才能训练")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载配置
    qlora_config = load_config(Path(args.qlora_config))
    training_config = load_config(Path(args.training_config))

    # WandB 设置
    use_wandb = os.environ.get("USE_WANDB", "false").lower() == "true"
    if not use_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    # 加载 tokenizer
    model_name = qlora_config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=qlora_config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建量化配置
    quant_config = qlora_config["quantization"]
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype_map.get(
            quant_config["bnb_4bit_compute_dtype"], torch.bfloat16
        ),
    )

    # 加载模型
    logger.info("加载量化模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # 应用 LoRA
    lora_cfg = qlora_config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据集
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = None
    val_path = Path(args.val_data)
    if val_path.exists():
        eval_dataset = load_dataset("json", data_files=args.val_data, split="train")

    # 构建训练参数
    train_cfg = qlora_config["training"]
    save_cfg = qlora_config["save_and_logging"]
    general = training_config["general"]
    precision = training_config["precision"]
    memory = training_config["memory"]
    sft_cfg = training_config.get("sft_trainer", {})

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        logging_dir=str(output_dir / "logs"),
        logging_steps=save_cfg["logging_steps"],
        save_steps=save_cfg["save_steps"],
        save_total_limit=save_cfg["save_total_limit"],
        eval_steps=save_cfg["eval_steps"],
        eval_strategy=save_cfg["eval_strategy"],
        save_strategy=save_cfg["save_strategy"],
        load_best_model_at_end=save_cfg["load_best_model_at_end"],
        metric_for_best_model=save_cfg["metric_for_best_model"],
        greater_is_better=save_cfg["greater_is_better"],
        fp16=precision.get("fp16", False),
        bf16=precision.get("bf16", True),
        gradient_checkpointing=memory.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=memory.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
        optim=train_cfg["optimizer"],
        seed=general.get("seed", 42),
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        label_names=["labels"],
        max_length=sft_cfg.get("max_seq_length", train_cfg["max_seq_length"]),
        packing=sft_cfg.get("packing", False),
        dataset_text_field=sft_cfg.get("dataset_text_field", "text"),
    )

    # 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 从 checkpoint 恢复训练
    logger.info(f"从 checkpoint 恢复训练: {checkpoint_path}")
    try:
        train_result = trainer.train(resume_from_checkpoint=str(checkpoint_path))

        # 保存最终 adapter
        final_output = output_dir / "checkpoint-final"
        trainer.save_model(str(final_output))
        tokenizer.save_pretrained(str(final_output))
        logger.info(f"最终 adapter 已保存: {final_output}")

        # 保存训练指标
        metrics = train_result.metrics
        metrics_path = output_dir / "resume_train_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"训练指标已保存: {metrics_path}")

        logger.info("断点续训完成！")

    except torch.cuda.OutOfMemoryError as e:
        logger.error("GPU 显存不足 (OOM)，请参考 train.py 中的 OOM 解决方案")
        raise


if __name__ == "__main__":
    main()
