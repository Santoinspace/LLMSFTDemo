"""
主训练脚本 - Qwen3-1.7B QLoRA 微调

使用 trl.SFTTrainer 进行监督微调，支持：
- 4-bit 量化 + LoRA
- 梯度检查点
- WandB 日志（可选）
- 显存使用量监控
"""
import argparse
import json
import logging
import os
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


def print_gpu_memory():
    """打印 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(
            f"GPU 显存: 已分配 {allocated:.2f}GB, "
            f"已预留 {reserved:.2f}GB, "
            f"总计 {total:.1f}GB"
        )


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_wandb(training_config: dict):
    """设置 WandB 日志（通过环境变量 USE_WANDB 控制）"""
    use_wandb = os.environ.get("USE_WANDB", "false").lower() == "true"
    if use_wandb:
        import wandb
        wandb_config = training_config.get("wandb", {})
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "qwen3-qlora-finetune")
        wandb.init(
            project=wandb_config.get("project", "qwen3-qlora-finetune"),
            name=wandb_config.get("run_name", "qwen3-1.7b-qlora"),
        )
        logger.info("WandB 日志已启用")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("WandB 日志已禁用（设置 USE_WANDB=true 启用）")
    return use_wandb


def build_quantization_config(qlora_config: dict) -> BitsAndBytesConfig:
    """构建量化配置"""
    quant_config = qlora_config["quantization"]
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(
        quant_config["bnb_4bit_compute_dtype"], torch.bfloat16
    )

    return BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(qlora_config: dict) -> LoraConfig:
    """构建 LoRA 配置"""
    lora_config = qlora_config["lora"]
    return LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
        target_modules=lora_config["target_modules"],
    )


def build_training_arguments(
    qlora_config: dict, training_config: dict, output_dir: Path
) -> SFTConfig:
    """构建训练参数"""
    train_cfg = qlora_config["training"]
    save_cfg = qlora_config["save_and_logging"]
    general = training_config["general"]
    precision = training_config["precision"]
    memory = training_config["memory"]
    sft_cfg = training_config.get("sft_trainer", {})

    use_wandb = os.environ.get("USE_WANDB", "false").lower() == "true"

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        max_steps=train_cfg.get("max_steps", -1),
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B QLoRA 微调训练")
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
    args = parser.parse_args()

    # 加载配置
    qlora_config = load_config(Path(args.qlora_config))
    training_config = load_config(Path(args.training_config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(" Qwen3-1.7B QLoRA 微调训练")
    logger.info("=" * 60)

    # 检查 GPU
    if not torch.cuda.is_available():
        logger.error("未检测到 CUDA GPU，请确认 GPU 驱动和 CUDA 已正确安装")
        raise RuntimeError("需要 GPU 环境才能训练")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    print_gpu_memory()

    # 设置 WandB
    use_wandb = setup_wandb(training_config)

    # 加载 tokenizer
    model_name = qlora_config["model"]["model_name"]
    logger.info(f"加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=qlora_config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载量化模型
    logger.info("加载 4-bit 量化模型...")
    bnb_config = build_quantization_config(qlora_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=qlora_config["model"].get("trust_remote_code", True),
        device_map="auto",
    )

    # 准备 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    print_gpu_memory()

    # 应用 LoRA
    lora_config = build_lora_config(qlora_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_gpu_memory()

    # 加载数据集
    logger.info(f"加载训练数据: {args.train_data}")
    train_dataset = load_dataset(
        "json",
        data_files=args.train_data,
        split="train",
    )
    logger.info(f"训练集样本数: {len(train_dataset)}")

    eval_dataset = None
    val_path = Path(args.val_data)
    if val_path.exists():
        logger.info(f"加载验证数据: {args.val_data}")
        eval_dataset = load_dataset(
            "json",
            data_files=args.val_data,
            split="train",
        )
        logger.info(f"验证集样本数: {len(eval_dataset)}")

    # 构建训练参数
    training_args = build_training_arguments(
        qlora_config, training_config, output_dir
    )

    # 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 开始训练
    logger.info("开始训练...")
    print_gpu_memory()

    try:
        train_result = trainer.train()

        # 保存最终 adapter
        final_output = output_dir / "checkpoint-final"
        trainer.save_model(str(final_output))
        tokenizer.save_pretrained(str(final_output))
        logger.info(f"最终 adapter 已保存: {final_output}")

        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)

        metrics_path = output_dir / "train_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"训练指标已保存: {metrics_path}")

        # 最终显存状态
        print_gpu_memory()
        logger.info("训练完成！")

    except torch.cuda.OutOfMemoryError as e:
        logger.error("=" * 60)
        logger.error(" GPU 显存不足 (OOM)")
        logger.error("=" * 60)
        logger.error("请尝试以下调整:")
        logger.error("  1. 减小 max_seq_length（如 256 或 128）")
        logger.error("  2. 确认 batch_size=1")
        logger.error("  3. 增加 gradient_accumulation_steps")
        logger.error("  4. 确认 gradient_checkpointing=True")
        logger.error("  5. 减小 LoRA rank r（如 8）")
        logger.error(f"原始错误: {e}")
        raise


if __name__ == "__main__":
    main()
