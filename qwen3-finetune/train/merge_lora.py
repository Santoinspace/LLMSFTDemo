"""
LoRA 权重合并脚本

将 LoRA adapter 合并回基础模型，以 float16 保存完整模型。
合并后验证模型可正常推理。
"""
import argparse
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def merge_lora_model(
    base_model_name: str,
    adapter_path: Path,
    output_dir: Path,
    push_to_hub: bool = False,
) -> None:
    """
    合并 LoRA adapter 到基础模型

    参数:
        base_model_name: 基础模型名称或路径
        adapter_path: LoRA adapter 路径
        output_dir: 合并后模型输出目录
        push_to_hub: 是否推送到 HuggingFace Hub
    """
    logger.info("=" * 60)
    logger.info(" LoRA 权重合并")
    logger.info("=" * 60)
    logger.info(f"基础模型: {base_model_name}")
    logger.info(f"Adapter 路径: {adapter_path}")
    logger.info(f"输出目录: {output_dir}")

    # 检查 adapter 路径是否存在
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter 路径不存在: {adapter_path}")

    # 加载 tokenizer
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型（全精度）
    logger.info("加载基础模型（float16）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu",  # 在 CPU 上合并，避免显存不足
    )

    # 加载 LoRA adapter
    logger.info("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # 合并权重
    logger.info("合并 LoRA 权重...")
    model = model.merge_and_unload()
    logger.info("合并完成！")

    # 保存合并后的模型
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"保存合并后的模型到: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 可选：推送到 Hub
    if push_to_hub:
        logger.info("推送模型到 HuggingFace Hub...")
        model.push_to_hub(str(output_dir.name))
        tokenizer.push_to_hub(str(output_dir.name))

    logger.info("模型保存完成！")

    # 验证合并后模型可正常推理
    logger.info("验证合并后模型...")
    verify_merged_model(output_dir)


def verify_merged_model(model_path: Path) -> None:
    """验证合并后的模型可以正常推理"""
    logger.info("加载合并后的模型进行验证...")

    # 将模型移到 GPU（如果可用）进行推理验证
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    # 测试推理
    test_input = "你好，请介绍一下自己。"
    messages = [
        {"role": "system", "content": "你是一个专业的领域知识助手。"},
        {"role": "user", "content": test_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    elapsed = time.time() - start_time

    # 解码新生成的 token
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    logger.info(f"测试输入: {test_input}")
    logger.info(f"生成输出: {generated_text[:100]}...")
    logger.info(f"推理耗时: {elapsed:.2f}s")
    logger.info("✅ 合并后模型验证通过！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 权重合并脚本")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-1.7B-Instruct",
        help="基础模型名称或路径",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/checkpoint-final",
        help="LoRA adapter 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/outputs_codealpacas/merged_model",
        help="合并后模型输出目录",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="是否推送到 HuggingFace Hub",
    )

    args = parser.parse_args()

    merge_lora_model(
        base_model_name=args.base_model,
        adapter_path=Path(args.adapter_path),
        output_dir=Path(args.output_dir),
        push_to_hub=args.push_to_hub,
    )
