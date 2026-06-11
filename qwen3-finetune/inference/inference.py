"""
单次推理接口

支持加载合并后的模型或 LoRA adapter 进行推理。
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class QwenInference:
    """Qwen3 推理封装"""

    def __init__(
        self,
        model_path: str,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
    ):
        """
        初始化推理模型

        参数:
            model_path: 模型路径（合并后的模型或 adapter 路径）
            load_in_4bit: 是否 4-bit 量化加载
            device: 指定设备（auto/cuda/cpu）
        """
        self.model_path = model_path
        logger.info(f"加载模型: {model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
        }

        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        if device:
            model_kwargs["device_map"] = device
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        self.device = next(self.model.parameters()).device
        logger.info(f"模型已加载，设备: {self.device}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Dict:
        """
        生成回复

        参数:
            messages: ChatML 格式消息列表
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            do_sample: 是否采样

        返回:
            {"text": 生成的文本, "tokens": token数, "time": 耗时(秒)}
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_time

        new_tokens = outputs[0][input_len:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "text": generated_text,
            "tokens": len(new_tokens),
            "time": round(elapsed, 3),
        }

    def chat(
        self,
        user_message: str,
        system_prompt: str = "你是一个专业的领域知识助手。",
        **kwargs,
    ) -> Dict:
        """简化接口：单轮对话"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.generate(messages, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 单次推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/outputs_codealpacas/merged_model",
        help="合并后的模型路径",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请介绍一下自己。",
        help="用户输入",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="你是一个专业的领域知识助手。",
        help="系统提示语",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="不使用 4-bit 量化",
    )

    args = parser.parse_args()

    # 初始化推理器
    infer = QwenInference(
        model_path=args.model_path,
        load_in_4bit=not args.no_4bit,
    )

    # 执行推理
    result = infer.chat(
        user_message=args.prompt,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"\n{'='*60}")
    print(f" Prompt: {args.prompt}")
    print(f" Response: {result['text']}")
    print(f"{'='*60}")
    print(f" Tokens: {result['tokens']}, 耗时: {result['time']}s, "
          f"速度: {result['tokens'] / max(result['time'], 0.001):.1f} tok/s")
