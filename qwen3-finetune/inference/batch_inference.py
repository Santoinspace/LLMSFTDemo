"""
批量推理脚本

从 jsonl 文件读取多条输入，批量生成并保存结果。
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

from inference import QwenInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_batch_inference(
    model_path: str,
    input_file: Path,
    output_file: Path,
    system_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    load_in_4bit: bool = True,
) -> None:
    """
    批量推理

    参数:
        model_path: 模型路径
        input_file: 输入文件（每行一个 json，包含 "question" 或 "messages" 字段）
        output_file: 输出文件
        system_prompt: 系统提示语
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        load_in_4bit: 是否 4-bit 量化加载
    """
    # 加载推理器
    logger.info(f"加载模型: {model_path}")
    infer = QwenInference(model_path=model_path, load_in_4bit=load_in_4bit)

    # 读取输入
    logger.info(f"读取输入文件: {input_file}")
    inputs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                inputs.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}")

    logger.info(f"共 {len(inputs)} 条输入")

    # 批量推理
    results = []
    total_tokens = 0
    total_time = 0.0

    for i, item in enumerate(inputs):
        # 支持两种输入格式
        if "messages" in item:
            messages = item["messages"]
        elif "question" in item:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
            ]
        else:
            logger.warning(f"第 {i + 1} 条输入格式不支持，跳过")
            continue

        result = infer.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        total_tokens += result["tokens"]
        total_time += result["time"]

        results.append({
            "index": i,
            "messages": messages,
            "response": result["text"],
            "tokens": result["tokens"],
            "time": result["time"],
        })

        if (i + 1) % 10 == 0:
            avg_speed = total_tokens / max(total_time, 0.001)
            logger.info(f"进度: {i + 1}/{len(inputs)}, "
                        f"平均速度: {avg_speed:.1f} tok/s")

    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印统计
    avg_time = total_time / max(len(results), 1)
    avg_speed = total_tokens / max(total_time, 0.001)
    logger.info(f"\n{'='*50}")
    logger.info(f"批量推理完成")
    logger.info(f"  总条数: {len(results)}")
    logger.info(f"  总 tokens: {total_tokens}")
    logger.info(f"  总耗时: {total_time:.1f}s")
    logger.info(f"  平均耗时: {avg_time:.2f}s/条")
    logger.info(f"  平均速度: {avg_speed:.1f} tok/s")
    logger.info(f"  结果已保存: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/outputs_codealpacas/merged_model",
        help="模型路径",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件（jsonl 格式）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/batch_results.jsonl",
        help="输出文件路径",
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
    run_batch_inference(
        model_path=args.model_path,
        input_file=Path(args.input),
        output_file=Path(args.output),
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=not args.no_4bit,
    )
