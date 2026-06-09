"""
数据预处理脚本

支持输入格式：
- alpaca: {"instruction": "...", "input": "...", "output": "..."}
- sharegpt: {"conversations": [{"from": "human", "value": "..."}, ...]}
- raw: {"question": "...", "answer": "..."}

统一输出为 Qwen3 ChatML 格式，并划分训练集/验证集（90%/10%）。
"""
import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 格式转换函数
# =============================================================================

def convert_alpaca_to_chatml(item: Dict, system_prompt: str) -> List[Dict]:
    """将 alpaca 格式转换为 ChatML messages 列表"""
    instruction = item.get("instruction", "").strip()
    input_text = item.get("input", "").strip()
    output_text = item.get("output", "").strip()

    if not instruction or not output_text:
        return []

    # 拼接 instruction 和 input
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    return messages


def convert_sharegpt_to_chatml(item: Dict, system_prompt: str) -> List[Dict]:
    """将 sharegpt 格式转换为 ChatML messages 列表"""
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return []

    messages = [{"role": "system", "content": system_prompt}]

    for turn in conversations:
        role_from = turn.get("from", "")
        value = turn.get("value", "").strip()
        if not value:
            continue

        if role_from in ("human", "user"):
            messages.append({"role": "user", "content": value})
        elif role_from in ("gpt", "assistant", "chatgpt"):
            messages.append({"role": "assistant", "content": value})
        elif role_from == "system":
            # sharegpt 中如果已有 system，替换为我们的 system_prompt
            messages[0]["content"] = value

    # 至少需要一轮 user+assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return []

    return messages


def convert_raw_to_chatml(item: Dict, system_prompt: str) -> List[Dict]:
    """将 raw 问答对格式转换为 ChatML messages 列表"""
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    if not question or not answer:
        return []

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return messages


# 格式映射表
FORMAT_CONVERTERS = {
    "alpaca": convert_alpaca_to_chatml,
    "sharegpt": convert_sharegpt_to_chatml,
    "raw": convert_raw_to_chatml,
}


# =============================================================================
# ChatML 格式化
# =============================================================================

def format_chatml(messages: List[Dict]) -> str:
    """
    将 messages 列表格式化为 ChatML 文本

    输出示例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Hello!<|im_end|>
    <|im_start|>assistant
    Hi! How can I help?<|im_end|>
    """
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    # 最后追加 assistant 的起始标记（用于生成）
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)


# =============================================================================
# 数据校验与过滤
# =============================================================================

# 乱码检测正则：匹配连续 3 个以上的替换字符或控制字符
GARBLED_PATTERN = re.compile(r"[�]{3,}|[\x00-\x08\x0b\x0c\x0e-\x1f]{3,}")


def is_valid_sample(messages: List[Dict], max_length: int) -> bool:
    """校验单条数据是否有效"""
    if not messages:
        return False

    # 检查是否包含乱码
    full_text = " ".join(m["content"] for m in messages)
    if GARBLED_PATTERN.search(full_text):
        return False

    # 检查长度限制（粗略估算：1 token ≈ 1-2 个中文字符）
    if len(full_text) > max_length * 3:
        return False

    # 检查 assistant 回复不为空
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if not assistant_msgs or not assistant_msgs[-1]["content"].strip():
        return False

    return True


# =============================================================================
# 主处理流程
# =============================================================================

def process_dataset(
    input_path: Path,
    output_dir: Path,
    data_format: str,
    system_prompt: str,
    max_length: int,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    主处理流程：读取 → 转换 → 过滤 → 划分 → 保存
    """
    if data_format not in FORMAT_CONVERTERS:
        raise ValueError(f"不支持的格式: {data_format}。支持: {list(FORMAT_CONVERTERS.keys())}")

    converter = FORMAT_CONVERTERS[data_format]

    # 读取原始数据
    logger.info(f"读取数据文件: {input_path}")
    raw_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                raw_data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")

    logger.info(f"原始数据条数: {len(raw_data)}")

    # 转换与过滤
    valid_samples = []
    skip_reasons = {"format_error": 0, "empty_content": 0, "too_long": 0, "garbled": 0}

    for item in raw_data:
        messages = converter(item, system_prompt)

        if not messages:
            skip_reasons["format_error"] += 1
            continue

        if not is_valid_sample(messages, max_length):
            # 细分跳过原因
            full_text = " ".join(m["content"] for m in messages)
            if GARBLED_PATTERN.search(full_text):
                skip_reasons["garbled"] += 1
            elif len(full_text) > max_length * 3:
                skip_reasons["too_long"] += 1
            else:
                skip_reasons["empty_content"] += 1
            continue

        text = format_chatml(messages)
        valid_samples.append({"text": text, "messages": messages})

    logger.info(f"有效数据条数: {len(valid_samples)}")
    logger.info(f"跳过统计: {skip_reasons}")

    if not valid_samples:
        logger.error("没有有效数据，请检查输入文件格式")
        return

    # 划分训练集/验证集
    random.seed(seed)
    random.shuffle(valid_samples)

    val_count = max(1, int(len(valid_samples) * val_ratio))
    train_data = valid_samples[val_count:]
    val_data = valid_samples[:val_count]

    logger.info(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps({"text": sample["text"]}, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_data:
            f.write(json.dumps({"text": sample["text"]}, ensure_ascii=False) + "\n")

    logger.info(f"训练集已保存: {train_path}")
    logger.info(f"验证集已保存: {val_path}")

    # 打印统计信息
    print_statistics(train_data, val_data)


def print_statistics(train_data: List[Dict], val_data: List[Dict]) -> None:
    """打印数据统计信息"""
    all_data = train_data + val_data

    print("\n" + "=" * 60)
    print(" 数据统计")
    print("=" * 60)
    print(f"  总有效条数:   {len(all_data)}")
    print(f"  训练集条数:   {len(train_data)}")
    print(f"  验证集条数:   {len(val_data)}")

    # 计算文本长度统计
    lengths = [len(s["text"]) for s in all_data]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n  文本长度统计（字符数）:")
    print(f"    平均长度:   {avg_len:.0f}")
    print(f"    最小长度:   {min(lengths) if lengths else 0}")
    print(f"    最大长度:   {max(lengths) if lengths else 0}")
    print(f"    中位数:     {sorted(lengths)[len(lengths) // 2] if lengths else 0}")

    # 粗略估算 token 分布（中文约 1.5 字符/token，英文约 4 字符/token）
    estimated_tokens = [int(l / 1.5) for l in lengths]
    avg_tokens = sum(estimated_tokens) / len(estimated_tokens) if estimated_tokens else 0

    print(f"\n  Token 估算（约 {avg_tokens:.0f} tokens/条）:")
    print(f"    总估算 tokens: {sum(estimated_tokens):,}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理脚本（转换为 ChatML 格式）")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据文件路径（jsonl 格式）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="输出目录（默认: data/processed）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "sharegpt", "raw"],
        default="alpaca",
        help="输入数据格式（默认: alpaca）",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="你是一个专业的领域知识助手，请准确回答用户的问题。",
        help="系统提示语",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="最大序列长度（token 数，默认: 1024）",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例（默认: 0.1）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    process_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        data_format=args.format,
        system_prompt=args.system_prompt,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
