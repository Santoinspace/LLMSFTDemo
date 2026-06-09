"""
测试用例生成脚本

从验证集中抽取测试样本，支持：
- 随机采样指定数量的测试用例
- 按长度分层采样（短/中/长）
- 输出格式化的测试用例文件
"""
import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_messages_from_chatml(text: str) -> List[Dict]:
    """从 ChatML 格式文本中提取 messages"""
    messages = []
    blocks = text.split("<|im_start|>")[1:]  # 跳过开头空白

    for block in blocks:
        if "<|im_end|>" not in block:
            continue
        role_content, _ = block.split("<|im_end|>", 1)
        lines = role_content.strip().split("\n", 1)
        role = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        if role and content:
            messages.append({"role": role, "content": content})

    return messages


def extract_qa_pair(text: str) -> dict:
    """从 ChatML 文本中提取问答对"""
    messages = extract_messages_from_chatml(text)

    question = ""
    reference = ""

    for msg in messages:
        if msg["role"] == "system":
            question = f"[System] {msg['content']}\n\n"
        elif msg["role"] == "user":
            question += f"[User] {msg['content']}"
        elif msg["role"] == "assistant":
            reference = msg["content"]

    return {
        "question": question,
        "reference": reference,
        "messages": messages,
    }


def stratified_sample(
    data: List[Dict],
    num_samples: int,
    seed: int = 42,
) -> List[Dict]:
    """
    按长度分层采样

    将数据按长度分为短/中/长三组，每组等比例采样
    """
    if num_samples >= len(data):
        return data

    # 计算长度分布
    lengths = [len(d["text"]) for d in data]
    sorted_pairs = sorted(enumerate(lengths), key=lambda x: x[1])

    n = len(data)
    group_size = n // 3

    short_idx = [i for i, _ in sorted_pairs[:group_size]]
    medium_idx = [i for i, _ in sorted_pairs[group_size : 2 * group_size]]
    long_idx = [i for i, _ in sorted_pairs[2 * group_size:]]

    random.seed(seed)
    per_group = max(1, num_samples // 3)

    sampled_short = random.sample(short_idx, min(per_group, len(short_idx)))
    sampled_medium = random.sample(medium_idx, min(per_group, len(medium_idx)))
    sampled_long = random.sample(long_idx, min(per_group, len(long_idx)))

    all_indices = sampled_short + sampled_medium + sampled_long
    random.shuffle(all_indices)

    # 如果不够 num_samples，补充采样
    remaining = set(range(n)) - set(all_indices)
    while len(all_indices) < num_samples and remaining:
        idx = random.choice(list(remaining))
        all_indices.append(idx)
        remaining.remove(idx)

    sampled = [data[i] for i in all_indices[:num_samples]]

    logger.info(f"分层采样完成: {len(sampled)} 条（短:{len(sampled_short)} 中:{len(sampled_medium)} 长:{len(sampled_long)}）")
    return sampled


def generate_test_cases(
    input_path: Path,
    output_path: Path,
    num_samples: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """生成测试用例"""
    logger.info(f"读取验证数据: {input_path}")

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}")

    logger.info(f"读取到 {len(data)} 条数据")

    # 采样
    sampled = stratified_sample(data, num_samples, seed)

    # 转换为测试用例
    test_cases = []
    for item in sampled:
        qa = extract_qa_pair(item.get("text", ""))

        test_cases.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "text": item.get("text", ""),
            "messages": qa["messages"],
        })

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for tc in test_cases:
            f.write(json.dumps(tc, ensure_ascii=False) + "\n")

    logger.info(f"测试用例已保存: {output_path} ({len(test_cases)} 条)")

    # 打印摘要
    print_test_case_summary(test_cases)

    return test_cases


def print_test_case_summary(test_cases: List[Dict]) -> None:
    """打印测试用例摘要"""
    lengths = [len(tc["reference"]) for tc in test_cases]
    print("\n" + "=" * 60)
    print(" 测试用例摘要")
    print("=" * 60)
    print(f"  总数: {len(test_cases)}")
    print(f"  参考答案平均长度: {sum(lengths) / max(len(lengths), 1):.0f} 字符")
    print(f"  最短: {min(lengths)} 字符")
    print(f"  最长: {max(lengths)} 字符")
    print()
    print("  示例问题:")
    for i, tc in enumerate(test_cases[:3], 1):
        question = tc["question"].replace("\n", " ")[:80]
        print(f"  [{i}] {question}...")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试用例生成脚本")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据文件路径（jsonl 格式）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/test_cases.jsonl",
        help="输出测试用例文件路径",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="采样数量（默认: 50）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()
    generate_test_cases(
        input_path=Path(args.input),
        output_path=Path(args.output),
        num_samples=args.num_samples,
        seed=args.seed,
    )
