"""
数据质量校验脚本

校验预处理后的 ChatML 格式数据，检查：
1. JSON 格式是否合法
2. ChatML 标记是否完整
3. 文本长度是否在合理范围内
4. 是否存在空内容或乱码
5. 数据分布统计
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ChatML 标记
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# 乱码检测
GARBLED_PATTERN = re.compile(r"[�]{3,}|[\x00-\x08\x0b\x0c\x0e-\x1f]{3,}")


def validate_chatml_format(text: str) -> Tuple[bool, str]:
    """
    校验 ChatML 格式是否正确

    返回: (是否有效, 错误信息)
    """
    # 检查基本标记
    if IM_START not in text:
        return False, "缺少 <|im_start|> 标记"
    if IM_END not in text:
        return False, "缺少 <|im_end|> 标记"

    # 检查标记配对
    start_count = text.count(IM_START)
    end_count = text.count(IM_END)

    # 最后一个 assistant 回复可能没有 im_end（用于生成），所以允许 start = end + 1
    if start_count != end_count and start_count != end_count + 1:
        return False, f"标记不配对: {start_count} 个 start, {end_count} 个 end"

    # 检查是否包含 system、user、assistant 角色
    has_system = f"{IM_START}system\n" in text
    has_user = f"{IM_START}user\n" in text
    has_assistant = f"{IM_START}assistant\n" in text

    if not has_user:
        return False, "缺少 user 角色"
    if not has_assistant:
        return False, "缺少 assistant 角色"

    # system 不是必须的，但如果有应该在最前面
    if has_system:
        first_start = text.index(IM_START)
        if text[first_start:first_start + len(IM_START) + 7] != f"{IM_START}system\n":
            return False, "system 消息应该在最前面"

    return True, ""


def validate_single_sample(
    line: str, line_num: int, max_length: int
) -> Tuple[bool, List[str]]:
    """
    校验单条数据

    返回: (是否有效, 错误信息列表)
    """
    errors = []

    # 1. JSON 格式校验
    try:
        item = json.loads(line)
    except json.JSONDecodeError as e:
        return False, [f"JSON 解析失败: {e}"]

    # 2. 必须包含 text 字段
    text = item.get("text", "")
    if not text:
        return False, ["text 字段为空"]

    # 3. ChatML 格式校验
    is_valid, err_msg = validate_chatml_format(text)
    if not is_valid:
        errors.append(f"ChatML 格式错误: {err_msg}")

    # 4. 长度校验
    if len(text) > max_length * 3:
        errors.append(f"文本过长: {len(text)} 字符（建议不超过 {max_length * 3}）")

    if len(text) < 10:
        errors.append(f"文本过短: {len(text)} 字符")

    # 5. 乱码检测
    if GARBLED_PATTERN.search(text):
        errors.append("检测到乱码")

    # 6. 检查空内容
    if text.strip() == "":
        errors.append("内容为空")

    return len(errors) == 0, errors


def validate_dataset(input_path: Path, max_length: int = 512) -> Dict:
    """
    校验整个数据集

    返回校验报告
    """
    logger.info(f"开始校验数据文件: {input_path}")

    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        return {"error": f"文件不存在: {input_path}"}

    total = 0
    valid = 0
    invalid = 0
    error_details = []
    lengths = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            is_valid, errors = validate_single_sample(line, line_num, max_length)

            if is_valid:
                valid += 1
                # 记录长度用于统计
                item = json.loads(line)
                lengths.append(len(item.get("text", "")))
            else:
                invalid += 1
                error_details.append({
                    "line": line_num,
                    "errors": errors,
                })

            # 每 10000 条报告一次进度
            if total % 10000 == 0:
                logger.info(f"已处理 {total} 条...")

    # 生成报告
    report = {
        "file": str(input_path),
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_ratio": f"{valid / total * 100:.2f}%" if total > 0 else "N/A",
        "error_samples": error_details[:20],  # 最多展示 20 条错误
    }

    if lengths:
        report["stats"] = {
            "avg_length": int(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "median_length": sorted(lengths)[len(lengths) // 2],
        }

    return report


def print_report(report: Dict) -> None:
    """打印校验报告"""
    print("\n" + "=" * 60)
    print(" 数据质量校验报告")
    print("=" * 60)
    print(f"  文件:       {report.get('file', 'N/A')}")
    print(f"  总条数:     {report.get('total', 0)}")
    print(f"  有效条数:   {report.get('valid', 0)}")
    print(f"  无效条数:   {report.get('invalid', 0)}")
    print(f"  有效比例:   {report.get('valid_ratio', 'N/A')}")

    if "stats" in report:
        stats = report["stats"]
        print(f"\n  长度统计（字符数）:")
        print(f"    平均: {stats['avg_length']}")
        print(f"    最小: {stats['min_length']}")
        print(f"    最大: {stats['max_length']}")
        print(f"    中位数: {stats['median_length']}")

    error_samples = report.get("error_samples", [])
    if error_samples:
        print(f"\n  错误示例（最多展示 20 条）:")
        for err in error_samples[:10]:
            print(f"    第 {err['line']} 行: {', '.join(err['errors'])}")

    print("=" * 60)

    if report.get("invalid", 0) == 0:
        print(" ✅ 所有数据校验通过！")
    else:
        print(f" ⚠️  发现 {report['invalid']} 条无效数据，建议重新预处理")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据质量校验脚本")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据文件路径（jsonl 格式）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="最大序列长度（默认: 1024）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="校验报告输出路径（json 格式，可选）",
    )

    args = parser.parse_args()

    report = validate_dataset(Path(args.input), args.max_length)
    print_report(report)

    # 保存报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"校验报告已保存: {output_path}")
