"""
数据集下载脚本

支持下载以下数据集：
- BELLE (BelleGroup/train_0.5M_CN)
- Alpaca-GPT4-zh
- 用户自定义 jsonl 文件路径
"""
import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 预定义数据集配置
DATASET_REGISTRY = {
    "belle": {
        "path": "BelleGroup/train_0.5M_CN",
        "format": "alpaca",
        "split": "train",
        "description": "BELLE 中文指令微调数据集（50万条）",
    },
    "alpaca_gpt4_zh": {
        "path": "shibing624/alpaca-zh",
        "format": "alpaca",
        "split": "train",
        "description": "Alpaca-GPT4 中文数据集",
    },
    "code_alpaca_20k": {
        "path": "sahil2801/CodeAlpaca-20k",
        "format": "alpaca",
        "split": "train",
        "description": "CodeAlpaca-20k 代码指令数据集（2万条）",
    },
}


def download_predefined_dataset(name: str, output_dir: Path) -> Path:
    """下载预定义数据集"""
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"未知数据集: {name}。可用数据集: {available}")

    config = DATASET_REGISTRY[name]
    logger.info(f"下载数据集: {config['description']}")
    logger.info(f"HuggingFace 路径: {config['path']}")

    # 尝试从 HuggingFace 下载，支持镜像
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    logger.info(f"使用 HuggingFace 端点: {hf_endpoint}")

    try:
        dataset = load_dataset(config["path"], split=config["split"])
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.info("提示: 如果网络问题，请设置 HuggingFace 镜像:")
        logger.info("  export HF_ENDPOINT=https://hf-mirror.com")
        raise

    # 保存为 jsonl 格式
    output_path = output_dir / f"{name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"下载完成: {count} 条数据 -> {output_path}")
    logger.info(f"数据格式: {config['format']}")
    return output_path


def copy_custom_dataset(input_path: Path, output_dir: Path) -> Path:
    """复制用户自定义数据集到输出目录"""
    if not input_path.exists():
        raise FileNotFoundError(f"自定义数据文件不存在: {input_path}")

    output_path = output_dir / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 验证文件是否为有效的 jsonl
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")

    if count == 0:
        raise ValueError(f"文件中没有有效数据: {input_path}")

    # 复制文件
    import shutil
    shutil.copy2(input_path, output_path)

    logger.info(f"自定义数据集已复制: {count} 条数据 -> {output_path}")
    return output_path


def show_available_datasets():
    """显示可用数据集列表"""
    print("\n可用数据集:")
    print("=" * 60)
    for name, config in DATASET_REGISTRY.items():
        print(f"  {name:20s} - {config['description']}")
        print(f"  {'':20s}   路径: {config['path']}")
        print(f"  {'':20s}   格式: {config['format']}")
        print()
    print(f"  {'custom':20s} - 使用自定义 jsonl 文件")
    print(f"  {'':20s}   通过 --custom_path 指定文件路径")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集下载脚本")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_REGISTRY.keys()) + ["custom"],
        default="code_alpaca_20k",
        help="选择要下载的数据集",
    )
    parser.add_argument(
        "--custom_path",
        type=str,
        default=None,
        help="自定义 jsonl 文件路径（仅 --dataset=custom 时使用）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="输出目录（默认: data/raw）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用数据集",
    )

    args = parser.parse_args()

    if args.list:
        show_available_datasets()
    else:
        output_dir = Path(args.output_dir)
        if args.dataset == "custom":
            if args.custom_path is None:
                parser.error("--dataset=custom 时必须指定 --custom_path")
            copy_custom_dataset(Path(args.custom_path), output_dir)
        else:
            download_predefined_dataset(args.dataset, output_dir)
