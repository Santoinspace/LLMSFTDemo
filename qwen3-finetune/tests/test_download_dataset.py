"""测试 data/download_dataset.py 的自定义数据集处理"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from download_dataset import (
    DATASET_REGISTRY,
    copy_custom_dataset,
    show_available_datasets,
    download_predefined_dataset,
)


class TestDatasetRegistry:
    """测试数据集注册表"""

    def test_belle_registered(self):
        assert "belle" in DATASET_REGISTRY
        assert DATASET_REGISTRY["belle"]["format"] == "alpaca"

    def test_alpaca_gpt4_zh_registered(self):
        assert "alpaca_gpt4_zh" in DATASET_REGISTRY
        assert DATASET_REGISTRY["alpaca_gpt4_zh"]["format"] == "alpaca"

    def test_code_alpaca_20k_registered(self):
        assert "code_alpaca_20k" in DATASET_REGISTRY
        assert DATASET_REGISTRY["code_alpaca_20k"]["format"] == "alpaca"
        assert DATASET_REGISTRY["code_alpaca_20k"]["path"] == "sahil2801/CodeAlpaca-20k"
        assert DATASET_REGISTRY["code_alpaca_20k"]["split"] == "train"

    def test_predefined_dataset_raises_for_unknown(self):
        with pytest.raises(ValueError, match="未知数据集"):
            download_predefined_dataset("unknown_dataset", Path("/tmp"))


class TestCopyCustomDataset:
    """测试自定义数据集复制"""

    def test_copy_valid_jsonl(self, temp_dir, sample_alpaca_data):
        # 创建源文件
        src = temp_dir / "custom.jsonl"
        with open(src, "w", encoding="utf-8") as f:
            for item in sample_alpaca_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        dest_dir = temp_dir / "output"
        result = copy_custom_dataset(src, dest_dir)

        assert result.exists()
        assert result.name == "custom.jsonl"
        # 读取并验证
        with open(result, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == len(sample_alpaca_data)

    def test_file_not_found(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            copy_custom_dataset(temp_dir / "nonexistent.jsonl", temp_dir / "output")

    def test_empty_file_raises_error(self, temp_dir):
        src = temp_dir / "empty.jsonl"
        src.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="没有有效数据"):
            copy_custom_dataset(src, temp_dir / "output")
