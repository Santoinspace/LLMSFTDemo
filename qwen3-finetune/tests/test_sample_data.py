"""测试 data/sample_data.jsonl 示例数据的正确性"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from validate_data import validate_single_sample


class TestSampleData:
    """验证 sample_data.jsonl 中的数据全部有效"""

    def test_sample_data_exists(self):
        sample_path = Path(__file__).parent.parent / "data" / "sample_data.jsonl"
        assert sample_path.exists(), f"示例数据文件不存在: {sample_path}"

    def test_all_samples_valid(self):
        sample_path = Path(__file__).parent.parent / "data" / "sample_data.jsonl"
        with open(sample_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        assert len(lines) == 10, f"应该有 10 条示例数据，实际 {len(lines)} 条"

        for i, line in enumerate(lines, 1):
            # JSON 可解析
            item = json.loads(line)
            assert "text" in item, f"第 {i} 条缺少 text 字段"

            # ChatML 格式正确
            text = item["text"]
            assert "<|im_start|>system" in text, f"第 {i} 条缺少 system 消息"
            assert "<|im_start|>user" in text, f"第 {i} 条缺少 user 消息"
            assert "<|im_start|>assistant" in text, f"第 {i} 条缺少 assistant 消息"

            # 通过 validate_data 校验
            valid, errors = validate_single_sample(line, i, max_length=512)
            assert valid is True, f"第 {i} 条未通过校验: {errors}"

    def test_sample_data_diverse_topics(self):
        """验证示例数据涵盖多个主题"""
        sample_path = Path(__file__).parent.parent / "data" / "sample_data.jsonl"
        topics_found = []
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item["text"]
                if "机器学习" in text:
                    topics_found.append("机器学习")
                if "深度学习" in text:
                    topics_found.append("深度学习")
                if "NLP" in text:
                    topics_found.append("NLP")
                if "RAG" in text or "检索增强" in text:
                    topics_found.append("RAG")
                if "LoRA" in text or "QLoRA" in text:
                    topics_found.append("LoRA")

        assert len(set(topics_found)) >= 4, "示例数据应涵盖至少 4 个不同主题"
