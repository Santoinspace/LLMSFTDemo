"""测试 eval/generate_test_cases.py 的测试用例生成逻辑"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from generate_test_cases import (
    extract_qa_pair,
    extract_messages_from_chatml,
    stratified_sample,
    generate_test_cases,
)


class TestChatMLParsing:
    """测试 ChatML 文本解析"""

    def test_extract_messages(self):
        text = (
            "<|im_start|>system\n你是助手。<|im_end|>\n"
            "<|im_start|>user\n什么是ML？<|im_end|>\n"
            "<|im_start|>assistant\nML是机器学习。<|im_end|>\n"
        )
        messages = extract_messages_from_chatml(text)

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "你是助手。"}
        assert messages[1] == {"role": "user", "content": "什么是ML？"}
        assert messages[2] == {"role": "assistant", "content": "ML是机器学习。"}

    def test_extract_empty_text(self):
        messages = extract_messages_from_chatml("")
        assert messages == []

    def test_extract_qa_pair(self):
        text = (
            "<|im_start|>system\n你是助手。<|im_end|>\n"
            "<|im_start|>user\n什么是AI？<|im_end|>\n"
            "<|im_start|>assistant\nAI是人工智能。<|im_end|>\n"
        )
        qa = extract_qa_pair(text)

        assert "question" in qa
        assert "reference" in qa
        assert "messages" in qa
        assert "[User] 什么是AI？" in qa["question"]
        assert qa["reference"] == "AI是人工智能。"


class TestStratifiedSample:
    """测试分层采样"""

    def test_sample_size(self):
        data = [{"text": "x" * i} for i in range(10, 110, 2)]

        sampled = stratified_sample(data, num_samples=10, seed=42)
        assert len(sampled) == 10
        # 所有元素应来自原始数据
        for s in sampled:
            assert s in data

    def test_sample_all_when_fewer(self):
        data = [{"text": "短"}] * 5
        sampled = stratified_sample(data, num_samples=20, seed=42)
        assert len(sampled) == 5  # 不超过总数

    def test_deterministic_with_seed(self):
        data = [{"text": str(i)} for i in range(100)]
        sample1 = stratified_sample(data, num_samples=20, seed=42)
        sample2 = stratified_sample(data, num_samples=20, seed=42)

        texts1 = [s["text"] for s in sample1]
        texts2 = [s["text"] for s in sample2]
        assert texts1 == texts2  # 相同 seed 应产生相同结果


class TestGenerateTestCases:
    """测试测试用例生成完整流程"""

    def test_generate_from_jsonl(self, temp_dir):
        # 创建包含 ChatML 数据的 jsonl 文件
        input_path = temp_dir / "val.jsonl"
        samples = []
        for i in range(20):
            text = (
                f"<|im_start|>system\n你是助手。<|im_end|>\n"
                f"<|im_start|>user\n问题{i}<|im_end|>\n"
                f"<|im_start|>assistant\n回答{i}<|im_end|>\n"
            )
            samples.append({"text": text})

        with open(input_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        output_path = temp_dir / "test_cases.jsonl"
        result = generate_test_cases(input_path, output_path, num_samples=10, seed=42)

        assert len(result) == 10
        assert output_path.exists()

        # 验证输出格式
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                tc = json.loads(line)
                assert "question" in tc
                assert "reference" in tc
                assert "messages" in tc
