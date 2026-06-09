"""测试 data/preprocess.py 的格式转换和数据处理逻辑"""
import json
import sys
from pathlib import Path

import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from preprocess import (
    convert_alpaca_to_chatml,
    convert_raw_to_chatml,
    convert_sharegpt_to_chatml,
    format_chatml,
    is_valid_sample,
    GARBLED_PATTERN,
)


class TestAlpacaConversion:
    """测试 alpaca 格式转换"""

    def test_basic_conversion(self):
        item = {"instruction": "什么是AI？", "input": "", "output": "AI是人工智能。"}
        sp = "你是一个助手。"
        result = convert_alpaca_to_chatml(item, sp)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": sp}
        assert result[1] == {"role": "user", "content": "什么是AI？"}
        assert result[2] == {"role": "assistant", "content": "AI是人工智能。"}

    def test_with_input_field(self):
        item = {"instruction": "解释AI", "input": "请用简单语言", "output": "AI即人工智能。"}
        sp = "你是助手。"
        result = convert_alpaca_to_chatml(item, sp)

        assert "解释AI\n\n请用简单语言" == result[1]["content"]

    def test_empty_instruction_returns_empty(self):
        item = {"instruction": "", "input": "", "output": "有输出但无指令"}
        sp = "你是助手。"
        result = convert_alpaca_to_chatml(item, sp)

        assert result == []

    def test_empty_output_returns_empty(self):
        item = {"instruction": "有指令", "input": "", "output": ""}
        sp = "你是助手。"
        result = convert_alpaca_to_chatml(item, sp)

        assert result == []

    def test_code_instruction_with_multiline_output(self):
        """验证代码指令格式：多行代码输出应完整保留"""
        item = {
            "instruction": "Write a function to reverse a string in Python",
            "input": "",
            "output": "def reverse_string(s):\n    return s[::-1]\n\n# Example usage\nprint(reverse_string('hello'))",
        }
        sp = "You are an expert coding assistant."
        result = convert_alpaca_to_chatml(item, sp)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[0]["content"] == sp
        assert result[1]["role"] == "user"
        assert "reverse a string" in result[1]["content"]
        assert result[2]["role"] == "assistant"
        assert "def reverse_string" in result[2]["content"]
        assert "return s[::-1]" in result[2]["content"]
        # 多行代码应保持换行
        assert "\n" in result[2]["content"]

    def test_code_instruction_with_input(self):
        """验证代码指令带 input 字段的场景"""
        item = {
            "instruction": "Complete the following Python code",
            "input": "def add(a, b):\n    # TODO: return the sum",
            "output": "def add(a, b):\n    return a + b",
        }
        sp = "You are an expert coding assistant."
        result = convert_alpaca_to_chatml(item, sp)

        assert len(result) == 3
        user_content = result[1]["content"]
        assert "Complete the following Python code" in user_content
        assert "def add(a, b):" in user_content
        assert result[2]["content"] == "def add(a, b):\n    return a + b"


class TestShareGPTConversion:
    """测试 sharegpt 格式转换"""

    def test_basic_conversion(self):
        item = {
            "conversations": [
                {"from": "human", "value": "什么是Python？"},
                {"from": "gpt", "value": "Python是编程语言。"},
            ]
        }
        sp = "你是助手。"
        result = convert_sharegpt_to_chatml(item, sp)

        assert len(result) == 3  # system + user + assistant
        assert result[0] == {"role": "system", "content": sp}
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_empty_conversations_returns_empty(self):
        item = {"conversations": []}
        sp = "你是助手。"
        result = convert_sharegpt_to_chatml(item, sp)
        assert result == []

    def test_missing_user_returns_empty(self):
        item = {
            "conversations": [
                {"from": "gpt", "value": "只有一个回复"},
            ]
        }
        sp = "你是助手。"
        result = convert_sharegpt_to_chatml(item, sp)
        assert result == []

    def test_multi_turn_conversation(self):
        item = {
            "conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"},
                {"from": "human", "value": "Q2"},
                {"from": "gpt", "value": "A2"},
            ]
        }
        sp = "你是助手。"
        result = convert_sharegpt_to_chatml(item, sp)

        assert len(result) == 5
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "user", "assistant"]


class TestRawConversion:
    """测试 raw 问答格式转换"""

    def test_basic_conversion(self):
        item = {"question": "你是谁？", "answer": "我是AI助手。"}
        sp = "你是助手。"
        result = convert_raw_to_chatml(item, sp)

        assert len(result) == 3
        assert result[1] == {"role": "user", "content": "你是谁？"}
        assert result[2] == {"role": "assistant", "content": "我是AI助手。"}

    def test_empty_question(self):
        item = {"question": "", "answer": "有答案"}
        sp = "你是助手。"
        result = convert_raw_to_chatml(item, sp)
        assert result == []

    def test_empty_answer(self):
        item = {"question": "有问题", "answer": ""}
        sp = "你是助手。"
        result = convert_raw_to_chatml(item, sp)
        assert result == []


class TestChatMLFormatting:
    """测试 ChatML 文本格式化"""

    def test_format_chatml(self):
        messages = [
            {"role": "system", "content": "你是助手。"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        text = format_chatml(messages)

        assert "<|im_start|>system\n你是助手。<|im_end|>" in text
        assert "<|im_start|>user\n你好<|im_end|>" in text
        assert "<|im_start|>assistant\n你好！<|im_end|>" in text
        # 最后一个 assistant 后应追加生成提示
        assert text.endswith("<|im_start|>assistant\n")


class TestValidation:
    """测试数据校验"""

    def test_valid_sample(self):
        messages = [
            {"role": "system", "content": "你是助手。"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ]
        assert is_valid_sample(messages, max_length=512) is True

    def test_empty_messages(self):
        assert is_valid_sample([], max_length=512) is False

    def test_garbled_text(self):
        messages = [
            {"role": "system", "content": "助手"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "正常回答���乱码"},
        ]
        # "���" 是3个乱码字符，但需要连续3个以上
        messages[2]["content"] = "正常回答�����测试"
        assert is_valid_sample(messages, max_length=512) is False

    def test_too_long(self):
        messages = [
            {"role": "system", "content": "助手"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "A" * 2000},
        ]
        # max_length=512, 限制 512*3 = 1536 字符
        assert is_valid_sample(messages, max_length=512) is False

    def test_empty_assistant_response(self):
        messages = [
            {"role": "system", "content": "助手"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "  "},
        ]
        assert is_valid_sample(messages, max_length=512) is False

    def test_no_assistant(self):
        messages = [
            {"role": "system", "content": "助手"},
            {"role": "user", "content": "问题"},
        ]
        assert is_valid_sample(messages, max_length=512) is False


class TestGarbledPattern:
    """测试乱码检测正则"""

    def test_normal_text(self):
        assert GARBLED_PATTERN.search("正常的中文文本") is None

    def test_replace_characters(self):
        assert GARBLED_PATTERN.search("包含乱码���的文本") is not None

    def test_control_characters(self):
        assert GARBLED_PATTERN.search("包含\x00\x01\x02\x03控制字符") is not None
