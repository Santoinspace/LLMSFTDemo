"""pytest 公共配置和 fixtures"""
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """临时目录 fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_alpaca_data():
    """示例 alpaca 格式数据"""
    return [
        {"instruction": "什么是机器学习？", "input": "", "output": "机器学习是AI的分支。"},
        {"instruction": "解释深度学习", "input": "请详细说明", "output": "深度学习使用多层神经网络。"},
        {"instruction": "", "input": "", "output": ""},  # 空数据，应被过滤
        {"instruction": "什么是NLP？", "input": "", "output": "自然语言处理。"},
        {"instruction": "Transformer原理", "input": "核心机制", "output": "自注意力机制是核心。"},
    ]


@pytest.fixture
def sample_sharegpt_data():
    """示例 sharegpt 格式数据"""
    return [
        {
            "conversations": [
                {"from": "human", "value": "什么是深度学习？"},
                {"from": "gpt", "value": "深度学习是机器学习的分支。"},
            ]
        },
        {
            "conversations": []  # 空对话，应被过滤
        },
    ]


@pytest.fixture
def sample_raw_data():
    """示例 raw 问答对格式数据"""
    return [
        {"question": "什么是Python？", "answer": "Python是一种编程语言。"},
        {"question": "", "answer": "空问题"},  # 空问题，应被过滤
        {"question": "空答案", "answer": ""},  # 空答案，应被过滤
    ]


@pytest.fixture
def sample_chatml_text():
    """示例 ChatML 格式文本"""
    return (
        "<|im_start|>system\n你是助手。<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n你好！有什么可以帮助你的？<|im_end|>\n"
    )


@pytest.fixture
def sample_jsonl_file(temp_dir, sample_alpaca_data):
    """创建临时 jsonl 文件，包含 alpaca 格式数据"""
    filepath = temp_dir / "test_data.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in sample_alpaca_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return filepath


@pytest.fixture
def sample_test_cases():
    """示例测试用例"""
    return [
        {
            "question": "什么是机器学习？",
            "reference": "机器学习是人工智能的分支，使计算机能从数据中学习。",
            "messages": [
                {"role": "system", "content": "你是助手。"},
                {"role": "user", "content": "什么是机器学习？"},
                {"role": "assistant", "content": "机器学习是人工智能的分支。"},
            ],
        },
        {
            "question": "什么是深度学习？",
            "reference": "深度学习使用多层神经网络来学习数据表示。",
            "messages": [
                {"role": "system", "content": "你是助手。"},
                {"role": "user", "content": "什么是深度学习？"},
                {"role": "assistant", "content": "深度学习使用多层神经网络。"},
            ],
        },
    ]
