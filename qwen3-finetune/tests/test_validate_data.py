"""测试 data/validate_data.py 的数据校验逻辑"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from validate_data import (
    validate_chatml_format,
    validate_single_sample,
)


class TestValidateChatMLFormat:
    """测试 ChatML 格式校验"""

    def test_valid_format(self):
        text = (
            "<|im_start|>system\n你是助手。<|im_end|>\n"
            "<|im_start|>user\n你好<|im_end|>\n"
            "<|im_start|>assistant\n你好！<|im_end|>\n"
        )
        valid, err = validate_chatml_format(text)
        assert valid is True
        assert err == ""

    def test_missing_im_start(self):
        text = "system\n你是助手。<|im_end|>\n"
        valid, err = validate_chatml_format(text)
        assert valid is False
        assert "im_start" in err

    def test_missing_im_end(self):
        text = "<|im_start|>system\n你是助手。\n"
        valid, err = validate_chatml_format(text)
        assert valid is False
        assert "im_end" in err

    def test_missing_user(self):
        text = (
            "<|im_start|>system\n你是助手。<|im_end|>\n"
            "<|im_start|>assistant\n只有system和assistant<|im_end|>\n"
        )
        valid, err = validate_chatml_format(text)
        assert valid is False
        assert "user" in err

    def test_missing_assistant(self):
        text = (
            "<|im_start|>system\n你是助手。<|im_end|>\n"
            "<|im_start|>user\n只有system和user<|im_end|>\n"
        )
        valid, err = validate_chatml_format(text)
        assert valid is False
        assert "assistant" in err

    def test_unbalanced_tags(self):
        # 4 个 start, 2 个 end → 差 2，应触发不配对
        text = (
            "<|im_start|>system\n你是助手。<|im_end|>\n"
            "<|im_start|>user\n你好<|im_end|>\n"
            "<|im_start|>assistant\n回答1\n"
            "<|im_start|>user\n问题2\n"
        )
        valid, err = validate_chatml_format(text)
        assert valid is False
        assert "不配对" in err

    def test_generation_mode_allows_extra_start(self):
        """最后一条 assistant 没有 im_end 时（用于生成），start 可以比 end 多 1"""
        text = (
            "<|im_start|>system\n你是助手。<|im_end|>\n"
            "<|im_start|>user\n你好<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        valid, err = validate_chatml_format(text)
        # start=3, end=2, 差 1 是允许的
        assert valid is True


class TestValidateSingleSample:
    """测试单条样本校验"""

    def test_valid_sample(self):
        line = json.dumps({
            "text": (
                "<|im_start|>system\n你是助手。<|im_end|>\n"
                "<|im_start|>user\n你好<|im_end|>\n"
                "<|im_start|>assistant\n你好！<|im_end|>\n"
            )
        }, ensure_ascii=False)
        valid, errors = validate_single_sample(line, 1, 512)
        assert valid is True
        assert len(errors) == 0

    def test_bad_json(self):
        line = "not a json string {{{"
        valid, errors = validate_single_sample(line, 1, 512)
        assert valid is False
        assert len(errors) == 1
        assert "JSON" in errors[0]

    def test_missing_text_field(self):
        line = json.dumps({"other_field": "some value"})
        valid, errors = validate_single_sample(line, 1, 512)
        assert valid is False
        assert any("text" in e for e in errors)

    def test_empty_text(self):
        line = json.dumps({"text": ""})
        valid, errors = validate_single_sample(line, 1, 512)
        assert valid is False
        assert any("为空" in e for e in errors)

    def test_too_short(self):
        line = json.dumps({"text": "hi"})
        valid, errors = validate_single_sample(line, 1, 512)
        assert valid is False
        assert any("过短" in e for e in errors)

    def test_garbled(self):
        line = json.dumps({
            "text": (
                "<|im_start|>system\n助手<|im_end|>\n"
                "<|im_start|>user\n问题<|im_end|>\n"
                "<|im_start|>assistant\n回答����乱码<|im_end|>\n"
            )
        }, ensure_ascii=False)
        valid, errors = validate_single_sample(line, 1, 512)
        assert valid is False
        assert any("乱码" in e for e in errors)
