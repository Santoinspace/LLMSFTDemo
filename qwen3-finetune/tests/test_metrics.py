"""测试 eval/metrics.py 的评估指标计算逻辑"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from metrics import (
    compute_bleu,
    compute_rouge,
    compute_domain_accuracy,
    compute_generation_stats,
    _tokenize_chinese,
)


class TestChineseTokenizer:
    """测试中文分词辅助函数"""

    def test_pure_chinese(self):
        tokens = _tokenize_chinese("机器学习很有意思")
        assert "机" in tokens
        assert "器" in tokens
        assert "学" in tokens
        assert "习" in tokens

    def test_pure_english(self):
        tokens = _tokenize_chinese("hello world foo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_mixed_text(self):
        tokens = _tokenize_chinese("Python是一种编程语言")
        assert "Python" in tokens
        assert "是" in tokens
        assert "一" in tokens
        assert "种" in tokens


class TestRouge:
    """测试 ROUGE 计算"""

    def test_perfect_match(self):
        # ROUGE scorer 需要以空格分隔的 token
        preds = ["machine learning is a branch of artificial intelligence"]
        refs = ["machine learning is a branch of artificial intelligence"]
        scores = compute_rouge(preds, refs)

        assert scores["rouge1"] > 0.9
        assert scores["rougeL"] > 0.9

    def test_no_overlap(self):
        preds = ["今天天气很好。"]
        refs = ["机器学习是人工智能的分支。"]
        scores = compute_rouge(preds, refs)

        assert scores["rouge1"] < 0.1

    def test_multiple_samples(self):
        preds = ["深度学习是ML分支。", "NLP是AI领域。"]
        refs = ["深度学习是机器学习的分支。", "自然语言处理是AI领域。"]
        scores = compute_rouge(preds, refs)

        for key in ["rouge1", "rouge2", "rougeL"]:
            assert 0.0 <= scores[key] <= 1.0

    def test_empty_predictions(self):
        scores = compute_rouge([], [])
        assert scores["rouge1"] == 0.0
        assert scores["rouge2"] == 0.0
        assert scores["rougeL"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            compute_rouge(["a", "b"], ["c"])


class TestBleu:
    """测试 BLEU-4 计算"""

    def test_perfect_match(self):
        preds = ["机器学习是人工智能的分支"]
        refs = ["机器学习是人工智能的分支"]
        score = compute_bleu(preds, refs)
        # 完全重叠应该接近 1
        assert 0.5 <= score <= 1.0

    def test_no_overlap(self):
        preds = ["今天天气很好今天天气很好今天天气很好今天天气很好"]
        refs = ["机器学习是人工智能的分支机器学习是人工智能的分支机器"]
        score = compute_bleu(preds, refs)
        assert score < 0.2

    def test_short_predictions_skipped(self):
        """少于 4 个 token 的预测应被跳过，返回 0"""
        preds = ["短"]
        refs = ["足够长的参考文本在这里"]
        score = compute_bleu(preds, refs)
        assert score == 0.0

    def test_multiple_samples(self):
        preds = ["深度学习是机器学习子领域。"] * 3
        refs = ["深度学习是机器学习的分支领域。"] * 3
        score = compute_bleu(preds, refs)
        assert 0.0 <= score <= 1.0


class TestDomainAccuracy:
    """测试领域准确率"""

    def test_keyword_match(self):
        preds = ["机器学习是人工智能的子领域，包含深度学习和强化学习。"]
        refs = ["机器学习是人工智能的分支，包含深度学习和强化学习。"]
        keywords = ["人工智能", "深度学习", "机器学习"]

        result = compute_domain_accuracy(preds, refs, keywords)
        assert "keyword_accuracy" in result
        assert 0.0 <= result["keyword_accuracy"] <= 1.0
        # "人工智能" 和 "深度学习" 都在 pred 中，所以至少有命中
        assert result["keyword_accuracy"] >= 0.5

    def test_overlap_calculation(self):
        preds = ["机器学习是AI领域的技术。"]
        refs = ["机器学习是人工智能的领域。"]
        result = compute_domain_accuracy(preds, refs)

        assert "overlap_accuracy" in result
        assert 0.0 <= result["overlap_accuracy"] <= 1.0

    def test_empty_input(self):
        result = compute_domain_accuracy([], [])
        assert result["overlap_accuracy"] == 0.0


class TestGenerationStats:
    """测试生成统计"""

    def test_basic_stats(self):
        preds = ["短回答", "这是一个较长的生成回答"]
        times = [0.5, 1.0]
        tokens = [10, 25]

        stats = compute_generation_stats(preds, times, tokens)

        assert stats["avg_length"] > 0
        assert stats["avg_tokens"] == 17.5
        assert stats["tokens_per_sec"] > 20  # 35 tokens / 1.5s

    def test_zero_time_handling(self):
        preds = ["测试"]
        times = [0.0]
        tokens = [5]

        stats = compute_generation_stats(preds, times, tokens)
        # 不应除零错误
        assert stats["tokens_per_sec"] > 0
