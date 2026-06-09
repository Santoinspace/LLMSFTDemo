"""测试 eval/eval_report.py 的 HTML 报告生成逻辑"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from eval_report import (
    _build_summary,
    _build_metrics_table,
    _build_sample_comparisons,
    _build_chart_script,
    _calc_change,
    create_html_report,
    HTML_TEMPLATE,
)


class TestCalcChange:
    """测试变化百分比计算"""

    def test_improvement(self):
        assert _calc_change(0.5, 0.6) == "+20.0%"

    def test_decline(self):
        assert _calc_change(0.6, 0.5) == "-16.7%"

    def test_reverse_mode(self):
        """PPL 下降是好事，所以反向计算"""
        change = _calc_change(100, 80, reverse=True)
        # PPL 从 100 降到 80, 变化是 -20%, 反向后是 +20%
        assert "+" in change

    def test_zero_base(self):
        assert _calc_change(0.0, 0.5) == "N/A"


class TestBuildSummary:
    """测试摘要卡片构建"""

    def test_build_summary(self):
        base = {
            "perplexity": 100.0,
            "rougeL": 0.3,
            "tokens_per_sec": 50.0,
            "per_sample": [{}] * 5,
        }
        finetuned = {
            "perplexity": 80.0,
            "rougeL": 0.45,
            "tokens_per_sec": 48.0,
            "per_sample": [{}] * 5,
        }
        items = _build_summary(base, finetuned)

        assert len(items) == 4
        # PPL 下降是改善
        ppi = items[0]
        assert "PPL" in ppi["label"]

    def test_build_summary_zero_perplexity(self):
        base = {"perplexity": 0, "rougeL": 0, "tokens_per_sec": 0, "per_sample": []}
        finetuned = {"perplexity": 0, "rougeL": 0, "tokens_per_sec": 0, "per_sample": []}
        items = _build_summary(base, finetuned)

        # 全零时不应崩溃
        for item in items:
            assert "value" in item
            assert "label" in item


class TestBuildMetricsTable:
    """测试指标表格构建"""

    def test_table_contains_all_metrics(self):
        base = {
            "perplexity": 100, "rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.35,
            "bleu4": 0.15, "keyword_accuracy": 0.5, "overlap_accuracy": 0.6,
            "avg_length": 80, "avg_tokens": 20, "tokens_per_sec": 45,
        }
        finetuned = {
            "perplexity": 80, "rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.48,
            "bleu4": 0.25, "keyword_accuracy": 0.7, "overlap_accuracy": 0.75,
            "avg_length": 100, "avg_tokens": 25, "tokens_per_sec": 44,
        }
        html = _build_metrics_table(base, finetuned)

        assert "<table>" in html
        assert "Perplexity" in html
        assert "ROUGE-1" in html
        assert "ROUGE-L" in html
        assert "BLEU-4" in html
        assert "推理速度" in html


class TestBuildSampleComparisons:
    """测试案例对比构建"""

    def test_build_comparisons(self):
        base_samples = [
            {"question": "Q1", "reference": "R1", "prediction": "Base猜的"},
            {"question": "Q2", "reference": "R2", "prediction": "Base猜的2"},
        ]
        ft_samples = [
            {"question": "Q1", "reference": "R1", "prediction": "FT猜的"},
            {"question": "Q2", "reference": "R2", "prediction": "FT猜的2"},
        ]
        html = _build_sample_comparisons(base_samples, ft_samples)

        assert "Q1" in html
        assert "Base猜的" in html
        assert "FT猜的" in html

    def test_unequal_samples(self):
        """不相等长度的样本列表"""
        base_samples = [{"question": "Q1", "reference": "R1", "prediction": "P1"}]
        ft_samples = []
        html = _build_sample_comparisons(base_samples, ft_samples)
        # 不应崩溃
        assert len(html) > 0


class TestBuildChartScript:
    """测试图表脚本构建"""

    def test_valid_javascript(self):
        base = {"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.35, "bleu4": 0.15,
                "keyword_accuracy": 0.5, "overlap_accuracy": 0.6}
        finetuned = {"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.48, "bleu4": 0.25,
                     "keyword_accuracy": 0.7, "overlap_accuracy": 0.75}

        script = _build_chart_script(base, finetuned)

        assert "Plotly.newPlot" in script
        assert "trace1" in script
        assert "trace2" in script


class TestCreateHTMLReport:
    """测试完整 HTML 报告生成"""

    def test_generate_report(self, temp_dir):
        results = {
            "base_model": {
                "perplexity": 100.0,
                "rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.35,
                "bleu4": 0.15, "keyword_accuracy": 0.5, "overlap_accuracy": 0.6,
                "avg_length": 80, "avg_tokens": 20, "tokens_per_sec": 45.0,
                "per_sample": [
                    {"question": "Q1", "reference": "R1", "prediction": "P1"},
                ],
            },
            "finetuned_model": {
                "perplexity": 80.0,
                "rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.48,
                "bleu4": 0.25, "keyword_accuracy": 0.7, "overlap_accuracy": 0.75,
                "avg_length": 100, "avg_tokens": 25, "tokens_per_sec": 44.0,
                "per_sample": [
                    {"question": "Q1", "reference": "R1", "prediction": "P1_ft"},
                ],
            },
        }

        output_path = temp_dir / "report.html"
        create_html_report(results, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # 验证 HTML 结构
        assert "<!DOCTYPE html>" in content
        assert "Plotly.newPlot" in content
        assert "Q1" in content
        assert "Base Model" in content
        assert "Fine-tuned" in content

    def test_html_template_formatting(self):
        """确保 HTML 模板格式化不会出错"""
        # 模板中的 {{ }} 是转义的花括号，测试 format 调用不抛异常
        try:
            HTML_TEMPLATE.format(
                summary_cards="test",
                metrics_table="test",
                sample_comparisons="test",
                chart_script="test",
            )
        except Exception:
            pytest.fail("HTML_TEMPLATE.format 抛出异常")
