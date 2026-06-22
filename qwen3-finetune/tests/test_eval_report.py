"""测试 eval/eval_report.py 的 HTML 报告生成逻辑（新版 + 旧版兼容）"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from eval_report import (
    _build_summary,
    _build_metrics_table,
    _build_sample_comparisons,
    _build_chart_script,
    _detect_format,
    _flatten_groups,
    create_html_report,
    HTML_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Format detection & normalisation
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_compare_validate_format(self):
        data = {"results": {"base": {}, "finetuned": {}}}
        assert _detect_format(data) == "compare_validate"

    def test_evaluate_format(self):
        data = {"base_model": {}, "finetuned_model": {}}
        assert _detect_format(data) == "evaluate"


class TestFlattenGroups:
    """分组归一化"""

    def test_compare_validate_groups(self):
        data = {
            "results": {
                "base": {
                    "model_path": "Qwen/Qwen3-1.7B",
                    "metrics": {
                        "no_rag": {"rouge1_f": 0.26},
                        "rag": {"rouge1_f": 0.39},
                    },
                    "samples": [
                        {"question": "Q1", "reference": "R1", "no_rag": "P1_no", "rag": "P1_rag"},
                    ],
                },
                "finetuned": {
                    "model_path": "outputs/merged_model",
                    "metrics": {
                        "no_rag": {"rouge1_f": 0.43},
                        "rag": {"rouge1_f": 0.48},
                    },
                    "samples": [
                        {"question": "Q1", "reference": "R1", "no_rag": "P1_ft_no", "rag": "P1_ft_rag"},
                    ],
                },
            },
        }
        groups = _flatten_groups(data)
        assert len(groups) == 4
        keys = [g["key"] for g in groups]
        assert "base:no_rag" in keys
        assert "base:rag" in keys
        assert "finetuned:no_rag" in keys
        assert "finetuned:rag" in keys

    def test_legacy_evaluate_groups(self):
        data = {
            "base_model": {"rouge1": 0.4, "rougeL": 0.35, "per_sample": []},
            "finetuned_model": {"rouge1": 0.5, "rougeL": 0.48, "per_sample": []},
        }
        groups = _flatten_groups(data)
        assert len(groups) == 2
        assert groups[0]["key"] == "base"
        assert groups[1]["key"] == "finetuned"


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_new_format(self):
        groups = [
            {"key": "base:no_rag", "label": "Base (no RAG)",
             "metrics": {"rougeL_f": 0.16}, "samples": [{}] * 5},
            {"key": "finetuned:no_rag", "label": "FT (no RAG)",
             "metrics": {"rougeL_f": 0.30, "bertscore_f1": 0.62}, "samples": [{}] * 5},
        ]
        html = _build_summary(groups)
        assert "ROUGE-L Delta" in html
        assert "BERTScore F1" in html
        assert "Groups" in html


# ---------------------------------------------------------------------------
# _build_metrics_table
# ---------------------------------------------------------------------------

class TestBuildMetricsTable:
    def test_new_format_table(self):
        groups = [
            {"key": "base:no_rag", "label": "Base (no RAG)",
             "metrics": {"rouge1_f": 0.264, "rougeL_f": 0.163, "bleu4": 0.023, "bertscore_f1": 0.45}},
            {"key": "finetuned:no_rag", "label": "FT (no RAG)",
             "metrics": {"rouge1_f": 0.429, "rougeL_f": 0.300, "bleu4": 0.105, "bertscore_f1": 0.60}},
        ]
        html = _build_metrics_table(groups)
        assert "<table>" in html
        assert "ROUGE-1 F1" in html
        assert "BLEU-4" in html
        assert "BERTScore F1" in html
        # Change column (2 groups only)
        assert "Change" in html


# ---------------------------------------------------------------------------
# _build_sample_comparisons
# ---------------------------------------------------------------------------

class TestBuildSampleComparisons:
    def test_new_format_samples(self):
        groups = [
            {"key": "base:no_rag", "label": "Base (no RAG)",
             "metrics": {}, "samples": [
                 {"question": "Q1", "reference": "R1", "prediction": "Base answer"},
             ]},
            {"key": "finetuned:no_rag", "label": "FT (no RAG)",
             "metrics": {}, "samples": [
                 {"question": "Q1", "reference": "R1", "prediction": "FT answer"},
             ]},
        ]
        html = _build_sample_comparisons(groups)
        assert "Q1" in html
        assert "Base answer" in html
        assert "FT answer" in html


# ---------------------------------------------------------------------------
# _build_chart_script
# ---------------------------------------------------------------------------

class TestBuildChartScript:
    def test_valid_javascript(self):
        groups = [
            {"key": "base:no_rag", "label": "Base (no RAG)",
             "metrics": {"rouge1_f": 0.26, "rougeL_f": 0.16, "bleu4": 0.02, "bertscore_f1": 0.45}},
            {"key": "finetuned:no_rag", "label": "FT (no RAG)",
             "metrics": {"rouge1_f": 0.43, "rougeL_f": 0.30, "bleu4": 0.10, "bertscore_f1": 0.60}},
        ]
        script = _build_chart_script(groups)
        assert "Plotly.newPlot" in script
        assert "Base (no RAG)" in script
        assert "FT (no RAG)" in script


# ---------------------------------------------------------------------------
# create_html_report
# ---------------------------------------------------------------------------

class TestCreateHTMLReport:
    def test_generate_new_format_report(self, temp_dir):
        data = {
            "results": {
                "base": {
                    "model_path": "Qwen/Qwen3-1.7B",
                    "metrics": {
                        "no_rag": {"rouge1_f": 0.26, "rouge2_f": 0.07, "rougeL_f": 0.16,
                                   "bleu4": 0.02, "bertscore_f1": 0.43, "avg_length": 800.0,
                                   "avg_tokens": 160.0, "tokens_per_sec": 81.0},
                    },
                    "samples": [
                        {"question": "Q1", "reference": "R1", "no_rag": "P1_base"},
                    ],
                },
                "finetuned": {
                    "model_path": "outputs/merged",
                    "metrics": {
                        "no_rag": {"rouge1_f": 0.43, "rouge2_f": 0.19, "rougeL_f": 0.30,
                                   "bleu4": 0.10, "bertscore_f1": 0.62, "avg_length": 376.0,
                                   "avg_tokens": 99.0, "tokens_per_sec": 58.0},
                    },
                    "samples": [
                        {"question": "Q1", "reference": "R1", "no_rag": "P1_ft"},
                    ],
                },
            },
        }
        output_path = temp_dir / "report_new.html"
        create_html_report(data, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Plotly.newPlot" in content
        assert "Q1" in content
        assert "Base (no RAG)" in content
        assert "FT (no RAG)" in content

    def test_generate_legacy_format_report(self, temp_dir):
        """Backward-compatible with old evaluate.py output"""
        data = {
            "base_model": {
                "rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.35,
                "bleu4": 0.15, "avg_length": 80, "avg_tokens": 20, "tokens_per_sec": 45.0,
                "per_sample": [{"question": "Q1", "reference": "R1", "prediction": "P1"}],
            },
            "finetuned_model": {
                "rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.48,
                "bleu4": 0.25, "avg_length": 100, "avg_tokens": 25, "tokens_per_sec": 44.0,
                "per_sample": [{"question": "Q1", "reference": "R1", "prediction": "P1_ft"}],
            },
        }
        output_path = temp_dir / "report_legacy.html"
        create_html_report(data, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Plotly.newPlot" in content
        assert "Q1" in content

    def test_html_template_formatting(self):
        try:
            HTML_TEMPLATE.format(
                summary_cards="test",
                metrics_table="test",
                sample_comparisons="test",
                chart_script="test",
            )
        except Exception:
            pytest.fail("HTML_TEMPLATE.format raised exception")
