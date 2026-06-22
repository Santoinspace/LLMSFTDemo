"""
Generate interactive HTML comparison reports from eval JSON output.

Supports both legacy format (evaluate.py) and current format (compare_validate.py).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen3-1.7B QLoRA Fine-tuning Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; margin: 20px 0; color: #1a1a2e; }}
        h2 {{ margin: 30px 0 15px; color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
        .chart-container {{ width: 100%; min-height: 450px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.92em; }}
        th {{ background: #0f3460; color: white; font-weight: 600; white-space: nowrap; }}
        tr:hover {{ background: #f8f9fa; }}
        .improvement {{ color: #00b894; font-weight: bold; }}
        .decline {{ color: #e17055; font-weight: bold; }}
        .neutral {{ color: #636e72; }}
        .sample-block {{ margin: 15px 0; padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #0f3460; }}
        .sample-block .question {{ font-weight: bold; color: #0f3460; margin-bottom: 8px; }}
        .sample-block .label {{ font-size: 0.82em; color: #888; margin-bottom: 2px; }}
        .sample-block .content {{ white-space: pre-wrap; background: white; padding: 10px; border-radius: 4px; margin-bottom: 10px; border: 1px solid #e0e0e0; font-size: 0.9em; max-height: 200px; overflow-y: auto; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0; }}
        .summary-item {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .summary-item .value {{ font-size: 1.6em; font-weight: bold; }}
        .summary-item .label {{ font-size: 0.82em; opacity: 0.9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Qwen3-1.7B QLoRA Fine-tuning Evaluation Report</h1>

        <div class="summary-grid">
            {summary_cards}
        </div>

        <div class="card">
            <h2>Metrics Comparison</h2>
            <div id="metrics-chart" class="chart-container"></div>
        </div>

        <div class="card">
            <h2>Metrics Table</h2>
            {metrics_table}
        </div>

        <div class="card">
            <h2>Sample Comparisons</h2>
            {sample_comparisons}
        </div>
    </div>

    <script>
        {chart_script}
    </script>
</body>
</html>"""

GROUP_COLORS = {
    "base:no_rag": "#95a5a6",
    "base:rag": "#3498db",
    "finetuned:no_rag": "#9b59b6",
    "finetuned:rag": "#2ecc71",
    "base": "#95a5a6",
    "finetuned": "#9b59b6",
}


def _detect_format(data: dict) -> str:
    """Return 'compare_validate' or 'evaluate' based on structure."""
    if "results" in data:
        return "compare_validate"
    if "base_model" in data or "finetuned_model" in data:
        return "evaluate"
    return "compare_validate"


def _flatten_groups(data: dict) -> List[dict]:
    """
    Normalise both formats into a flat list of groups:
    [{"key": "base:no_rag", "label": "Base (no RAG)", "metrics": {...}, "samples": [...]}, ...]
    """
    fmt = _detect_format(data)

    if fmt == "compare_validate":
        results = data.get("results", {})
        groups = []
        for model_label in ["base", "finetuned"]:
            model_data = results.get(model_label)
            if not model_data:
                continue
            metrics = model_data.get("metrics", {})
            samples = model_data.get("samples", [])
            for mode in ["no_rag", "rag"]:
                if mode not in metrics:
                    continue
                key = f"{model_label}:{mode}"
                if model_label == "base" and mode == "no_rag":
                    group_label = "Base (no RAG)"
                elif model_label == "base" and mode == "rag":
                    group_label = "Base + RAG"
                elif model_label == "finetuned" and mode == "no_rag":
                    group_label = "FT (no RAG)"
                else:
                    group_label = "FT + RAG"
                groups.append({
                    "key": key,
                    "label": group_label,
                    "metrics": metrics[mode],
                    "samples": [
                        {"question": s.get("question", ""),
                         "reference": s.get("reference", ""),
                         "prediction": s.get(mode, s.get("prediction", ""))}
                        for s in samples
                    ],
                })
        return groups

    # Legacy evaluate.py format
    groups = []
    for model_key, model_label in [("base_model", "Base"), ("finetuned_model", "Finetuned")]:
        model_data = data.get(model_key, {})
        if not model_data:
            continue
        groups.append({
            "key": model_label.lower(),
            "label": model_label,
            "metrics": {k: v for k, v in model_data.items() if not isinstance(v, list)},
            "samples": model_data.get("per_sample", []),
        })
    return groups


def create_html_report(report_data: dict, output_path: Path) -> None:
    groups = _flatten_groups(report_data)
    if not groups:
        logger.warning("No groups found in report data; skipping HTML.")
        return

    summary_html = _build_summary(groups)
    metrics_html = _build_metrics_table(groups)
    samples_html = _build_sample_comparisons(groups)
    chart_js = _build_chart_script(groups)

    html = HTML_TEMPLATE.format(
        summary_cards=summary_html,
        metrics_table=metrics_html,
        sample_comparisons=samples_html,
        chart_script=chart_js,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved: %s", output_path)


def _build_summary(groups: List[dict]) -> str:
    items = []
    if len(groups) >= 2:
        base_metrics = groups[0]["metrics"]
        ft_metrics = groups[-1]["metrics"]  # Last group is usually the best finetuned
        rl_key = next((k for k in ["rougeL_f", "rougeL"] if k in ft_metrics), None)
        if rl_key:
            delta = ft_metrics[rl_key] - base_metrics.get(rl_key, 0)
            items.append({"label": "ROUGE-L Delta", "value": f"{delta:+.3f}"})
        bs_key = next((k for k in ["bertscore_f1"] if k in ft_metrics), None)
        if bs_key:
            items.append({"label": "BERTScore F1 (best)", "value": f'{ft_metrics[bs_key]:.3f}'})

    items.append({"label": "Groups", "value": str(len(groups))})
    items.append({"label": "Samples", "value": str(len(groups[0]["samples"]) if groups else 0)})

    return "\n".join(
        f'<div class="summary-item"><div class="value">{item["value"]}</div>'
        f'<div class="label">{item["label"]}</div></div>'
        for item in items
    )


def _pct_change(base_val: float, ft_val: float) -> str:
    if base_val == 0:
        return "N/A"
    return f"{((ft_val - base_val) / base_val * 100):+.1f}%"


def _build_metrics_table(groups: List[dict]) -> str:
    # Collect all metric keys across all groups
    metric_keys = []
    seen = set()
    for g in groups:
        for k in g["metrics"]:
            if k not in seen:
                metric_keys.append(k)
                seen.add(k)

    # Sort: F1/P/R groups
    priority_order = [
        "rouge1_f", "rouge1_p", "rouge1_r", "rouge2_f", "rouge2_p", "rouge2_r",
        "rougeL_f", "rougeL_p", "rougeL_r", "bleu4", "bertscore_f1",
        "avg_length", "avg_tokens", "tokens_per_sec",
    ]
    ordered = [k for k in priority_order if k in seen]
    ordered += [k for k in metric_keys if k not in ordered]
    # Legacy aliases
    key_labels = {
        "rouge1_f": "ROUGE-1 F1", "rouge1_p": "ROUGE-1 P", "rouge1_r": "ROUGE-1 R",
        "rouge2_f": "ROUGE-2 F1", "rouge2_p": "ROUGE-2 P", "rouge2_r": "ROUGE-2 R",
        "rougeL_f": "ROUGE-L F1", "rougeL_p": "ROUGE-L P", "rougeL_r": "ROUGE-L R",
        "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
        "bleu4": "BLEU-4", "bertscore_f1": "BERTScore F1",
        "overlap_accuracy": "Overlap Acc", "keyword_accuracy": "Keyword Acc",
        "avg_length": "Avg Length", "avg_tokens": "Avg Tokens", "tokens_per_sec": "Tok/s",
    }

    header = "<tr><th>Metric</th>" + "".join(f"<th>{g['label']}</th>" for g in groups)
    if len(groups) == 2:
        header += "<th>Change</th>"
    header += "</tr>"

    rows = []
    for key in ordered:
        vals = [g["metrics"].get(key, "-") for g in groups]
        label = key_labels.get(key, key)
        row = f"<tr><td>{label}</td>"
        for v in vals:
            if isinstance(v, float):
                row += f"<td>{v:.4f}</td>"
            else:
                row += f"<td>{v}</td>"
        if len(groups) == 2 and isinstance(vals[0], (int, float)) and isinstance(vals[1], (int, float)):
            if vals[0] == 0:
                row += '<td class="neutral">N/A</td>'
            else:
                pct = (vals[1] - vals[0]) / vals[0] * 100
                css = "improvement" if pct > 0 else "decline"
                row += f'<td class="{css}">{pct:+.1f}%</td>'
        rows.append(row + "</tr>")

    return f"<table>{header}{''.join(rows)}</table>"


def _build_sample_comparisons(groups: List[dict]) -> str:
    if not groups or not groups[0].get("samples"):
        return "<p>No sample-level data available.</p>"

    samples_by_idx = groups[0]["samples"]
    max_show = min(10, len(samples_by_idx))

    parts = []
    for i in range(max_show):
        q = samples_by_idx[i].get("question", "")
        if len(q) > 150:
            q = q[:150] + "..."

        parts.append(f'<div class="sample-block"><div class="question">[{i + 1}] {q}</div>')
        for g in groups:
            s = g["samples"][i] if i < len(g["samples"]) else {}
            pred = s.get("prediction", s.get(g["key"].split(":")[-1] if ":" in g["key"] else "prediction", ""))
            parts.append(
                f'<div class="label">{g["label"]}:</div>'
                f'<div class="content">{pred[:400] if pred else "(empty)"}</div>'
            )
        parts.append("</div>")

    return "\n".join(parts)


def _build_chart_script(groups: List[dict]) -> str:
    # Chart-worthy metrics: F1 scores only
    chart_metrics = []
    for k in ["rouge1_f", "rouge2_f", "rougeL_f", "bleu4", "bertscore_f1"]:
        if any(k in g["metrics"] for g in groups):
            chart_metrics.append(k)
    if not chart_metrics:
        # Fallback for legacy format
        chart_metrics = ["rouge1", "rouge2", "rougeL", "bleu4"]

    metric_labels = {
        "rouge1_f": "ROUGE-1", "rouge2_f": "ROUGE-2", "rougeL_f": "ROUGE-L",
        "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
        "bleu4": "BLEU-4", "bertscore_f1": "BERTScore",
    }

    traces = []
    for g in groups:
        traces.append({
            "x": [metric_labels.get(k, k) for k in chart_metrics],
            "y": [g["metrics"].get(k, 0) for k in chart_metrics],
            "name": g["label"],
            "type": "bar",
            "marker": {"color": GROUP_COLORS.get(g["key"], "#636e72")},
        })

    return f"""
    var data = {json.dumps(traces)};
    var layout = {{
        barmode: 'group',
        margin: {{ t: 20, r: 20, l: 50, b: 80 }},
        yaxis: {{ title: 'Score', range: [0, 1] }},
        legend: {{ orientation: 'h', y: 1.15 }},
    }};
    Plotly.newPlot('metrics-chart', data, layout, {{ responsive: true }});
    """


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation HTML report")
    parser.add_argument("--results", type=str, required=True, help="Path to eval JSON file")
    parser.add_argument("--output", type=str, default=None, help="HTML output path (default: same dir, .html)")

    args = parser.parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = Path(args.output) if args.output else results_path.with_suffix(".html")
    create_html_report(data, output)
    logger.info("Done: %s", output)
