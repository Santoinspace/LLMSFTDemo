"""
评估报告生成模块

生成包含以下内容的 HTML 报告：
- 指标对比柱状图（base vs finetuned）
- 逐条案例对比
- 指标变化百分比汇总表
- 使用 plotly 生成交互式图表
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

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
    <title>Qwen3-1.7B QLoRA 微调评估报告</title>
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
        th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #0f3460; color: white; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .improvement {{ color: #00b894; font-weight: bold; }}
        .decline {{ color: #e17055; font-weight: bold; }}
        .neutral {{ color: #636e72; }}
        .sample-block {{ margin: 15px 0; padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #0f3460; }}
        .sample-block .question {{ font-weight: bold; color: #0f3460; margin-bottom: 8px; }}
        .sample-block .label {{ font-size: 0.85em; color: #888; margin-bottom: 2px; }}
        .sample-block .content {{ white-space: pre-wrap; background: white; padding: 10px; border-radius: 4px; margin-bottom: 10px; border: 1px solid #e0e0e0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .summary-item {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .summary-item .value {{ font-size: 2em; font-weight: bold; }}
        .summary-item .label {{ font-size: 0.85em; opacity: 0.9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Qwen3-1.7B QLoRA 微调评估报告</h1>

        <div class="summary-grid">
            {summary_cards}
        </div>

        <div class="card">
            <h2>指标对比</h2>
            <div id="metrics-chart" class="chart-container"></div>
        </div>

        <div class="card">
            <h2>指标变化汇总</h2>
            {metrics_table}
        </div>

        <div class="card">
            <h2>逐条案例对比</h2>
            {sample_comparisons}
        </div>
    </div>

    <script>
        {chart_script}
    </script>
</body>
</html>"""


def create_html_report(results: Dict, output_path: Path) -> None:
    """
    生成完整的 HTML 评估报告

    参数:
        results: evaluate.py 输出的结果字典，包含 base_model 和 finetuned_model
        output_path: HTML 报告输出路径
    """
    base = results.get("base_model", {})
    finetuned = results.get("finetuned_model", {})

    # 生成摘要卡片
    summary_items = _build_summary(base, finetuned)
    summary_html = "\n".join(
        f'<div class="summary-item"><div class="value">{item["value"]}</div><div class="label">{item["label"]}</div></div>'
        for item in summary_items
    )

    # 生成指标对比表
    metrics_html = _build_metrics_table(base, finetuned)

    # 生成案例对比
    base_samples = base.get("per_sample", [])
    ft_samples = finetuned.get("per_sample", [])
    samples_html = _build_sample_comparisons(base_samples, ft_samples)

    # 生成图表脚本
    chart_js = _build_chart_script(base, finetuned)

    # 组装 HTML
    html = HTML_TEMPLATE.format(
        summary_cards=summary_html,
        metrics_table=metrics_html,
        sample_comparisons=samples_html,
        chart_script=chart_js,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML 报告已生成: {output_path}")


def _build_summary(base: Dict, finetuned: Dict) -> List[Dict]:
    """构建摘要卡片"""
    ppl_change = _calc_change(base.get("perplexity", 0), finetuned.get("perplexity", 0), reverse=True)
    rouge_change = _calc_change(base.get("rougeL", 0), finetuned.get("rougeL", 0))
    speed_change = _calc_change(base.get("tokens_per_sec", 0), finetuned.get("tokens_per_sec", 0))

    return [
        {"label": "PPL 变化", "value": ppl_change},
        {"label": "ROUGE-L 变化", "value": rouge_change},
        {"label": "推理速度变化", "value": speed_change},
        {"label": "总测试样本", "value": str(len(base.get("per_sample", [])))},
    ]


def _calc_change(base_val: float, ft_val: float, reverse: bool = False) -> str:
    """计算变化百分比"""
    if base_val == 0:
        return "N/A"
    pct = (ft_val - base_val) / base_val * 100
    if reverse:
        pct = -pct
    arrow = "+" if pct > 0 else ""
    return f"{arrow}{pct:.1f}%"


def _build_metrics_table(base: Dict, finetuned: Dict) -> str:
    """构建指标对比表"""
    metrics_list = [
        ("Perplexity", "perplexity", "down"),
        ("ROUGE-1", "rouge1", "up"),
        ("ROUGE-2", "rouge2", "up"),
        ("ROUGE-L", "rougeL", "up"),
        ("BLEU-4", "bleu4", "up"),
        ("关键词准确率", "keyword_accuracy", "up"),
        ("内容重叠率", "overlap_accuracy", "up"),
        ("平均生成长度", "avg_length", "neutral"),
        ("平均Token数", "avg_tokens", "neutral"),
        ("推理速度(tok/s)", "tokens_per_sec", "up"),
    ]

    rows = []
    for label, key, direction in metrics_list:
        b_val = base.get(key, 0)
        ft_val = finetuned.get(key, 0)
        change = _calc_change(b_val, ft_val, reverse=(direction == "down"))

        css_class = "neutral"
        if change != "N/A":
            try:
                val = float(change.replace("%", "").replace("+", ""))
                if val > 2:
                    css_class = "improvement"
                elif val < -2:
                    css_class = "decline"
            except ValueError:
                pass

        rows.append(
            f"<tr><td>{label}</td><td>{b_val}</td><td>{ft_val}</td>"
            f"<td class='{css_class}'>{change}</td></tr>"
        )

    header = "<tr><th>指标</th><th>Base Model</th><th>Fine-tuned</th><th>变化</th></tr>"
    return f"<table>{header}{''.join(rows)}</table>"


def _build_sample_comparisons(base_samples: List, ft_samples: List) -> str:
    """构建案例对比 HTML"""
    html_parts = []
    max_n = max(len(base_samples), len(ft_samples))
    max_show = min(10, max_n)

    for i in range(max_show):
        bs = base_samples[i] if i < len(base_samples) else {}
        fs = ft_samples[i] if i < len(ft_samples) else {}

        question = bs.get("question", fs.get("question", ""))
        # 截断过长的问题
        if len(question) > 200:
            question = question[:200] + "..."

        html_parts.append(f"""
        <div class="sample-block">
            <div class="question">[{i + 1}] {question}</div>

            <div class="label">参考答案:</div>
            <div class="content">{bs.get("reference", "")[:300]}</div>

            <div class="label">基座模型输出:</div>
            <div class="content">{bs.get("prediction", "")[:300]}</div>

            <div class="label">微调模型输出:</div>
            <div class="content">{fs.get("prediction", "")[:300]}</div>
        </div>
        """)

    return "\n".join(html_parts)


def _build_chart_script(base: Dict, finetuned: Dict) -> str:
    """构建 Plotly 图表脚本"""
    metrics = [
        ("ROUGE-1", "rouge1"),
        ("ROUGE-2", "rouge2"),
        ("ROUGE-L", "rougeL"),
        ("BLEU-4", "bleu4"),
        ("关键词准确率", "keyword_accuracy"),
        ("内容重叠率", "overlap_accuracy"),
    ]

    labels = [m[0] for m in metrics]
    base_values = [base.get(m[1], 0) for m in metrics]
    ft_values = [finetuned.get(m[1], 0) for m in metrics]

    return f"""
    var trace1 = {{
        x: {json.dumps(labels)},
        y: {json.dumps(base_values)},
        name: 'Base Model',
        type: 'bar',
        marker: {{ color: '#636e72' }}
    }};

    var trace2 = {{
        x: {json.dumps(labels)},
        y: {json.dumps(ft_values)},
        name: 'Fine-tuned',
        type: 'bar',
        marker: {{ color: '#0f3460' }}
    }};

    var layout = {{
        barmode: 'group',
        margin: {{ t: 20, r: 20, l: 50, b: 80 }},
        yaxis: {{ title: 'Score', range: [0, 1] }},
        legend: {{ orientation: 'h', y: 1.1 }},
    }};

    Plotly.newPlot('metrics-chart', [trace1, trace2], layout, {{ responsive: true }});
    """


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成评估报告")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="评估结果 JSON 文件路径（evaluate.py 输出）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_outputs/eval_report.html",
        help="HTML 报告输出路径",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"结果文件不存在: {results_path}")
        exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    create_html_report(results, Path(args.output))
    logger.info(f"报告已生成: {args.output}")
