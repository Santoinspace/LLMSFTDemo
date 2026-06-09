"""
微调前后对比评估主脚本

在测试集上分别推理 base_model 和 finetuned_model，计算并对比：
1. Perplexity (PPL)
2. ROUGE-1/2/L
3. BLEU-4
4. 领域准确率
5. 平均生成长度
6. 推理速度
"""
import argparse
import json
import logging
import time
from pathlib import Path

import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, load_in_4bit: bool = True):
    """加载模型和 tokenizer"""
    logger.info(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()

    return model, tokenizer


def load_test_cases(test_path: Path) -> list:
    """加载测试用例"""
    test_cases = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                test_cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    logger.info(f"加载了 {len(test_cases)} 条测试用例")
    return test_cases


def generate_response(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 256,
) -> tuple:
    """生成回复，返回 (生成的文本, 耗时, token数)"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time

    # 只解码新生成的部分
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_tokens = len(new_tokens)

    return generated_text, elapsed, num_tokens


def run_evaluation(
    model,
    tokenizer,
    test_cases: list,
    model_name: str,
) -> dict:
    """在测试集上运行评估"""
    logger.info(f"开始评估 {model_name}...")

    predictions = []
    references = []
    generation_times = []
    token_counts = []
    test_texts = []

    for i, tc in enumerate(test_cases):
        messages = tc.get("messages", [])
        # 构建 user+assistant 对话格式用于测试
        filtered_msgs = [m for m in messages if m["role"] != "system"]
        # 只保留 user 消息
        user_msgs = [m for m in messages if m["role"] == "user"]

        if not user_msgs:
            continue

        reference = tc.get("reference", "")

        # 生成回答
        prediction, elapsed, num_tokens = generate_response(model, tokenizer, messages[:2])

        predictions.append(prediction)
        references.append(reference)
        generation_times.append(elapsed)
        token_counts.append(num_tokens)
        test_texts.append(tc.get("text", ""))

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i + 1}/{len(test_cases)}")

    # 计算指标
    metrics = compute_all_metrics(
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        predictions=predictions,
        references=references,
        generation_times=generation_times,
        token_counts=token_counts,
        max_length=512,
    )

    # 添加逐条结果
    metrics["per_sample"] = [
        {
            "question": test_cases[i].get("question", ""),
            "reference": references[i],
            "prediction": predictions[i],
            "time": round(generation_times[i], 2),
            "tokens": token_counts[i],
        }
        for i in range(len(predictions))
    ]

    logger.info(f"{model_name} 评估完成")
    return metrics


def print_comparison_table(base_metrics: dict, finetuned_metrics: dict) -> None:
    """打印对比表格"""
    metrics_list = [
        "perplexity", "rouge1", "rouge2", "rougeL", "bleu4",
        "keyword_accuracy", "overlap_accuracy",
        "avg_length", "avg_tokens", "tokens_per_sec",
    ]
    labels = [
        "Perplexity", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-4",
        "关键词准确率", "内容重叠率",
        "平均生成长度", "平均Token数", "推理速度(tok/s)",
    ]
    directions = [
        "down", "up", "up", "up", "up", "up", "up", "neutral", "neutral", "up",
    ]

    table_data = []
    for metric, label, direction in zip(metrics_list, labels, directions):
        base_val = base_metrics.get(metric, 0)
        ft_val = finetuned_metrics.get(metric, 0)

        if base_val == 0:
            change = "N/A"
        else:
            pct = (ft_val - base_val) / base_val * 100
            arrow = "+" if pct > 0 else ""
            change = f"{arrow}{pct:.1f}%"

        table_data.append([label, f"{base_val}", f"{ft_val}", change])

    print("\n" + "=" * 70)
    print(" 微调前后评估指标对比")
    print("=" * 70)
    print(tabulate(
        table_data,
        headers=["指标", "Base Model", "Fine-tuned", "变化"],
        tablefmt="grid",
        numalign="right",
    ))
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="微调前后对比评估")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-1.7B-Instruct",
        help="基座模型路径",
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="outputs/merged_model",
        help="微调后模型路径",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="eval/test_cases.jsonl",
        help="测试数据文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs",
        help="评估结果输出目录",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试用例
    test_cases = load_test_cases(Path(args.test_data))
    if not test_cases:
        logger.error("测试用例为空，请先运行 generate_test_cases.py")
        return

    # 评估基座模型
    logger.info("=" * 60)
    logger.info(" 评估基座模型")
    logger.info("=" * 60)
    base_model, base_tokenizer = load_model(args.base_model)
    base_metrics = run_evaluation(base_model, base_tokenizer, test_cases, "Base")
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 评估微调模型
    logger.info("=" * 60)
    logger.info(" 评估微调模型")
    logger.info("=" * 60)
    ft_model, ft_tokenizer = load_model(args.finetuned_model)
    finetuned_metrics = run_evaluation(ft_model, ft_tokenizer, test_cases, "Fine-tuned")
    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 合并结果并保存
    results = {
        "base_model": base_metrics,
        "finetuned_model": finetuned_metrics,
    }
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {results_path}")

    # 打印对比表格
    print_comparison_table(base_metrics, finetuned_metrics)

    # 生成 HTML 报告
    logger.info("生成 HTML 报告...")
    generate_html_report(results, output_dir)
    logger.info(f"HTML 报告已保存: {output_dir / 'eval_report.html'}")


def generate_html_report(results: dict, output_dir: Path):
    """
    调用 eval_report.py 生成 HTML 可视化报告
    如果 eval_report 模块不可用则跳过
    """
    try:
        from eval_report import create_html_report
        report_path = output_dir / "eval_report.html"
        create_html_report(results, report_path)
    except ImportError as e:
        logger.warning(f"无法导入 eval_report 模块: {e}")
        logger.info("请确认 eval/eval_report.py 在当前目录")


if __name__ == "__main__":
    main()
