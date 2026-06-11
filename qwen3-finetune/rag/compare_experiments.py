"""
四组交叉对比实验：微调 × RAG

对比 (base | fine-tuned) × (no_RAG | with_RAG) 四种组合在测试集上的表现。
复用 RAGPipeline + eval/metrics.py。

使用方式:
    python rag/compare_experiments.py --max_samples 5
    python rag/compare_experiments.py --test_cases eval/test_cases.jsonl
"""
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 导入 eval/metrics（需加父目录到 path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from metrics import compute_bleu, compute_rouge


def load_test_cases(path: str, max_samples: int = None) -> List[Dict]:
    """加载测试用例，返回 [{"question": str, "reference": str}]"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            question_raw = item.get("question", "")
            # 提取 [User] 部分
            user_match = re.search(r"\[User\]\s*(.*)", question_raw, re.DOTALL)
            user_question = user_match.group(1).strip() if user_match else question_raw
            cases.append({
                "question": user_question,
                "reference": item.get("reference", ""),
            })
    if max_samples:
        cases = cases[:max_samples]
    return cases


def run_model_experiments(
    model_path: str,
    model_label: str,
    test_cases: List[Dict],
    retriever,
    max_new_tokens: int = 256,
    top_k: int = 5,
) -> Dict:
    """
    对单个模型运行所有测试，返回 with_rag 和 without_rag 的结果。
    """
    from rag_pipeline import RAGPipeline

    logger.info(f"\n{'='*60}")
    logger.info(f" [{model_label}] 加载模型: {model_path}")
    logger.info(f"{'='*60}")

    pipeline = RAGPipeline(
        model_path=model_path,
        retriever=retriever,
        top_k=top_k,
    )

    results_no_rag = []
    results_with_rag = []

    for i, case in enumerate(test_cases):
        logger.info(f"[{model_label}] {i+1}/{len(test_cases)}: {case['question'][:80]}...")

        try:
            compare = pipeline.compare(
                case["question"],
                max_new_tokens=max_new_tokens,
            )
            results_no_rag.append(compare["without_rag"]["answer"])
            results_with_rag.append(compare["with_rag"]["answer"])
        except Exception as e:
            logger.warning(f"推理失败 #{i+1}: {e}")
            results_no_rag.append("")
            results_with_rag.append("")

    # 释放模型显存
    del pipeline
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_label,
        "no_rag": results_no_rag,
        "with_rag": results_with_rag,
    }


def main():
    parser = argparse.ArgumentParser(description="四组交叉实验：微调 × RAG")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B",
                        help="基座模型")
    parser.add_argument("--finetuned_model", type=str,
                        default="outputs/outputs_codealpacas/merged_model",
                        help="微调模型路径")
    parser.add_argument("--retriever_path", type=str, default="bm25_index_swebench",
                        help="BM25 索引目录")
    parser.add_argument("--retriever_collection", type=str, default="swebench_instances",
                        help="集合名")
    parser.add_argument("--test_cases", type=str, default="eval/test_cases.jsonl",
                        help="测试用例文件")
    parser.add_argument("--top_k", type=int, default=5,
                        help="检索数量")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="生成最大 token 数")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制测试题数（调试用）")
    parser.add_argument("--output", type=str,
                        default="eval_outputs/experiment_4group.json",
                        help="结果输出路径")
    args = parser.parse_args()

    # 加载测试用例
    test_cases = load_test_cases(args.test_cases, args.max_samples)
    logger.info(f"加载 {len(test_cases)} 道测试题")
    references = [c["reference"] for c in test_cases]

    # 加载 BM25 检索器
    from bm25_store import BM25Store
    retriever = BM25Store(
        persist_directory=args.retriever_path,
        collection_name=args.retriever_collection,
    )
    logger.info(f"BM25 文档数: {retriever.count()}")

    # 实验 1+2: base model
    base_results = run_model_experiments(
        model_path=args.base_model,
        model_label="base",
        test_cases=test_cases,
        retriever=retriever,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
    )

    # 实验 3+4: fine-tuned model
    ft_results = run_model_experiments(
        model_path=args.finetuned_model,
        model_label="finetuned",
        test_cases=test_cases,
        retriever=retriever,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
    )

    # 四组答案
    groups = {
        "base":       base_results["no_rag"],
        "base+RAG":   base_results["with_rag"],
        "ft":         ft_results["no_rag"],
        "ft+RAG":     ft_results["with_rag"],
    }

    # 计算指标
    metrics = {}
    logger.info("\n计算指标...")
    for name, preds in groups.items():
        rouge = compute_rouge(preds, references)
        bleu = compute_bleu(preds, references)
        avg_len = round(sum(len(p) for p in preds) / max(len(preds), 1), 1)
        metrics[name] = {
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bleu4": bleu,
            "avg_length": avg_len,
        }

    # 打印对比表
    print(f"\n{'='*80}")
    print(" 4-Group Experiment: Fine-tuning x RAG")
    print(f"{'='*80}")
    header = f"{'Metric':<14} | {'base':<10} | {'base+RAG':<10} | {'ft':<10} | {'ft+RAG':<10}"
    print(header)
    print("-" * len(header))

    for metric_key, metric_label in [
        ("rouge1", "ROUGE-1"),
        ("rouge2", "ROUGE-2"),
        ("rougeL", "ROUGE-L"),
        ("bleu4", "BLEU-4"),
        ("avg_length", "Avg Length"),
    ]:
        vals = [f"{metrics[g][metric_key]:<10}" for g in ["base", "base+RAG", "ft", "ft+RAG"]]
        print(f"{metric_label:<14} | " + " | ".join(vals))

    print(f"{'='*80}\n")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "base_model": args.base_model,
                "finetuned_model": args.finetuned_model,
                "num_test_cases": len(test_cases),
            },
            "metrics": metrics,
            "answers": {
                "base": {"no_rag": base_results["no_rag"], "with_rag": base_results["with_rag"]},
                "finetuned": {"no_rag": ft_results["no_rag"], "with_rag": ft_results["with_rag"]},
            },
            "references": references,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
