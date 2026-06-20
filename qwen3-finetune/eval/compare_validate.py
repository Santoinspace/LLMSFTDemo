"""Unified model comparison and validation script.

Compares base and fine-tuned models on JSONL test cases, optionally with RAG.
Input rows may contain:
- question + reference
- question + expected_answer
- messages + reference

The output is a compact JSON report with metrics and per-sample predictions.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
RAG_DIR = PROJECT_DIR / "rag"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(RAG_DIR))

from metrics import compute_bleu, compute_generation_stats, compute_rouge  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


RAG_SYSTEM_TEMPLATE = """You are a helpful customer support assistant.
Answer the user using the reference information below. If the references are
insufficient, say so instead of inventing details.

Reference information:
{context}
"""


def first_content(messages: List[dict], role: str) -> str:
    for message in messages:
        if message.get("role") == role and str(message.get("content", "")).strip():
            return message["content"]
    return ""


def prompt_messages_from_case(case: dict) -> List[dict]:
    messages = case.get("messages")
    if isinstance(messages, list) and messages:
        prompt = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                prompt.append({"role": "system", "content": content})
            elif role == "user":
                prompt.append({"role": "user", "content": content})
                break
        if prompt and prompt[-1]["role"] == "user":
            return prompt

    question = case.get("question") or first_content(messages or [], "user")
    return [{"role": "user", "content": question}]


def normalize_question(question: str) -> str:
    marker = "[User]"
    if marker in question:
        return question.split(marker, 1)[1].strip()
    return question.strip()


def load_test_cases(path: Path, max_samples: Optional[int] = None) -> List[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages") if isinstance(item.get("messages"), list) else []
            question = normalize_question(item.get("question") or first_content(messages, "user"))
            reference = (
                item.get("reference")
                or item.get("expected_answer")
                or item.get("answer")
                or first_content(messages, "assistant")
            )
            if not question or not reference:
                continue
            item["question"] = question
            item["reference"] = reference
            item["prompt_messages"] = prompt_messages_from_case(item)
            cases.append(item)
            if max_samples and len(cases) >= max_samples:
                break
    return cases


def load_model(model_path: str, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "cpu"

    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: List[dict], max_new_tokens: int, temperature: float):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    new_tokens = outputs[0][input_len:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return prediction, elapsed, len(new_tokens)


def build_retriever(args):
    if args.retriever == "none":
        return None
    if args.retriever == "bm25":
        from bm25_store import BM25Store

        return BM25Store(
            persist_directory=args.retriever_path,
            collection_name=args.retriever_collection,
        )
    if args.retriever == "chromadb":
        from vector_store import VectorStore

        return VectorStore(
            persist_directory=args.retriever_path,
            collection_name=args.retriever_collection,
        )
    raise ValueError(f"Unknown retriever: {args.retriever}")


def rag_messages(question: str, retriever, top_k: int) -> List[dict]:
    results = retriever.similarity_search(question, k=top_k)
    contexts = [row.get("content", "") for row in results]
    context_text = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{context}" for i, context in enumerate(contexts)
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_TEMPLATE.format(context=context_text)},
        {"role": "user", "content": question},
    ]


def compute_group_metrics(predictions: List[str], references: List[str], times: List[float], tokens: List[int]):
    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    stats = compute_generation_stats(predictions, times, tokens)
    return {
        **rouge,
        "bleu4": bleu,
        **stats,
    }


def evaluate_model(
    model_path: str,
    label: str,
    cases: List[dict],
    retriever,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    load_in_4bit: bool,
) -> Dict:
    logger.info("=" * 60)
    logger.info("Evaluating %s: %s", label, model_path)
    logger.info("=" * 60)
    model, tokenizer = load_model(model_path, load_in_4bit=load_in_4bit)

    groups = {"no_rag": {"predictions": [], "times": [], "tokens": []}}
    if retriever is not None:
        groups["rag"] = {"predictions": [], "times": [], "tokens": []}

    samples = []
    for i, case in enumerate(cases, 1):
        logger.info("[%s] %s/%s: %s", label, i, len(cases), case["question"][:80])

        pred, elapsed, ntok = generate(
            model,
            tokenizer,
            case["prompt_messages"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        groups["no_rag"]["predictions"].append(pred)
        groups["no_rag"]["times"].append(elapsed)
        groups["no_rag"]["tokens"].append(ntok)

        sample = {
            "question": case["question"],
            "reference": case["reference"],
            "no_rag": pred,
        }

        if retriever is not None:
            pred_rag, elapsed_rag, ntok_rag = generate(
                model,
                tokenizer,
                rag_messages(case["question"], retriever, top_k),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            groups["rag"]["predictions"].append(pred_rag)
            groups["rag"]["times"].append(elapsed_rag)
            groups["rag"]["tokens"].append(ntok_rag)
            sample["rag"] = pred_rag

        samples.append(sample)

    references = [case["reference"] for case in cases]
    metrics = {
        name: compute_group_metrics(
            data["predictions"],
            references,
            data["times"],
            data["tokens"],
        )
        for name, data in groups.items()
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"model_path": model_path, "metrics": metrics, "samples": samples}


def print_metrics_table(results: Dict[str, dict]) -> None:
    metric_names = ["rouge1", "rouge2", "rougeL", "bleu4", "avg_length", "avg_tokens", "tokens_per_sec"]
    print("\n" + "=" * 100)
    print("Comparison Metrics")
    print("=" * 100)
    header = ["group"] + metric_names
    print(" | ".join(f"{h:<14}" for h in header))
    print("-" * 100)
    for label, result in results.items():
        for mode, metrics in result["metrics"].items():
            group = f"{label}:{mode}"
            values = [group] + [metrics.get(name, 0) for name in metric_names]
            print(" | ".join(f"{str(v):<14}" for v in values))
    print("=" * 100 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified base/finetuned and optional RAG comparison.")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--finetuned_model", type=str, default=None)
    parser.add_argument("--test_cases", "--test_data", dest="test_cases", required=True)
    parser.add_argument("--output", type=str, default="eval_outputs/compare_validate.json")
    parser.add_argument("--retriever", choices=["none", "bm25", "chromadb"], default="none")
    parser.add_argument("--retriever_path", type=str, default="bm25_index_swebench")
    parser.add_argument("--retriever_collection", type=str, default="swebench_instances")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.base_model and not args.finetuned_model:
        raise ValueError("At least one of --base_model or --finetuned_model is required.")

    cases = load_test_cases(Path(args.test_cases), args.max_samples)
    if not cases:
        raise ValueError(f"No valid test cases found: {args.test_cases}")
    logger.info("Loaded %s test cases", len(cases))

    retriever = build_retriever(args)
    results = {}
    if args.base_model:
        results["base"] = evaluate_model(
            args.base_model,
            "base",
            cases,
            retriever,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            load_in_4bit=not args.no_4bit,
        )
    if args.finetuned_model:
        results["finetuned"] = evaluate_model(
            args.finetuned_model,
            "finetuned",
            cases,
            retriever,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            load_in_4bit=not args.no_4bit,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "base_model": args.base_model,
            "finetuned_model": args.finetuned_model,
            "test_cases": args.test_cases,
            "num_test_cases": len(cases),
            "retriever": args.retriever,
            "retriever_path": args.retriever_path,
            "retriever_collection": args.retriever_collection,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "results": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_metrics_table(results)
    logger.info("Saved report: %s", output)


if __name__ == "__main__":
    main()
