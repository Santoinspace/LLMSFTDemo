"""Build leakage-free domain and general evaluation sets for replay experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_forgetting_replay15 import (  # noqa: E402
    GROUPS,
    format_chatml,
    load_rows,
    normalize_messages,
    raw_path,
    stable_rng,
)


CHATML_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL
)

GENERAL_EVAL_COUNTS = {
    "english_general_smoltalk": 20,
    "english_general_ultrachat": 20,
    "english_constraints": 40,
    "chinese_general": 50,
    "rewrite": 25,
    "summarize": 25,
    "math": 40,
    "code_python": 40,
    "safety_boundaries": 40,
}

CATEGORY_NAMES = {
    "english_general_smoltalk": "english_general",
    "english_general_ultrachat": "english_general",
    "english_constraints": "constraints",
    "chinese_general": "chinese_general",
    "rewrite": "rewrite",
    "summarize": "summarize",
    "math": "math",
    "code_python": "code_python",
    "safety_boundaries": "safety_boundaries",
}


def read_text_rows(path: Path) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["text"])
    return rows


def parse_chatml(text: str) -> list[dict[str, str]]:
    return [
        {"role": match.group(1), "content": match.group(2)}
        for match in CHATML_PATTERN.finditer(text)
    ]


def first_exchange(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str, str] | None:
    prompt = []
    found_user = False
    question = ""
    for message in messages:
        role = message["role"]
        if role == "system" and not found_user:
            prompt.append(message)
        elif role == "user" and not found_user:
            prompt.append(message)
            question = message["content"].strip()
            found_user = True
        elif role == "assistant" and found_user:
            reference = message["content"].strip()
            if question and reference:
                return prompt, question, reference
            return None
    return None


def final_exchange(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str, str] | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] != "assistant":
            continue
        prompt = messages[:index]
        if not prompt or prompt[-1]["role"] != "user":
            continue
        question = prompt[-1]["content"].strip()
        reference = messages[index]["content"].strip()
        if question and reference:
            return prompt, question, reference
    return None


def all_user_questions(text: str) -> set[str]:
    return {
        normalized_question(message["content"])
        for message in parse_chatml(text)
        if message["role"] == "user" and message["content"].strip()
    }


def normalized_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def build_general_eval(
    raw_root: Path,
    replay_texts: set[str],
    train_questions: set[str],
    tokenizer: Any,
    max_seq_length: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    seen_questions = set(train_questions)
    seen_pairs = set()
    stats = {}

    for group in GROUPS:
        requested = GENERAL_EVAL_COUNTS[group.name]
        source_rows = load_rows(raw_path(raw_root, group.raw_file))
        indices = list(range(len(source_rows)))
        stable_rng(seed, f"eval:{group.name}").shuffle(indices)
        selected = []
        rejected_train_row = 0
        rejected_question = 0
        rejected_length = 0

        for original_index in indices:
            full_messages = normalize_messages(group.converter(source_rows[original_index]))
            if not full_messages:
                continue
            if format_chatml(full_messages) in replay_texts:
                rejected_train_row += 1
                continue
            exchange = final_exchange(full_messages)
            if exchange is None:
                continue
            prompt, question, reference = exchange
            question_key = normalized_question(question)
            if question_key in seen_questions:
                rejected_question += 1
                continue
            pair_text = format_chatml(prompt + [{"role": "assistant", "content": reference}])
            pair_hash = hashlib.sha256(pair_text.encode("utf-8")).hexdigest()
            if pair_hash in seen_pairs:
                continue
            if len(tokenizer(pair_text, add_special_tokens=False)["input_ids"]) > max_seq_length:
                rejected_length += 1
                continue

            row = {
                "id": f"general_{group.name}_{len(selected):04d}",
                "category": CATEGORY_NAMES[group.name],
                "source_group": group.name,
                "source_dataset": group.raw_file.repo_id,
                "source_original_index": original_index,
                "question": question,
                "reference": reference,
                "messages": prompt + [{"role": "assistant", "content": reference}],
                "text": pair_text,
            }
            selected.append(row)
            seen_questions.add(question_key)
            seen_pairs.add(pair_hash)
            if len(selected) == requested:
                break

        if len(selected) != requested:
            raise ValueError(f"{group.name}: selected {len(selected)} of {requested}")
        output.extend(selected)
        stats[group.name] = {
            "requested": requested,
            "selected": len(selected),
            "rejected_replay_train_row": rejected_train_row,
            "rejected_train_question": rejected_question,
            "rejected_over_max_length": rejected_length,
        }

    random.Random(seed).shuffle(output)
    return output, stats


def build_clean_domain_set(
    source: Path,
    train_texts: set[str],
    train_questions: set[str],
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    seen_questions = set()
    seen_texts = set()
    stats = {
        "source_rows": 0,
        "rejected_blocked_text": 0,
        "rejected_blocked_question": 0,
        "rejected_duplicate": 0,
        "invalid": 0,
    }

    for original_index, text in enumerate(read_text_rows(source)):
        stats["source_rows"] += 1
        if text in train_texts:
            stats["rejected_blocked_text"] += 1
            continue
        messages = parse_chatml(text)
        exchange = first_exchange(messages)
        if exchange is None:
            stats["invalid"] += 1
            continue
        prompt, question, reference = exchange
        question_key = normalized_question(question)
        if question_key in train_questions:
            stats["rejected_blocked_question"] += 1
            continue
        if text in seen_texts or question_key in seen_questions:
            stats["rejected_duplicate"] += 1
            continue
        seen_texts.add(text)
        seen_questions.add(question_key)
        output.append(
            {
                "id": f"domain_{split_name}_{len(output):04d}",
                "category": "customer_support",
                "source_dataset": "W-L/Customer-service-tickets-qwen-qa",
                "source_original_index": original_index,
                "question": question,
                "reference": reference,
                "messages": prompt + [{"role": "assistant", "content": reference}],
                "text": text,
            }
        )
    stats["selected"] = len(output)
    return output, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed_train", default="data/preprocessed/wl_customer_support_replay15/mixed_train.jsonl")
    parser.add_argument("--replay_train", default="data/preprocessed/wl_customer_support_replay15/general_replay.jsonl")
    parser.add_argument("--domain_val", default="data/wl_customer_support_split/sft_val.jsonl")
    parser.add_argument("--domain_test", default="data/wl_customer_support_split/sft_test.jsonl")
    parser.add_argument("--raw_dir", default="data/raw/forgetting_replay15")
    parser.add_argument("--output_dir", default="eval/forgetting")
    parser.add_argument("--tokenizer_path", default="outputs/outputs_wl_customer_support/merged_model")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--domain_sample_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    mixed_texts = read_text_rows(Path(args.mixed_train))
    replay_texts = set(read_text_rows(Path(args.replay_train)))
    train_texts = set(mixed_texts)
    train_questions = set().union(*(all_user_questions(text) for text in mixed_texts))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    general_eval, general_stats = build_general_eval(
        Path(args.raw_dir),
        replay_texts,
        train_questions,
        tokenizer,
        args.max_seq_length,
        args.seed,
    )
    domain_val, domain_val_stats = build_clean_domain_set(
        Path(args.domain_val), train_texts, train_questions, "val"
    )
    raw_val_texts = set(read_text_rows(Path(args.domain_val)))
    raw_val_questions = set().union(*(all_user_questions(text) for text in raw_val_texts))
    domain_test, domain_test_stats = build_clean_domain_set(
        Path(args.domain_test),
        train_texts | raw_val_texts,
        train_questions | raw_val_questions,
        "test",
    )
    sampled_domain = list(domain_test)
    random.Random(args.seed).shuffle(sampled_domain)
    sampled_domain = sampled_domain[: args.domain_sample_size]

    write_jsonl(output_dir / "general_heldout_300.jsonl", general_eval)
    write_jsonl(output_dir / "domain_val_clean.jsonl", domain_val)
    write_jsonl(output_dir / "domain_test_clean_all.jsonl", domain_test)
    write_jsonl(output_dir / "domain_test_clean_300.jsonl", sampled_domain)

    reference_lengths = [
        len(tokenizer(row["reference"], add_special_tokens=False)["input_ids"])
        for row in general_eval
    ]
    manifest = {
        "seed": args.seed,
        "leakage_policy": "Exclude exact training texts and normalized training questions; domain test also excludes the full validation split; deduplicate evaluation questions.",
        "train": {
            "mixed_rows": len(mixed_texts),
            "unique_texts": len(train_texts),
            "unique_questions": len(train_questions),
        },
        "general_eval": {
            "rows": len(general_eval),
            "counts": GENERAL_EVAL_COUNTS,
            "group_stats": general_stats,
            "reference_tokens": {
                "p95": percentile(reference_lengths, 0.95),
                "p99": percentile(reference_lengths, 0.99),
                "max": max(reference_lengths),
            },
        },
        "domain_val_clean": domain_val_stats,
        "domain_test_clean": domain_test_stats,
        "domain_test_sample_rows": len(sampled_domain),
        "files": {
            "general": "general_heldout_300.jsonl",
            "domain_val": "domain_val_clean.jsonl",
            "domain_test_all": "domain_test_clean_all.jsonl",
            "domain_test_sample": "domain_test_clean_300.jsonl",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
