"""Deterministically split W-L customer support data for SFT and RAG.

This script simulates a pseudo-temporal setup:
- base model cutoff: 2026-04
- SFT increment month: 2026-05
- RAG latest month: 2026-06

The source dataset has no real timestamp field, so the split is produced with a
seeded shuffle. Samples are never shared across splits.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm


SFT_TRAIN_RATIO = 0.70
SFT_VAL_RATIO = 0.05
SFT_TEST_RATIO = 0.05
RAG_CORPUS_RATIO = 0.15
RAG_EVAL_RATIO = 0.05

DEFAULT_DATASET_NAME = "W-L/Customer-service-tickets-qwen-qa"
DEFAULT_OUTPUT_DIR = "data/wl_customer_support_split"
INVALID_THRESHOLD = 0.01


def select_dataset_split(dataset):
    """Return train split when available, otherwise the first split."""
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        first_key = next(iter(dataset.keys()))
        return dataset[first_key]
    return dataset


def _load_local_file(path: Path) -> Dataset:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path), split="train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    raise ValueError(f"Unsupported local dataset file type: {path}")


def _find_data_files(path: Path) -> Tuple[str, List[str]]:
    json_files = sorted(
        str(p) for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl"}
    )
    parquet_files = sorted(str(p) for p in path.rglob("*") if p.suffix.lower() == ".parquet")

    if json_files and parquet_files:
        raise ValueError(
            f"Local directory contains both JSON and Parquet files; choose one format: {path}"
        )
    if json_files:
        return "json", json_files
    if parquet_files:
        return "parquet", parquet_files
    raise FileNotFoundError(f"No json/jsonl/parquet files found under: {path}")


def _load_local_directory(path: Path) -> Dataset:
    try:
        return select_dataset_split(load_from_disk(str(path)))
    except Exception:
        dataset_type, files = _find_data_files(path)
        return select_dataset_split(load_dataset(dataset_type, data_files={"train": files}))


def load_raw_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Dataset:
    """Load a dataset from local path first, then HF Hub name.

    Supports HF Hub datasets, local Dataset/DatasetDict directories saved with
    ``save_to_disk``, and local json/jsonl/parquet files or directories.
    """
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {path}")
        if path.is_file():
            return _load_local_file(path)
        if path.is_dir():
            return _load_local_directory(path)
        raise ValueError(f"Unsupported dataset_path: {path}")

    if not dataset_name:
        raise ValueError("Either dataset_name or dataset_path must be provided.")

    return select_dataset_split(load_dataset(dataset_name))


def _content_is_non_empty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_example(example) -> bool:
    """Validate that an example has usable Qwen-style messages."""
    messages = example.get("messages") if isinstance(example, dict) else None
    if not isinstance(messages, list) or not messages:
        return False

    has_user = False
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        content = message.get("content")
        if not _content_is_non_empty(content):
            return False
        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True

    return has_user and has_assistant


def _first_message_content(messages: List[dict], role: str) -> str:
    for message in messages:
        if message.get("role") == role and _content_is_non_empty(message.get("content")):
            return message["content"]
    raise ValueError(f"Missing non-empty {role} message")


def _title_from_user_query(user_query: str) -> str:
    for line in user_query.splitlines():
        title = line.strip()
        if title:
            return title[:120]
    return user_query.strip()[:120]


def _pseudo_june_date(row_number: int) -> str:
    day = row_number % 30 + 1
    return f"2026-06-{day:02d}"


def _format_chatml(messages: List[dict]) -> str:
    lines = []
    for message in messages:
        lines.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def collect_valid_rows(dataset: Dataset, invalid_path: Path) -> Tuple[List[dict], int, int]:
    """Collect valid rows and write invalid rows to disk without cleaning them."""
    valid_rows = []
    invalid_rows = []

    for original_index, example in tqdm(
        enumerate(dataset), total=len(dataset), desc="Validating examples"
    ):
        row = dict(example)
        if validate_example(row):
            valid_rows.append(
                {
                    "original_index": original_index,
                    "messages": row["messages"],
                }
            )
        else:
            invalid_rows.append({"original_index": original_index, "example": row})

    _write_jsonl(invalid_path, invalid_rows)

    total_rows = len(dataset)
    invalid_count = len(invalid_rows)
    if total_rows and invalid_count / total_rows > INVALID_THRESHOLD:
        raise ValueError(
            f"Invalid examples exceed 1%: {invalid_count}/{total_rows} "
            f"({invalid_count / total_rows:.2%}). See {invalid_path}"
        )

    return valid_rows, total_rows, invalid_count


def compute_split_counts(total_valid: int) -> Dict[str, int]:
    sft_train = int(total_valid * SFT_TRAIN_RATIO)
    sft_val = int(total_valid * SFT_VAL_RATIO)
    sft_test = int(total_valid * SFT_TEST_RATIO)
    rag_corpus = int(total_valid * RAG_CORPUS_RATIO)
    rag_eval = total_valid - sft_train - sft_val - sft_test - rag_corpus
    return {
        "sft_train": sft_train,
        "sft_val": sft_val,
        "sft_test": sft_test,
        "rag_corpus": rag_corpus,
        "rag_eval": rag_eval,
    }


def build_split_positions(total_valid: int, seed: int) -> Dict[str, List[int]]:
    positions = list(range(total_valid))
    random.Random(seed).shuffle(positions)
    counts = compute_split_counts(total_valid)

    split_to_positions = {}
    cursor = 0
    for split_name in ["sft_train", "sft_val", "sft_test", "rag_corpus", "rag_eval"]:
        next_cursor = cursor + counts[split_name]
        split_to_positions[split_name] = positions[cursor:next_cursor]
        cursor = next_cursor

    return split_to_positions


def check_no_overlap(split_to_indices: Dict[str, List[int]], expected_total: Optional[int] = None):
    """Ensure no original_index appears in more than one split."""
    seen = set()
    total = 0
    for split_name, indices in split_to_indices.items():
        split_indices = set(indices)
        if len(split_indices) != len(indices):
            raise ValueError(f"Duplicate original_index inside split: {split_name}")
        overlap = seen.intersection(split_indices)
        if overlap:
            raise ValueError(f"Overlapping original_index values found in {split_name}: {overlap}")
        seen.update(split_indices)
        total += len(indices)

    if expected_total is not None and total != expected_total:
        raise ValueError(f"Split total mismatch: {total} != {expected_total}")


def _row_at(valid_rows: List[dict], position: int) -> dict:
    return valid_rows[position]


def _sft_rows(valid_rows: List[dict], positions: List[int]) -> Iterable[dict]:
    for position in positions:
        yield {"text": _format_chatml(_row_at(valid_rows, position)["messages"])}


def _rag_corpus_rows(
    valid_rows: List[dict],
    positions: List[int],
    source_dataset: str,
) -> Iterable[dict]:
    for row_number, position in enumerate(positions):
        row = _row_at(valid_rows, position)
        user_query = _first_message_content(row["messages"], "user")
        support_answer = _first_message_content(row["messages"], "assistant")
        yield {
            "id": f"rag_{row_number:06d}",
            "content": f"Customer ticket:\n{user_query}\n\nSupport answer:\n{support_answer}",
            "metadata": {
                "source_dataset": source_dataset,
                "pseudo_date": _pseudo_june_date(row_number),
                "original_index": row["original_index"],
                "title": _title_from_user_query(user_query),
                "user_query": user_query,
                "support_answer": support_answer,
            },
        }


def _rag_eval_rows(
    valid_rows: List[dict],
    positions: List[int],
    source_dataset: str,
) -> Iterable[dict]:
    for row_number, position in enumerate(positions):
        row = _row_at(valid_rows, position)
        user_query = _first_message_content(row["messages"], "user")
        support_answer = _first_message_content(row["messages"], "assistant")
        yield {
            "id": f"rag_eval_{row_number:06d}",
            "source_dataset": source_dataset,
            "pseudo_date": _pseudo_june_date(row_number),
            "original_index": row["original_index"],
            "question": user_query,
            "reference": support_answer,
            "notes": "This sample is held out from rag_corpus and all SFT splits.",
        }


def write_readme(output_dir: Path) -> None:
    readme = """# W-L Customer Support Pseudo Temporal Split

This directory contains a deterministic split of `W-L/Customer-service-tickets-qwen-qa`.

## Scenario

- Base model cutoff: 2026-04
- SFT increment month: 2026-05
- RAG latest month: 2026-06

The source dataset has no real timestamp field. This is a pseudo temporal split
created with a seeded shuffle, so it is reproducible but not chronological.

## Files

- `sft_train.jsonl`: incremental SFT training data, one `text` field per row.
- `sft_val.jsonl`: validation data used during SFT, one `text` field per row.
- `sft_test.jsonl`: held-out SFT test data for checking parameter learning.
- `rag_corpus.jsonl`: June-style retrieval corpus with `content` and `metadata`.
- `rag_eval.jsonl`: June-style held-out RAG evaluation data with `question` and `reference`.
- `invalid_examples.jsonl`: invalid source rows, preserved for audit.
- `split_manifest.json`: source, ratios, counts, simulation metadata, and file names.

## Recommended Use

Train only on `sft_train.jsonl`, validate with `sft_val.jsonl`, and test SFT
behavior with `sft_test.jsonl`. These files match the project's SFT convention:
`dataset_text_field = "text"`. Build the retrieval index only from
`rag_corpus.jsonl`; it matches the project's retriever convention: `content`
for searchable text and `metadata` for source fields.

Do not add `rag_eval.jsonl` to the retrieval index. Its expected answers are for
offline evaluation only, and including it in the corpus would leak answers into
retrieval. Likewise, `rag_corpus.jsonl` must not be used for SFT.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(
    output_dir: Path,
    dataset_name: str,
    dataset_path: Optional[str],
    seed: int,
    total_rows: int,
    valid_rows: int,
    invalid_rows: int,
    counts: Dict[str, int],
) -> None:
    manifest = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "seed": seed,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "ratios": {
            "sft_train": SFT_TRAIN_RATIO,
            "sft_val": SFT_VAL_RATIO,
            "sft_test": SFT_TEST_RATIO,
            "rag_corpus": RAG_CORPUS_RATIO,
            "rag_eval": RAG_EVAL_RATIO,
        },
        "counts": counts,
        "simulation": {
            "base_model_cutoff": "2026-04",
            "sft_increment_month": "2026-05",
            "rag_latest_month": "2026-06",
            "split_type": "pseudo_temporal_seeded_shuffle",
        },
        "files": {
            "sft_train": "sft_train.jsonl",
            "sft_val": "sft_val.jsonl",
            "sft_test": "sft_test.jsonl",
            "rag_corpus": "rag_corpus.jsonl",
            "rag_eval": "rag_eval.jsonl",
            "invalid_examples": "invalid_examples.jsonl",
            "readme": "README.md",
        },
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_split(
    dataset_name: str,
    output_dir: Path,
    seed: int,
    dataset_path: Optional[str] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_raw_dataset(dataset_name=dataset_name, dataset_path=dataset_path)

    valid_rows, total_rows, invalid_count = collect_valid_rows(
        dataset, output_dir / "invalid_examples.jsonl"
    )
    split_to_positions = build_split_positions(len(valid_rows), seed)
    split_to_original_indices = {
        split_name: [_row_at(valid_rows, position)["original_index"] for position in positions]
        for split_name, positions in split_to_positions.items()
    }
    check_no_overlap(split_to_original_indices, expected_total=len(valid_rows))

    counts = {
        "sft_train": _write_jsonl(
            output_dir / "sft_train.jsonl",
            _sft_rows(valid_rows, split_to_positions["sft_train"]),
        ),
        "sft_val": _write_jsonl(
            output_dir / "sft_val.jsonl",
            _sft_rows(valid_rows, split_to_positions["sft_val"]),
        ),
        "sft_test": _write_jsonl(
            output_dir / "sft_test.jsonl",
            _sft_rows(valid_rows, split_to_positions["sft_test"]),
        ),
        "rag_corpus": _write_jsonl(
            output_dir / "rag_corpus.jsonl",
            _rag_corpus_rows(valid_rows, split_to_positions["rag_corpus"], dataset_name),
        ),
        "rag_eval": _write_jsonl(
            output_dir / "rag_eval.jsonl",
            _rag_eval_rows(valid_rows, split_to_positions["rag_eval"], dataset_name),
        ),
    }

    if sum(counts.values()) != len(valid_rows):
        raise ValueError(f"Written split total mismatch: {sum(counts.values())} != {len(valid_rows)}")

    write_manifest(
        output_dir=output_dir,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        seed=seed,
        total_rows=total_rows,
        valid_rows=len(valid_rows),
        invalid_rows=invalid_count,
        counts=counts,
    )
    write_readme(output_dir)

    print("Done.")
    print(f"Total valid rows: {len(valid_rows)}")
    print(f"Invalid rows: {invalid_count}")
    print(
        "sft_train: {sft_train} / sft_val: {sft_val} / sft_test: {sft_test} / "
        "rag_corpus: {rag_corpus} / rag_eval: {rag_eval}".format(**counts)
    )
    print(f"Output dir: {output_dir}")

    return {
        "total_rows": total_rows,
        "valid_rows": len(valid_rows),
        "invalid_rows": invalid_count,
        **counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split W-L customer support messages into SFT and RAG datasets."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="HF dataset name. Ignored when --dataset_path is provided.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Local dataset path. Supports save_to_disk directories, json/jsonl, and parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for split files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_split(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
