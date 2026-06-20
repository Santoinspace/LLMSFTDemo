import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from split_wl_customer_support import (  # noqa: E402
    check_no_overlap,
    run_split,
    validate_example,
)


def _messages(i):
    return [
        {"role": "user", "content": f"Ticket {i}\nNeed help with product {i}."},
        {"role": "assistant", "content": f"Support answer {i}."},
    ]


def _write_source(path: Path, count: int = 20) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({"messages": _messages(i)}, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_example_requires_messages():
    assert validate_example({"messages": _messages(1)}) is True
    assert validate_example({"messages": []}) is False
    assert validate_example({"messages": [{"role": "user", "content": "Q"}]}) is False
    assert validate_example({"messages": [{"role": "assistant", "content": "A"}]}) is False
    assert validate_example({"messages": [{"role": "user", "content": "  "}, {"role": "assistant", "content": "A"}]}) is False


def test_check_no_overlap_rejects_duplicate_across_splits():
    with pytest.raises(ValueError, match="Overlapping original_index"):
        check_no_overlap({"a": [1, 2], "b": [2, 3]})


def test_split_outputs_are_complete_and_non_overlapping(tmp_path):
    source = tmp_path / "source.jsonl"
    output_dir = tmp_path / "split"
    _write_source(source, count=20)

    result = run_split(
        dataset_name="W-L/Customer-service-tickets-qwen-qa",
        dataset_path=str(source),
        output_dir=output_dir,
        seed=42,
    )

    assert result["valid_rows"] == 20
    assert result["invalid_rows"] == 0
    assert result["sft_train"] == 14
    assert result["sft_val"] == 1
    assert result["sft_test"] == 1
    assert result["rag_corpus"] == 3
    assert result["rag_eval"] == 1

    sft_train = _read_jsonl(output_dir / "sft_train.jsonl")
    sft_val = _read_jsonl(output_dir / "sft_val.jsonl")
    sft_test = _read_jsonl(output_dir / "sft_test.jsonl")
    rag_corpus = _read_jsonl(output_dir / "rag_corpus.jsonl")
    rag_eval = _read_jsonl(output_dir / "rag_eval.jsonl")

    for row in sft_train + sft_val + sft_test:
        assert set(row.keys()) == {"text"}
        assert "<|im_start|>user\n" in row["text"]
        assert "<|im_start|>assistant\n" in row["text"]

    corpus_required = {
        "id",
        "content",
        "metadata",
    }
    eval_required = {
        "id",
        "source_dataset",
        "pseudo_date",
        "original_index",
        "question",
        "reference",
        "notes",
    }
    assert all(corpus_required == set(row.keys()) for row in rag_corpus)
    assert all(eval_required == set(row.keys()) for row in rag_eval)
    assert rag_corpus[0]["id"] == "rag_000000"
    assert rag_eval[0]["id"] == "rag_eval_000000"
    assert rag_corpus[0]["metadata"]["pseudo_date"] == "2026-06-01"
    assert "Customer ticket:\n" in rag_corpus[0]["content"]
    assert "\n\nSupport answer:\n" in rag_corpus[0]["content"]
    assert "support_answer" in rag_corpus[0]["metadata"]
    assert rag_eval[0]["notes"] == "This sample is held out from rag_corpus and all SFT splits."

    text_to_index = {
        "<|im_start|>user\n"
        f"Ticket {i}\nNeed help with product {i}.<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"Support answer {i}.<|im_end|>\n"
        "<|im_start|>assistant\n": i
        for i in range(20)
    }
    original_indices = {
        "sft_train": [text_to_index[row["text"]] for row in sft_train],
        "sft_val": [text_to_index[row["text"]] for row in sft_val],
        "sft_test": [text_to_index[row["text"]] for row in sft_test],
        "rag_corpus": [row["metadata"]["original_index"] for row in rag_corpus],
        "rag_eval": [row["original_index"] for row in rag_eval],
    }
    check_no_overlap(original_indices, expected_total=20)

    manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_rows"] == 20
    assert manifest["valid_rows"] == 20
    assert manifest["counts"] == {
        "sft_train": 14,
        "sft_val": 1,
        "sft_test": 1,
        "rag_corpus": 3,
        "rag_eval": 1,
    }
    assert (output_dir / "README.md").exists()
    assert (output_dir / "invalid_examples.jsonl").read_text(encoding="utf-8") == ""


def test_same_seed_repeated_run_is_deterministic(tmp_path):
    source = tmp_path / "source.jsonl"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _write_source(source, count=20)

    kwargs = {
        "dataset_name": "W-L/Customer-service-tickets-qwen-qa",
        "dataset_path": str(source),
        "seed": 123,
    }
    run_split(output_dir=out_a, **kwargs)
    run_split(output_dir=out_b, **kwargs)

    for filename in [
        "sft_train.jsonl",
        "sft_val.jsonl",
        "sft_test.jsonl",
        "rag_corpus.jsonl",
        "rag_eval.jsonl",
        "invalid_examples.jsonl",
    ]:
        assert _digest(out_a / filename) == _digest(out_b / filename)
