"""Download and preprocess the 15% general replay mixture.

Raw upstream files are kept under data/raw/forgetting_replay15. Training-ready
ChatML JSONL files are written under data/preprocessed/wl_customer_support_replay15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import pyarrow.parquet as pq
import requests
from tqdm import tqdm
from transformers import AutoTokenizer


HF_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class RawFile:
    repo_id: str
    revision: str
    filename: str

    @property
    def repo_dir(self) -> str:
        return self.repo_id.lower().replace("/", "__").replace("-", "_")


@dataclass(frozen=True)
class SampleGroup:
    name: str
    raw_file: RawFile
    count: int
    converter: Callable[[dict[str, Any]], list[dict[str, str]]]


SMOLTALK_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
COIG_REVISION = "8b55868c6168adf86c30e7ca0f782cca1c514297"


def smoltalk_file(config: str) -> RawFile:
    return RawFile(
        "HuggingFaceTB/smoltalk",
        SMOLTALK_REVISION,
        f"data/{config}/train-00000-of-00001.parquet",
    )


def messages_converter(row: dict[str, Any]) -> list[dict[str, str]]:
    return row.get("messages") or []


def coig_converter(row: dict[str, Any]) -> list[dict[str, str]]:
    instruction = str(row.get("instruction") or "").strip()
    extra_input = str(row.get("input") or "").strip()
    prompt = f"{instruction}\n\n{extra_input}" if extra_input else instruction
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(row.get("output") or "").strip()},
    ]


def coconot_converter(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(row.get("prompt") or "").strip()},
        {"role": "assistant", "content": str(row.get("response") or "").strip()},
    ]


GROUPS = (
    SampleGroup(
        "english_general_smoltalk",
        smoltalk_file("everyday-conversations"),
        600,
        messages_converter,
    ),
    SampleGroup(
        "english_general_ultrachat",
        RawFile(
            "HuggingFaceH4/ultrachat_200k",
            ULTRACHAT_REVISION,
            "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        ),
        600,
        messages_converter,
    ),
    SampleGroup(
        "english_constraints",
        smoltalk_file("smol-constraints"),
        600,
        messages_converter,
    ),
    SampleGroup(
        "chinese_general",
        RawFile(
            "m-a-p/COIG-CQIA",
            COIG_REVISION,
            "coig_pc/coig_pc_core_sample.jsonl",
        ),
        800,
        coig_converter,
    ),
    SampleGroup(
        "rewrite",
        smoltalk_file("smol-rewrite"),
        200,
        messages_converter,
    ),
    SampleGroup(
        "summarize",
        smoltalk_file("smol-summarize"),
        200,
        messages_converter,
    ),
    SampleGroup(
        "math",
        smoltalk_file("metamathqa-50k"),
        300,
        messages_converter,
    ),
    SampleGroup(
        "code_python",
        RawFile(
            "allenai/tulu-3-sft-personas-code",
            "1412abe88dd2976af977260788e033013449f7b2",
            "data/train-00000-of-00001.parquet",
        ),
        100,
        messages_converter,
    ),
    SampleGroup(
        "safety_boundaries",
        RawFile(
            "allenai/coconot",
            "main",
            "original/train-00000-of-00001.parquet",
        ),
        89,
        coconot_converter,
    ),
)


README_FILES = (
    RawFile("HuggingFaceTB/smoltalk", SMOLTALK_REVISION, "README.md"),
    RawFile("HuggingFaceH4/ultrachat_200k", ULTRACHAT_REVISION, "README.md"),
    RawFile("m-a-p/COIG-CQIA", COIG_REVISION, "README.md"),
    RawFile("allenai/tulu-3-sft-personas-code", "main", "README.md"),
    RawFile("allenai/coconot", "main", "README.md"),
)


def raw_path(raw_root: Path, item: RawFile) -> Path:
    return raw_root / item.repo_dir / item.filename


def is_complete_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix == ".parquet":
        try:
            pq.read_metadata(path)
        except Exception:
            return False
    return True


def download_file(item: RawFile, destination: Path, endpoint: str) -> None:
    if is_complete_file(destination):
        print(f"Using existing raw file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and not destination.exists():
        partial.replace(destination)
    url = (
        f"{endpoint.rstrip('/')}/datasets/{item.repo_id}/resolve/"
        f"{item.revision}/{quote(item.filename, safe='/')}"
    )

    session = requests.Session()
    probe_headers = {"Range": "bytes=0-0", "Accept-Encoding": "identity"}
    for probe_attempt in range(1, 61):
        try:
            with session.get(url, headers=probe_headers, timeout=(15, 300)) as response:
                response.raise_for_status()
                content_range = response.headers.get("content-range", "")
                if response.status_code == 206 and "/" in content_range:
                    total_bytes = int(content_range.rsplit("/", 1)[1])
                else:
                    total_bytes = int(response.headers["content-length"])
            break
        except requests.RequestException:
            if probe_attempt == 60:
                raise
            time.sleep(min(probe_attempt * 2, 10))

    downloaded = destination.stat().st_size if destination.exists() else 0
    if downloaded > total_bytes:
        destination.unlink()
        downloaded = 0

    chunk_size = 8 * 1024 * 1024
    with tqdm(
        total=total_bytes,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        desc=destination.name,
    ) as progress:
        while downloaded < total_bytes:
            end = min(downloaded + chunk_size, total_bytes) - 1
            headers = {
                "Range": f"bytes={downloaded}-{end}",
                "Accept-Encoding": "identity",
            }
            for attempt in range(1, 61):
                try:
                    with session.get(
                        url,
                        headers=headers,
                        stream=True,
                        timeout=(15, 300),
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError(
                                f"Server ignored Range {headers['Range']}"
                            )
                        expected_range = f"bytes {downloaded}-{end}/"
                        actual_range = response.headers.get("content-range", "")
                        if not actual_range.startswith(expected_range):
                            raise RuntimeError(
                                f"Unexpected Content-Range: {actual_range}"
                            )
                        block = b"".join(response.iter_content(chunk_size=1024 * 1024))
                        if len(block) != end - downloaded + 1:
                            raise RuntimeError(
                                f"Short range: expected {end - downloaded + 1}, got {len(block)}"
                            )
                    with destination.open("ab") as output:
                        output.write(block)
                    downloaded = end + 1
                    progress.update(len(block))
                    break
                except (requests.RequestException, OSError, RuntimeError) as exc:
                    if attempt == 60:
                        raise RuntimeError(
                            f"Failed to download range {headers['Range']} from {url}: {exc}"
                        ) from exc
                    time.sleep(min(attempt * 2, 10))

    if not is_complete_file(destination):
        raise RuntimeError(f"Downloaded file is incomplete: {destination}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            return []
        normalized.append({"role": role, "content": content})
    roles = {message["role"] for message in normalized}
    if "user" not in roles or "assistant" not in roles:
        return []
    return normalized


def format_chatml(messages: list[dict[str, str]]) -> str:
    turns = [
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
        for message in messages
    ]
    turns.append("<|im_start|>assistant\n")
    return "\n".join(turns)


def stable_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def select_group(
    group: SampleGroup,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    seed: int,
    seen: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    indices = list(range(len(rows)))
    stable_rng(seed, group.name).shuffle(indices)
    selected = []
    stats = {"raw_rows": len(rows), "invalid": 0, "too_long": 0, "duplicate": 0}

    for index in indices:
        messages = normalize_messages(group.converter(rows[index]))
        if not messages:
            stats["invalid"] += 1
            continue
        text = format_chatml(messages)
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            stats["duplicate"] += 1
            continue
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if token_count > max_seq_length:
            stats["too_long"] += 1
            continue
        seen.add(fingerprint)
        selected.append({"text": text})
        if len(selected) == group.count:
            break

    if len(selected) != group.count:
        raise ValueError(
            f"{group.name}: requested {group.count}, but only {len(selected)} valid rows "
            f"fit max_seq_length={max_seq_length}."
        )
    stats["selected"] = len(selected)
    return selected, stats


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_readme(output_dir: Path, total: int, max_seq_length: int) -> None:
    content = f"""# WL Customer Support Replay 15%

This directory contains a deterministic {total}-row general replay set. Relative
to 19,777 customer-support rows, replay is 15.00% of the final mixed training
set (`{total} / (19777 + {total})`).

- `general_replay.jsonl`: all categories, deterministically shuffled.
- Category JSONL files: training-ready `{{\"text\": ChatML}}` subsets.
- `manifest.json`: sources, revisions, counts, filtering, and raw checksums.

Rows longer than {max_seq_length} tokens are excluded before sampling. Upstream
files and dataset cards are preserved in `data/raw/forgetting_replay15`; this
directory contains only derived files. Existing replay10 data is unchanged.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw/forgetting_replay15")
    parser.add_argument(
        "--output_dir", default="data/preprocessed/wl_customer_support_replay15"
    )
    parser.add_argument(
        "--tokenizer_path", default="outputs/outputs_wl_customer_support/merged_model"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", default=HF_ENDPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_files = {group.raw_file for group in GROUPS}
    for item in sorted(unique_files | set(README_FILES), key=lambda value: (value.repo_id, value.filename)):
        download_file(item, raw_path(raw_root, item), args.endpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    seen: set[str] = set()
    all_rows: list[dict[str, str]] = []
    group_stats = {}

    for group in GROUPS:
        rows = load_rows(raw_path(raw_root, group.raw_file))
        selected, stats = select_group(
            group, rows, tokenizer, args.max_seq_length, args.seed, seen
        )
        write_jsonl(output_dir / f"{group.name}.jsonl", selected)
        all_rows.extend(selected)
        group_stats[group.name] = stats
        print(f"{group.name}: {len(selected)}")

    stable_rng(args.seed, "combined").shuffle(all_rows)
    write_jsonl(output_dir / "general_replay.jsonl", all_rows)

    raw_manifest = []
    for item in sorted(unique_files, key=lambda value: (value.repo_id, value.filename)):
        path = raw_path(raw_root, item)
        raw_manifest.append(
            {
                "repo_id": item.repo_id,
                "revision": item.revision,
                "filename": item.filename,
                "local_path": str(path.as_posix()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "tokenizer_path": args.tokenizer_path,
        "ratio_definition": "replay_rows / (ticket_train_rows + replay_rows)",
        "ticket_train_rows": 19777,
        "replay_rows": len(all_rows),
        "final_replay_fraction": len(all_rows) / (19777 + len(all_rows)),
        "counts": {group.name: group.count for group in GROUPS},
        "group_stats": group_stats,
        "raw_files": raw_manifest,
        "output_files": [f"{group.name}.jsonl" for group in GROUPS]
        + ["general_replay.jsonl"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(output_dir, len(all_rows), args.max_seq_length)

    print("Done.")
    print(f"Replay rows: {len(all_rows)}")
    print(f"Final replay fraction: {manifest['final_replay_fraction']:.2%}")
    print(f"Raw dir: {raw_root.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
