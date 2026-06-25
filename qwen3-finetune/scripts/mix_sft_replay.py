"""Deterministically mix an SFT dataset with a general replay dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"text"} or not isinstance(row["text"], str):
                raise ValueError(f"{path}:{line_number} must contain only a text field")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sft_path = Path(args.sft)
    replay_path = Path(args.replay)
    output_path = Path(args.output)

    sft_rows = read_jsonl(sft_path)
    replay_rows = read_jsonl(replay_path)
    mixed_rows = [("sft", row) for row in sft_rows]
    mixed_rows.extend(("replay", row) for row in replay_rows)
    random.Random(args.seed).shuffle(mixed_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for _, row in mixed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "sft": {"path": str(sft_path.as_posix()), "rows": len(sft_rows)},
        "replay": {"path": str(replay_path.as_posix()), "rows": len(replay_rows)},
        "mixed": {
            "path": str(output_path.as_posix()),
            "rows": len(mixed_rows),
            "replay_fraction": len(replay_rows) / len(mixed_rows),
            "sha256": sha256(output_path),
        },
    }
    manifest_path = output_path.with_name("mixed_train_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("Done.")
    print(f"SFT rows: {len(sft_rows)}")
    print(f"Replay rows: {len(replay_rows)}")
    print(f"Mixed rows: {len(mixed_rows)}")
    print(f"Replay fraction: {manifest['mixed']['replay_fraction']:.2%}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
