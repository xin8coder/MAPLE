#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ANNOTATION = re.compile(
    r"^\s*\((?:simple|memory|maintenance|objective|large|local[- ]numeric|referential|regime[- ]shift)\)\s*",
    re.IGNORECASE,
)
TEXT_KEYS = {"natural_language_update", "public_update", "selected_stage_prompt"}


def main() -> int:
    args = parse_args()
    leaks: list[str] = []
    errors: list[str] = []
    checked = 0
    checked_by_key: Counter[str] = Counter()
    sources: list[dict[str, Any]] = []
    for path in args.path:
        if not path.exists():
            errors.append(f"missing input: {path}")
            continue
        input_checked = 0
        files = list_json_files(path)
        for child in files:
            child_checked = 0
            for source, payload in load_records(child):
                for key, text in iter_prompt_text(payload):
                    checked += 1
                    input_checked += 1
                    child_checked += 1
                    checked_by_key[key.rsplit(".", 1)[-1]] += 1
                    if ANNOTATION.search(text):
                        leaks.append(f"{source}:{key}: {text[:120]}")
            sources.append(
                {
                    "path": str(child),
                    "sha256": sha256_file(child),
                    "checked_strings": child_checked,
                }
            )
        if input_checked == 0:
            errors.append(f"no prompt/update strings found in input: {path}")
    if checked < args.min_checked:
        errors.append(f"expected at least {args.min_checked} checked strings, found {checked}")
    report = {
        "schema_version": "nldo_prompt_annotation_audit_v2",
        "checked_strings": checked,
        "checked_by_key": dict(sorted(checked_by_key.items())),
        "annotation_pattern": ANNOTATION.pattern,
        "annotation_leaks": leaks,
        "sources": sources,
        "errors": errors,
        "passed": not leaks and not errors,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(rendered, encoding="utf-8")
    return 1 if leaks or errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that appendix-only NLDO update-type annotations never enter prompts or public streams."
    )
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        default=[
            Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"),
            Path("release_artifacts/public_preview/data/public_streams/nldo_public_streams.jsonl"),
        ],
    )
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--min-checked", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl"))


def load_records(path: Path) -> Iterable[tuple[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield f"{path}:{line_number}", json.loads(line)
        return
    yield str(path), json.loads(path.read_text(encoding="utf-8"))


def iter_prompt_text(value: Any, prefix: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}"
            if str(key) in TEXT_KEYS and isinstance(item, str):
                yield child, item
            yield from iter_prompt_text(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_prompt_text(item, f"{prefix}[{index}]")


if __name__ == "__main__":
    raise SystemExit(main())
