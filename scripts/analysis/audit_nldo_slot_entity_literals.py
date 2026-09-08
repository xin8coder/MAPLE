#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable


ENTITY_LITERAL = re.compile(r"^(?:[A-Z]{1,4}\d{1,4}|[A-Z]\d{1,3}_\d{1,3}|B\d+_\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit generated NLDO Workbench slots for hard-coded public entity identifiers."
    )
    parser.add_argument("--run-jsonl", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    selected = set(args.episode_id or [])
    finding_map: dict[tuple[str, str, str, str, int], set[int | str | None]] = {}
    audited_rows = 0
    audited_slots = 0
    for row in load_jsonl(args.run_jsonl, selected):
        episode_id = str(row.get("episode_id") or "")
        audited_rows += 1
        stage_slots = [("initial", row.get("setup_code"), row.get("fitness_code"))]
        for stage in row.get("update_results") or []:
            if not isinstance(stage, dict):
                continue
            stage_slots.append(
                (
                    str(stage.get("update_id") or ""),
                    stage.get("setup_code"),
                    stage.get("fitness_code"),
                )
            )
        for update_id, setup_code, fitness_code in stage_slots:
            for slot_name, code in (("setup.py", setup_code), ("fitness.py", fitness_code)):
                if not isinstance(code, str) or not code.strip():
                    continue
                audited_slots += 1
                for literal, lineno in entity_literals(code):
                    key = (episode_id, update_id, slot_name, literal, lineno)
                    finding_map.setdefault(key, set()).add(row.get("run_seed"))

    findings = [
        {
            "episode_id": episode_id,
            "update_id": update_id,
            "slot": slot_name,
            "literal": literal,
            "lineno": lineno,
            "seed_count": len(seeds),
            "seeds": sorted(seeds, key=lambda value: (value is None, str(value))),
        }
        for (episode_id, update_id, slot_name, literal, lineno), seeds in sorted(finding_map.items())
    ]

    payload = {
        "schema_version": "nldo_slot_entity_literal_audit_v1",
        "run_jsonl": str(args.run_jsonl),
        "audited_rows": audited_rows,
        "audited_slots": audited_slots,
        "finding_count": len(findings),
        "passed": not findings,
        "findings": findings,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and findings else 0


def load_jsonl(path: Path, selected: set[str]) -> Iterable[dict[str, Any]]:
    handle = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")
    with handle:
        for line in handle:
            if not line.strip():
                continue
            if selected and not any(f'"episode_id":"{episode_id}"' in line or f'"episode_id": "{episode_id}"' in line for episode_id in selected):
                continue
            row = json.loads(line)
            if selected and str(row.get("episode_id") or "") not in selected:
                continue
            yield row


def entity_literals(code: str) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.strip()
        if ENTITY_LITERAL.fullmatch(value):
            found.append((value, int(getattr(node, "lineno", 0) or 0)))
    return found


if __name__ == "__main__":
    raise SystemExit(main())
