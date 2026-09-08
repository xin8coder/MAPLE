#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ENTITY_KEYS = {"item", "facility", "customer", "set", "machine", "job", "nurse", "day", "order", "vehicle", "jobs"}


def main() -> int:
    args = parse_args()
    episodes = {str(row["episode_id"]): row for row in load_jsonl(args.episodes_jsonl)}
    audited: dict[tuple[str, str], dict[str, Any]] = {}
    for run in load_jsonl(args.run_jsonl):
        episode_id = str(run.get("episode_id") or "")
        episode = episodes.get(episode_id)
        if not episode:
            continue
        oracle = list(episode.get("hidden_update_oracle") or [])
        public_updates = list(episode.get("update_stream") or [])
        for index, result in enumerate(run.get("update_results") or []):
            if index >= len(oracle) or index >= len(public_updates):
                continue
            difficulty = str(public_updates[index].get("difficulty") or "")
            if difficulty != "memory":
                continue
            update_id = str(result.get("update_id") or public_updates[index].get("update_id") or f"t{index + 1:02d}")
            expected = sorted(expected_entity_tokens((oracle[index].get("hidden_delta") or {})))
            patch = ((result.get("restart") or {}).get("data_patch") or {})
            patch_text = json.dumps(patch, ensure_ascii=False, sort_keys=True)
            missing = [token for token in expected if token not in patch_text]
            key = (episode_id, update_id)
            item = audited.setdefault(
                key,
                {
                    "episode_id": episode_id,
                    "update_id": update_id,
                    "expected_entities": expected,
                    "patch": patch,
                    "seeds": 0,
                    "seeds_with_missing_entity": 0,
                },
            )
            item["seeds"] += 1
            item["seeds_with_missing_entity"] += int(bool(missing))
            item["missing_entities"] = missing
    rows = list(audited.values())
    failed = [row for row in rows if row["seeds_with_missing_entity"]]
    report = {
        "schema_version": "nldo_public_patch_grounding_audit_v1",
        "run_jsonl": str(args.run_jsonl),
        "memory_update_states": len(rows),
        "states_with_missing_expected_entity": len(failed),
        "passed": not failed,
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit memory-update public patches against withheld grounding entities.")
    parser.add_argument("--episodes-jsonl", type=Path, default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"))
    parser.add_argument("--run-jsonl", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_entity_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in ENTITY_KEYS:
                if isinstance(item, list):
                    tokens.update(str(entry) for entry in item)
                elif item is not None:
                    tokens.add(str(item))
            if str(key) == "changes":
                tokens.update(expected_entity_tokens(item))
    elif isinstance(value, list):
        for item in value:
            tokens.update(expected_entity_tokens(item))
    return tokens


if __name__ == "__main__":
    raise SystemExit(main())
