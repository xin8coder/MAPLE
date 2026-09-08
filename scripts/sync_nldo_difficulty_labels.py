#!/usr/bin/env python3
"""Retag NLDO difficulty labels: only t11--t12 regime replacements stay "large".

The deliberately disruptive replacements are the fixed t11--t12 subset; other
updates previously labeled "large" are compound objective/priority edits over
an unchanged entity set and are retagged "objective" so the annotation matches
the paper's narrative.  Only ``difficulty`` fields are rewritten.

Usage: python scripts/sync_nldo_difficulty_labels.py [--jsonl PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = ROOT / "data" / "evo2_dynoptbench" / "public_csv" / "nldo_15episodes_12updates_csv.jsonl"
REGIME_STAGES = {11, 12}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    changed = 0
    for episode in lines:
        for update in episode.get("update_stream", []):
            stage = int(update.get("time_index") or str(update.get("update_id"))[-2:])
            difficulty = str(update.get("difficulty") or "")
            before[difficulty] += 1
            if difficulty == "large" and stage not in REGIME_STAGES:
                update["difficulty"] = "objective"
                changed += 1
            after[update["difficulty"]] += 1
    if not args.dry_run and changed:
        args.jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in lines) + "\n", encoding="utf-8")
    print(json.dumps({"jsonl": str(args.jsonl), "retagged": changed, "before": dict(before), "after": dict(after)}, indent=2))


if __name__ == "__main__":
    main()
