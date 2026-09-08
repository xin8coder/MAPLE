#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.baselines.run_static_calibration import evaluate_official_static, evaluate_reference, resolve_manifest_case_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute local static solving evaluations without new LLM calls.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rows", required=True, help="Input static_calibration_rows.jsonl.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases_path = resolve_manifest_case_file(manifest_path, str(manifest.get("case_file") or ""))
    cases = {str(row.get("case_id")): row for row in load_jsonl(cases_path)}
    rows = load_jsonl(Path(args.rows))

    updated: list[dict[str, Any]] = []
    for row in rows:
        case = cases.get(str(row.get("case_id")), {})
        parsed = row.get("parsed_output") if isinstance(row.get("parsed_output"), dict) else {}
        row = dict(row)
        row["reference_evaluation"] = evaluate_reference(case, parsed)
        row["official_evaluation"] = evaluate_official_static(case, parsed)
        updated.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "static_calibration_rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in updated) + ("\n" if updated else ""),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(updated), "rows_path": str(rows_path)}, indent=2, sort_keys=True))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
