#!/usr/bin/env python3
"""Check that the frozen LiveOpt artifact matches the current paper sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "release_artifacts/anonymous_frozen_results"
DEFAULT_PAPER_TABLES = ROOT / "release_artifacts/paper_table_exports"
DEFAULT_MAIN_SUMMARY = ROOT / "outputs/main_dynamic_hv_heatmap_20260715/summary.json"
DEFAULT_STRATA_SUMMARY = ROOT / "logs/analysis/liveopt_update_strata_current_20260715/summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--paper-tables", type=Path, default=DEFAULT_PAPER_TABLES)
    parser.add_argument("--main-summary", type=Path, default=DEFAULT_MAIN_SUMMARY)
    parser.add_argument("--strata-summary", type=Path, default=DEFAULT_STRATA_SUMMARY)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items() if key not in {"benchmark"}}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT.resolve()) + "/", "")
    return value


def json_equal(left: Path, right: Path) -> bool:
    return normalized(json.loads(left.read_text(encoding="utf-8"))) == normalized(
        json.loads(right.read_text(encoding="utf-8"))
    )


def find_row(rows: list[dict[str, Any]], view: str, stratum: str, scope: str = "all15") -> dict[str, Any]:
    for row in rows:
        if row.get("view") == view and row.get("stratum") == stratum and row.get("scope") == scope:
            return row
    raise KeyError((view, stratum, scope))


def main() -> int:
    args = parse_args()
    artifact = args.artifact.resolve()
    paper_tables = args.paper_tables.resolve()
    errors: list[str] = []

    artifact_tables = artifact / "paper_tables"
    for frozen in sorted(artifact_tables.glob("*.tex")):
        current = paper_tables / frozen.name
        if not current.is_file():
            errors.append(f"paper table missing: {frozen.name}")
        elif frozen.read_bytes() != current.read_bytes():
            errors.append(f"paper table differs: {frozen.name}")

    frozen_main = artifact / "evidence/main_dynamic_summary.json"
    if not json_equal(frozen_main, args.main_summary.resolve()):
        errors.append("main dynamic summary differs from the current source")

    frozen_strata = artifact / "evidence/strata/summary.json"
    if not json_equal(frozen_strata, args.strata_summary.resolve()):
        errors.append("update-strata summary differs from the current source")

    main_summary = json.loads(frozen_main.read_text(encoding="utf-8"))
    liveopt = main_summary["methods"]["LiveOpt"]
    if liveopt["ds"]["cells"] != 117 or liveopt["dm"]["cells"] != 78:
        errors.append("Table 2 state counts are not 117 DS and 78 DM")
    if round(float(liveopt["ds"]["mean_quality"]), 3) != 0.951:
        errors.append("Table 2 DS quality is not 0.951")
    if round(float(liveopt["dm"]["mean_hv"]), 3) != 0.779:
        errors.append("Table 2 DM quality is not 0.779")

    strata = json.loads(frozen_strata.read_text(encoding="utf-8"))
    ds_large = find_row(strata["rows"], "DS", "large")
    dm_large = find_row(strata["rows"], "DM", "large")
    if int(ds_large["states"]) != 6 or round(float(ds_large["quality"]), 3) != 0.684:
        errors.append("DS regime-shift stratum is not 6 states at 0.684")
    if int(dm_large["states"]) != 12 or round(float(dm_large["quality"]), 3) != 0.494:
        errors.append("DM regime-shift stratum is not 12 states at 0.494")

    claims = json.loads(
        (artifact / "metadata/claim_registry.json").read_text(encoding="utf-8")
    )["claims"]
    claim_map = {row["claim_id"]: row["claim"] for row in claims}
    for claim_id, tokens in {
        "C01": ("117 DS", "78 DM", "0.951", "0.779"),
        "C09": ("0.684", "0.494"),
        "C20": ("100.0%", "91.7%", "0.774", "0.618"),
        "C12": ("127", "135", "candidate population", "no row contains LiveOpt state"),
        "C21": ("P007-t03", "P008-t06", "P009-t07"),
    }.items():
        claim = claim_map.get(claim_id, "")
        for token in tokens:
            if token not in claim:
                errors.append(f"{claim_id} is missing current token {token}")

    report = {
        "status": "failed" if errors else "passed",
        "artifact": str(artifact),
        "paper_tables_checked": len(list(artifact_tables.glob("*.tex"))),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
