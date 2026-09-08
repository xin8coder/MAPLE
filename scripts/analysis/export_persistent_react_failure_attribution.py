#!/usr/bin/env python3
"""Export first-break attribution for Persistent ReAct rostering runs.

The diagnostic is intentionally narrow: it summarizes the three recorded
P007--P009 controller trajectories used by the paper. It does not estimate a
population-level failure distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "logs/llm_tests/tss_dynamic_public_persistent_react_final_merged_20260721"
    / "dynamic_public_baseline_stage_rows.jsonl"
)
DEFAULT_OUT = ROOT / "logs/analysis/persistent_react_failure_attribution_20260727"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/persistent_react_failure_attribution_rows.tex"
EPISODES = ("NLDO-P007", "NLDO-P008", "NLDO-P009")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_TEX)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") != "persistent_react" or row.get("episode_id") not in EPISODES:
                continue
            key = (str(row["episode_id"]), int(row["stage_index"]))
            if key in rows:
                raise ValueError(f"duplicate stage row: {key}")
            rows[key] = row
    return rows


def is_valid(row: dict[str, Any]) -> bool:
    hidden = row.get("hidden_evaluation") or {}
    return row.get("status") == "completed" and parse_bool(hidden.get("feasible"))


def first_break(
    rows: dict[tuple[str, int], dict[str, Any]], episode_id: str
) -> tuple[int, dict[str, Any]]:
    for stage in range(13):
        row = rows.get((episode_id, stage))
        if row is None:
            raise ValueError(f"missing row before first break: {episode_id} t{stage:02d}")
        if not is_valid(row):
            return stage, row
    raise ValueError(f"trajectory has no recorded break: {episode_id}")


def local_coverage_shortage(row: dict[str, Any]) -> float | None:
    output = (row.get("code_execution") or {}).get("output") or {}
    diagnostics = output.get("score_components")
    if not isinstance(diagnostics, dict):
        diagnostics = (
            ((output.get("final_answer") or {}).get("solution") or {}).get("diagnostics")
        )
    if not isinstance(diagnostics, dict) or diagnostics.get("coverage_shortage") is None:
        return None
    return float(diagnostics["coverage_shortage"])


def classify(stage: int, row: dict[str, Any]) -> dict[str, Any]:
    hidden = row.get("hidden_evaluation") or {}
    heldout_shortage = float(hidden.get("score_coverage_shortage") or 0.0)
    local_shortage = local_coverage_shortage(row)
    status = str(row.get("status") or "")
    reason = str(row.get("failure_reason") or "")

    if status == "schema_failed" and "react_trace" in reason:
        category = "Missing output field"
        evidence = (
            "The code compiled and ran, but a required reasoning field "
            "was still missing after repair."
        )
    elif (
        status == "hidden_rejected"
        and local_shortage is not None
        and local_shortage <= 1e-12
        and heldout_shortage > 0
    ):
        category = "Incorrect constraint check"
        evidence = (
            f"The executable reported zero shortage; held-out evaluation found "
            f"{heldout_shortage:g} uncovered roster slots."
        )
    elif status == "hidden_rejected" and heldout_shortage > 0:
        category = "Infeasible search result"
        local_text = (
            f"{local_shortage:g}" if local_shortage is not None else "an unspecified number of"
        )
        evidence = (
            f"The executable reported {local_text} uncovered slots and held-out "
            f"evaluation confirmed {heldout_shortage:g}."
        )
    else:
        raise ValueError(
            f"unrecognized first break at t{stage:02d}: status={status!r}, reason={reason!r}"
        )

    code = row.get("code_execution") or {}
    return {
        "episode_id": str(row["episode_id"]),
        "first_break_stage": stage,
        "valid_prefix_states": stage,
        "status": status,
        "failure_reason": reason,
        "category": category,
        "evidence": evidence,
        "code_compile_success": parse_bool(code.get("compile_success")),
        "code_runtime_success": parse_bool(code.get("runtime_success")),
        "local_coverage_shortage": local_shortage,
        "heldout_coverage_shortage": heldout_shortage,
        "repair_attempts_used": int((row.get("code_output_repair") or {}).get("attempts_used") or 0),
        "trace_path": str(row.get("trace_path") or ""),
    }


def write_tex(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "% Generated by scripts/analysis/export_persistent_react_failure_attribution.py.",
        r"\newcommand{\PersistentReactFailureAttributionRows}{%",
    ]
    for index, record in enumerate(records):
        if index % 2:
            lines.append(r"\rowcolor{LiveOptBaselineRowB}")
        episode = record["episode_id"].replace("NLDO-", "")
        lines.append(
            f"{episode} & t{record['first_break_stage']:02d} & "
            f"{record['category']} & {record['evidence']} " + r"\\"
        )
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    rows = load_rows(source)
    records = [classify(*first_break(rows, episode_id)) for episode_id in EPISODES]

    expected = {
        "NLDO-P007": (3, "Incorrect constraint check", 119.0),
        "NLDO-P008": (6, "Missing output field", 1.0),
        "NLDO-P009": (7, "Infeasible search result", 2.0),
    }
    for record in records:
        stage, category, heldout_shortage = expected[record["episode_id"]]
        if (
            record["first_break_stage"] != stage
            or record["category"] != category
            or record["heldout_coverage_shortage"] != heldout_shortage
        ):
            raise ValueError(f"unexpected attribution record: {record}")

    summary = {
        "schema_version": "persistent_react_first_break_attribution_v1",
        "scope": {
            "method": "persistent_react",
            "episodes": list(EPISODES),
            "controller_trajectories": 3,
            "interpretation": "observed first breaks; not an estimated failure distribution",
        },
        "sequential_validity_rule": (
            "A stage is valid only when status=completed and held-out feasibility=true; "
            "the first invalid stage terminates the scored prefix."
        ),
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_tex(args.paper_table, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
