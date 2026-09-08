#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


TEXT_KEYS = (
    "en_question",
    "question",
    "question_text",
    "problem",
    "problem_text",
    "statement",
    "natural_language",
    "description",
    "text",
    "src",
    "prompt",
)
REFERENCE_KEYS = (
    "en_answer",
    "answer",
    "solution",
    "solution_status",
    "status",
    "model",
    "mathematical_model",
    "label",
    "target",
    "optimum",
    "objective_value",
)
ID_KEYS = ("id", "index", "case_id", "problem_id", "name")
PUBLIC_CONTEXT_KEYS = (
    "parameters",
    "metadata",
    "keywords",
    "type",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize static NL optimization solving benchmarks such as NLP4LP, OptiBench, or IndustryOR "
            "into a cache-friendly calibration manifest."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=[],
        help="Input JSONL/JSON/CSV file. May be repeated.",
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=[],
        help="Source name for each input file, e.g. NLP4LP, OptiBench, or IndustryOR. May be repeated.",
    )
    parser.add_argument("--limit-per-source", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--out-dir", default="logs/static_calibration/manifest_latest")
    parser.add_argument(
        "--methods",
        default="liveopt_static,optimai_2025,or_llm_agent_2025,nl4opt,orlm,optimus",
        help="Comma-separated method ids to include in the manifest.",
    )
    args = parser.parse_args()

    inputs = [Path(item) for item in args.input_jsonl]
    if not inputs:
        raise SystemExit("At least one --input-jsonl file is required.")
    if args.source_name and len(args.source_name) != len(inputs):
        raise SystemExit("--source-name must be omitted or repeated once per input file.")
    source_names = args.source_name or [path.stem for path in inputs]

    rng = random.Random(args.seed)
    cases: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path, source in zip(inputs, source_names):
        records = load_records(path)
        normalized = [normalize_record(row, source, idx, path) for idx, row in enumerate(records)]
        normalized = [row for row in normalized if row.get("problem_text")]
        if args.limit_per_source > 0 and len(normalized) > args.limit_per_source:
            normalized = rng.sample(normalized, args.limit_per_source)
            normalized.sort(key=lambda row: row["case_id"])
        cases.extend(normalized)
        source_counts[source] = len(normalized)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "static_calibration_cases.jsonl"
    manifest_path = out_dir / "static_calibration_manifest.json"
    readme_path = out_dir / "static_calibration_readme.md"

    cases_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in cases) + ("\n" if cases else ""),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "liveopt_static_calibration_manifest_v1",
        "description": "Static natural-language optimization calibration manifest.",
        "case_file": str(cases_path),
        "case_count": len(cases),
        "source_counts": source_counts,
        "methods": methods,
        "selection": {
            "limit_per_source": args.limit_per_source,
            "seed": args.seed,
        },
        "inputs": [
            {"path": str(path), "source": source, "sha256": file_sha256(path)}
            for path, source in zip(inputs, source_names)
        ],
        "evaluation_policy": {
            "primary_metrics": [
                "parse_success",
                "compile_success",
                "runtime_success",
                "objective_exact",
                "solution_exact",
                "official_accuracy",
                "tokens",
                "latency_seconds",
            ],
            "no_dynamic_memory": True,
            "no_restart_advantage": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    readme_path.write_text(render_readme(manifest), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "case_count": len(cases), "source_counts": source_counts}, indent=2))
    return 0


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "rows", "examples", "instances"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [payload]
    rows = []
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def normalize_record(row: dict[str, Any], source: str, index: int, path: Path) -> dict[str, Any]:
    problem_text = first_string(row, TEXT_KEYS)
    source_id = first_string(row, ID_KEYS) or str(index)
    reference = {key: row[key] for key in REFERENCE_KEYS if key in row and row[key] not in (None, "")}
    public_context = {key: parse_possible_json(row[key]) for key in PUBLIC_CONTEXT_KEYS if key in row and row[key] not in (None, "")}
    case_hash = stable_hash({"source": source, "source_id": source_id, "problem_text": problem_text})
    source_lower = source.lower()
    metadata: dict[str, Any] = {
        "row_index": index,
        "available_fields": sorted(row.keys()),
        "static_only": True,
        "dynamic_updates": [],
    }
    if source_lower == "bwor":
        metadata.update(
            {
                "official_metric": "objective_or_status_accuracy",
                "official_tolerance_abs": 0.1,
                "official_tolerance_rel": 0.0,
                "reference_solution_required": False,
                "official_source_repository": "https://github.com/bwz96sco/or_llm_agent",
                "official_dataset_file": "data/datasets/bwor.jsonl",
            }
        )
    return {
        "schema_version": "liveopt_static_calibration_case_v1",
        "case_id": f"{source.lower()}_{case_hash[:12]}",
        "source": source,
        "source_file": str(path),
        "source_id": source_id,
        "problem_text": problem_text,
        "public_context": public_context,
        "reference": reference,
        "metadata": metadata,
    }


def parse_possible_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_readme(manifest: dict[str, Any]) -> str:
    rows = [
        "# Static Calibration Manifest",
        "",
        f"Cases: `{manifest['case_count']}`",
        "",
        "## Sources",
        "",
    ]
    for source, count in sorted(manifest["source_counts"].items()):
        rows.append(f"- `{source}`: {count} cases")
    rows.extend(
        [
            "",
            "## Methods",
            "",
        ]
    )
    for method in manifest["methods"]:
        rows.append(f"- `{method}`")
    rows.extend(
        [
            "",
            "## Evaluation Boundary",
            "",
            "- Static-only: no dynamic memory, no accepted-state carryover, no restart advantage.",
            "- Public problem text and optional public labels/reference fields only.",
            "- Report parse success, solver execution success, feasible/label match, objective gap when available, tokens, and latency.",
        ]
    )
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
