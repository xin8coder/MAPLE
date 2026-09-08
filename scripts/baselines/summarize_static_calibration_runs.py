#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "liveopt_static": r"\method{} static",
    "react_tools": "ReAct",
    "optimai": "Retired ORLM alias",
    "orlm": "ORLM",
    "optimus": "OptiMUS",
    "optimai_2025": "OptimAI",
    "or_llm_agent_2025": "OR-LLM-Agent",
    "nl4opt": "NL4Opt",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge static solving rows and export summary artifacts.")
    parser.add_argument("--run-dir", action="append", default=[], help="Directory containing static_calibration_rows.jsonl.")
    parser.add_argument("--glob", action="append", default=[], help="Optional glob for run directories. May be repeated.")
    parser.add_argument("--out-dir", default="logs/static_calibration/merged_latest")
    parser.add_argument("--paper-tex", default="")
    parser.add_argument("--tex-macro", default="StaticSolvingNLPFourLPRows")
    parser.add_argument("--omit-solution-column", action="store_true")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for pattern in args.glob:
        run_dirs.extend(sorted(Path().glob(pattern)))
    # Explicit run directories are commonly used for repair/retry rows and
    # should override earlier globbed first-pass rows during latest-row merge.
    run_dirs.extend(Path(path) for path in args.run_dir)
    if not run_dirs:
        raise SystemExit("Provide --run-dir or --glob.")
    rows = merge_latest_rows(load_rows(run_dirs))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "static_calibration_merged_rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    summary = summarize(rows)
    summary["rows_path"] = str(rows_path)
    summary_path = out_dir / "static_calibration_merged_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.paper_tex:
        tex_path = Path(args.paper_tex)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(
            render_tex(summary, macro_name=args.tex_macro, include_solution_column=not args.omit_solution_column),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def load_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / "static_calibration_rows.jsonl"
        if not path.exists():
            path = run_dir / "static_calibration_merged_rows.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def merge_latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("method")))
        latest[key] = row
    return [latest[key] for key in sorted(latest)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    cases = {str(row.get("case_id")) for row in rows}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    methods: dict[str, dict[str, Any]] = {}
    ambiguous_rows = detect_common_nonmatches(rows)
    ambiguous_ids = {item["case_id"] for item in ambiguous_rows}
    for method, method_rows in sorted(by_method.items()):
        numeric = [row for row in method_rows if (row.get("reference_evaluation") or {}).get("status") == "numeric_compared"]
        matches = [row for row in numeric if (row.get("reference_evaluation") or {}).get("numeric_match") is True]
        strict_numeric = [row for row in numeric if str(row.get("case_id")) not in ambiguous_ids]
        strict_matches = [row for row in strict_numeric if (row.get("reference_evaluation") or {}).get("numeric_match") is True]
        official_rows = [row for row in method_rows if isinstance(row.get("official_evaluation"), dict)]
        official_matches = [row for row in official_rows if (row.get("official_evaluation") or {}).get("official_accuracy") is True]
        compile_success = [row for row in official_rows if (row.get("official_evaluation") or {}).get("compile_success") is True]
        runtime_success = [row for row in official_rows if (row.get("official_evaluation") or {}).get("runtime_success") is True]
        objective_exact = [row for row in official_rows if (row.get("official_evaluation") or {}).get("objective_exact") is True]
        solution_applicable = [
            row for row in official_rows if (row.get("official_evaluation") or {}).get("reference_solution_available") is True
        ]
        solution_exact = [row for row in solution_applicable if (row.get("official_evaluation") or {}).get("solution_exact") is True]
        compile_errors = [row for row in official_rows if (row.get("official_evaluation") or {}).get("compile_success") is False]
        runtime_errors = [
            row
            for row in official_rows
            if (row.get("official_evaluation") or {}).get("compile_success") is True
            and (row.get("official_evaluation") or {}).get("runtime_success") is False
        ]
        tokens = [int((row.get("token_usage") or {}).get("total_tokens") or 0) for row in method_rows]
        latency = [float(row.get("latency_seconds") or 0.0) for row in method_rows]
        methods[method] = {
            "rows": len(method_rows),
            "completed": sum(1 for row in method_rows if row.get("status") == "completed"),
            "schema_failed": sum(1 for row in method_rows if row.get("status") == "schema_failed"),
            "failed": sum(1 for row in method_rows if row.get("status") not in {"completed", "schema_failed"}),
            "numeric_total": len(numeric),
            "numeric_matches": len(matches),
            "end_to_end_numeric_success_rate": len(matches) / len(method_rows) if method_rows else None,
            "numeric_match_rate": len(matches) / len(numeric) if numeric else None,
            "reference_audited_numeric_total": len(strict_numeric),
            "reference_audited_numeric_matches": len(strict_matches),
            "reference_audited_numeric_match_rate": len(strict_matches) / len(strict_numeric) if strict_numeric else None,
            "ambiguity_adjusted_total": len(strict_numeric),
            "ambiguity_adjusted_matches": len(strict_matches),
            "ambiguity_adjusted_match_rate": len(strict_matches) / len(strict_numeric) if strict_numeric else None,
            "official_total": len(official_rows),
            "official_accuracy": len(official_matches),
            "official_accuracy_rate": len(official_matches) / len(official_rows) if official_rows else None,
            "compile_success": len(compile_success),
            "compile_success_rate": len(compile_success) / len(official_rows) if official_rows else None,
            "runtime_success": len(runtime_success),
            "runtime_success_rate": len(runtime_success) / len(official_rows) if official_rows else None,
            "objective_exact": len(objective_exact),
            "objective_exact_rate": len(objective_exact) / len(official_rows) if official_rows else None,
            "solution_exact_total": len(solution_applicable),
            "solution_exact": len(solution_exact),
            "solution_exact_rate": len(solution_exact) / len(solution_applicable) if solution_applicable else None,
            "compile_errors": len(compile_errors),
            "compile_error_rate": len(compile_errors) / len(official_rows) if official_rows else None,
            "runtime_errors": len(runtime_errors),
            "runtime_error_rate": len(runtime_errors) / len(official_rows) if official_rows else None,
            "mean_tokens": statistics.mean(tokens) if tokens else None,
            "median_tokens": statistics.median(tokens) if tokens else None,
            "mean_latency_seconds": statistics.mean(latency) if latency else None,
            "median_latency_seconds": statistics.median(latency) if latency else None,
        }
    return {
        "schema_version": "liveopt_static_solving_merged_summary_v2",
        "row_count": len(rows),
        "case_count": len(cases),
        "methods": methods,
        "common_nonmatch_ambiguities": ambiguous_rows,
    }


def detect_common_nonmatches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row.get("case_id")), []).append(row)
    ambiguous: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(by_case.items()):
        numeric = [row for row in case_rows if (row.get("reference_evaluation") or {}).get("status") == "numeric_compared"]
        if len(numeric) < 2:
            continue
        nonmatches = [row for row in numeric if (row.get("reference_evaluation") or {}).get("numeric_match") is False]
        if len(nonmatches) == len(numeric):
            sample = numeric[0]
            evaluations = [
                {
                    "method": row.get("method"),
                    "reference": (row.get("reference_evaluation") or {}).get("reference_value"),
                    "prediction": (row.get("reference_evaluation") or {}).get("predicted_value"),
                }
                for row in numeric
            ]
            ambiguous.append(
                {
                    "case_id": case_id,
                    "source_id": sample.get("source_id"),
                    "problem_text": str(sample.get("problem_text") or "")[:600],
                    "evaluations": evaluations,
                }
            )
    return ambiguous


def render_tex(
    summary: dict[str, Any],
    *,
    macro_name: str = "StaticSolvingNLPFourLPRows",
    include_solution_column: bool = True,
) -> str:
    rows = [
        "% Auto-generated by scripts/baselines/summarize_static_calibration_runs.py",
        rf"\newcommand{{\{macro_name}}}{{%",
    ]
    for method, metrics in sorted(summary["methods"].items()):
        label = METHOD_LABELS.get(method, method.replace("_", r"\_"))
        official = fmt_rate(metrics.get("official_accuracy_rate"))
        compile_rate = fmt_rate(metrics.get("compile_success_rate"))
        runtime_rate = fmt_rate(metrics.get("runtime_success_rate"))
        objective_rate = fmt_rate(metrics.get("objective_exact_rate"))
        solution_rate = fmt_rate(metrics.get("solution_exact_rate"))
        mean_tokens = fmt_number(metrics.get("mean_tokens"))
        mean_latency = fmt_number(metrics.get("mean_latency_seconds"))
        cells = [
            label,
            f"{metrics.get('official_accuracy', 0)}/{metrics.get('official_total', 0)} ({official})",
            f"{metrics.get('compile_success', 0)}/{metrics.get('official_total', 0)} ({compile_rate})",
            f"{metrics.get('runtime_success', 0)}/{metrics.get('official_total', 0)} ({runtime_rate})",
            f"{metrics.get('objective_exact', 0)}/{metrics.get('official_total', 0)} ({objective_rate})",
        ]
        if include_solution_column:
            solution_total = metrics.get("solution_exact_total", metrics.get("official_total", 0))
            cells.append(f"{metrics.get('solution_exact', 0)}/{solution_total} ({solution_rate})")
        cells.extend([mean_tokens, f"{mean_latency}s"])
        rows.append(" & ".join(cells) + r" \\")
    rows.append("}")
    return "\n".join(rows) + "\n"


def fmt_rate(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}\\%"


def fmt_number(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):,.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
