#!/usr/bin/env python3
"""Audit and export the P010/P015 state-binding factorial evidence.

The language controller is evaluated over three independently worded requests.
The numerical replay freezes the controller output from wording 0 and reruns
the six branches with three 200x200 search seeds.  Missing-state arms are
expected to fail only when the requested binding cannot be recovered from the
state made visible to that arm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_liveopt_state_binding_challenge import BRANCHES


DEFAULT_CONTROLLERS = (
    ROOT / "logs/llm_tests/liveopt_state_binding_controller_pilot_v3_20260717/controller_results.jsonl",
    ROOT / "logs/llm_tests/liveopt_state_binding_controller_q1_v3_20260717/controller_results.jsonl",
    ROOT / "logs/llm_tests/liveopt_state_binding_controller_q2_v3_20260717/controller_results.jsonl",
)
DEFAULT_FORMAL = (
    ROOT
    / "outputs/liveopt_state_binding_formal_q0_200x200x3_20260717/formal_results_enriched.jsonl"
)
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_state_binding_evidence_20260717"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_state_binding_factorial.tex"

VIEW_ORDER = ("full", "ledger_only", "accepted_only", "current_only", "oracle")
VIEW_LABELS = {
    "full": r"Full LSM",
    "ledger_only": r"Stored events only",
    "accepted_only": r"Accepted state only",
    "current_only": r"Current request only",
    "oracle": r"Direct relation",
}
VIEW_CHANNELS = {
    "full": frozenset(("event_ledger", "accepted_output")),
    "ledger_only": frozenset(("event_ledger",)),
    "accepted_only": frozenset(("accepted_output",)),
    "current_only": frozenset(),
    "oracle": frozenset(("event_ledger", "accepted_output")),
}
TASK_ORDER = ("event_only", "accepted_only", "event_and_accepted")
TASK_LABELS = {
    "event_only": "Event",
    "accepted_only": "Accepted",
    "event_and_accepted": "Linked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-results", type=Path, action="append", default=[])
    parser.add_argument("--formal-results", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def expected_success(branch_id: str, view: str) -> bool:
    spec = next(spec for spec in BRANCHES if spec.branch_id == branch_id)
    return set(spec.required_channels).issubset(VIEW_CHANNELS[view])


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def controller_evidence(
    paths: tuple[Path, ...], errors: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    branch_ids = {spec.branch_id for spec in BRANCHES}
    for expected_index, path in enumerate(paths):
        if not path.exists():
            errors.append(f"controller result missing: {repo_path(path)}")
            continue
        current = load_jsonl(path)
        if len(current) != len(BRANCHES) * len(VIEW_ORDER):
            errors.append(
                f"{repo_path(path)}: row count {len(current)}, expected {len(BRANCHES) * len(VIEW_ORDER)}"
            )
        indices = {int(row.get("paraphrase_index", -1)) for row in current}
        if indices != {expected_index}:
            errors.append(f"{repo_path(path)}: paraphrase indices {sorted(indices)}, expected [{expected_index}]")
        rows.extend(current)

    expected_keys = {
        (index, spec.branch_id, view)
        for index in range(len(paths))
        for spec in BRANCHES
        for view in VIEW_ORDER
    }
    observed_keys = [
        (int(row.get("paraphrase_index", -1)), str(row.get("branch_id")), str(row.get("memory_view")))
        for row in rows
    ]
    if len(set(observed_keys)) != len(observed_keys):
        errors.append("controller results contain duplicate paraphrase/branch/view keys")
    missing = sorted(expected_keys - set(observed_keys))
    extra = sorted(set(observed_keys) - expected_keys)
    if missing:
        errors.append(f"controller results missing {len(missing)} keys")
    if extra:
        errors.append(f"controller results contain {len(extra)} unexpected keys")

    for row in rows:
        branch_id = str(row.get("branch_id"))
        view = str(row.get("memory_view"))
        if branch_id not in branch_ids or view not in VIEW_ORDER:
            continue
        expected = expected_success(branch_id, view)
        if bool(row.get("binding_exact")) != expected:
            errors.append(
                f"controller expectation mismatch: q{row.get('paraphrase_index')} {branch_id}/{view} "
                f"binding_exact={row.get('binding_exact')} expected={expected}"
            )
        if row.get("controller_errors"):
            errors.append(
                f"controller returned errors: q{row.get('paraphrase_index')} {branch_id}/{view}: "
                f"{row.get('controller_errors')}"
            )

    restart_groups: dict[tuple[int, str, int], set[str]] = {}
    for row in rows:
        key = (
            int(row.get("paraphrase_index", -1)),
            str(row.get("branch_id")),
            int(row.get("numerical_seed", -1)),
        )
        restart_groups.setdefault(key, set()).add(str(row.get("restart_sha256")))
    unequal = [key for key, hashes in restart_groups.items() if len(hashes) != 1]
    if unequal:
        errors.append(f"controller arms have unequal restart populations in {len(unequal)} groups")

    aggregate: list[dict[str, Any]] = []
    for view in VIEW_ORDER:
        view_rows = [row for row in rows if row.get("memory_view") == view]
        record: dict[str, Any] = {
            "memory_view": view,
            "trials": len(view_rows),
            "binding_exact": sum(bool(row.get("binding_exact")) for row in view_rows),
        }
        for task in TASK_ORDER:
            subset = [row for row in view_rows if row.get("task_type") == task]
            record[f"{task}_trials"] = len(subset)
            record[f"{task}_binding_exact"] = sum(bool(row.get("binding_exact")) for row in subset)
        aggregate.append(record)
    return rows, aggregate


def formal_evidence(
    path: Path, errors: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        errors.append(f"formal result missing: {repo_path(path)}")
        return [], []
    rows = load_jsonl(path)
    expected_keys = {
        (spec.branch_id, view, seed)
        for spec in BRANCHES
        for view in VIEW_ORDER
        for seed in range(3)
    }
    observed_keys = [
        (str(row.get("branch_id")), str(row.get("memory_view")), int(row.get("numerical_seed", -1)))
        for row in rows
    ]
    if len(set(observed_keys)) != len(observed_keys):
        errors.append("formal results contain duplicate branch/view/seed keys")
    missing = expected_keys - set(observed_keys)
    extra = set(observed_keys) - expected_keys
    if missing:
        errors.append(f"formal results missing {len(missing)} keys")
    if extra:
        errors.append(f"formal results contain {len(extra)} unexpected keys")

    for row in rows:
        branch_id = str(row.get("branch_id"))
        view = str(row.get("memory_view"))
        if (branch_id, view, int(row.get("numerical_seed", -1))) not in expected_keys:
            continue
        expected = expected_success(branch_id, view)
        valid = bool(row.get("binding_exact")) and bool(row.get("hidden_contract_pass"))
        if valid != expected:
            errors.append(
                f"formal expectation mismatch: {branch_id}/{view}/seed{row.get('numerical_seed')} "
                f"valid={valid} expected={expected}"
            )
        if int(row.get("population_size", -1)) != 200 or int(row.get("generation_cap", -1)) != 200:
            errors.append(
                f"formal budget mismatch: {branch_id}/{view}/seed{row.get('numerical_seed')} "
                f"has {row.get('population_size')}x{row.get('generation_cap')}"
            )

    restart_groups: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = (str(row.get("branch_id")), int(row.get("numerical_seed", -1)))
        restart_groups.setdefault(key, set()).add(str(row.get("restart_sha256")))
    unequal = [key for key, hashes in restart_groups.items() if len(hashes) != 1]
    if unequal:
        errors.append(f"formal arms have unequal restart populations in {len(unequal)} groups")

    aggregate: list[dict[str, Any]] = []
    for view in VIEW_ORDER:
        subset = [row for row in rows if row.get("memory_view") == view]
        ratios = [float(row["oracle_relative_hv"]) for row in subset if row.get("oracle_relative_hv") is not None]
        valid = [
            row
            for row in subset
            if bool(row.get("binding_exact")) and bool(row.get("hidden_contract_pass"))
        ]
        task_adjusted_reference_hv = [
            float(row.get("filtered_normalized_hv") or 0.0)
            if bool(row.get("binding_exact")) and bool(row.get("hidden_contract_pass"))
            else 0.0
            for row in subset
        ]
        aggregate.append(
            {
                "memory_view": view,
                "trials": len(subset),
                "valid": len(valid),
                "binding_exact": sum(bool(row.get("binding_exact")) for row in subset),
                "hidden_contract_pass": sum(bool(row.get("hidden_contract_pass")) for row in subset),
                "mean_archive_compliance_rate": mean(
                    [float(row.get("archive_compliance_rate") or 0.0) for row in subset]
                ),
                "mean_task_adjusted_reference_hv": mean(task_adjusted_reference_hv),
                "mean_task_adjusted_oracle_relative_hv": mean(ratios),
            }
        )
    return rows, aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_tex(
    controller: list[dict[str, Any]], formal: list[dict[str, Any]],
    *, online_representatives: bool = False,
) -> str:
    controller_map = {row["memory_view"]: row for row in controller}
    formal_map = {row["memory_view"]: row for row in formal}
    lines = [
        "% Generated by scripts/analysis/export_liveopt_state_binding_evidence.py.",
        r"\begin{table}[!htbp]", r"\centering", r"\begingroup",
        r"\DenseResultTableSetup",
        r"\begin{tabularx}{\linewidth}{@{}l*{5}{>{\centering\arraybackslash}X}@{}}",
        r"\toprule", r"\TableHead",
        r"\textbf{State view} & \TightStack{\textbf{Event}\\\textbf{request}} & \TightStack{\textbf{Accepted}\\\textbf{request}} & \TightStack{\textbf{Linked}\\\textbf{request}} & \TightStack{\textbf{Valid search}\\\textbf{results}} & \textbf{Mean HV} \\",
        r"\midrule",
    ]
    for index, view in enumerate(VIEW_ORDER):
        c_row, f_row = controller_map.get(view, {}), formal_map.get(view, {})
        counts = [f"{int(c_row.get(f'{task}_binding_exact', 0))}/{int(c_row.get(f'{task}_trials', 0))}" for task in TASK_ORDER]
        valid = f"{int(f_row.get('valid', 0))}/{int(f_row.get('trials', 0))}"
        hv = f_row.get("mean_task_adjusted_reference_hv")
        hv_text = "--" if hv is None else f"{float(hv):.3f}"
        if index % 2:
            lines.append(r"\rowcolor{LiveOptBaselineRowB}")
        lines.append(f"{VIEW_LABELS[view]} & {counts[0]} & {counts[1]} & {counts[2]} & {valid} & {hv_text} " + r"\\")
    scope = (
        "The request columns retain original language-grounding results with minimum-recorded-fitness archive representatives across three wordings. Valid search results and mean HV use the corrected replay against original online plans: six cases, three numerical seeds, and the saved q0 queries per condition."
        if online_representatives else
        "The request columns count language-grounding results across three wordings. Valid search results and mean HV use six cases with three numerical seeds per condition, with minimum-recorded-fitness archive representatives."
    )
    lines.extend([
        r"\bottomrule", r"\end{tabularx}", r"\endgroup",
        r"\caption{\textbf{Using stored events and the accepted state.} " + scope +
        " The recorded evaluator is retained; missing or incorrect bindings and infeasible archives score zero.}",
        r"\label{tab:state-binding-factorial}", r"\end{table}", "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    controller_paths = tuple(args.controller_results or DEFAULT_CONTROLLERS)
    errors: list[str] = []
    controller_rows, controller_aggregate = controller_evidence(controller_paths, errors)
    formal_rows, formal_aggregate = formal_evidence(args.formal_results, errors)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_tex.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "controller_aggregate.csv", controller_aggregate)
    write_csv(args.out_dir / "formal_aggregate.csv", formal_aggregate)
    args.paper_tex.write_text(render_tex(controller_aggregate, formal_aggregate, online_representatives=bool(formal_rows) and all(row.get("representative_rule") == "original_exported_online_plan" for row in formal_rows)), encoding="utf-8")

    sources = [*controller_paths, args.formal_results]
    evidence = {
        "status": "passed" if not errors else "failed",
        "protocol": "liveopt_state_binding_factorial_v1",
        "design": {
            "profiles": ["NLDO-P010", "NLDO-P015"],
            "task_types": list(TASK_ORDER),
            "controller_paraphrases": len(controller_paths),
            "formal_numerical_seeds": [0, 1, 2],
            "formal_population_size": 200,
            "formal_generation_cap": 200,
            "common_incoming_population": True,
            "formal_controller_paraphrase": 0,
        },
        "inputs": [
            {
                "path": repo_path(path),
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for path in sources
        ],
        "controller_aggregate": controller_aggregate,
        "formal_aggregate": formal_aggregate,
        "paper_tex": repo_path(args.paper_tex),
        "errors": errors,
    }
    (args.out_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
