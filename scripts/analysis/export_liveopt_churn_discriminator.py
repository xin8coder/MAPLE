#!/usr/bin/env python3
"""Export the churn-discriminator suite as a compact paper table.

Reads logs/analysis/liveopt_churn_discriminator_20260721/ (branch_specs.json,
decisions.jsonl, stage_metrics.csv) and writes
release_artifacts/paper_table_exports/liveopt_churn_discriminator_rows.tex.

Validate the recorded rule decisions, then report matched Warm/Full outcomes
and paired-seed dispersion from the same incoming population. The compact
main-text table retains relabeling and task-addition mean differences;
rescaling controls are reported in the appendix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "logs" / "analysis" / "liveopt_churn_discriminator_20260721"
DEFAULT_OUT = ROOT / "release_artifacts" / "paper_table_exports" / "liveopt_churn_discriminator_rows.tex"

BRANCH_ORDER = [
    "NLDO-P010-unit_scale",
    "NLDO-P010-relabel",
    "NLDO-P010-easy_inject",
    "NLDO-P015-unit_scale",
    "NLDO-P015-relabel",
    "NLDO-P015-easy_inject",
]
KIND_LABEL = {
    "unit_scale": "parameter rescaling",
    "relabel": "ID relabel",
    "easy_inject": "easy-row inject",
}


def signed_value(value: float) -> str:
    return "0.000" if round(value, 3) == 0 else f"{value:+.3f}"


def paired_statistics(records: list[dict[str, str]]) -> dict[str, dict]:
    index = {}
    for row in records:
        key = (row["branch_id"], row["arm"], int(row["seed"]))
        if key in index:
            raise ValueError(f"Duplicate branch/action/seed: {key}")
        index[key] = row
    expected = {(branch, arm, seed) for branch in BRANCH_ORDER
                for arm in ("fixed_warm", "fixed_full") for seed in range(3)}
    if set(index) != expected:
        raise ValueError("Expected all six branches, both actions, and seeds 0--2")

    statistics_by_branch = {}
    for branch in BRANCH_ORDER:
        values = {"fixed_warm": [], "fixed_full": []}
        generations = {"fixed_warm": [], "fixed_full": []}
        union_hvs = []
        for seed in range(3):
            pair = [index[branch, arm, seed] for arm in values]
            episode = branch.rsplit("-", 1)[0]
            kind = branch.rsplit("-", 1)[1]
            for row in pair:
                if (row["episode_id"] != episode or row["kind"] != kind or
                        any(int(row[k]) != n for k, n in (
                            ("fork_stage", 5), ("population_size", 200),
                            ("incoming_population_count", 200), ("generation_cap", 200)))):
                    raise ValueError(f"Mismatched recorded protocol: {branch}/{seed}")
                hv, union_hv, ratio = (float(row[k]) for k in ("hv", "union_hv", "hv_ratio"))
                if (not all(math.isfinite(v) for v in (hv, union_hv, ratio)) or
                        hv < 0 or union_hv <= 0 or ratio < 0 or
                        not math.isclose(hv / union_hv, ratio, rel_tol=1e-10, abs_tol=1e-10)):
                    raise ValueError(f"Invalid recorded HV ratio: {branch}/{seed}")
                generation = int(row["executed_generations"])
                if not 0 <= generation <= 200:
                    raise ValueError(f"Invalid generation count: {branch}/{seed}")
                values[row["arm"]].append(ratio)
                generations[row["arm"]].append(generation)
                union_hvs.append(union_hv)
        if len(set(union_hvs)) != 1:
            raise ValueError(f"Branch actions do not share one reference: {branch}")
        deltas = [w - f for w, f in zip(values["fixed_warm"], values["fixed_full"])]
        statistics_by_branch[branch] = {
            "warm_hv": statistics.mean(values["fixed_warm"]),
            "full_hv": statistics.mean(values["fixed_full"]),
            "delta_mean": statistics.mean(deltas),
            "delta_sd": statistics.stdev(deltas),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "warm_generations": statistics.mean(generations["fixed_warm"]),
            "full_generations": statistics.mean(generations["fixed_full"]),
            **{f"delta_seed_{seed}": delta for seed, delta in enumerate(deltas)},
        }
    return statistics_by_branch


def detail_rows(stats: dict[str, dict]) -> str:
    lines = [r"\newcommand{\LiveOptChurnDiscriminatorRows}{%"]
    for branch in BRANCH_ORDER:
        row = stats[branch]
        _, episode, kind = branch.split("-", 2)
        lines.append(
            f"{episode} {KIND_LABEL[kind]} & {row['warm_hv']:.3f} & {row['full_hv']:.3f} & "
            f"${signed_value(row['delta_mean'])}\\pm{row['delta_sd']:.3f}$ & "
            f"{signed_value(row['delta_min'])} & {signed_value(row['delta_max'])} " + r"\\"
        )
    lines.extend((r"\midrule", r"\rowcolor{LiveOptRowAccent}"))
    warm_mean = statistics.mean(row["warm_hv"] for row in stats.values())
    full_mean = statistics.mean(row["full_hv"] for row in stats.values())
    lines.append(f"All six cases & {warm_mean:.3f} & {full_mean:.3f} & "
                 f"{signed_value(warm_mean - full_mean)} & -- & -- " + r"\\")
    return "\n".join(lines + ["}"]) + "\n"


def compact_rows(warm: dict[str, list[float]], full: dict[str, list[float]]) -> str:
    lines = [r"\newcommand{\LiveOptChurnCompactRows}{%"]
    for kind, label in (("relabel", "IDs"), ("easy_inject", "Added tasks")):
        values = []
        for episode in ("P010", "P015"):
            key = f"NLDO-{episode}-{kind}"
            if len(warm[key]) != 3 or len(full[key]) != 3:
                raise ValueError(f"Expected three seeds per action: {key}")
            delta = statistics.mean(warm[key]) - statistics.mean(full[key])
            values.append(signed_value(delta))
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.append("}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats-csv", type=Path, help="Optional paired-seed statistics, separate from source records")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = json.loads((args.run_dir / "branch_specs.json").read_text(encoding="utf-8"))
    churn_by_branch = {b["branch_id"]: b["churn"]["churn_ratio"] for b in specs["branches"]}

    gate_decision: dict[str, str] = {}
    sensor_decisions: dict[str, set[str]] = {}
    for line in (args.run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        policy = row.get("policy")
        if policy == "semantic_restart_gate":
            gate_decision[row["branch_id"]] = "Full" if row.get("full_vote") else "Warm"
        elif policy == "sensor_landscape":
            sensor_decisions.setdefault(row["branch_id"], set()).add(str(row.get("selected_action")))

    warm: dict[str, list[float]] = {}
    full: dict[str, list[float]] = {}
    source = args.run_dir / "stage_metrics.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    stats = paired_statistics(records)
    for row in records:
        target = warm if row["arm"] == "fixed_warm" else full
        target.setdefault(row["branch_id"], []).append(float(row["hv_ratio"]))

    errors = []
    for branch_id in BRANCH_ORDER:
        churn_ratio = churn_by_branch.get(branch_id)
        gate = gate_decision.get(branch_id)
        sensor = sensor_decisions.get(branch_id, set())
        if churn_ratio is None or churn_ratio < 0.5 or gate != "Warm" or sensor != {"Full"}:
            errors.append(f"recorded decisions differ from the stated six-branch comparison: {branch_id}")

    if errors:
        raise SystemExit("export failed: " + "; ".join(errors))

    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    content = "% Generated by scripts/analysis/export_liveopt_churn_discriminator.py.\n"
    content += f"% Source CSV SHA256: {checksum}. Paired SD across seeds 0--2.\n"
    content += detail_rows(stats)
    content += compact_rows(warm, full)
    args.out.write_text(content, encoding="utf-8")
    if args.stats_csv:
        args.stats_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["branch_id", "source_sha256", *next(iter(stats.values()))])
            writer.writeheader()
            for branch, row in stats.items():
                writer.writerow({"branch_id": branch, "source_sha256": checksum, **row})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
