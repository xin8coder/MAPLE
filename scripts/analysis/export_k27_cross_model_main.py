#!/usr/bin/env python3
"""Export the cross-model controller comparison on the full NLDO benchmark.

Reads the DeepSeek-V4-Pro Preview main-run reference metrics (the three canonical
main run dirs) and the Kimi k2.7 run's reference metrics, aggregates
prefix-zero solve rate and feasibility-gated online quality per view
(DS = P001--P009 scalar, DM = P010--P015 Pareto), and writes
release_artifacts/paper_table_exports/liveopt_cross_model_main_rows.tex plus a JSON summary.

Protocol note: both controllers contribute one controller trajectory per
episode; numerical seeds estimate search variation only (they do not
multiply the episode denominator).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEEPSEEK_DIRS = [
    ROOT / "outputs/liveopt_objective_shift_p001_p009_full_200x200_10seed_20260713/reference_metrics/reference_stage_metrics.csv",
    ROOT / "outputs/liveopt_ablation_llmbackbone_sequential_p007_p009_200x200x10_trace_t11pop_20260714/reference_metrics/reference_stage_metrics.csv",
    ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/reference_metrics/reference_stage_metrics.csv",
]
K27_CSV = ROOT / "outputs/liveopt_k27_main_nldo_full_200x200x10_20260723/reference_metrics/reference_stage_metrics.csv"
OUT_TEX = ROOT / "release_artifacts" / "paper_table_exports" / "liveopt_cross_model_main_rows.tex"
OUT_JSON = ROOT / "logs" / "analysis" / "liveopt_cross_model_main_20260723.json"

DS_EPISODES = {f"NLDO-P{i:03d}" for i in range(1, 10)}
DM_EPISODES = {f"NLDO-P{i:03d}" for i in range(10, 16)}
STAGES = list(range(13))
SEEDS = list(range(10))


def load_rows(paths: list[Path]) -> dict[tuple[str, int, int], dict]:
    """(episode, seed, stage) -> row; later files override duplicates."""
    rows: dict[tuple[str, int, int], dict] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["episode_id"], int(row["run_seed"]), int(row["stage_index"]))
                rows[key] = row
    return rows


def gated_value(row: dict) -> float | None:
    if str(row.get("feasible")) != "True":
        return None
    for field in ("normalized_score", "normalized_hv"):
        value = row.get(field)
        if value not in (None, ""):
            return float(value)
    return 0.0


def require_complete(
    rows: dict[tuple[str, int, int], dict], episodes: set[str], label: str
) -> None:
    expected = {
        (episode, seed, stage)
        for episode in episodes
        for seed in SEEDS
        for stage in STAGES
    }
    missing = sorted(expected - set(rows))
    if missing:
        preview = ", ".join(f"{ep}/seed{seed}/t{stage:02d}" for ep, seed, stage in missing[:5])
        raise SystemExit(
            f"{label} is not a complete 15-episode x 10-seed x 13-state run; "
            f"missing {len(missing)} rows (first: {preview})"
        )


def view_metrics(rows: dict[tuple[str, int, int], dict], episodes: set[str]) -> dict:
    """Prefix-zero solve rate and online quality, averaged over episodes."""
    solve_rates: list[float] = []
    qualities: list[float] = []
    for episode in sorted(episodes):
        seeds = sorted({seed for (ep, seed, _stage) in rows if ep == episode})
        if not seeds:
            continue
        per_seed_prefix: list[int] = []
        per_seed_quality: list[float] = []
        for seed in seeds:
            prefix = 0
            quality_sum = 0.0
            for stage in STAGES:
                row = rows.get((episode, seed, stage))
                if row is None:
                    break
                value = gated_value(row)
                if value is None:
                    break
                prefix += 1
                quality_sum += value
            per_seed_prefix.append(prefix)
            per_seed_quality.append(quality_sum / len(STAGES))
        solve_rates.append(statistics.mean(per_seed_prefix) / len(STAGES))
        qualities.append(statistics.mean(per_seed_quality))
    return {
        "solve_rate": statistics.mean(solve_rates),
        "online_quality": statistics.mean(qualities),
        "episodes": len(solve_rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k27-csv", type=Path, default=K27_CSV)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    deepseek = load_rows(list(DEEPSEEK_DIRS))
    k27 = load_rows([args.k27_csv])
    if not k27:
        raise SystemExit(f"no rows found in {args.k27_csv}")
    all_episodes = DS_EPISODES | DM_EPISODES
    require_complete(deepseek, all_episodes, "DeepSeek-V4-Pro Preview")
    require_complete(k27, all_episodes, "Kimi k2.7")

    liveopt_summary = {
        "DeepSeek-V4-Pro": {"DS": view_metrics(deepseek, DS_EPISODES), "DM": view_metrics(deepseek, DM_EPISODES)},
        "Kimi k2.7": {"DS": view_metrics(k27, DS_EPISODES), "DM": view_metrics(k27, DM_EPISODES)},
    }
    def pct(value: float) -> str:
        return f"{100 * value:.1f}\\%"

    lines = [
        "% Generated by scripts/analysis/export_k27_cross_model_main.py.",
        "\\newcommand{\\LiveOptCrossModelMainRows}{%",
        "\\rowcolor{LiveOptRowAccent}",
        f"\\method{{}} (DeepSeek-V4-Pro Preview) & {pct(liveopt_summary['DeepSeek-V4-Pro']['DS']['solve_rate'])} & {liveopt_summary['DeepSeek-V4-Pro']['DS']['online_quality']:.3f} & {pct(liveopt_summary['DeepSeek-V4-Pro']['DM']['solve_rate'])} & {liveopt_summary['DeepSeek-V4-Pro']['DM']['online_quality']:.3f} \\\\",
        f"\\method{{}} (Kimi k2.7) & {pct(liveopt_summary['Kimi k2.7']['DS']['solve_rate'])} & {liveopt_summary['Kimi k2.7']['DS']['online_quality']:.3f} & {pct(liveopt_summary['Kimi k2.7']['DM']['solve_rate'])} & {liveopt_summary['Kimi k2.7']['DM']['online_quality']:.3f} \\\\",
        "}",
    ]
    args.out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "liveopt": liveopt_summary,
        "reporting_rule": "complete full-benchmark runs only",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
