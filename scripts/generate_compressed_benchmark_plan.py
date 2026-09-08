#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXACT_SMALL = [
    {
        "benchmark": "FJSP-Exact-Small",
        "domain": "fjsp",
        "source": "SchedulingLab/fjsp-instances small optimum/bounds subset",
        "reference_policy": "exact_optimum_or_certified_bound",
        "solver_candidates": ["CP-SAT", "MILP", "linear_program_solver_v1 for linearized subproblems"],
        "updates": [
            "machine downtime",
            "urgent job / due date tightening",
            "operation-machine eligibility change",
            "processing time change",
            "memory reference to earlier blocked machine",
        ],
    },
    {
        "benchmark": "CO-Bench-Exact-Small",
        "domain": "cobench",
        "source": "CO-Bench style exact subfamilies: knapsack + facility_location",
        "reference_policy": "exact_optimum",
        "solver_candidates": ["dynamic programming", "MILP", "linear_program_solver_v1"],
        "updates": [
            "capacity or budget change",
            "mandatory item/facility",
            "forbidden item/facility",
            "cost/value drift",
            "memory reference to earlier policy item",
        ],
    },
]


REALISTIC_DYNAMIC = [
    {
        "benchmark": "Dynamic-VRP/DPDP",
        "domain": "routing",
        "source": "ICAPS 2021 DPDP / Olist VRP / VRP-REP",
        "reference_policy": "solver_best_known_or_literature_best_known",
        "solver_candidates": ["OR-Tools/PyVRP/HGS", "GA/MOEA", "rolling horizon"],
        "updates": [
            "new pickup-delivery order",
            "order cancellation",
            "vehicle delay/unavailable",
            "time-window tightening",
            "memory reference to customer in earlier dispatch note",
            "priority order",
            "capacity pressure",
            "route stability request",
            "traffic/weather delay",
            "late-day replanning",
        ],
    },
    {
        "benchmark": "INRC-II-Dynamic-Rostering",
        "domain": "inrc2",
        "source": "INRC-II public JSON datasets",
        "reference_policy": "official_score_or_solver_best_known",
        "solver_candidates": ["CP-SAT", "MILP", "GA/MOEA", "official evaluator"],
        "updates": [
            "nurse absence",
            "coverage increase",
            "skill demand change",
            "preference request",
            "memory reference to replacement nurse",
            "fairness policy change",
            "published roster stability",
            "rest-after-night change",
            "temporary overtime cap",
            "weekend coverage change",
        ],
    },
    {
        "benchmark": "Green-Dynamic-VRP-MO",
        "domain": "green_vrp_multiobjective",
        "source": "Olist VRP / DPDP with emission and service objectives",
        "reference_policy": "pareto_solver_best_known",
        "solver_candidates": ["NSGA-II/MOEA-D", "HGS variants", "GA/MOEA"],
        "objectives": ["distance", "lateness", "emission"],
        "auxiliary_metrics": ["disruption", "priority_satisfaction"],
        "updates": [
            "urgent order",
            "traffic delay",
            "vehicle unavailable",
            "time-window change",
            "carbon policy tightened",
            "memory reference to affected route",
            "fuel price increase",
            "priority customer request",
            "driver shift constraint",
            "late-day stability objective",
        ],
    },
    {
        "benchmark": "Dynamic-Cloud-Scheduling-MO",
        "domain": "cloud_scheduling_multiobjective",
        "source": "Google cluster trace / Alibaba cluster trace slices",
        "reference_policy": "pareto_solver_best_known",
        "solver_candidates": ["rolling horizon MILP", "NSGA-II/MOEA-D", "GA/MOEA"],
        "objectives": ["sla_violations", "energy"],
        "auxiliary_metrics": ["latency", "migration_disruption", "utilization"],
        "updates": [
            "new job burst",
            "machine capacity drop",
            "deadline tightening",
            "energy price change",
            "priority job insertion",
            "memory reference to earlier workload burst",
            "GPU scarcity",
            "carbon intensity change",
            "node maintenance",
            "analytics deadline relaxation",
            "resource/objective reset",
        ],
    },
]


def build_plan(exact_scenarios: int = 6, realistic_scenarios: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in EXACT_SMALL:
        for idx in range(exact_scenarios):
            rows.append(_episode(spec, "exact_small", idx, 5))
    for spec in REALISTIC_DYNAMIC:
        for idx in range(realistic_scenarios):
            rows.append(_episode(spec, "realistic_dynamic", idx, 10))
    return rows


def _episode(spec: dict[str, Any], layer: str, idx: int, update_count: int) -> dict[str, Any]:
    update_templates = spec["updates"]
    return {
        "episode_id": f"{layer}_{spec['domain']}_{idx:03d}",
        "layer": layer,
        "benchmark": spec["benchmark"],
        "domain": spec["domain"],
        "source_dataset": spec["source"],
        "source_instance_policy": "select concrete source instances during materialization",
        "scenario_index": idx,
        "update_count": update_count,
        "public_initial_problem_policy": "native natural language; no hand-written domain skill exposed to agent",
        "update_stream_policy": [
            {
                "update_id": f"u{t + 1:03d}",
                "template": update_templates[t % len(update_templates)],
                "requires_memory": "memory reference" in update_templates[t % len(update_templates)],
            }
            for t in range(update_count)
        ],
        "reference_policy": spec["reference_policy"],
        "solver_candidates": spec["solver_candidates"],
        "objectives": spec.get("objectives", ["single_objective_quality", "disruption"]),
        "agent_allowed_solvers": ["linear_program_solver_v1", "ga_or_moea", "generated_repair_operator"],
        "hidden_evaluator_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compressed two-layer EVO² benchmark plan manifest.")
    parser.add_argument("--output", default="data/evo2_dynoptbench/compressed_plan/evo2_compressed_benchmark_plan.jsonl")
    args = parser.parse_args()
    rows = build_plan()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "episodes": len(rows), "updates": sum(r["update_count"] for r in rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
