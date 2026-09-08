#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.agents.liveopt_dynamic_impl import (
    LiveOptDataPatcher,
    LiveOptDynamicRunner,
    LiveOptUpdateLocalizer,
    LiveOptWorkbenchPatcher,
)
from evo2.agents.liveopt_workbench_impl import compile_scaffold_project
from evo2.core.template_optimizer import EvolutionConfig


SETUP_CODE = """
def build_problem(public_context):
    raw = public_context["public_data"]
    tasks = [{"id": row["job_code"], "load": int(row["hours"])} for row in raw["jobs"]]
    workers = [
        {"id": row["worker_code"], "capacity": int(row["max_hours"]), "cost": float(row["rate"])}
        for row in raw["staff"]
    ]
    data = {
        "tasks": tasks,
        "workers": workers,
        "worker_by_id": {row["id"]: row for row in workers},
    }
    return {
        "data": data,
        "segments": [
            {
                "name": "assign",
                "kind": "assignment",
                "demands": [row["id"] for row in tasks],
                "resources": [row["id"] for row in workers],
            }
        ],
        "sense": "min",
        "objective_names": ["cost"],
        "constraint_names": ["capacity"],
    }
"""


FITNESS_CODE = """
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    workers = data["worker_by_id"]
    used = {worker_id: 0 for worker_id in workers}
    cost = 0.0
    missing = 0
    for task in data["tasks"]:
        worker_id = assignments.get(task["id"])
        if worker_id not in workers:
            missing += 1
            continue
        used[worker_id] += task["load"]
        cost += task["load"] * workers[worker_id]["cost"]
    excess = sum(max(0, used[worker_id] - workers[worker_id]["capacity"]) for worker_id in workers)
    violations = {}
    if missing:
        violations["missing_tasks"] = missing
    if excess:
        violations["capacity_excess"] = excess
    solution = {"assignments": assignments, "used_capacity": used, "objective_version": "cost_v1"}
    return penalty_result(cost, violations, objectives=[cost], solution=solution)
"""


FITNESS_PATCH_DOUBLE_COST = """
### fitness.py
```python
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    workers = data["worker_by_id"]
    used = {worker_id: 0 for worker_id in workers}
    cost = 0.0
    missing = 0
    for task in data["tasks"]:
        worker_id = assignments.get(task["id"])
        if worker_id not in workers:
            missing += 1
            continue
        used[worker_id] += task["load"]
        cost += 2.0 * task["load"] * workers[worker_id]["cost"]
    excess = sum(max(0, used[worker_id] - workers[worker_id]["capacity"]) for worker_id in workers)
    violations = {}
    if missing:
        violations["missing_tasks"] = missing
    if excess:
        violations["capacity_excess"] = excess
    solution = {"assignments": assignments, "used_capacity": used, "objective_version": "double_cost_v2"}
    return penalty_result(cost, violations, objectives=[cost], solution=solution)
```
"""


SETUP_AND_FITNESS_PATCH_OPTIONAL_TASKS = """
### setup.py
```python
def build_problem(public_context):
    raw = public_context["public_data"]
    policy_rows = raw.get("policy") or []
    policy = policy_rows[0] if isinstance(policy_rows, list) and policy_rows else policy_rows
    tasks = [{"id": row["job_code"], "load": int(row["hours"])} for row in raw["jobs"]]
    workers = [
        {"id": row["worker_code"], "capacity": int(row["max_hours"]), "cost": float(row["rate"])}
        for row in raw["staff"]
    ]
    data = {
        "tasks": tasks,
        "workers": workers,
        "worker_by_id": {row["id"]: row for row in workers},
        "unassigned_cost": float(policy.get("unassigned_cost", 5.0)),
    }
    return {
        "data": data,
        "segments": [
            {
                "name": "assign",
                "kind": "optional_assignment",
                "demands": [row["id"] for row in tasks],
                "resources": [row["id"] for row in workers],
                "allow_none": True,
            }
        ],
        "sense": "min",
        "objective_names": ["service_cost"],
        "constraint_names": ["capacity"],
    }
```
### fitness.py
```python
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    workers = data["worker_by_id"]
    used = {worker_id: 0 for worker_id in workers}
    service_cost = 0.0
    skipped = []
    invalid = 0
    for task in data["tasks"]:
        worker_id = assignments.get(task["id"])
        if worker_id is None:
            skipped.append(task["id"])
            service_cost += data["unassigned_cost"] * task["load"]
            continue
        if worker_id not in workers:
            invalid += 1
            continue
        used[worker_id] += task["load"]
        service_cost += task["load"] * workers[worker_id]["cost"]
    excess = sum(max(0, used[worker_id] - workers[worker_id]["capacity"]) for worker_id in workers)
    violations = {}
    if invalid:
        violations["invalid_assignments"] = invalid
    if excess:
        violations["capacity_excess"] = excess
    solution = {
        "assignments": assignments,
        "used_capacity": used,
        "skipped_tasks": skipped,
        "objective_version": "optional_tasks_v3",
    }
    return penalty_result(service_cost, violations, objectives=[service_cost], solution=solution)
```
"""


class QueuePatchClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def chat(self, messages, temperature=0.0, max_tokens=0, **kwargs):
        if not self.responses:
            raise RuntimeError("no queued patch response left")
        response = self.responses.pop(0)
        self.payloads.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs})
        return {"choices": [{"message": {"content": response}}], "usage": {"total_tokens": len(response.split())}}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_client = QueuePatchClient([FITNESS_PATCH_DOUBLE_COST, SETUP_AND_FITNESS_PATCH_OPTIONAL_TASKS])
    project = compile_scaffold_project("dynamic_smoke", public_context(), {"setup.py": SETUP_CODE, "fitness.py": FITNESS_CODE})
    initial = project.run(EvolutionConfig(population_size=args.population_size, generations=args.generations, seed=1, archive_limit=args.archive_limit))
    runner = LiveOptDynamicRunner(
        project,
        public_context=public_context(),
        result=initial,
        # Every scenario supplies its impact and tables. Fail locally if the
        # smoke unexpectedly asks either component for a model response.
        localizer=LiveOptUpdateLocalizer(model="fake", client=QueuePatchClient([])),
        data_patcher=LiveOptDataPatcher(model="fake", client=QueuePatchClient([])),
        patcher=LiveOptWorkbenchPatcher(model="fake", client=patch_client),
    )
    started = time.perf_counter()
    stages = []
    scenarios = [
        {
            "update_id": "u001",
            "text": "Job A now takes 3 hours; objective and encoding stay unchanged.",
            "context": public_context(hours_a="3"),
            "impact": {
                "data_update": True,
                "patch_setup": False,
                "patch_fitness": False,
                "restart_skill": "warm_restart_v1",
                "reason": "only public workload data changed",
            },
        },
        {
            "update_id": "u002",
            "text": "Operating cost is now counted twice, while data schema and assignment encoding stay unchanged.",
            "context": public_context(hours_a="3"),
            "impact": {
                "data_update": False,
                "patch_setup": False,
                "patch_fitness": True,
                "restart_skill": "warm_restart_v1",
                "reason": "objective changed but encoding remains compatible",
            },
        },
        {
            "update_id": "u003",
            "text": "A new task C arrives and tasks may be skipped with a public penalty instead of forcing every task to be assigned.",
            "context": public_context(hours_a="3", extra_jobs=[{"job_code": "C", "hours": "6"}], policy={"unassigned_cost": 4.0}),
            "impact": {
                "data_update": True,
                "patch_setup": True,
                "patch_fitness": True,
                "restart_skill": "full_restart_v1",
                "reason": "encoding changes from mandatory assignment to optional assignment",
            },
        },
        {
            "update_id": "u004",
            "text": "Task C is clarified to require only 2 hours; optional assignment policy remains unchanged.",
            "context": public_context(hours_a="3", extra_jobs=[{"job_code": "C", "hours": "2"}], policy={"unassigned_cost": 4.0}),
            "impact": {
                "data_update": True,
                "patch_setup": False,
                "patch_fitness": False,
                "restart_skill": "warm_restart_v1",
                "reason": "only public task load changed under the same optional encoding",
            },
        },
    ]
    for idx, scenario in enumerate(scenarios, start=2):
        stage = runner.update(
            update_id=scenario["update_id"],
            natural_language_update=scenario["text"],
            public_context=scenario["context"],
            impact=scenario["impact"],
            config=EvolutionConfig(
                population_size=args.population_size,
                generations=args.generations,
                seed=idx,
                archive_limit=args.archive_limit,
                structured_initialization=False,
            ),
        )
        stages.append(stage.to_record())
    record = {
        "status": "completed",
        "protocol": "liveopt_dynamic_impl_smoke_v1",
        "latency_seconds": time.perf_counter() - started,
        "initial": {
            "best": candidate_record(initial.best),
            "archive_count": len(initial.archive),
            "population_count": len(initial.population),
            "segments": [segment.__dict__ for segment in project.segments],
        },
        "stages": stages,
        "patch_call_count": len(patch_client.payloads),
        "output_dir": str(out_dir),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(console_summary(record, summary_path), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-API multi-update smoke test for the LiveOpt Workbench.")
    parser.add_argument("--output-dir", default="outputs/liveopt_dynamic_impl_smoke/latest")
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--archive-limit", type=int, default=24)
    return parser.parse_args()


def public_context(hours_a: str = "2", hours_b: str = "1", extra_jobs: list[dict[str, str]] | None = None, policy: dict[str, Any] | None = None):
    jobs = [{"job_code": "A", "hours": hours_a}, {"job_code": "B", "hours": hours_b}]
    jobs.extend(extra_jobs or [])
    data = {
        "jobs": jobs,
        "staff": [{"worker_code": "U1", "max_hours": "4", "rate": "1.0"}, {"worker_code": "U2", "max_hours": "4", "rate": "2.0"}],
    }
    if policy:
        data["policy"] = dict(policy)
    return {"public_data": data}


def candidate_record(candidate) -> dict[str, Any]:
    result = candidate.result
    return {
        "genome": candidate.genome,
        "scalar": result.scalar if result else None,
        "objectives": result.objectives if result else [],
        "feasible": result.feasible if result else None,
        "violations": result.violations if result else {},
        "solution": result.solution if result else {},
    }


def console_summary(record: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    rows = [
        {
            "stage": "initial",
            "feasible": record["initial"]["best"]["feasible"],
            "scalar": record["initial"]["best"]["scalar"],
            "objective_version": record["initial"]["best"]["solution"].get("objective_version"),
            "assignments": record["initial"]["best"]["solution"].get("assignments"),
            "restart": "random_initialization",
            "initial_seed_count": 0,
            "patch_setup": False,
            "patch_fitness": False,
        }
    ]
    for stage in record["stages"]:
        best = stage["best"]
        impact = stage["impact"]
        rows.append(
            {
                "stage": stage["update_id"],
                "feasible": best["feasible"],
                "scalar": best["scalar"],
                "objective_version": best["solution"].get("objective_version"),
                "assignments": best["solution"].get("assignments"),
                "skipped_tasks": best["solution"].get("skipped_tasks"),
                "restart": stage["restart_metadata"].get("restart_skill"),
                "initial_seed_count": stage["initial_seed_count"],
                "patch_setup": impact.get("patch_setup"),
                "patch_fitness": impact.get("patch_fitness"),
            }
        )
    return {
        "status": record["status"],
        "summary_json": str(summary_path),
        "patch_call_count": record["patch_call_count"],
        "stages": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
