from __future__ import annotations

import pytest

from evo2.agents.liveopt_dynamic_impl import (
    DynamicUpdateImpact,
    LiveOptDynamicRunner,
    build_update_localizer_prompt,
)
from evo2.agents.liveopt_state_binding import (
    accepted_state_prompt_record,
    apply_state_bindings,
    materialize_state_bindings,
    memory_view_exposes_accepted,
    memory_view_exposes_ledger,
    normalize_state_binding_queries,
    validate_memory_view,
)
from evo2.agents.liveopt_workbench_impl import ScaffoldTrace, compile_scaffold_project
from evo2.core.template_optimizer import (
    Candidate,
    EvolutionConfig,
    EvolutionResult,
    coerce_fitness_result,
)


SETUP_CODE = """
def build_problem(public_context):
    return {
        "data": public_context,
        "segments": [{
            "name": "assign",
            "kind": "assignment",
            "demands": ["A", "B"],
            "resources": ["U1", "U2"],
        }],
        "solver_mode": "scalar_ga",
        "objective_names": ["cost"],
    }
"""


FITNESS_CODE = """
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    cost = sum(1.0 if worker == "U1" else 2.0 for worker in assignments.values())
    return penalty_result(cost, {}, objectives=[cost], solution={"assignments": assignments})
"""


def make_project():
    return compile_scaffold_project(
        "state_binding_test",
        {"public_data": {}},
        {"setup.py": SETUP_CODE, "fitness.py": FITNESS_CODE},
    )


def accepted_result(project=None) -> EvolutionResult:
    project = project or make_project()
    genome = {"assign": {"A": "U1", "B": "U2"}}
    result = coerce_fitness_result(project.evaluate(genome, project.data), genome)
    candidate = Candidate(genome=genome, result=result)
    return EvolutionResult(
        best=candidate,
        population=[candidate],
        archive=[],
        history=[],
        metadata={"selection": "scalar_ga"},
    )


def preserve_u1_query():
    return [
        {
            "binding_id": "keep_u1",
            "kind": "preserve_resource_assignments",
            "segment": "assign",
            "resource_id": "U1",
            "depends_on": ["accepted_output"],
        }
    ]


def preserve_a_query():
    return [
        {
            "binding_id": "keep_a",
            "kind": "preserve_demand_assignments",
            "segment": "assign",
            "demands": ["A"],
            "depends_on": ["accepted_output"],
        }
    ]


def test_memory_views_expose_only_the_declared_state_channels() -> None:
    assert memory_view_exposes_ledger("full")
    assert memory_view_exposes_accepted("full")
    assert memory_view_exposes_ledger("ledger_only")
    assert not memory_view_exposes_accepted("ledger_only")
    assert not memory_view_exposes_ledger("accepted_only")
    assert memory_view_exposes_accepted("accepted_only")
    assert not memory_view_exposes_ledger("current_only")
    assert not memory_view_exposes_accepted("current_only")
    with pytest.raises(ValueError, match="unsupported memory_view"):
        validate_memory_view("population")


def test_accepted_prompt_record_contains_the_committed_output_not_search_state() -> None:
    record = accepted_state_prompt_record(accepted_result())

    assert record == {
        "available": True,
        "representative": {
            "solution": {"assignments": {"A": "U1", "B": "U2"}},
            "scalar": 3.0,
            "objectives": [3.0],
            "feasible": True,
        },
    }
    assert "population" not in str(record)
    assert "archive" not in str(record)


def test_fixed_materializer_resolves_preserve_query_against_accepted_decision() -> None:
    project = make_project()
    queries = normalize_state_binding_queries(preserve_u1_query())

    bindings, diagnostics = materialize_state_bindings(
        queries,
        accepted_result=accepted_result(project),
        segments=project.segments,
    )

    assert bindings[0]["assignments"] == {"A": "U1"}
    assert diagnostics == {
        "query_count": 1,
        "materialized_count": 1,
        "locked_assignment_count": 1,
        "errors": [],
    }

    missing, missing_diagnostics = materialize_state_bindings(
        queries,
        accepted_result=None,
        segments=project.segments,
    )
    assert missing == []
    assert "accepted result is unavailable" in missing_diagnostics["errors"][0]

    demand_bindings, demand_diagnostics = materialize_state_bindings(
        preserve_a_query(),
        accepted_result=accepted_result(project),
        segments=project.segments,
    )
    assert demand_bindings[0]["assignments"] == {"A": "U1"}
    assert demand_diagnostics["errors"] == []


def test_fixed_evaluator_enforces_binding_without_changing_matching_objective() -> None:
    project = make_project()
    bindings, _ = materialize_state_bindings(
        preserve_u1_query(),
        accepted_result=accepted_result(project),
        segments=project.segments,
    )
    bound = apply_state_bindings(project, bindings)

    matching = bound.evaluate({"assign": {"A": "U1", "B": "U2"}}, bound.data)
    mismatch = bound.evaluate({"assign": {"A": "U2", "B": "U1"}}, bound.data)

    assert matching.feasible
    assert matching.scalar == 3.0
    assert not mismatch.feasible
    assert mismatch.violations["state_binding_mismatch"] == 1
    assert mismatch.diagnostics["state_binding_mismatches"][0]["demand"] == "A"


class RecordingLocalizer:
    def __init__(self):
        self.calls = []

    def classify(self, **kwargs):
        self.calls.append(kwargs)
        return (
            DynamicUpdateImpact.from_mapping(
                {
                    "restart_skill": "warm_restart_v1",
                    "reason": "typed state-binding test",
                    "state_binding_queries": preserve_u1_query(),
                }
            ),
            ScaffoldTrace(model="recording_localizer"),
        )


@pytest.mark.parametrize(
    ("memory_view", "sees_accepted", "sees_ledger", "materialized"),
    [
        ("full", True, True, 1),
        ("ledger_only", False, True, 0),
        ("accepted_only", True, False, 1),
        ("current_only", False, False, 0),
    ],
)
def test_runner_masks_controller_state_without_changing_restart_input(
    memory_view: str,
    sees_accepted: bool,
    sees_ledger: bool,
    materialized: int,
) -> None:
    project = make_project()
    initial = accepted_result(project)
    localizer = RecordingLocalizer()
    runner = LiveOptDynamicRunner(
        project,
        public_context={"public_data": {}},
        result=initial,
        localizer=localizer,
        public_update_history=[
            {
                "update_id": "t01",
                "natural_language_update": "Public event E1 identified resource U1.",
                "public_data_patch": {},
            }
        ],
    )

    stage = runner.update(
        update_id="t02",
        natural_language_update="Preserve assignments currently committed to U1.",
        memory_view=memory_view,
        force_restart_skill="warm_restart_v1",
        config=EvolutionConfig(
            population_size=2,
            generations=0,
            seed=7,
            structured_initialization=False,
        ),
        smoke_test=False,
    )

    call = localizer.calls[0]
    assert (call["previous_result"] is not None) is sees_accepted
    assert bool(call["public_update_history"]) is sees_ledger
    assert stage.restart_metadata["previous_candidate_count"] == 1
    assert stage.restart_metadata["state_binding_diagnostics"]["materialized_count"] == materialized
    assert stage.restart_metadata["state_binding_memory_view"] == memory_view


def test_memory_view_arms_receive_byte_identical_restart_genomes() -> None:
    project = make_project()
    initial = accepted_result(project)
    common = {
        "project": project,
        "public_context": {"public_data": {}},
        "result": initial,
        "public_update_history": [
            {
                "update_id": "t01",
                "natural_language_update": "Public event E1 identified resource U1.",
                "public_data_patch": {},
            }
        ],
    }
    initial_genomes = {}
    for memory_view in ("full", "ledger_only", "accepted_only", "current_only"):
        runner = LiveOptDynamicRunner(**common, localizer=RecordingLocalizer())
        stage = runner.update(
            update_id="t02",
            natural_language_update="Preserve assignments currently committed to U1.",
            memory_view=memory_view,
            force_restart_skill="warm_restart_v1",
            config=EvolutionConfig(
                population_size=4,
                generations=0,
                seed=11,
                structured_initialization=False,
            ),
            smoke_test=False,
        )
        initial_genomes[memory_view] = stage.initial_genomes

    assert initial_genomes["full"] == initial_genomes["ledger_only"]
    assert initial_genomes["full"] == initial_genomes["accepted_only"]
    assert initial_genomes["full"] == initial_genomes["current_only"]


def test_memory_view_cannot_be_mixed_with_legacy_lsm_ablation() -> None:
    project = make_project()
    runner = LiveOptDynamicRunner(project, result=accepted_result(project))

    with pytest.raises(ValueError, match="cannot be combined"):
        runner.update(
            update_id="t02",
            natural_language_update="Preserve the current assignment.",
            memory_view="ledger_only",
            disable_lsm=True,
        )


def test_localizer_prompt_makes_typed_binding_the_only_assignment_hold_path() -> None:
    project = make_project()
    prompt = build_update_localizer_prompt(
        natural_language_update="Keep A on its currently accepted worker.",
        project=project,
        public_context={"public_data": {}},
        previous_result=accepted_result(project),
        public_update_history=[],
    )

    assert "state_binding_queries is the sole representation" in prompt
    assert "Never substitute a slot patch" in prompt
