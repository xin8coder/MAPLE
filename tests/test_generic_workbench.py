from __future__ import annotations

import pytest

from evo2.agents.generic_workbench_impl import (
    GenericWorkbenchBuilder,
    build_generic_workbench_from_scratch_prompt,
    compile_generic_workbench,
    extract_python_artifact,
    generic_artifact_from_scaffold,
    generic_restart_seeds,
)
from evo2.agents.liveopt_workbench_impl import ScaffoldTrace, compile_scaffold_project
from evo2.core.template_optimizer import Candidate, EvolutionConfig, EvolutionResult, FitnessResult


ARTIFACT = """
def build_state(public_context):
    return {"target": float(public_context["target"]), "lower": 0.0, "upper": 10.0}

def initialize(state, rng):
    return {"x": rng.uniform(state["lower"], state["upper"])}

def evaluate_decode(candidate, state):
    x = float(candidate["x"])
    value = abs(x - state["target"])
    return {"scalar": value, "objectives": [], "feasible": True, "solution": {"x": x}}

def crossover(parent_a, parent_b, state, rng):
    return {"x": 0.5 * (float(parent_a["x"]) + float(parent_b["x"]))}

def mutate(candidate, state, rng):
    x = float(candidate["x"]) + rng.uniform(-0.5, 0.5)
    return {"x": min(state["upper"], max(state["lower"], x))}

def coerce_repair(old_candidate, state, rng):
    x = float(old_candidate.get("x", state["lower"]))
    return {"x": min(state["upper"], max(state["lower"], x))}
"""


def test_generic_workbench_uses_fixed_driver_with_untyped_hooks() -> None:
    project = compile_generic_workbench("generic-test", {"target": 7.0}, ARTIFACT)
    result = project.run(
        EvolutionConfig(
            population_size=20,
            generations=20,
            seed=3,
            archive_limit=20,
            selection_mode="scalar_ga",
            structured_initialization=False,
        ),
        initial_candidates=[{"x": -100.0}],
    )

    assert result.metadata["workbench_interface"] == "generic_operator_hooks_v1"
    assert result.metadata["selection"] == "scalar_ga"
    assert 0.0 <= result.best.genome["x"] <= 10.0
    assert result.best.result.feasible
    assert result.best.result.scalar < 1.0


def test_generic_workbench_rejects_typed_or_incomplete_artifacts() -> None:
    with pytest.raises(ValueError, match="missing hooks"):
        compile_generic_workbench("missing", {}, "def build_state(public_context): return {}", smoke_test=False)
    with pytest.raises(ValueError, match="typed/driver internals"):
        compile_generic_workbench("typed", {}, ARTIFACT + "\ndef bad(): return SegmentSpec\n", smoke_test=False)


def test_accepted_typed_artifact_is_materialized_as_concrete_generic_hooks() -> None:
    scaffold = compile_scaffold_project(
        "assignment",
        {"tables": {"items": [{"id": "a"}, {"id": "b"}], "resources": [{"id": "x"}, {"id": "y"}]}},
        {
            "setup.py": """
def build_problem(public_context):
    tables = public_context['tables']
    items = [row['id'] for row in tables['items']]
    resources = [row['id'] for row in tables['resources']]
    return {'data': {'items': items, 'resources': resources}, 'segments': [{'name': 'assignment', 'kind': 'assignment', 'demands': items, 'resources': resources}], 'solver_mode': 'scalar_ga', 'objective_names': ['cost']}
""",
            "fitness.py": """
def evaluate(genome, data):
    value = sum(0 if genome['assignment'].get(item) == 'x' else 1 for item in data['items'])
    return {'scalar': value, 'objectives': [], 'feasible': True, 'solution': genome}
""",
        },
        ScaffoldTrace(model="test"),
    )
    code = generic_artifact_from_scaffold(scaffold)
    project = compile_generic_workbench(
        "assignment",
        {"tables": {"items": [{"id": "a"}, {"id": "b"}], "resources": [{"id": "x"}, {"id": "y"}]}},
        code,
    )
    result = project.run(EvolutionConfig(population_size=12, generations=5, seed=2))

    assert project.segments == []
    assert result.metadata["workbench_interface"] == "generic_operator_hooks_v1"
    assert set(result.best.genome["assignment"]) == {"a", "b"}


def test_generic_response_requires_one_complete_code_block() -> None:
    assert extract_python_artifact(f"```python\n{ARTIFACT}\n```").strip() == ARTIFACT.strip()
    with pytest.raises(ValueError, match="exactly one"):
        extract_python_artifact(ARTIFACT)


def test_from_scratch_builder_receives_only_public_problem_and_bare_abi() -> None:
    class Client:
        def chat(self, messages, **kwargs):
            prompt = messages[-1]["content"]
            assert "public task" in prompt.lower() or "public natural-language problem" in prompt.lower()
            assert "setup.py" not in prompt
            assert "fitness.py" not in prompt
            assert "typed scaffold" not in prompt.lower()
            return {
                "choices": [{"message": {"content": f"```python\n{ARTIFACT}\n```"}}],
                "usage": {"total_tokens": 1, "prompt_tokens": 1, "completion_tokens": 0},
            }

    project = GenericWorkbenchBuilder(client=Client()).build_project(
        task_id="blank",
        public_problem="Choose a public scalar x close to the target.",
        public_context={"target": 7.0},
    )
    assert project.run(EvolutionConfig(population_size=10, generations=2, seed=0)).best.result.feasible

    prompt = build_generic_workbench_from_scratch_prompt(
        public_problem="Choose x.",
        public_context={"target": 7.0},
    )
    assert "No encoding template" in prompt
    assert "candidate-field schema" in prompt


def test_blank_generic_warm_transfer_reencodes_decoded_public_solutions() -> None:
    project = compile_generic_workbench("blank-transfer", {"target": 7.0}, ARTIFACT)
    source = Candidate(
        genome={"typed_private_key": [1, 2, 3]},
        result=FitnessResult(scalar=1.0, feasible=True, solution={"x": 8.0}),
    )
    previous = EvolutionResult(best=source, population=[source], archive=[], history=[], metadata={})
    seeds, metadata = generic_restart_seeds(
        project=project,
        previous_result=previous,
        restart_skill="warm_restart_v1",
        population_size=10,
        seed=0,
        decoded_solution_transfer=True,
    )
    assert seeds == [{"x": 8.0}]
    assert metadata["source_transfer_kind"] == "decoded_public_solution_reencoding"
