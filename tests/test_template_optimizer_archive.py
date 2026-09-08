from __future__ import annotations

from evo2.core.template_optimizer import Candidate, FitnessResult, _archive


def candidate(name: str, objectives: list[float]) -> Candidate:
    return Candidate(
        genome={"name": name},
        result=FitnessResult(
            scalar=sum(objectives),
            objectives=objectives,
            feasible=True,
            solution={"name": name},
        ),
    )


def test_pareto_archive_excludes_dominated_candidates() -> None:
    diagnostics: dict[str, object] = {}
    archive = _archive(
        [candidate("left", [0.0, 2.0]), candidate("right", [2.0, 0.0]), candidate("dominated", [3.0, 3.0])],
        [],
        10,
        diagnostics=diagnostics,
    )

    assert {item.genome["name"] for item in archive} == {"left", "right"}
    assert diagnostics == {
        "combined_candidates": 3,
        "feasible_candidates": 3,
        "unique_objective_vectors": 3,
        "nondominated_before_truncation": 2,
        "archive_size": 2,
        "archive_limit": 10,
        "cap_active": False,
    }


def test_pareto_archive_deduplicates_equal_objective_vectors() -> None:
    diagnostics: dict[str, object] = {}
    archive = _archive(
        [candidate("first", [1.0, 1.0]), candidate("duplicate", [1.0, 1.0]), candidate("tradeoff", [0.5, 2.0])],
        [],
        10,
        diagnostics=diagnostics,
    )

    assert len(archive) == 2
    assert sum(item.result.objectives == [1.0, 1.0] for item in archive if item.result) == 1
    assert diagnostics["unique_objective_vectors"] == 2
    assert diagnostics["archive_size"] == 2
    assert diagnostics["cap_active"] is False
