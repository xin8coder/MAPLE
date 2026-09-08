from __future__ import annotations

import pytest

from evo2.core.template_optimizer import EvolutionConfig, FitnessResult, SegmentSpec, run_evolution


def _evaluate(genome, _data):
    route = list(genome["route"])
    assignment = dict(genome["assignment"])
    scalar = float(sum(abs(index - int(value)) for index, value in enumerate(route)))
    scalar += float(sum(resource == "R2" for resource in assignment.values()))
    return FitnessResult(scalar=scalar, solution={"route": route, "assignment": assignment})


def test_type_agnostic_resampling_preserves_typed_interfaces() -> None:
    segments = [
        SegmentSpec(name="route", kind="permutation", values=[0, 1, 2, 3]),
        SegmentSpec(
            name="assignment",
            kind="assignment",
            demands=["D1", "D2", "D3"],
            resources=["R1", "R2"],
        ),
    ]

    result = run_evolution(
        segments,
        _evaluate,
        config=EvolutionConfig(
            population_size=18,
            generations=5,
            seed=7,
            structured_initialization=False,
            variation_mode="type_agnostic_resampling",
        ),
    )

    assert result.metadata["variation_mode"] == "type_agnostic_resampling"
    for candidate in result.population:
        assert sorted(candidate.genome["route"]) == [0, 1, 2, 3]
        assignment = dict(candidate.genome["assignment"])
        assert set(assignment) == {"D1", "D2", "D3"}
        assert set(assignment.values()) <= {"R1", "R2"}


def test_unknown_variation_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported variation_mode"):
        run_evolution(
            [SegmentSpec(name="x", kind="int_vector", length=1, lower=0, upper=1)],
            _evaluate,
            config=EvolutionConfig(variation_mode="flat"),
        )
