import json

import pytest

from scripts.analysis.evaluate_reference_metrics import validate_nldo_mo_reference_grid


def write_episode(tmp_path, step):
    path = tmp_path / "nldo_reference.jsonl"
    episode = {
        "episode_id": "NLDO-P010",
        "domain": "green_vrp_multiobjective",
        "update_stream": [],
        "evaluation": {"reference_trajectory": [step]},
    }
    path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    return path


def strong_step():
    return {
        "update_id": "initial",
        "reference_type": "pareto_ga_best_known",
        "solver": "offline_multi_seed_nsga2_style_ga",
        "solver_config": {"population_size": 500, "generations": 500, "seeds": 10},
        "pareto_archive": [{"solution": {}, "objectives": {"distance": 1.0}}],
        "reference_hv": 0.8,
        "hypervolume_reference_point": {"distance": 2.0},
        "hypervolume_ideal_point": {"distance": 0.0},
    }


def test_formal_reference_accepts_complete_500x500x10_step(tmp_path) -> None:
    validate_nldo_mo_reference_grid(write_episode(tmp_path, strong_step()))


def test_formal_reference_rejects_constructive_bootstrap_step(tmp_path) -> None:
    step = strong_step()
    step.pop("solver_config")
    step.pop("reference_hv")
    step.pop("hypervolume_reference_point")
    step.pop("hypervolume_ideal_point")
    step["solver"] = "deterministic_weighted_insertion_pareto_proxy"
    step["reference_type"] = "pareto_solver_best_known"

    with pytest.raises(SystemExit, match="constructive/bootstrap references are forbidden"):
        validate_nldo_mo_reference_grid(write_episode(tmp_path, step))
