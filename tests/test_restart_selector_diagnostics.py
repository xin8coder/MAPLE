from scripts.analysis.export_nldo_restart_selector_diagnostics import collect_records, summarize


def test_selector_diagnostics_use_public_strata_and_population_mix() -> None:
    episodes = {
        "NLDO-P010": {
            "episode_id": "NLDO-P010",
            "update_stream": [{"difficulty": "memory"}],
        }
    }
    runs = [
        {
            "episode_id": "NLDO-P010",
            "run_seed": 0,
            "update_results": [
                {
                    "impact": {
                        "restart_skill": "full_restart_v1",
                        "raw": {
                            "objective_space_shift": {
                                "objective_space_shift": 1.0,
                                "fresh_population_ratio": 1.0,
                                "history_population_ratio": 0.0,
                                "direct_feasible_ratio": 0.40,
                                "public_state_churn": 0.30,
                                "public_state_changed_fact_count": 20,
                                "segment_distance": 0.25,
                                "objective_rank_disruption": 0.10,
                            }
                        },
                    },
                    "solver_result": {"metadata": {"runtime": {"executed_generations": 80}}},
                }
            ],
        }
    ]
    records = collect_records(runs, episodes)
    assert records[0]["stratum"] == "memory"
    assert records[0]["fresh_population_ratio"] == 1.0
    rows = summarize(records)
    overall = rows[0]
    assert overall["full_count"] == 1
    assert overall["mean_objective_space_shift"] == 1.0
    assert overall["mean_history_population_ratio"] == 0.0
    assert overall["mean_public_state_churn"] == 0.30
    assert overall["mean_public_state_changed_fact_count"] == 20.0
    assert overall["mean_executed_generations"] == 80.0
