from argparse import Namespace

from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import evolution_config


def test_formal_evolution_config_propagates_trace_and_reference_free_stop() -> None:
    args = Namespace(
        population_size=200,
        initial_population_size=200,
        generations=200,
        initial_generations=200,
        archive_limit=200,
        record_metric_trace=True,
        record_trace_archives=False,
        metric_trace_interval=1,
        metric_trace_archive_limit=200,
        reference_free_early_stop=True,
        early_stop_min_generations=40,
        early_stop_patience=20,
        early_stop_hv_epsilon=5e-4,
        early_stop_scalar_epsilon=5e-4,
        early_stop_epsilon_box=0.01,
        early_stop_min_archive_size=8,
    )

    config = evolution_config(args, seed=7, stage_index=0)

    assert config.population_size == 200
    assert config.generations == 200
    assert config.seed == 7
    assert config.record_metric_history is True
    assert config.reference_free_early_stopping is True
    assert config.early_stop_min_generations == 40
    assert config.early_stop_patience == 20
    assert config.early_stop_hv_epsilon == 5e-4
    assert config.early_stop_scalar_epsilon == 5e-4
    assert config.early_stop_epsilon_box == 0.01
    assert config.early_stop_min_archive_size == 8
