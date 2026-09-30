from pathlib import Path
import builtins

import numpy as np
import pytest

from dpct_run import DHPCTIndividual
import dpct_run.individual as individual_module


def test_runtime_individual_has_no_evolution_api():
    assert not hasattr(DHPCTIndividual, "mate")
    assert not hasattr(DHPCTIndividual, "mutate")


def test_load_fixture_and_run_short_rollout(monkeypatch):
    class FixtureResultsProcessor:
        def get_task_success(self, individual, history, env_name=None):
            return None

    monkeypatch.setattr(
        individual_module,
        "_resolve_environment_results_processor",
        lambda env_name: FixtureResultsProcessor(),
    )
    cfg = DHPCTIndividual.load_config(str(Path(__file__).parent / "fixtures" / "MountainCarContinuous-cdf7cc.json"))
    ind = DHPCTIndividual.from_config(cfg)
    fitness = ind.evaluate(steps=5, early_termination=True)
    assert np.isfinite(fitness)
    assert ind.run_steps > 0
    assert ind.total_reward is not None


def test_cli_show_config_json(capsys):
    from dpct_run.applications.run_individual import main

    path = Path(__file__).parent / "fixtures" / "MountainCarContinuous-cdf7cc.json"
    rc = main([str(path), "--show-config-json"])
    captured = capsys.readouterr()
    assert rc == 0
    assert '"env_name"' in captured.out
    assert "Command complete" in captured.err


def test_config_round_trip_preserves_evaluation_seed():
    ind = DHPCTIndividual("CartPole-v1", [2, 1], random_seed=11)
    ind.compile()
    ind.evaluation_seed = 42

    config = ind.config()
    restored = DHPCTIndividual.from_config(config)

    assert config["metadata"]["random_seed"] == 11
    assert config["metadata"]["evaluation_seed"] == 42
    assert restored.random_seed == 11
    assert restored.evaluation_seed == 42


def test_evaluate_uses_saved_evaluation_seed_and_restores_genome_seed(monkeypatch):
    ind = DHPCTIndividual("CartPole-v1", [2, 1], random_seed=11)
    ind.evaluation_seed = 42
    seen = []
    monkeypatch.setattr(ind, "run", lambda **kwargs: seen.append(ind.random_seed) or 1.0)

    assert ind.evaluate(nevals=2) == 1.0
    assert seen == [42, 43]
    assert ind.random_seed == 11
    assert ind.evaluation_seed == 42


def test_explicit_evaluation_seed_overrides_saved_seed(monkeypatch):
    ind = DHPCTIndividual("CartPole-v1", [2, 1], random_seed=11)
    ind.evaluation_seed = 42
    seen = []
    monkeypatch.setattr(ind, "run", lambda **kwargs: seen.append(ind.random_seed) or 1.0)

    assert ind.evaluate(nevals=2, evaluation_seed=7) == 1.0
    assert seen == [7, 8]
    assert ind.random_seed == 11
    assert ind.evaluation_seed == 7


def test_old_config_without_evaluation_seed_falls_back_to_genome_seed(monkeypatch):
    ind = DHPCTIndividual("CartPole-v1", [2, 1], random_seed=13)
    ind.compile()
    config = ind.config()
    config["metadata"].pop("evaluation_seed", None)
    restored = DHPCTIndividual.from_config(config)
    seen = []
    monkeypatch.setattr(restored, "run", lambda **kwargs: seen.append(restored.random_seed) or 1.0)

    assert restored.evaluate(nevals=2) == 1.0
    assert seen == [13, 14]
    assert restored.random_seed == 13
    assert restored.evaluation_seed is None


def test_evaluate_restores_genome_seed_when_run_raises(monkeypatch):
    ind = DHPCTIndividual("CartPole-v1", [2, 1], random_seed=11)
    ind.evaluation_seed = 42

    def fail(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ind, "run", fail)

    try:
        ind.evaluate()
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert ind.random_seed == 11
    assert ind.evaluation_seed == 42


def test_cli_seed_explicitly_overrides_saved_evaluation_seed(monkeypatch, capsys):
    from dpct_run.applications import run_individual

    class FakeIndividual:
        random_seed = 11
        evaluation_seed = 42
        run_steps = 7
        success = True

        def __init__(self):
            self.evaluate_calls = []

        def evaluate(self, **kwargs):
            self.evaluate_calls.append(kwargs)
            return 1.0

    fake = FakeIndividual()
    monkeypatch.setattr(run_individual, "_individual_from_config_file", lambda path: fake)
    path = Path(__file__).parent / "fixtures" / "MountainCarContinuous-cdf7cc.json"

    rc = run_individual.main([str(path), "--run", "--seed", "7"])
    captured = capsys.readouterr()

    assert rc == 0
    assert fake.evaluate_calls[0]["evaluation_seed"] == 7
    assert fake.random_seed == 11
    assert "Seed: 7" in captured.out


def test_cli_reports_saved_evaluation_seed_when_no_override(monkeypatch, capsys):
    from dpct_run.applications import run_individual

    class FakeIndividual:
        random_seed = 11
        evaluation_seed = 42
        run_steps = 7
        success = True

        def evaluate(self, **kwargs):
            return 1.0

    monkeypatch.setattr(run_individual, "_individual_from_config_file", lambda path: FakeIndividual())
    path = Path(__file__).parent / "fixtures" / "MountainCarContinuous-cdf7cc.json"

    rc = run_individual.main([str(path), "--run"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Seed: 42" in captured.out


def test_task_success_requires_dpct_env_when_unavailable(monkeypatch):
    original_import = builtins.__import__

    def fail_dpct_env_import(name, *args, **kwargs):
        if name == "dpct_env.processors":
            raise ImportError("dpct-env unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_dpct_env_import)
    individual = type(
        "FakeIndividual",
        (),
        {"env_name": "MountainCarContinuous-v0", "last_environment_history": object()},
    )()

    with pytest.raises(individual_module.DpctEnvRequiredError) as exc_info:
        individual_module._get_environment_task_success(individual)

    message = str(exc_info.value)
    assert "dpct-env is required" in message
    assert "MountainCarContinuous-v0" in message
    assert "python -m pip install" in message


def test_generic_task_success_falls_back_when_dpct_env_is_unavailable(monkeypatch):
    original_import = builtins.__import__

    def fail_dpct_env_import(name, *args, **kwargs):
        if name == "dpct_env.processors":
            raise ImportError("dpct-env unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_dpct_env_import)
    individual = type(
        "FakeIndividual",
        (),
        {"env_name": "Pendulum-v1", "last_environment_history": object()},
    )()

    assert individual_module._get_environment_task_success(individual) is None


def test_task_success_reports_broken_dpct_env_dependency(monkeypatch):
    original_import = builtins.__import__

    def fail_dpct_env_import(name, *args, **kwargs):
        if name == "dpct_env.processors":
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_dpct_env_import)
    individual = type(
        "FakeIndividual",
        (),
        {"env_name": "MountainCarContinuous-v0", "last_environment_history": object()},
    )()

    with pytest.raises(individual_module.DpctEnvRequiredError) as exc_info:
        individual_module._get_environment_task_success(individual)

    message = str(exc_info.value)
    assert "installed but could not be imported" in message
    assert "matplotlib" in message


def test_environment_specific_fitness_requires_dpct_env_when_unavailable(monkeypatch):
    original_import = builtins.__import__

    def fail_dpct_env_import(name, *args, **kwargs):
        if name == "dpct_env.fitness":
            raise ImportError("dpct-env unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_dpct_env_import)

    with pytest.raises(individual_module.DpctEnvRequiredError) as exc_info:
        individual_module._resolve_environment_fitness_function(
            "height_achieved",
            env_name="MountainCarContinuous-v0",
        )

    message = str(exc_info.value)
    assert "dpct-env is required" in message
    assert "height_achieved" in message
    assert "python -m pip install" in message


def test_environment_specific_fitness_dependency_is_checked_before_rollout(monkeypatch):
    individual = DHPCTIndividual("CartPole-v1", [2, 1], random_seed=11)
    individual.compile()

    def fail_fitness_resolution(*args, **kwargs):
        raise individual_module.DpctEnvRequiredError("missing dpct-env")

    monkeypatch.setattr(
        individual_module,
        "_resolve_environment_fitness_function",
        fail_fitness_resolution,
    )

    import gymnasium as gym

    monkeypatch.setattr(
        gym,
        "make",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rollout started")),
    )

    with pytest.raises(individual_module.DpctEnvRequiredError, match="missing dpct-env"):
        individual.run(fitness_method="height_achieved")


def test_dpct_env_required_error_is_exported():
    from dpct_run import DpctEnvRequiredError

    assert DpctEnvRequiredError is individual_module.DpctEnvRequiredError
