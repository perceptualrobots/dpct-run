from pathlib import Path

import numpy as np

from dpct_run import DHPCTIndividual


def test_runtime_individual_has_no_evolution_api():
    assert not hasattr(DHPCTIndividual, "mate")
    assert not hasattr(DHPCTIndividual, "mutate")


def test_load_fixture_and_run_short_rollout():
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
