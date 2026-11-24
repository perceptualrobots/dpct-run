"""Public runner uses Pendulum success from the final post-step observation."""

import gymnasium as gym
import numpy as np
import pytest

from dpct_run.individual import DHPCTIndividual


class ScriptedPendulum(gym.Env):
    observation_space = gym.spaces.Box(
        low=np.array([-1., -1., -8.], dtype=np.float32),
        high=np.array([1., 1., 8.], dtype=np.float32),
    )
    action_space = gym.spaces.Box(-2., 2., shape=(1,), dtype=np.float32)

    def __init__(self, settled):
        self.settled = settled
        self.count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.array([-1., 0., 0.], dtype=np.float32), {}

    def step(self, action):
        self.count += 1
        settled = self.settled(self.count)
        return np.array([1., 0., 0.] if settled else [-1., 0., 0.], dtype=np.float32), -1., False, False, {}


@pytest.mark.parametrize("early,settled,expected_steps,expected_success,expected_fitness", [
    (True, lambda n: True, 10, True, 10),
    (False, lambda n: n <= 10, 25, False, 26),
    (False, lambda n: n >= 11, 25, True, 25),
])
def test_pendulum_final_step(monkeypatch, early, settled, expected_steps, expected_success, expected_fitness):
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: ScriptedPendulum(settled))
    individual = DHPCTIndividual(env_name="Pendulum-v1", levels=[1, 2],
        top_level_obs_indices=[0, 1], reference_values=[1., 0.], evolve_references=False)
    individual.compile()
    value = individual.run(steps=25, early_termination=early, fitness_method="evaluation_steps")
    assert individual.run_steps == expected_steps
    assert individual.success is expected_success
    assert value == expected_fitness


def test_pendulum_config_restores_success_thresholds():
    individual = DHPCTIndividual(env_name="Pendulum-v1", levels=[1, 2],
        top_level_obs_indices=[0, 1], reference_values=[1., 0.], evolve_references=False)
    individual.pendulum_settling_window_steps = 12
    individual.pendulum_min_settled_ratio = .75
    individual.pendulum_observation_threshold = [.2, .3, .4]
    individual.pendulum_gradient_threshold = .05
    individual.compile()
    restored = DHPCTIndividual.from_config(individual.config())
    assert restored.pendulum_settling_window_steps == 12
    assert restored.pendulum_min_settled_ratio == .75
    assert restored.pendulum_observation_threshold == [.2, .3, .4]
    assert restored.pendulum_gradient_threshold == .05
