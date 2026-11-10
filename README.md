# dpct-run

Runtime-only package for running saved Deep Perceptual Control Theory (DPCT) individuals.

`dpct-run` is intended for external users who need to replay/evaluate trained DPCT individuals in Gymnasium environments without access to the full private DPCT evolution and optimization stack.

## What is included

- Load saved `DHPCTIndividual.config()` / `best_individual.json` files
- Reconstruct the Keras runtime controller
- Run/evaluate individuals in Gymnasium environments
- Optional model summaries, network images, rollout history graphs, and videos
- Optional Comet `best_individual.json` download helper
- Optional legacy `pct` config loading/conversion support

## What is deliberately not included

- Evolution (`DHPCTEvolver`)
- Optuna optimization (`DHPCTOptimizer`)
- Genetic mutation/mating APIs
- DEAP/Optuna dependencies
- Comet experiment logging for evolutionary runs

## Install

From PyPI:

```bash
pip install dpct-run
```

Optional extras:

```bash
pip install "dpct-run[plots,video,comet,legacy]"
```

From GitHub:

```bash
pip install git+https://github.com/perceptualrobots/dpct-run.git
```

### Environment-specific support

Rollout task-success interpretation and environment-specific fitness methods are
provided by `dpct-env`. Install a compatible `dpct-env` checkout into the same
Python environment when running individuals:

```bash
python -m pip install -e /path/to/dpct-env
```

`dpct-run` raises `DpctEnvRequiredError` with installation guidance rather than
silently reporting generic or misleading success when `dpct-env` is unavailable.

## CLI examples

Show a saved config summary:

```bash
dpct-run-individual best_individual.json --show-config
```

Run a saved individual:

```bash
dpct-run-individual best_individual.json --run --steps 500 --seed 42
```

Save rollout history graphs:

```bash
dpct-run-individual best_individual.json --run --history-dir ./history --history-graphs all
```

Use the short alias:

```bash
dpct-run best_individual.json --run --steps 500
```

## Python example

```python
from dpct_run import DHPCTIndividual

config = DHPCTIndividual.load_config("best_individual.json")
individual = DHPCTIndividual.from_config(config)
fitness = individual.evaluate(steps=500, early_termination=True)
print(fitness, individual.success, individual.total_reward)
```

## Compatibility notes

`dpct-run` follows the saved DPCT config schema produced by the full DPCT package. Treat `best_individual.json` as the primary interchange artifact.

This initial extraction targets classic Gymnasium-style environments and generic built-in fitness methods (`cumulative_reward`, `evaluation_steps`, `rms`, `mae`, adjusted RMS/MAE). Environment-specific scoring and task-success interpretation are delegated to `dpct-env`; runtime evaluation fails clearly if that package is required but unavailable.
