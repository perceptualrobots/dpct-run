"""Rollout history graph parsing and export for dpct-run-individual."""

from __future__ import annotations

import re
from pathlib import Path

from dpct_run.visualization import draw_traditional_pcnn


_HISTORY_GRAPH_ALIASES = {
    "obs": "observations",
    "observation": "observations",
    "observations": "observations",
    "action": "actions",
    "actions": "actions",
    "reward": "rewards",
    "rewards": "rewards",
    "error": "errors",
    "errors": "errors",
    "network": "network",
    "hierarchy": "network",
    "level0": "level0-output",
    "level0-output": "level0-output",
    "level0_output": "level0-output",
    "reference": "reference-perception",
    "ref-perception": "reference-perception",
    "reference-perception": "reference-perception",
    "reference_perception": "reference-perception",
    "output": "outputs",
    "outputs": "outputs",
}
_HISTORY_GRAPH_DEFAULTS = {
    "network",
    "observations",
    "actions",
    "rewards",
    "errors",
    "level0-output",
    "reference-perception",
}



def _history_node_label(level: int | None = None, column: int | None = None) -> str:
    """Return network-diagram-style labels such as L01C02 or C01."""
    if level is None:
        if column is None:
            raise ValueError("column is required when level is omitted")
        return f"C{column:02d}"
    if column is None:
        return f"L{level:02d}"
    return f"L{level:02d}C{column:02d}"

def _parse_history_column_selector(selector: str) -> int:
    """Parse action column selectors such as C01 or 1."""
    selector = selector.strip().lower().replace("-", "_")
    match = re.fullmatch(r"(?:c|col|column)?(?P<column>\d+)", selector)
    if not match:
        raise ValueError(f"selector '{selector}' must look like C01")
    return int(match.group("column"))

def _parse_history_level_column_selector(selector: str, *, require_column: bool) -> tuple[int, int | None]:
    """Parse selectors such as L02 or L01C02.

    Legacy U/unit spellings are still accepted for compatibility, but help and
    output filenames use network-diagram Levels/Columns terminology.
    """
    selector = selector.strip().lower().replace("-", "_")
    match = re.fullmatch(
        r"(?:l|level)?(?P<level>\d+)(?:(?:c|col|column|u|unit)(?P<column>\d+))?",
        selector,
    )
    if not match:
        raise ValueError(f"selector '{selector}' must look like L02 or L01C02")
    level = int(match.group("level"))
    column_text = match.group("column")
    if require_column and column_text is None:
        raise ValueError(f"selector '{selector}' must include a column, e.g. L01C02")
    return level, None if column_text is None else int(column_text)

def _parse_history_graphs(text: str) -> set[str]:
    """Parse --history-graphs into canonical graph names/selectors.

    Simple entries are graph names (e.g. ``actions``). Selector entries use
    ``graph:selector`` syntax, e.g. ``actions:C01``,
    ``reference-perception:L01C02``, or ``outputs:L02``.
    Indices are zero-based to match DPCT layer/column names.
    """
    raw_items = [item.strip().lower() for item in (text or "all").split(",") if item.strip()]
    if not raw_items or "all" in raw_items:
        return set(_HISTORY_GRAPH_DEFAULTS)

    graphs: set[str] = set()
    invalid: list[str] = []
    for item in raw_items:
        if ":" in item:
            prefix, selector = item.split(":", 1)
            canonical = _HISTORY_GRAPH_ALIASES.get(prefix)
            if canonical == "actions":
                try:
                    graphs.add(f"action:{_parse_history_column_selector(selector)}")
                except ValueError:
                    invalid.append(item)
            elif canonical == "reference-perception":
                try:
                    level, column = _parse_history_level_column_selector(selector, require_column=True)
                    graphs.add(f"reference-perception:{level}:{column}")
                except ValueError:
                    invalid.append(item)
            elif canonical == "outputs":
                try:
                    level, column = _parse_history_level_column_selector(selector, require_column=False)
                    if column is None:
                        graphs.add(f"outputs:{level}")
                    else:
                        graphs.add(f"output:{level}:{column}")
                except ValueError:
                    invalid.append(item)
            else:
                invalid.append(item)
            continue

        canonical = _HISTORY_GRAPH_ALIASES.get(item)
        if canonical is None or canonical == "outputs":
            invalid.append(item)
        else:
            graphs.add(canonical)
    if invalid:
        valid = sorted(set(_HISTORY_GRAPH_ALIASES) | {"all", "actions:C01", "reference-perception:L01C02", "outputs:L02"})
        raise ValueError(f"Unknown --history-graphs value(s): {', '.join(invalid)}. Valid values: {', '.join(valid)}")
    return graphs

def _array_from_history_layer(history, layer_name: str):
    import numpy as np

    layer_activations = getattr(history, "layer_activations", {}) or {}
    values = layer_activations.get(layer_name)
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def _final_activation_layer_name(history, prefix: str, level: int) -> str | None:
    """Return final active layer name for a role/level, preferring wrappers."""
    layer_activations = getattr(history, "layer_activations", {}) or {}
    level_name = f"{level:02d}"
    return next(
        (name for name in (
            f"{prefix}{level_name}_smooth",
            f"{prefix}{level_name}_sigmoid",
            f"{prefix}{level_name}_activation",
            f"{prefix}{level_name}",
        ) if name in layer_activations),
        None,
    )

def _history_steps(length: int):
    """Return human-readable one-based step numbers for recorded rollout data."""
    import numpy as np

    return np.arange(1, int(length) + 1)

def _save_series_graph(data, output_path: Path, *, title: str, ylabel: str, label_prefix: str, column_idx: int | None = None) -> list[Path]:
    """Save all columns or one selected column from a 1D/2D time series."""
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return []
    if column_idx is not None:
        if column_idx < 0 or column_idx >= arr.shape[1]:
            return []
        arr = arr[:, [column_idx]]
        labels = [f"{label_prefix}[{column_idx}]"]
    else:
        labels = [f"{label_prefix}[{idx}]" for idx in range(arr.shape[1])]

    steps = _history_steps(arr.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, label in enumerate(labels):
        ax.plot(steps, arr[:, idx], label=label, linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(labels) <= 12:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [output_path] if output_path.exists() else []

def _save_action_column_graph(history, output_dir: Path, column_idx: int) -> list[Path]:
    label = _history_node_label(column=column_idx)
    path = output_dir / f"action_{label}.png"
    return _save_series_graph(
        history.get_actions_array(),
        path,
        title=f"Action {label}",
        ylabel="Action",
        label_prefix="action",
        column_idx=column_idx,
    )

def _save_output_level_graph(history, output_dir: Path, level: int, column_idx: int | None = None) -> list[Path]:
    layer_name = _final_activation_layer_name(history, "OL", level)
    if layer_name is None:
        return []
    data = _array_from_history_layer(history, layer_name)
    if data is None:
        return []
    if column_idx is None:
        label = _history_node_label(level=level)
        path = output_dir / f"output_{label}.png"
        title = f"Output {label} ({layer_name})"
    else:
        label = _history_node_label(level=level, column=column_idx)
        path = output_dir / f"output_{label}.png"
        title = f"Output {label} ({layer_name})"
    return _save_series_graph(
        data,
        path,
        title=title,
        ylabel="Output",
        label_prefix=layer_name,
        column_idx=column_idx,
    )

def _save_reference_perception_graphs(
    individual,
    history,
    output_dir: Path,
    level: int | None = None,
    column: int | None = None,
) -> list[Path]:
    """Save reference/perception graphs for a level/column selection.

    Perception/reference layer activations are computed before ``env.step``.
    When top-level units are wired directly to observations, overlay the matching
    post-step environment observation so terminal target crossings are visible.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    saved_paths: list[Path] = []
    levels = list(getattr(individual, "levels", []) or [])
    if not levels:
        return saved_paths

    selected_level = len(levels) - 1 if level is None else int(level)
    if selected_level < 0 or selected_level >= len(levels):
        return saved_paths

    rl_name = _final_activation_layer_name(history, "RL", selected_level)
    pl_name = _final_activation_layer_name(history, "PL", selected_level)
    if rl_name is None or pl_name is None:
        return saved_paths

    ref_arr = _array_from_history_layer(history, rl_name)
    perception_arr = _array_from_history_layer(history, pl_name)
    if ref_arr is None or perception_arr is None:
        return saved_paths

    top_level_obs_indices = getattr(individual, "top_level_obs_indices", None)
    if top_level_obs_indices is not None:
        if isinstance(top_level_obs_indices, np.ndarray):
            top_level_obs_indices = top_level_obs_indices.tolist()
        elif isinstance(top_level_obs_indices, (list, tuple)):
            top_level_obs_indices = list(top_level_obs_indices)
        else:
            top_level_obs_indices = None

    post_step_observations = None
    if hasattr(history, "get_observations_array"):
        post_step_observations = np.asarray(history.get_observations_array())
    elif hasattr(history, "observations"):
        post_step_observations = np.asarray(getattr(history, "observations", []), dtype=np.float32)

    unit_count = min(ref_arr.shape[1], perception_arr.shape[1], int(levels[selected_level]))
    columns = range(unit_count) if column is None else [int(column)]
    for column_idx in columns:
        if column_idx < 0 or column_idx >= unit_count:
            continue
        label = _history_node_label(level=selected_level, column=column_idx)
        steps = _history_steps(min(ref_arr.shape[0], perception_arr.shape[0]))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, ref_arr[:len(steps), column_idx], label=f"reference {rl_name}[{column_idx}]", linewidth=1.5)
        ax.plot(steps, perception_arr[:len(steps), column_idx], label=f"perception {pl_name}[{column_idx}] before action", linewidth=1.5)
        if (
            selected_level == len(levels) - 1
            and top_level_obs_indices is not None
            and column_idx < len(top_level_obs_indices)
            and post_step_observations is not None
            and post_step_observations.ndim == 2
        ):
            obs_idx = int(top_level_obs_indices[column_idx])
            if 0 <= obs_idx < post_step_observations.shape[1]:
                post_steps = np.arange(1, post_step_observations.shape[0] + 1)
                ax.plot(
                    post_steps,
                    post_step_observations[:, obs_idx],
                    label=f"env obs[{obs_idx}] after action",
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.9,
                )
                ax.scatter(
                    [post_steps[-1]],
                    [post_step_observations[-1, obs_idx]],
                    s=28,
                    zorder=3,
                )
        ax.set_xlabel("Controller step")
        ax.set_ylabel("Value")
        ax.set_title(f"Reference vs perception {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        filename = f"reference_perception_{label}.png"
        path = output_dir / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths

def _history_metric_data(history, metric: str):
    """Return data/y-axis labels for standard rollout metric graphs."""
    if metric == "observations":
        if hasattr(history, "get_complete_observations_array"):
            return history.get_complete_observations_array(), "Observations", "observation"
        return history.get_observations_array(), "Observations", "observation"
    if metric == "actions":
        return history.get_actions_array(), "Actions", "action"
    if metric == "rewards":
        return history.get_rewards_array(), "Rewards", "reward"
    if metric == "errors":
        return history.get_errors_array(), "Errors", "error"
    raise ValueError(f"Unknown history metric: {metric}")

def _save_history_graphs(individual, output_dir: Path, graphs: set[str]) -> list[Path]:
    """Save selected rollout/history graphs for a completed individual run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    history = getattr(individual, "history", None) or getattr(individual, "last_environment_history", None)

    if "network" in graphs:
        network_path = output_dir / "hierarchy_network.png"
        draw_traditional_pcnn(individual, filename=str(network_path), expand_units=True, show_weights=True)
        if network_path.exists():
            saved_paths.append(network_path)

    if history is None:
        return saved_paths

    metric_graphs = {
        "observations": ("observations", "observation_history.png"),
        "actions": ("actions", "action_history.png"),
        "rewards": ("rewards", "reward_history.png"),
        "errors": ("errors", "error_history.png"),
    }
    for graph_name, (metric, filename) in metric_graphs.items():
        if graph_name not in graphs:
            continue
        path = output_dir / filename
        data, ylabel, label_prefix = _history_metric_data(history, metric)
        saved_paths.extend(
            _save_series_graph(
                data,
                path,
                title=ylabel,
                ylabel=ylabel,
                label_prefix=label_prefix,
            )
        )

    if "level0-output" in graphs:
        saved_paths.extend(_save_output_level_graph(history, output_dir, 0))

    for graph in sorted(graphs):
        if graph.startswith("action:"):
            _, column_text = graph.split(":", 1)
            saved_paths.extend(_save_action_column_graph(history, output_dir, int(column_text)))
        elif graph.startswith("outputs:"):
            _, level_text = graph.split(":", 1)
            saved_paths.extend(_save_output_level_graph(history, output_dir, int(level_text)))
        elif graph.startswith("output:"):
            _, level_text, column_text = graph.split(":", 2)
            saved_paths.extend(_save_output_level_graph(history, output_dir, int(level_text), int(column_text)))
        elif graph.startswith("reference-perception:"):
            _, level_text, column_text = graph.split(":", 2)
            saved_paths.extend(
                _save_reference_perception_graphs(
                    individual,
                    history,
                    output_dir,
                    level=int(level_text),
                    column=int(column_text),
                )
            )

    if "reference-perception" in graphs:
        saved_paths.extend(_save_reference_perception_graphs(individual, history, output_dir))

    return saved_paths


class HistoryGraphExporter:
    """Parse graph selections and export rollout history images."""

    parse_graphs = staticmethod(_parse_history_graphs)
    save = staticmethod(_save_history_graphs)
