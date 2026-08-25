"""Human-readable hierarchy summary formatting for dpct-run-individual."""

from __future__ import annotations

import re
from pathlib import Path


def _nested_shape(value) -> list[int]:
    """Return the shape of nested list-like values without importing numpy here."""
    shape: list[int] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape

def _format_shape(value) -> str:
    shape = _nested_shape(value)
    if not shape:
        return "scalar"
    return "x".join(str(dim) for dim in shape)

def _format_sequence(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)

def _summary_activation_value(value, *, prefer_last: bool = False) -> str:
    """Pick one representative activation function for compact summaries."""
    if isinstance(value, list):
        if not value:
            return "unknown"
        return str(value[-1] if prefer_last else value[0])
    if value is None:
        return "unknown"
    return str(value)

def _format_activation_funcs(activation_funcs) -> list[str]:
    if not isinstance(activation_funcs, dict):
        return [f"activation_funcs: {activation_funcs}"]

    lines = ["function types:"]
    if "perception" in activation_funcs:
        lines.append(
            f"  perception function: {_summary_activation_value(activation_funcs['perception'])}"
        )
    if "reference" in activation_funcs:
        lines.append(
            f"  reference function: {_summary_activation_value(activation_funcs['reference'])}"
        )
    if "output" in activation_funcs:
        lines.append(
            f"  output function: {_summary_activation_value(activation_funcs['output'], prefer_last=True)}"
        )

    for key in ["perception", "reference", "output", "actions"]:
        if key in activation_funcs:
            lines.append(f"  {key}: {_format_sequence(activation_funcs[key])}")
    for key in sorted(k for k in activation_funcs.keys() if k not in {"perception", "reference", "output", "actions"}):
        lines.append(f"  {key}: {_format_sequence(activation_funcs[key])}")
    return lines

def _unit_type_for_role(role: str, level_index: int, num_levels: int, top_level_obs_indices) -> str:
    """Derive the Keras layer class name used for a given role and level."""
    if role == "perception":
        if top_level_obs_indices is not None and level_index == num_levels - 1:
            return "ElementWiseMultiply"
        return "Dense"
    if role == "reference":
        return "ElementWiseMultiply" if level_index == num_levels - 1 else "Dense"
    if role == "output":
        return "ElementWiseMultiply"
    if role == "actions":
        return "Dense"
    return "Unknown"

def _activation_for_level(activation_funcs, name: str, level_index: int) -> str:
    """Return a named function type for one hierarchy level."""
    if not isinstance(activation_funcs, dict):
        return str(activation_funcs)
    value = activation_funcs.get(name)
    if isinstance(value, list):
        if level_index < len(value):
            return str(value[level_index])
        return "unknown"
    if value is None:
        return "unknown"
    return str(value)

def _layer_level(layer_name: str) -> int | None:
    """Extract the hierarchy level index from names such as PL00/RL01/OL02/CL00."""
    match = re.search(r"(\d+)$", layer_name)
    if not match:
        return None
    return int(match.group(1))

def _format_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

def _format_weight_value(value, indent: str = "    ") -> list[str]:
    """Pretty-print nested scalar/list weight values."""
    if isinstance(value, list):
        if not value:
            return [f"{indent}[]"]
        if all(not isinstance(item, list) for item in value):
            return [f"{indent}[{', '.join(_format_number(item) for item in value)}]"]
        lines = []
        for row in value:
            lines.extend(_format_weight_value(row, indent=indent))
        return lines
    return [f"{indent}{_format_number(value)}"]

def _layer_role(layer_name: str) -> str:
    if layer_name.startswith("PL"):
        return "perception"
    if layer_name.startswith("RL"):
        return "reference"
    if layer_name.startswith("OL"):
        return "output"
    if layer_name.startswith("CL"):
        return "comparator"
    if layer_name == "Actions":
        return "actions"
    return "other"

def _format_level_details(hierarchy: dict, weights: dict, biases: dict) -> list[str]:
    """Render per-level function types and actual weights/biases."""
    levels = hierarchy.get("levels") or []
    activation_funcs = hierarchy.get("activation_funcs")
    all_layer_names = sorted(set(weights) | set(biases))
    lines = ["levels detail:"]

    for level_index, units in enumerate(levels):
        lines.append(f"  level {level_index} ({units} units):")
        num_levels = len(levels)
        top_obs = hierarchy.get("top_level_obs_indices")
        for role in ("perception", "reference", "output"):
            unit_type = _unit_type_for_role(role, level_index, num_levels, top_obs)
            func = _activation_for_level(activation_funcs, role, level_index)
            lines.append(f"    {role}: {unit_type}, function: {func}")
        layer_names = [name for name in all_layer_names if _layer_level(name) == level_index]
        if not layer_names:
            lines.append("    weights: None")
            continue
        lines.append("    weights:")
        for name in layer_names:
            role = _layer_role(name)
            if name in weights:
                lines.append(f"      {name} ({role}, shape {_format_shape(weights[name])}):")
                lines.extend(_format_weight_value(weights[name], indent="        "))
            if name in biases:
                lines.append(f"      {name} bias ({role}, shape {_format_shape(biases[name])}):")
                lines.extend(_format_weight_value(biases[name], indent="        "))

    if "Actions" in weights or "Actions" in biases:
        lines.append("  action output:")
        actions_func = _activation_for_level(activation_funcs, "actions", 0)
        lines.append(f"    actions: Dense, function: {actions_func}")
        if "Actions" in weights:
            lines.append(f"    Actions weights (shape {_format_shape(weights['Actions'])}):")
            lines.extend(_format_weight_value(weights["Actions"], indent="      "))
        if "Actions" in biases:
            lines.append(f"    Actions bias (shape {_format_shape(biases['Actions'])}):")
            lines.extend(_format_weight_value(biases["Actions"], indent="      "))

    return lines

def _format_hierarchy_summary(config: dict, source_path: Path | None = None) -> str:
    """Render a concise, human-friendly summary of a DPCT hierarchy config."""
    hierarchy = config.get("hierarchy", {}) if isinstance(config, dict) else {}
    env_properties = config.get("env_properties", {}) if isinstance(config, dict) else {}
    metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
    weights = config.get("weights", {}) if isinstance(config, dict) else {}
    biases = config.get("biases", {}) if isinstance(config, dict) else {}

    lines: list[str] = ["Hierarchy config"]
    if source_path is not None:
        lines.append(f"source: {source_path}")
    lines.extend(
        [
            f"environment: {config.get('env_name', 'unknown') if isinstance(config, dict) else 'unknown'}",
            f"observation_space: {env_properties.get('observation_space', 'unknown')}",
            f"action_space: {env_properties.get('action_space', 'unknown')}",
            f"levels: {_format_sequence(hierarchy.get('levels'))}",
            f"weight_types: {_format_sequence(hierarchy.get('weight_types'))}",
            f"obs_connection_level: {hierarchy.get('obs_connection_level', 'unknown')}",
            f"top_reference_weight_init: {hierarchy.get('top_reference_weight_init', 'unknown')}",
            f"evolve_references: {hierarchy.get('evolve_references', False)}",
            f"reference_values: {_format_sequence(hierarchy.get('reference_values'))}",
            f"top_level_obs_indices: {_format_sequence(hierarchy.get('top_level_obs_indices'))}",
            f"zerolevel_inputs_indexes: {_format_sequence(hierarchy.get('zerolevel_inputs_indexes'))}",
        ]
    )
    lines.extend(_format_activation_funcs(hierarchy.get("activation_funcs")))
    lines.extend(_format_level_details(hierarchy, weights, biases))

    if metadata:
        lines.append("metadata:")
        for key in [
            "fitness",
            "fitness_method",
            "fitness_source",
            "fitness_observation_indices",
            "fitness_targets",
            "run_steps",
            "total_reward",
            "last_reward",
            "success",
            "random_seed",
        ]:
            if key in metadata:
                lines.append(f"  {key}: {metadata[key]}")

    return "\n".join(lines)


class HierarchySummaryFormatter:
    """Format DPCT config dictionaries for CLI display."""

    format = staticmethod(_format_hierarchy_summary)
