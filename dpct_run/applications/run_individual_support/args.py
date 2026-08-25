"""Argument parsing helpers for dpct-run-individual."""

from __future__ import annotations

import argparse
import ast
import json


_HISTORY_GRAPH_FORMAT = """\
History graph export format:
  dpct-run-individual CONFIG --run --history-dir DIR \\
    [--history-graphs GRAPH[,GRAPH...]]

Video export format:
    dpct-run-individual CONFIG --run --video-out rollout.mp4

Graph names:
  all, network, observations, actions, rewards, errors,
  level0-output, reference-perception

Selectors use zero-based network diagram labels:
  actions:C01                    one action column
  reference-perception:L01C02    one reference/perception level+column
  outputs:L02                    all output columns at one level
  output:L02C03                  one output level+column

Example:
  dpct-run-individual samples/configs/dpct/MountainCarContinuous-cdf7cc.json \\
    --run --steps 999 --seed 42 \\
    --history-dir /tmp/dpct-history \\
    --history-graphs actions,observations,level0-output,reference-perception
"""



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpct-run-individual",
        description="Inspect or run a DHPCTIndividual from a config file or Comet experiment URL.",
        epilog=_HISTORY_GRAPH_FORMAT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_file",
        metavar="config-file-or-comet-url",
        type=str,
        help=(
            "Path to config file (.json/.txt/.properties/.pkl/.pickle), or a Comet "
            "experiment URL whose best_individual.json asset will be cached under /tmp/comet"
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the hierarchy in its configured environment",
    )
    parser.add_argument(
        "--show-config-json",
        action="store_true",
        help="Print the loaded hierarchy config as JSON without running it",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print a user-friendly hierarchy config summary without running it",
    )
    parser.add_argument(
        "--show-model-summary",
        action="store_true",
        help="Compile and print the Keras model summary without running it",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum number of environment steps (default: 500)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during execution",
    )
    parser.add_argument(
        "--early-termination",
        action="store_true",
        help="Stop when environment episode ends",
    )
    parser.add_argument(
        "--fitness-method",
        default=None,
        help=(
            "Optional fitness calculation method override. Built-ins include cumulative_reward, "
            "evaluation_steps, rms, mae, and adjusted error methods; dpct-env "
            "registered methods are also accepted. If omitted, saved config metadata is used "
            "when available, otherwise DHPCTIndividual.run() falls back to cumulative_reward."
        ),
    )
    parser.add_argument(
        "--fitness-kwargs",
        type=str,
        default=None,
        help="JSON/Python dict of configuration kwargs for environment-specific fitness functions",
    )
    parser.add_argument(
        "--fitness-source",
        choices=["comparator_error", "top_level_comparator_error", "observation"],
        default=None,
        help=(
            "Optional source override for RMS/MAE fitness methods: all comparator errors, "
            "top-level comparator errors only, or observation deltas. If omitted, saved "
            "config metadata is used when available, otherwise DHPCTIndividual.run() "
            "falls back to comparator_error."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step observations, actions, errors, and available layer outputs",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save DPCT config to samples/configs/dpct/<stem>.json after the run",
    )
    parser.add_argument(
        "--network-image",
        type=str,
        default=None,
        help=(
            "Optional path to save a rendered network architecture image "
            "(e.g. samples/images/my_network.png)"
        ),
    )
    parser.add_argument(
        "--network-line-thickness",
        type=float,
        default=5.0,
        help="Connection line thickness for --network-image output (default: 5.0)",
    )
    parser.add_argument(
        "--network-label-fontsize",
        type=int,
        default=16,
        help="Unit label font size for --network-image output (default: 16)",
    )
    parser.add_argument(
        "--network-weight-fontsize",
        type=int,
        default=14,
        help="Weight label font size for --network-image output (default: 14)",
    )
    parser.add_argument(
        "--network-smooth-factor-fontsize",
        type=int,
        default=12,
        help="Smooth-factor label font size for --network-image output (default: 12)",
    )
    parser.add_argument(
        "--network-bottom-label-fontsize",
        type=int,
        default=14,
        help="Bottom metadata label font size for --network-image output (default: 14)",
    )
    parser.add_argument(
        "--network-legend-fontsize",
        type=int,
        default=16,
        help="Legend label font size for --network-image output (default: 16)",
    )
    parser.add_argument(
        "--network-legend-title-fontsize",
        type=int,
        default=15,
        help="Legend title font size for --network-image output (default: 15)",
    )
    parser.add_argument(
        "--network-unit-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show per-unit labels in expanded network diagrams (default: true). "
            "Use --no-network-unit-labels to hide them."
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for rollout history graphs. Requires --run. "
            "Use --history-graphs to choose which graphs to save."
        ),
    )
    parser.add_argument(
        "--history-graphs",
        type=str,
        default="all",
        help=(
            "Comma-separated graph names/selectors to save with --history-dir. "
            "Choices: all, network, observations, actions, rewards, errors, "
            "level0-output, reference-perception. Selectors use network diagram "
            "labels, e.g. actions:C01, reference-perception:L01C02, outputs:L02. Default: all."
        ),
    )
    parser.add_argument(
        "--video-out",
        type=str,
        default=None,
        help=(
            "Optional rollout video output path (.mp4 or .gif). Requires --run. "
            "When set, rollout frames are captured from Gymnasium's rgb_array renderer."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frames per second for --video-out output (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed override. If omitted, the seed restored by loading the config is used"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of runs to execute; if --seed is set, seed is incremented each run (default: 1)",
    )
    parser.add_argument(
        "--nevals",
        type=int,
        default=1,
        help="Number of evaluation runs per iteration (default: 1, must be >= 1)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "max", "min", "median"],
        default="mean",
        help="Aggregation method when nevals > 1 (default: mean)",
    )
    parser.add_argument(
        "--show-tf-warnings",
        action="store_true",
        help="Keep TensorFlow startup warnings visible for debugging",
    )
    return parser

def _parse_mapping_text(config_text: str | None, arg_name: str) -> dict:
    if not config_text:
        return {}
    try:
        parsed = json.loads(config_text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(config_text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"{arg_name} must be valid JSON or Python dict text") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must parse to a dictionary")
    return parsed

def _has_requested_action(args: argparse.Namespace) -> bool:
    """Return True when the CLI invocation requests work beyond loading the file."""
    return any(
        [
            args.run,
            args.show_config_json,
            args.show_config,
            args.show_model_summary,
            args.save_config,
            bool(args.network_image),
            bool(args.history_dir),
            bool(args.video_out),
        ]
    )

def _print_action_hint(parser: argparse.ArgumentParser) -> None:
    """Explain that config-file alone is inspect-only and list useful actions."""
    print("No action requested; the hierarchy was not run.")
    print("")
    print("Pass one or more options after config-file, for example:")
    print("  --run                    Run the hierarchy in its environment")
    print("  --show-config-json       Print the loaded hierarchy config as JSON")
    print("  --show-config            Print a user-friendly hierarchy config summary")
    print("  --show-model-summary     Compile and print the Keras model summary")
    print("  --save-config            Save the loaded config as DPCT JSON")
    print("  --network-image PATH     Save a hierarchy architecture image")
    print("  --video-out PATH         Save an environment rollout video (.mp4/.gif)")
    print("")
    print(f"Use '{parser.prog} --help' for all options.")
