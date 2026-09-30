"""CLI for running a DHPCTIndividual from a saved config file.

Runtime examples live in ``dpct/applications/run_individual.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from dpct_run.applications.run_individual_support.environment import (
    _configure_tensorflow_logging,
    _load_local_env_files,
    _should_show_tf_warnings,
)

_configure_tensorflow_logging()

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")

from dpct_run.applications.run_individual_support.args import (  # noqa: E402
    _build_parser,
    _has_requested_action,
    _parse_mapping_text,
    _print_action_hint,
)
from dpct_run.applications.run_individual_support.config_loader import (  # noqa: E402
    IndividualConfigLoader,
    _comet_best_individual_cache_path,
    _download_comet_best_individual,
    _extract_legacy_config_dict,
    _individual_from_config_file,
    _is_comet_experiment_url,
    _parse_comet_experiment_url,
)
from dpct_run.applications.run_individual_support.history_graphs import (  # noqa: E402
    HistoryGraphExporter,
    _history_node_label,
    _history_steps,
    _parse_history_graphs,
    _save_history_graphs,
)
from dpct_run.applications.run_individual_support.summary import (  # noqa: E402
    HierarchySummaryFormatter,
    _format_hierarchy_summary,
)
from dpct_run.visualization import draw_traditional_pcnn  # noqa: E402


def _resolve_config_source(source: str) -> Path:
    """Resolve a local config file or Comet URL to a local config path.

    Kept as a thin wrapper so tests and callers can monkeypatch the public
    helpers exposed by this CLI module while implementation lives in
    ``run_individual_support.config_loader``.
    """
    if _is_comet_experiment_url(source):
        cache_path = _comet_best_individual_cache_path(source)
        if cache_path.exists():
            print(f"Using cached Comet best_individual.json: {cache_path}", file=sys.stderr)
            return cache_path
        print(f"Downloading Comet best_individual.json to: {cache_path}", file=sys.stderr)
        _download_comet_best_individual(source, cache_path)
        return cache_path

    config_path = Path(source)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path


def _resolve_video_output_path(base_output: str, run_index: int, total_runs: int) -> Path:
    """Resolve per-run rollout video output path.

    Adds a ``.mp4`` suffix when omitted and appends ``_run_XXX`` for batches.
    """
    base_path = Path(base_output)
    if not base_path.suffix:
        base_path = base_path.with_suffix(".mp4")

    if total_runs <= 1:
        return base_path

    return base_path.with_name(f"{base_path.stem}_run_{run_index:03d}{base_path.suffix}")


class _RolloutVideoRecorder:
    """Simple frame writer for rollout captures."""

    def __init__(self, output_path: Path, fps: int = 30):
        self.output_path = output_path
        self.fps = int(fps)
        self.frame_count = 0
        self._writer = None
        self._imageio = None

    def __enter__(self):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Video export requires imageio. Install dependencies with: "
                "pip install imageio imageio-ffmpeg"
            ) from exc

        self._imageio = imageio
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._writer = self._imageio.get_writer(str(self.output_path), fps=self.fps)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize video writer for '{self.output_path}'. "
                "Use a .mp4 or .gif output path and ensure imageio/imageio-ffmpeg are available."
            ) from exc
        return self

    def add_frame(self, frame) -> None:
        if self._writer is None or frame is None:
            return
        array = np.asarray(frame)
        if array.ndim != 3:
            return
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        self._writer.append_data(array)
        self.frame_count += 1

    def __exit__(self, exc_type, exc, tb):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return False


def main(argv: list[str] | None = None) -> int:
    command_start = time.perf_counter()
    try:
        _load_local_env_files()

        parser = _build_parser()
        args = parser.parse_args(argv)

        try:
            config_path = _resolve_config_source(args.config_file)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if args.iterations < 1:
            print("--iterations must be >= 1.", file=sys.stderr)
            return 2

        if args.nevals < 1:
            print("--nevals must be >= 1.", file=sys.stderr)
            return 2

        if not _has_requested_action(args):
            _print_action_hint(parser)
            return 0

        if args.history_dir and not args.run:
            print("--history-dir requires --run so rollout history is available.", file=sys.stderr)
            return 2

        if args.video_out and not args.run:
            print("--video-out requires --run so rollout frames are available.", file=sys.stderr)
            return 2

        if args.video_fps < 1:
            print("--video-fps must be >= 1.", file=sys.stderr)
            return 2

        if args.video_out and args.render:
            print(
                "--render is ignored when --video-out is set; using rgb_array capture for video export.",
                file=sys.stderr,
            )

        try:
            history_graphs = _parse_history_graphs(args.history_graphs)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if args.seed is not None:
            seed_values: list[int | None] = [args.seed + i for i in range(args.iterations)]
        else:
            seed_values = [None for _ in range(args.iterations)]

        try:
            fitness_kwargs = _parse_mapping_text(args.fitness_kwargs, "--fitness-kwargs")
            run_results: list[tuple[int | None, float, int, bool | None]] = []
            individual = None
            if args.show_config_json or args.show_config or args.show_model_summary or not args.run:
                individual = _individual_from_config_file(config_path)

            if args.show_config_json:
                assert individual is not None
                print(json.dumps(individual.config(), indent=2))

            if args.show_config:
                assert individual is not None
                print(_format_hierarchy_summary(individual.config(), source_path=config_path))

            if args.show_model_summary:
                assert individual is not None
                individual.compile()
                individual.model.summary()

            if args.run:
                run_results = []
                individual = None
                for idx, seed in enumerate(seed_values, start=1):
                    run_start = time.perf_counter()
                    individual = _individual_from_config_file(config_path)
                    video_output_path: Optional[Path] = None
                    evaluate_kwargs = {
                        "nevals": args.nevals,
                        "aggregate": args.aggregate,
                        "steps": args.steps,
                        "early_termination": args.early_termination,
                        "render": args.render,
                        "fitness_method": args.fitness_method,
                        "fitness_source": args.fitness_source,
                        "fitness_kwargs": fitness_kwargs,
                        "verbose": args.verbose,
                    }
                    if seed is not None:
                        evaluate_kwargs["evaluation_seed"] = seed
                    if args.history_dir:
                        evaluate_kwargs["record_history"] = True

                    if args.video_out:
                        video_output_path = _resolve_video_output_path(args.video_out, idx, len(seed_values))
                        evaluate_kwargs["render"] = False
                        evaluate_kwargs["render_mode"] = "rgb_array"
                        with _RolloutVideoRecorder(video_output_path, fps=args.video_fps) as recorder:
                            evaluate_kwargs["frame_callback"] = recorder.add_frame
                            fitness = individual.evaluate(**evaluate_kwargs)
                        if recorder.frame_count > 0:
                            print(
                                f"Video saved to {video_output_path} "
                                f"({recorder.frame_count} frames @ {args.video_fps} fps)"
                            )
                        else:
                            print(
                                f"Video writer created at {video_output_path}, but no frames were captured.",
                                file=sys.stderr,
                            )
                    else:
                        fitness = individual.evaluate(**evaluate_kwargs)

                    run_elapsed = time.perf_counter() - run_start
                    actual_steps = getattr(individual, "run_steps", 0)
                    success = getattr(individual, "success", None)
                    effective_seed = seed
                    if effective_seed is None:
                        effective_seed = getattr(individual, "evaluation_seed", None)
                    if effective_seed is None:
                        effective_seed = getattr(individual, "random_seed", None)
                    run_results.append((effective_seed, fitness, actual_steps, success))
                    seed_str = "None" if effective_seed is None else str(effective_seed)
                    success_str = "None" if success is None else str(bool(success))
                    if len(seed_values) == 1:
                        print(
                            f"Run complete. Seed: {seed_str} | Fitness: {fitness} | "
                            f"Success: {success_str} | Configured steps: {args.steps} | "
                            f"Actual steps run: {actual_steps} | "
                            f"N evals: {args.nevals} ({args.aggregate}) | "
                            f"Elapsed seconds: {run_elapsed:.3f}"
                        )
                    else:
                        print(
                            f"Run {idx}: Seed: {seed_str} | Fitness: {fitness} | "
                            f"Success: {success_str} | Configured steps: {args.steps} | "
                            f"Actual steps run: {actual_steps} | "
                            f"N evals: {args.nevals} ({args.aggregate}) | "
                            f"Elapsed seconds: {run_elapsed:.3f}"
                        )
                    if args.history_dir:
                        history_base = Path(args.history_dir)
                        history_output_dir = history_base if len(seed_values) == 1 else history_base / f"run_{idx:03d}"
                        saved_graphs = _save_history_graphs(individual, history_output_dir, history_graphs)
                        if saved_graphs:
                            print(f"History graphs saved to {history_output_dir}")
                            for path in saved_graphs:
                                print(f"  {path.name}")
                        else:
                            print(f"No history graphs were saved to {history_output_dir}", file=sys.stderr)
        except Exception as exc:
            print(f"Failed to run individual: {exc}", file=sys.stderr)
            return 1

        if len(run_results) > 1:
            print(f"Run batch complete. Total runs: {len(run_results)}")

            fitness_values = [fitness for _, fitness, _, _ in run_results]
            mean_fitness = sum(fitness_values) / len(fitness_values)
            success_values = [success for _, _, _, success in run_results]
            succeeded = sum(1 for success in success_values if success is True)
            failed = sum(1 for success in success_values if success is False)
            unknown = sum(1 for success in success_values if success is None)
            print(
                "Batch fitness summary: "
                f"mean={mean_fitness}, min={min(fitness_values)}, max={max(fitness_values)}"
            )
            print(
                "Batch success summary: "
                f"succeeded={succeeded}, failed={failed}, unknown={unknown}"
            )

        if args.save_config:
            assert individual is not None
            dpct_dir = Path("samples") / "configs" / "dpct"
            dpct_dir.mkdir(parents=True, exist_ok=True)
            if len(run_results) == 1:
                save_path = dpct_dir / f"{config_path.stem}.json"
            elif run_results:
                last_seed = run_results[-1][0]
                if last_seed is None:
                    save_path = dpct_dir / f"{config_path.stem}-last.json"
                else:
                    save_path = dpct_dir / f"{config_path.stem}-seed{last_seed}.json"
            else:
                save_path = dpct_dir / f"{config_path.stem}.json"
            individual.save_config(str(save_path))
            print(f"Config saved to {save_path}")

        if args.network_image:
            assert individual is not None
            image_path = Path(args.network_image)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            draw_traditional_pcnn(
                individual,
                filename=str(image_path),
                expand_units=True,
                show_weights=True,
                line_thickness=args.network_line_thickness,
                circle_label_fontsize=args.network_label_fontsize,
                weight_fontsize=args.network_weight_fontsize,
                smooth_factor_fontsize=args.network_smooth_factor_fontsize,
                bottom_label_fontsize=args.network_bottom_label_fontsize,
                legend_fontsize=args.network_legend_fontsize,
                legend_title_fontsize=args.network_legend_title_fontsize,
                show_unit_labels=args.network_unit_labels,
            )
            print(f"Network image saved to {image_path}")

        return 0
    finally:
        command_elapsed = time.perf_counter() - command_start
        print(f"Command complete. Elapsed seconds: {command_elapsed:.3f}", file=sys.stderr)



if __name__ == "__main__":
    raise SystemExit(main())
