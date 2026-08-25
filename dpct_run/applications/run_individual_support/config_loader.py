"""Config and Comet loading helpers for dpct-run-individual."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from dpct_run.individual import DHPCTIndividual


def _is_comet_experiment_url(source: str) -> bool:
    """Return True when source looks like a Comet experiment URL."""
    parsed = urlparse(source)
    host = parsed.netloc.lower()
    return parsed.scheme in {"http", "https"} and host in {"comet.com", "www.comet.com"}

def _parse_comet_experiment_url(url: str) -> tuple[str, str, str]:
    """Extract workspace, project name, and experiment key from a Comet URL."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if parsed.scheme not in {"http", "https"} or host not in {"comet.com", "www.comet.com"}:
        raise ValueError("Comet experiment URL must use https://www.comet.com/<workspace>/<project>/<experiment-key>")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3:
        raise ValueError("Comet experiment URL must include workspace, project, and experiment key")
    workspace, project_name, experiment_key = parts[:3]
    return workspace, project_name, experiment_key

def _comet_best_individual_cache_path(url: str, *, cache_root: Path = Path("/tmp/comet")) -> Path:
    """Return /tmp/comet/<workspace>/<project>/<experiment-key>/best_individual.json."""
    workspace, project_name, experiment_key = _parse_comet_experiment_url(url)
    return cache_root / workspace / project_name / experiment_key / "best_individual.json"

def _download_comet_best_individual(url: str, output_path: Path) -> None:
    """Download best_individual.json from a Comet experiment URL."""
    workspace, project_name, experiment_key = _parse_comet_experiment_url(url)
    try:
        from comet_ml import API
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("comet_ml is required to download best_individual.json from Comet") from exc

    api = API()
    experiment = api.get_experiment(workspace, project_name, experiment_key)
    if experiment is None:
        raise RuntimeError(
            f"Comet experiment not found: workspace={workspace}, "
            f"project={project_name}, experiment={experiment_key}"
        )

    if hasattr(experiment, "get_assets_by_name"):
        assets = experiment.get_assets_by_name("best_individual.json", return_type="binary")
        asset = assets[0] if assets else None
    else:  # pragma: no cover - compatibility with older comet_ml
        asset = experiment.get_asset_by_name("best_individual.json", return_type="binary")
    if asset is None:
        available = []
        try:
            available = [str(item.get("fileName")) for item in experiment.get_asset_list() if item.get("fileName")]
        except Exception:
            available = []
        suffix = f" Available assets: {', '.join(sorted(available))}" if available else ""
        raise RuntimeError(f"best_individual.json asset not found in Comet experiment.{suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(asset, str):
        output_path.write_text(asset, encoding="utf-8")
    elif isinstance(asset, bytes):
        output_path.write_bytes(asset)
    else:
        output_path.write_text(json.dumps(asset), encoding="utf-8")

def _resolve_config_source(source: str) -> Path:
    """Resolve a local config file or Comet URL to a local config path."""
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

def _extract_legacy_config_dict(raw_text: str) -> dict:
    """Extract and parse the `config = {...}` dict from a legacy .properties file."""
    marker = "config"
    idx = raw_text.find(marker)
    while idx != -1:
        # Ensure this is a top-level assignment like: config = {...}
        eq_idx = raw_text.find("=", idx + len(marker))
        if eq_idx == -1:
            break

        name = raw_text[idx:eq_idx].strip()
        if name != "config":
            idx = raw_text.find(marker, idx + len(marker))
            continue

        brace_start = raw_text.find("{", eq_idx)
        if brace_start == -1:
            break

        depth = 0
        in_string = False
        quote_char = ""
        escaped = False

        for i in range(brace_start, len(raw_text)):
            ch = raw_text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote_char:
                    in_string = False
                continue

            if ch in ("'", '"'):
                in_string = True
                quote_char = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    literal = raw_text[brace_start:i + 1]
                    try:
                        parsed = ast.literal_eval(literal)
                    except Exception as exc:
                        raise ValueError("Failed to parse legacy 'config = {...}' block") from exc
                    if not isinstance(parsed, dict):
                        raise ValueError("Legacy 'config = {...}' block is not a dictionary")
                    return parsed

        break

    raise ValueError("Could not find a valid legacy 'config = {...}' block")

def _individual_from_config_file(config_path: Path) -> DHPCTIndividual:
    config = None

    if config_path.suffix.lower() == ".properties":
        return DHPCTIndividual.load_from_legacy_config(str(config_path))
    elif config_path.suffix.lower() in {".txt", ".json"}:
        with open(config_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        try:
            config = json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                config = ast.literal_eval(raw_text)
            except Exception as exc:
                raise ValueError(
                    "Could not parse config file as JSON or legacy Python-dict text format"
                ) from exc
    else:
        config = DHPCTIndividual.load_config(str(config_path))

    if isinstance(config, dict) and "env_name" in config and "hierarchy" in config:
        return DHPCTIndividual.from_config(config)

    if isinstance(config, dict) and config.get("type") in {"PCTHierarchy", "Individual"}:
        return DHPCTIndividual.from_legacy_config(config)

    raise ValueError(
        "Unsupported config format. Expected DPCT config with 'env_name' and 'hierarchy' "
        "or legacy config with type 'PCTHierarchy'/'Individual'."
    )


class IndividualConfigLoader:
    """Resolve config sources and build ``DHPCTIndividual`` instances."""

    resolve_source = staticmethod(_resolve_config_source)
    load_individual = staticmethod(_individual_from_config_file)
    comet_best_individual_cache_path = staticmethod(_comet_best_individual_cache_path)
