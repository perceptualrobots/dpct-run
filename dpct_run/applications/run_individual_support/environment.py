"""Environment and logging setup helpers for dpct-run-individual."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _should_show_tf_warnings(argv: list[str]) -> bool:
    """Detect whether TensorFlow warning output should remain enabled."""
    return "--show-tf-warnings" in argv

def _configure_tensorflow_logging() -> None:
    """Reduce TensorFlow/absl warning noise for CLI runs."""
    if _should_show_tf_warnings(sys.argv[1:]):
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        # absl may be unavailable in minimal environments.
        pass

def _load_local_env_files() -> None:
    """Load KEY=VALUE pairs from local .env/.enc files into os.environ if unset."""
    candidate_paths = [Path(".env"), Path(".enc")]

    for env_path in candidate_paths:
        if not env_path.exists() or not env_path.is_file():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
