"""Runtime environment capture for manifests and environment.json."""

from __future__ import annotations

import platform
import sys
from importlib import metadata as _metadata


def _package_version(name: str) -> str:
    try:
        return _metadata.version(name)
    except Exception:
        return "unknown"


def collect_environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit-learn": _package_version("scikit-learn"),
        "pyyaml": _package_version("PyYAML"),
        "pytest": _package_version("pytest"),
        # Logical interpreter path inside the project; recorded for
        # provenance without embedding machine-specific absolute paths.
        "executable": "venv/bin/python3",
    }
