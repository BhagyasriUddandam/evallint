"""Minimal .env loader for the demo scripts.

NOT part of the shipped library.

Hand-rolled rather than pulling in python-dotenv: this reads one file with a
handful of KEY=VALUE lines, and a dependency that exists to avoid fifteen
lines is a dependency that has to be installed, locked, and explained.

Two rules that matter:
  - An already-set environment variable ALWAYS wins. A key you exported
    deliberately must not be silently replaced by a stale one in a file.
  - Values are never printed or returned — only key NAMES — so nothing here
    can leak a secret into a log, a traceback, or a terminal recording.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DEFAULT_ENV_PATH", "load_env_file"]

DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_env_file(path: Path | None = None) -> list[str]:
    """Load KEY=VALUE pairs from ``path`` into os.environ.

    Returns the names of the keys it set, in file order — names only, never
    values. Missing file is not an error: the environment may already be
    configured, which is the normal case in CI.
    """
    path = DEFAULT_ENV_PATH if path is None else path
    if not path.is_file():
        return []

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        # `export FOO=bar` is a common shape for a file people also source.
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue

        value = value.strip()
        # Strip one matching pair of surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        # Explicitly exported values win over the file.
        if name in os.environ:
            continue

        os.environ[name] = value
        loaded.append(name)

    return loaded
