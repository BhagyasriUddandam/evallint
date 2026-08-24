"""Config-file discovery, so thresholds can live in version control.

Flags on a command line are fine for one person at a terminal. A team wants the
thresholds reviewed in a pull request like any other decision, and a CI job
should not encode them in a YAML step where nobody looking at the repo can find
them. So evallint reads them from a file, and the CLI flags override it.

Two locations, checked in this order from the starting directory upward:

    evallint.toml       ->  top-level keys
    pyproject.toml      ->  [tool.evallint]

`evallint.toml` wins if both exist in the same directory. Searching upward means
a config at the repository root applies to an eval set in a subdirectory, which
is how every other Python tool behaves.

Nothing here has a default for any setting: the defaults live with the CLI
options and the check constructors, in one place. This module only reports what
a file asked for.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

__all__ = ["ConfigError", "KNOWN_KEYS", "find_config", "load_config"]

log = logging.getLogger(__name__)

# Every key a config file may set, mapped to the type it must be. Anything else
# is rejected rather than ignored: a silently-ignored `min_class_shrae` typo
# means a team believes a threshold is in force when it is not, which is the
# same class of quiet wrongness this tool exists to report.
KNOWN_KEYS: dict[str, type | tuple[type, ...]] = {
    "skip_duplicates": bool,
    "duplicate_threshold": (int, float),
    "min_class_share": (int, float),
    "min_class_count": int,
    "fail_on": str,
    # A nested table, not a scalar. The coverage spec is a DECLARATION about
    # what your eval is meant to span, so it belongs in version control and gets
    # reviewed like code -- there is no sensible command-line form for it, and
    # inventing one would encourage typing the reference by hand each run.
    "coverage": dict,
}

_FAIL_ON_VALUES = ("never", "warning", "any")


class ConfigError(ValueError):
    """A config file was found but could not be used."""


def find_config(start: Path | None = None) -> Path | None:
    """Return the nearest config file at or above ``start``, or None."""
    directory = (start or Path.cwd()).resolve()
    if directory.is_file():
        directory = directory.parent

    for candidate_dir in (directory, *directory.parents):
        own = candidate_dir / "evallint.toml"
        if own.is_file():
            return own
        pyproject = candidate_dir / "pyproject.toml"
        if pyproject.is_file():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
            except tomllib.TOMLDecodeError:
                # A pyproject.toml we cannot parse is not necessarily ours to
                # complain about — keep looking rather than failing the run.
                log.debug("skipping unparseable %s", pyproject)
                continue
            if isinstance(data.get("tool", {}).get("evallint"), dict):
                return pyproject
    return None


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate settings from ``path``.

    Raises:
        ConfigError: the file is unparseable, holds an unknown key, or holds a
            value of the wrong type. Loudly, because a config a user wrote and
            evallint quietly disregarded is worse than no config at all.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML ({exc})") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: could not be read ({exc})") from exc

    if path.name == "pyproject.toml":
        section = data.get("tool", {}).get("evallint", {})
        where = f"{path} [tool.evallint]"
    else:
        section = data
        where = str(path)

    if not isinstance(section, dict):
        raise ConfigError(f"{where}: expected a table of settings")

    unknown = sorted(set(section) - set(KNOWN_KEYS))
    if unknown:
        raise ConfigError(
            f"{where}: unknown setting(s) {', '.join(unknown)}. "
            f"Valid settings: {', '.join(sorted(KNOWN_KEYS))}"
        )

    for key, value in section.items():
        expected = KNOWN_KEYS[key]
        # bool is a subclass of int in Python, so an explicit guard is needed or
        # `min_class_count = true` would sail through as the integer 1.
        if isinstance(value, bool) is not (expected is bool):
            raise ConfigError(
                f"{where}: {key} must be {_describe(expected)}, got "
                f"{type(value).__name__}"
            )
        if not isinstance(value, expected):
            raise ConfigError(
                f"{where}: {key} must be {_describe(expected)}, got "
                f"{type(value).__name__}"
            )

    coverage = section.get("coverage")
    if coverage is not None:
        _validate_coverage(coverage, where)

    fail_on = section.get("fail_on")
    if fail_on is not None and fail_on not in _FAIL_ON_VALUES:
        raise ConfigError(
            f"{where}: fail_on must be one of {', '.join(_FAIL_ON_VALUES)}, "
            f"got {fail_on!r}"
        )

    log.debug("loaded %d setting(s) from %s", len(section), where)
    return dict(section)


#: Keys allowed inside [coverage]. Same reasoning as KNOWN_KEYS: a typo that is
#: silently ignored means a team believes a coverage space is declared when it
#: is not, and the resulting occupancy figure is about a different space.
_COVERAGE_KEYS: dict[str, type | tuple[type, ...]] = {
    "dimensions": dict,
    "reference": dict,
    "impossible": list,
    "min_cell": int,
    "max_divergence": (int, float),
}


def _validate_coverage(section: Any, where: str) -> None:
    """Shape-check [coverage] before the check builds a spec from it.

    Only shape. The spec's own `__post_init__` enforces the semantics -- an axis
    with one level, a typo in `impossible`, a reference naming a cell the
    dimensions cannot produce -- because those rules must hold for a
    hand-constructed spec too.
    """
    if not isinstance(section, dict):
        raise ConfigError(f"{where}: [coverage] must be a table")

    unknown = sorted(set(section) - set(_COVERAGE_KEYS))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) in [coverage]: {', '.join(unknown)}. "
            f"Valid: {', '.join(sorted(_COVERAGE_KEYS))}"
        )
    for key, value in section.items():
        expected = _COVERAGE_KEYS[key]
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ConfigError(
                f"{where}: coverage.{key} must be {_describe(expected)}, got "
                f"{type(value).__name__}"
            )

    dimensions = section.get("dimensions")
    if not dimensions:
        raise ConfigError(
            f"{where}: [coverage] needs a 'dimensions' table. Coverage of an "
            "unstated space is not a measurable quantity, so there is no "
            "default and no inferred spec."
        )
    for name, values in dimensions.items():
        if not isinstance(values, list) or not all(
            isinstance(v, str) for v in values
        ):
            raise ConfigError(
                f"{where}: coverage.dimensions.{name} must be a list of "
                f"strings naming the levels that should be present"
            )


def _describe(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__
