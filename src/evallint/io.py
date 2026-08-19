"""Loading eval sets from disk. Pure I/O — no analysis lives here.

Supported formats, chosen by file extension:
    .jsonl / .ndjson  one JSON object per line
    .json             a list of objects, or {"cases": [...]}
    .csv              a header row plus one row per case

Canonical fields are ``id``, ``input``, ``expected`` and ``label``. Any other
column or key is preserved in ``EvalCase.metadata`` rather than dropped, so a
user's own bookkeeping survives a round trip through the tool.

Errors are raised as LoadError with the file and location baked into the
message ("sample.jsonl line 4: ..."), because a bad row in a 5000-line eval set
is useless to report without saying which row.

Every reader uses ``utf-8-sig``, not ``utf-8``. Excel, PowerShell and several
Windows tools write a UTF-8 BOM, and a BOM is the worst kind of input problem
here because it does not fail: read as plain utf-8, a BOM'd CSV parses with its
first column named "\ufeffid" instead of "id", so the real ids are silently
demoted to metadata and every case is renamed case_1, case_2, ... Findings then
point at case ids that do not exist in the user's file. ``utf-8-sig`` strips a
BOM when present and is a no-op when it is not.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from .schema import EvalCase, EvalSet, SchemaError

__all__ = ["LoadError", "load"]

log = logging.getLogger(__name__)

CANONICAL_FIELDS = ("id", "input", "expected", "label")
_JSONL_SUFFIXES = {".jsonl", ".ndjson"}


class LoadError(ValueError):
    """A file could not be read, parsed, or turned into a valid EvalSet."""


def load(path: str | Path) -> EvalSet:
    """Read an eval set from disk.

    Args:
        path: Path to a .jsonl, .ndjson, .json or .csv file.

    Returns:
        A validated EvalSet whose ``source`` is the path it was read from.

    Raises:
        LoadError: The file is missing, has an unsupported extension, is not
            parseable, or contains a case that violates the schema.
    """
    started = time.perf_counter()
    path = Path(path)
    if not path.is_file():
        raise LoadError(f"{path}: no such file")

    suffix = path.suffix.lower()
    if suffix in _JSONL_SUFFIXES:
        records = _read_jsonl(path)
    elif suffix == ".json":
        records = _read_json(path)
    elif suffix == ".csv":
        records = _read_csv(path)
    else:
        raise LoadError(
            f"{path.name}: unsupported file type '{suffix or path.name}' "
            "(expected .jsonl, .ndjson, .json or .csv)"
        )

    cases = [
        _to_case(record, where, position)
        for position, (where, record) in enumerate(records, start=1)
    ]

    try:
        eval_set = EvalSet(cases=tuple(cases), source=str(path))
    except SchemaError as exc:
        raise LoadError(f"{path.name}: {exc}") from exc

    log.info(
        "loaded %d cases from %s (%s) in %.3fs",
        len(eval_set), path, suffix.lstrip(".") or "?", time.perf_counter() - started,
    )
    return eval_set


def _read_jsonl(path: Path) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []
    text = path.read_text(encoding="utf-8-sig")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue  # blank lines are common padding, not an error
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LoadError(
                f"{path.name} line {lineno}: invalid JSON ({exc.msg})"
            ) from exc
        records.append((f"{path.name} line {lineno}", obj))
    return records


def _read_json(path: Path) -> list[tuple[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise LoadError(
            f"{path.name}: invalid JSON ({exc.msg} at line {exc.lineno})"
        ) from exc

    if isinstance(payload, dict):
        # Both shapes are common in the wild, so accept both rather than make
        # the user rewrite their file.
        if "cases" not in payload:
            raise LoadError(
                f"{path.name}: a JSON object must have a 'cases' key holding "
                "the list of cases"
            )
        payload = payload["cases"]

    if not isinstance(payload, list):
        raise LoadError(
            f"{path.name}: expected a list of cases, or an object with a "
            f"'cases' list, got {type(payload).__name__}"
        )
    return [(f"{path.name} item {i}", obj) for i, obj in enumerate(payload, start=1)]


def _read_csv(path: Path) -> list[tuple[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LoadError(f"{path.name}: file is empty (no header row)")

        records: list[tuple[str, Any]] = []
        for row_number, row in enumerate(reader, start=1):
            # CSV has no null, so an empty cell means "absent". Dropping the
            # key here is what makes a blank optional column behave the same
            # as a missing JSON key. It is also why CSV is slightly lossy:
            # every value that survives is a string.
            record = {
                key: value
                for key, value in row.items()
                if key is not None and value is not None and value.strip() != ""
            }
            records.append((f"{path.name} row {row_number}", record))
    return records


def _to_case(record: Any, where: str, position: int) -> EvalCase:
    if not isinstance(record, dict):
        raise LoadError(
            f"{where}: expected an object, got {type(record).__name__}"
        )

    data = dict(record)  # copy: we pop canonical fields and keep the remainder
    raw_id = data.pop("id", None)
    case_id = f"case_{position}" if raw_id is None else str(raw_id)

    if "input" not in data:
        raise LoadError(f"{where}: missing required field 'input'")

    try:
        return EvalCase(
            id=case_id,
            input=data.pop("input"),
            expected=data.pop("expected", None),
            label=data.pop("label", None),
            metadata=data,
        )
    except SchemaError as exc:
        raise LoadError(f"{where}: {exc}") from exc
