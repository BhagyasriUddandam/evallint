"""Loading eval sets from disk. Pure I/O — no analysis lives here.

Supported formats, chosen by file extension:
    .jsonl / .ndjson  one JSON object per line
    .json             a list of objects, or {"cases": [...]}
    .csv              a header row plus one row per case

Canonical fields are ``id``, ``input``, ``expected`` and ``label``, plus the
schema-2 additions ``messages``, ``system``, ``acceptable``, ``expected_schema``,
``rubric``, ``tools`` and ``expected_tool_calls``. Any other column or key is
preserved in ``EvalCase.metadata`` rather than dropped, so a user's own
bookkeeping survives a round trip through the tool.

## Declaring a schema version is optional, and only changes strictness

    .json    {"evallint_schema": 2, "cases": [...]}
    .jsonl   a first line of {"evallint_schema": 2}, then one case per line
    .csv     cannot declare one; CSV is always read leniently

A file that declares nothing is read as version 1: rich fields are still parsed
when they are well-formed, but a ``messages`` value that is not a conversation
is demoted to metadata with a note rather than rejected. That is what keeps
existing datasets loading -- including one that happens to have a column called
``messages`` meaning something else entirely. Declaring version 2 turns those
demotions into errors. See :mod:`evallint.validation`.

The version is never INFERRED from the fields present. A v1 file read under v2
rules would fail on data that was always legal, and guessing here is the one
thing this loader must not do.

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

from .mapping import FieldMapping, apply_mapping, resolve_mapping
from .schema import EvalSet, SchemaError
from .validation import CANONICAL_FIELDS, check_version, parse_cases

__all__ = ["CANONICAL_FIELDS", "LoadError", "VERSION_KEY", "load", "load_with_mapping"]

log = logging.getLogger(__name__)

#: The key a file uses to declare its schema version.
VERSION_KEY = "evallint_schema"

_JSONL_SUFFIXES = {".jsonl", ".ndjson"}


class LoadError(ValueError):
    """A file could not be read, parsed, or turned into a valid EvalSet."""


def load(path: str | Path, field_map: dict[str, str] | None = None) -> EvalSet:
    """Read an eval set from disk.

    Args:
        path: Path to a .jsonl, .ndjson, .json or .csv file.
        field_map: Optional explicit ``field -> column`` mapping, e.g.
            ``{"input": "question", "label": "subject"}``. When omitted,
            common aliases are inferred -- see :mod:`evallint.mapping`.

    Returns:
        A validated EvalSet whose ``source`` is the path it was read from.

    Raises:
        LoadError: The file is missing, has an unsupported extension, is not
            parseable, or contains a case that violates the schema.
    """
    return load_with_mapping(path, field_map)[0]


def load_with_mapping(
    path: str | Path, field_map: dict[str, str] | None = None
) -> tuple[EvalSet, FieldMapping]:
    """Like :func:`load`, but also returns how columns were mapped.

    The CLI uses this so it can print the mapping. A silently wrong inference
    is the one failure mode this feature must not have, and the only defence
    is showing the user which columns were actually used.
    """
    started = time.perf_counter()
    path = Path(path)
    if not path.is_file():
        raise LoadError(f"{path}: no such file")

    suffix = path.suffix.lower()
    if suffix in _JSONL_SUFFIXES:
        records, declared = _read_jsonl(path)
    elif suffix == ".json":
        records, declared = _read_json(path)
    elif suffix == ".csv":
        records, declared = _read_csv(path), None
    else:
        raise LoadError(
            f"{path.name}: unsupported file type '{suffix or path.name}' "
            "(expected .jsonl, .ndjson, .json or .csv)"
        )

    # An unreadable version is refused before anything else: parsing 5000
    # records under the wrong rules and then complaining would waste the work
    # and could produce issues that are artefacts of the wrong version.
    # Wrapped in LoadError like every other load failure. Left unwrapped, a
    # SchemaValidationError escaped past the CLI's `except LoadError`, so a
    # single mistyped version line produced a traceback instead of "Error: ...".
    try:
        version = 1 if declared is None else check_version(declared, source=path.name)
    except SchemaError as exc:
        raise LoadError(str(exc)) from exc

    # Column names come from the first record; JSONL in the wild is ragged, so
    # union the first few rather than trusting row 1 to have every field.
    sample: list[str] = []
    for _, record in records[:25]:
        if isinstance(record, dict):
            for key in record:
                if key not in sample:
                    sample.append(key)

    if sample or field_map:
        try:
            mapping = resolve_mapping(sample, field_map)
        except ValueError as exc:
            raise LoadError(f"{path.name}: {exc}") from exc
    else:
        # No columns to map means no records, or no records that are objects.
        # Both already have their own precise error further down; a mapping
        # complaint here ("no column could be used as 'input'. Columns found: ")
        # would just be a worse-worded version of it.
        mapping = FieldMapping({})

    mapped = [
        (where, apply_mapping(record, mapping) if isinstance(record, dict) else record)
        for where, record in records
    ]

    try:
        parsed = parse_cases(mapped, strict=version >= 2, source=path.name)
    except SchemaError as exc:
        # Already carries the file name and every issue's location.
        raise LoadError(str(exc)) from exc

    try:
        eval_set = EvalSet(
            cases=tuple(parsed.cases),
            source=str(path),
            schema_version=version,
            load_notes=tuple(parsed.notes),
        )
    except SchemaError as exc:
        raise LoadError(f"{path.name}: {exc}") from exc

    log.info(
        "loaded %d cases from %s (%s, schema %d, %d chat) in %.3fs",
        len(eval_set), path, suffix.lstrip(".") or "?", version,
        eval_set.chat_cases, time.perf_counter() - started,
    )
    for note in eval_set.load_notes:
        log.warning("%s", note)
    for line in mapping.explain():
        log.debug("field map: %s", line)
    return eval_set, mapping


def _read_jsonl(path: Path) -> tuple[list[tuple[str, Any]], Any]:
    records: list[tuple[str, Any]] = []
    declared: Any = None
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
        if _is_header(obj, first=not records and declared is None):
            # JSONL has no envelope, so the version goes on an optional first
            # line. Recognised only when the object declares a version and
            # carries nothing that could make it a case -- otherwise a case
            # with an `evallint_schema` field of its own would vanish.
            declared = obj[VERSION_KEY]
            continue
        records.append((f"{path.name} line {lineno}", obj))
    return records, declared


def _is_header(obj: Any, *, first: bool) -> bool:
    return (
        first
        and isinstance(obj, dict)
        and VERSION_KEY in obj
        and not ({"id", "input", "messages"} & set(obj))
    )


def _read_json(path: Path) -> tuple[list[tuple[str, Any]], Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise LoadError(
            f"{path.name}: invalid JSON ({exc.msg} at line {exc.lineno})"
        ) from exc

    declared: Any = None
    if isinstance(payload, dict):
        # Both shapes are common in the wild, so accept both rather than make
        # the user rewrite their file.
        declared = payload.get(VERSION_KEY)
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
    return (
        [(f"{path.name} item {i}", obj) for i, obj in enumerate(payload, start=1)],
        declared,
    )


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
