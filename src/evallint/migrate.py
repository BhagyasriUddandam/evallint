"""Converting an existing eval set to the declared schema-2 format.

## What migration is for, given that it is never required

A version-1 file is valid forever. Nothing in this package will ever refuse
one, and `evallint old.jsonl` will keep working. So migration exists for two
concrete reasons, both about the file rather than the tool:

  1. **It makes the column mapping permanent.** A GSM8K file needs
     ``--map input=question`` on every single run, and an inferred mapping is
     re-inferred (and re-reported) every time. Migrating writes the canonical
     names once, and the flag goes away.
  2. **It declares the version**, which turns the lenient demotions described
     in :mod:`evallint.validation` into errors. A file you intend to maintain
     is better off failing loudly on a typo than quietly keeping it as
     metadata.

## What migration deliberately does NOT do

It does not turn ``input`` into a one-turn ``messages`` list. That conversion
would be lossless -- a single user turn flattens back to exactly the same text --
but it would make a simple file harder to read in exchange for nothing, and this
schema's whole claim is that the four-field form stays first-class. Converting a
single-prompt case into a conversation is a decision about the eval, so it is
left to whoever is adding the second turn. `docs/schema-v2.md` shows how.

It also does not touch the data. Field names change, the version line is added,
and nothing else: no reordering, no normalising, no dropping of unrecognised
columns.

## The output is verified before it is written

Migration reloads what it produced and compares every case, field by field,
against the input. If anything differs, it raises and writes nothing. A
migration tool that silently corrupts one case in a thousand is worse than no
migration tool, and the check costs one extra parse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import VERSION_KEY, LoadError, load_with_mapping
from .schema import SCHEMA_VERSION, EvalCase, EvalSet

__all__ = ["MigrationError", "MigrationReport", "migrate_file", "to_payload"]

_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
#: Fields that only exist in schema 2, for reporting what the file gained.
_V2_FIELDS = (
    "messages",
    "system",
    "acceptable",
    "expected_schema",
    "rubric",
    "tools",
    "expected_tool_calls",
)


class MigrationError(RuntimeError):
    """The migration could not be completed safely, so nothing was written."""


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """What a migration changed. Everything here is a fact about the two files."""

    source: str
    dest: str
    cases: int
    renamed: dict[str, str]
    chat_cases: int
    v2_fields_used: tuple[str, ...]
    load_notes: tuple[str, ...]
    verified: bool

    def render(self) -> str:
        lines = [
            f"{self.source} -> {self.dest}",
            f"  {self.cases} case(s), schema version {SCHEMA_VERSION} declared",
        ]
        if self.renamed:
            lines.append("  columns renamed to canonical fields:")
            lines.extend(
                f"    {column} -> {field}" for field, column in self.renamed.items()
            )
            lines.append(
                "    (so --map is no longer needed for this file)"
            )
        else:
            lines.append("  no columns needed renaming")
        if self.chat_cases:
            lines.append(f"  {self.chat_cases} case(s) carry a conversation")
        if self.v2_fields_used:
            lines.append(
                "  schema-2 fields present: " + ", ".join(self.v2_fields_used)
            )
        for note in self.load_notes:
            lines.append(f"  note: {note}")
        lines.append(
            "  verified: the written file reloads to identical cases"
            if self.verified
            else "  NOT VERIFIED"
        )
        return "\n".join(lines)


def to_payload(eval_set: EvalSet) -> dict[str, Any]:
    """The full versioned envelope for a set, ready to serialise."""
    return {
        VERSION_KEY: SCHEMA_VERSION,
        "cases": [case.as_dict() for case in eval_set],
    }


def migrate_file(
    source: str | Path,
    dest: str | Path,
    *,
    field_map: dict[str, str] | None = None,
    overwrite: bool = False,
) -> MigrationReport:
    """Read ``source``, write the declared schema-2 form to ``dest``.

    The output format follows ``dest``'s extension: a ``.json`` file gets the
    ``{"evallint_schema": 2, "cases": [...]}`` envelope, a ``.jsonl`` file gets
    a version header line followed by one case per line. CSV is not a valid
    destination -- it cannot hold a nested conversation or a version, and
    writing one would silently drop both.

    Raises:
        MigrationError: the destination exists (and ``overwrite`` is False), the
            extension is unsupported, or the written file did not reload to
            identical cases.
    """
    source, dest = Path(source), Path(dest)
    suffix = dest.suffix.lower()
    if suffix not in _JSONL_SUFFIXES and suffix != ".json":
        raise MigrationError(
            f"{dest.name}: cannot migrate to '{suffix or dest.name}'. The "
            "schema-2 format needs nesting and a version declaration, which "
            "CSV cannot hold; use .json or .jsonl."
        )
    if dest.resolve() == source.resolve():
        raise MigrationError(
            f"{dest.name}: destination is the source file. Migration writes a "
            "new file so the original stays intact if anything goes wrong."
        )
    if dest.exists() and not overwrite:
        raise MigrationError(
            f"{dest.name}: already exists. Pass overwrite to replace it."
        )

    try:
        eval_set, mapping = load_with_mapping(source, field_map)
    except LoadError as exc:
        raise MigrationError(str(exc)) from exc

    text = _serialise(eval_set, suffix)

    # Verify BEFORE writing to the real destination. Writing first and checking
    # afterwards would leave a corrupt file behind on failure.
    _verify(eval_set, text, suffix, dest)

    dest.write_text(text, encoding="utf-8")

    renamed = {
        field: column
        for field, column in mapping.resolved.items()
        if field != column
    }
    used = tuple(
        name
        for name in _V2_FIELDS
        if any(name in case.as_dict() for case in eval_set)
    )
    return MigrationReport(
        source=str(source),
        dest=str(dest),
        cases=len(eval_set),
        renamed=renamed,
        chat_cases=eval_set.chat_cases,
        v2_fields_used=used,
        load_notes=eval_set.load_notes,
        verified=True,
    )


def _serialise(eval_set: EvalSet, suffix: str) -> str:
    payload = to_payload(eval_set)
    if suffix == ".json":
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    lines = [json.dumps({VERSION_KEY: SCHEMA_VERSION}, ensure_ascii=False)]
    lines.extend(
        json.dumps(case, ensure_ascii=False) for case in payload["cases"]
    )
    return "\n".join(lines) + "\n"


def _verify(eval_set: EvalSet, text: str, suffix: str, dest: Path) -> None:
    """Reload the serialised text and compare every case to the original."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / f"verify{suffix}"
        probe.write_text(text, encoding="utf-8")
        try:
            reloaded, _ = load_with_mapping(probe)
        except LoadError as exc:
            raise MigrationError(
                f"{dest.name} NOT written: the migrated form does not load "
                f"back ({exc}). This is a bug in evallint, not in your file — "
                "please report it with the input that caused it."
            ) from exc

    if len(reloaded) != len(eval_set):
        raise MigrationError(
            f"{dest.name} NOT written: {len(eval_set)} cases in, "
            f"{len(reloaded)} out."
        )
    for original, copy in zip(eval_set, reloaded, strict=True):
        difference = _first_difference(original, copy)
        if difference is not None:
            raise MigrationError(
                f"{dest.name} NOT written: case '{original.id}' changed during "
                f"migration ({difference}). This is a bug in evallint, not in "
                "your file — please report it with the input that caused it."
            )


def _first_difference(a: EvalCase, b: EvalCase) -> str | None:
    """Name the first field that differs, or None. Reports the field, not just
    a boolean, because "something changed" is not actionable."""
    for field in (
        "id",
        "input",
        "expected",
        "label",
        "metadata",
        "messages",
        "system",
        "acceptable",
        "expected_schema",
        "rubric",
        "tools",
        "expected_tool_calls",
    ):
        before, after = getattr(a, field), getattr(b, field)
        if before != after:
            return f"{field}: {before!r} became {after!r}"
    return None
