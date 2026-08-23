"""Mapping a real eval set's column names onto evallint's fields.

Why this exists: evallint expects ``id``, ``input``, ``expected`` and ``label``.
Five widely-used public eval datasets were tried cold, and **none** of them use
``input``. All five failed on line 1 of the first file:

    TruthfulQA   question / best_answer / category
    GSM8K        question / answer
    MMLU         question / answer / subject
    HellaSwag    ctx      / label  / activity_label
    MT-Bench     prompt   / reference / category

So the schema was not a small friction, it was a wall. This module turns "cannot
load" into "loads, and tells you exactly which columns it used".

THE RULE THAT MATTERS: never guess when guessing could be wrong.

A loader that silently picks the wrong column is worse than one that refuses,
because the run then produces a plausible report about the wrong data — the
exact failure mode this project exists to catch. So:

  * A column already named ``input``/``expected``/``label`` is used as-is.
    Canonical names are never re-mapped.
  * Otherwise aliases are considered. Exactly one candidate is accepted and
    recorded. Two or more candidates is an error naming them, not a coin flip.
  * The resolved mapping is always reported, including the alternatives that
    were passed over, so a wrong inference is visible rather than buried.

HellaSwag is the cautionary case. It HAS a column called ``label``, but there
it means the index of the correct ending, not a class. evallint cannot know
that, so it uses the canonical name and reports that ``activity_label`` was
also available — which is how a user catches it.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ALIASES",
    "MAPPABLE",
    "AmbiguousMappingError",
    "FieldMapping",
    "MappingError",
    "resolve_mapping",
]

# Ordered by how strongly the name implies the field. Sourced from the schemas
# of real datasets rather than invented: HuggingFace eval sets, promptfoo,
# OpenAI evals, and the five listed above.
ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("case_id", "example_id", "question_id", "prompt_id", "uid", "ind", "index"),
    "input": (
        "question",
        "prompt",
        "instruction",
        "query",
        "text",
        "input_text",
        "ctx",
        "context",
        "user_message",
    ),
    "expected": (
        "expected_answer",
        "best_answer",
        "reference",
        "reference_answer",
        "answer",
        "gold",
        "gold_answer",
        "target",
        "ideal",
        "output",
        "completion",
    ),
    "messages": (
        "conversation",
        "turns",
        "dialogue",
        "dialog",
        "chat",
        "chat_history",
        "conversations",
    ),
    "system": (
        "system_prompt",
        "system_message",
        "system_instruction",
    ),
    "label": (
        "category",
        "subject",
        "topic",
        "domain",
        "task",
        "task_type",
        "activity_label",
        "class",
        "group",
        "type",
    ),
}

#: Every field that can be mapped from a column. Kept in one place because
#: three loops below have to agree on it -- explain(), apply_mapping() and the
#: displaced-column pass. When they disagreed, a mapped value got silently
#: overwritten by the column that shared its name.
MAPPABLE = ("id", "input", "expected", "label", "messages", "system")

#: A case needs SOMETHING to send to a model. Either a text prompt or a
#: conversation will do, and a chat dataset has only the latter -- so requiring
#: `input` outright would refuse every multi-turn file, which is exactly the
#: wall this module exists to remove.
_REQUIRED_ANY_OF = ("input", "messages")


class MappingError(ValueError):
    """A column mapping could not be resolved."""


class AmbiguousMappingError(MappingError):
    """Several columns could plausibly fill one field, so evallint refuses."""


class FieldMapping:
    """The resolved column-name mapping, plus how each choice was made.

    ``explain()`` is not decoration: a user has to be able to see that
    ``expected`` came from ``best_answer`` and that ``incorrect_answers`` was
    ignored, or a wrong inference is undetectable.
    """

    def __init__(
        self,
        resolved: dict[str, str],
        *,
        inferred: dict[str, str] | None = None,
        alternatives: dict[str, tuple[str, ...]] | None = None,
        displaced: dict[str, str] | None = None,
    ) -> None:
        self.resolved = dict(resolved)
        self.inferred = dict(inferred or {})
        self.alternatives = dict(alternatives or {})
        # column name -> where it was moved to, for columns whose own name is a
        # canonical field that the mapping filled from somewhere else.
        self.displaced = dict(displaced or {})

    def column_for(self, field: str) -> str | None:
        return self.resolved.get(field)

    def is_identity(self) -> bool:
        """True when every field came from a column of the same name."""
        return not self.inferred

    def explain(self) -> list[str]:
        lines = []
        for field in MAPPABLE:
            column = self.resolved.get(field)
            if column is None:
                continue
            if field in self.inferred:
                note = f"{field} <- '{column}' (inferred)"
                also = self.alternatives.get(field, ())
                if also:
                    note += f"; also present: {', '.join(also)}"
            else:
                note = f"{field} <- '{column}'"
                also = self.alternatives.get(field, ())
                if also:
                    note += f"  [WARNING: '{column}' used as-is, but "
                    note += f"{', '.join(also)} also present — check the meaning]"
            lines.append(note)
        for column, moved_to in self.displaced.items():
            lines.append(
                f"column '{column}' was NOT used as {column} (the mapping took "
                f"it from elsewhere); kept as metadata '{moved_to}'"
            )
        return lines

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FieldMapping({self.resolved!r}, inferred={self.inferred!r})"


def resolve_mapping(
    columns: list[str] | tuple[str, ...],
    overrides: dict[str, str] | None = None,
) -> FieldMapping:
    """Work out which columns fill which fields.

    Args:
        columns: the column names present in the data.
        overrides: explicit ``field -> column``, which always wins.

    Raises:
        MappingError: an override names a column that is not present, or a
            required field could not be filled.
        AmbiguousMappingError: two or more aliases could fill one field.
    """
    present = list(columns)
    lower = {c.lower(): c for c in present}
    overrides = dict(overrides or {})

    unknown_targets = sorted(set(overrides) - set(ALIASES))
    if unknown_targets:
        raise MappingError(
            f"cannot map to unknown field(s) {', '.join(unknown_targets)}. "
            f"Mappable fields: {', '.join(sorted(ALIASES))}"
        )
    for field, column in overrides.items():
        if column not in present:
            raise MappingError(
                f"--map {field}={column}: no column named '{column}'. "
                f"Available: {', '.join(present)}"
            )

    resolved: dict[str, str] = {}
    inferred: dict[str, str] = {}
    alternatives: dict[str, tuple[str, ...]] = {}

    for field, aliases in ALIASES.items():
        if field in overrides:
            resolved[field] = overrides[field]
            continue

        # A column already carrying the canonical name is used as-is; renaming
        # it based on a guess would be a worse surprise than the reverse.
        if field in lower and lower[field] not in overrides.values():
            resolved[field] = lower[field]
            # Still surface plausible alternatives. HellaSwag's 'label' means
            # the correct-ending index while 'activity_label' is the real
            # class -- evallint cannot tell, so it must at least say so.
            others = tuple(lower[a] for a in aliases if a in lower)
            if others:
                alternatives[field] = others
            continue

        candidates = [lower[a] for a in aliases if a in lower]
        candidates = [c for c in candidates if c not in resolved.values()]
        candidates = [c for c in candidates if c not in overrides.values()]
        if not candidates:
            continue
        if len(candidates) > 1:
            raise AmbiguousMappingError(
                f"cannot tell which column is '{field}': {', '.join(candidates)} "
                f"are all plausible. Choose one explicitly, e.g. "
                f"--map {field}={candidates[0]}"
            )
        resolved[field] = candidates[0]
        inferred[field] = candidates[0]

    if not any(f in resolved for f in _REQUIRED_ANY_OF):
        raise MappingError(
            "no column could be used as 'input' (or 'messages' for a chat "
            f"dataset). Columns found: {', '.join(present)}. Map one "
            f"explicitly, e.g. --map input={present[0] if present else 'COLUMN'}"
        )

    # A column can be NAMED like a canonical field while the mapping fills that
    # field from a different column -- `--map label=activity_label` on HellaSwag,
    # which also has a `label` column meaning the correct-ending index. That
    # column must not be allowed to overwrite the mapped value, and it must not
    # be dropped either, so it is moved aside under a name that is reported.
    displaced: dict[str, str] = {}
    for field in MAPPABLE:
        # A distinct name: `column` is bound to a str earlier in this function,
        # so reusing it here hides the Optional from a type checker.
        named = lower.get(field)
        if named is None or resolved.get(field) == named:
            continue
        moved_to = f"unmapped_{named}"
        while moved_to in present or moved_to in displaced.values():
            moved_to = f"{moved_to}_"
        displaced[named] = moved_to

    return FieldMapping(
        resolved,
        inferred=inferred,
        alternatives=alternatives,
        displaced=displaced,
    )


def apply_mapping(record: dict[str, Any], mapping: FieldMapping) -> dict[str, Any]:
    """Rewrite one record's keys into evallint's field names.

    Columns that are not mapped are kept under their original names, so a
    user's own bookkeeping survives the round trip exactly as before.
    """
    out: dict[str, Any] = {}
    taken = set()
    for field in MAPPABLE:
        column = mapping.column_for(field)
        if column is not None and column in record:
            out[field] = record[column]
            taken.add(column)
    for key, value in record.items():
        if key in taken:
            continue
        # Without this, a leftover column called 'label' would land on the
        # 'label' key and silently replace the value the mapping just put
        # there -- a wrong answer with no error, which is the worst outcome
        # available. mapping.displaced says where it goes instead.
        out[mapping.displaced.get(key, key)] = value
    return out
