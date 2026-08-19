"""The internal representation of an eval set.

Everything upstream (io.py) converts a file into these two types; everything
downstream (the checks) reads only these two types. That boundary is the whole
point of this module: a check should never know or care whether the data came
from CSV or JSONL.

Validation lives here rather than in io.py so that a hand-constructed EvalCase
is held to exactly the same rules as one loaded from disk.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EvalCase", "EvalSet", "SchemaError"]


class SchemaError(ValueError):
    """A case or set violates the schema.

    Subclasses ValueError because that is what a caller would naturally catch
    for "you gave me bad data".
    """


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One test case in an eval set.

    Only ``id`` and ``input`` are required. Everything else is optional because
    real eval sets in the wild are ragged, and a check that needs a field it
    does not have should say so rather than refuse to load the file.

    Attributes:
        id: Unique identifier within the set. Reports refer to cases by this.
        input: The prompt / question given to the model. This is what the
            duplicate check embeds.
        expected: Ground truth. Deliberately untyped: it can be a string, a
            bool, a number, or a list, and none of the v1 checks interpret it.
        label: The case's category / class. This is what the imbalance check
            counts. Non-string scalars are normalised to str (see below).
        metadata: Any fields the source file had that are not canonical. Kept
            rather than dropped so nothing is silently lost on load.
    """

    id: str
    input: str
    expected: Any = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.id, "id")
        _require_nonempty_str(self.input, "input")

        if not isinstance(self.metadata, dict):
            raise SchemaError(
                f"metadata must be a dict, got {type(self.metadata).__name__}"
            )

        if self.label is not None:
            # A label is a category name, so it must be a scalar we can group
            # and print. We normalise 1 and "1" to the same class on purpose:
            # a CSV writes labels as text and a JSON file may write them as
            # numbers, and those should not count as two different classes.
            if isinstance(self.label, (str, int, float, bool)):
                normalised = str(self.label).strip()
                if not normalised:
                    raise SchemaError(
                        f"case '{self.id}': label must be non-empty when present"
                    )
                # frozen dataclasses block plain assignment, so write through
                # object.__setattr__ — the standard idiom for normalising a
                # field in __post_init__.
                object.__setattr__(self, "label", normalised)
            else:
                raise SchemaError(
                    f"case '{self.id}': label must be a string or number, "
                    f"got {type(self.label).__name__}"
                )


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A collection of eval cases, plus where it came from.

    ``cases`` is stored as a tuple: an eval set is a fixed snapshot being
    audited, and no check should ever mutate the thing it is measuring.
    """

    cases: tuple[EvalCase, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cases, (str, bytes)) or not isinstance(
            self.cases, Sequence
        ):
            raise SchemaError("cases must be a sequence of EvalCase objects")

        object.__setattr__(self, "cases", tuple(self.cases))

        if not self.cases:
            raise SchemaError("an eval set must contain at least one case")

        for position, case in enumerate(self.cases, start=1):
            if not isinstance(case, EvalCase):
                raise SchemaError(
                    f"case {position} is a {type(case).__name__}, expected EvalCase"
                )

        seen: set[str] = set()
        duplicates: list[str] = []
        for case in self.cases:
            if case.id in seen and case.id not in duplicates:
                duplicates.append(case.id)
            seen.add(case.id)
        if duplicates:
            # Duplicate ids are a load-time error, not a finding. The duplicate
            # CHECK is about semantically similar inputs; this is just two rows
            # claiming the same name, which would make any report ambiguous.
            raise SchemaError(
                "case ids must be unique, repeated: " + ", ".join(sorted(duplicates))
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)


def _require_nonempty_str(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise SchemaError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
