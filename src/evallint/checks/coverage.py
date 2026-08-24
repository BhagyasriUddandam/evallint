"""Coverage of a declared space — and a refusal to guess what that space is.

## The question, and why the obvious version of it is unanswerable

"Does my eval cover what matters?" is the most-wanted thing in eval tooling and
the easiest to fake. Answering it needs a reference: a statement of what the eval
is supposed to span. There are three places that reference could come from, and
two of them are disqualifying:

  * **Invent a taxonomy** — "reasoning, safety, tool use, long context". Nothing
    justifies those categories over any others, so the resulting percentage is
    arbitrary. This is what a score would be.
  * **Ask an LLM to generate one** — reintroduces exactly the judge problem this
    package removed. The taxonomy would be unmeasurable, unreproducible, and the
    coverage number would inherit all of it.
  * **Require the user to supply it.** Then the reference is theirs, the
    arithmetic is a census, and the tool has added measurement rather than
    opinion.

This module does the third, and **refuses to report coverage without a
reference**. There is no `coverage: 68%` that comes from nowhere.

## What is measured

Given a `CoverageSpec` — dimensions with the values that ought to be present —
the eval is cross-tabulated into cells:

    dimension "topic"  : billing, shipping, returns, account
    dimension "length" : short, medium, long
                                        -> 12 cells

For every cell: how many cases occupy it, whether it is EMPTY, and whether it is
too THIN to support a per-cell claim. The last is the same argument the imbalance
check makes: a cell of n cases has an accuracy that can only take n+1 distinct
values, so its Wilson interval is quoted rather than a bare percentage.

Given an optional `reference` distribution — production traffic shares, say — the
observed distribution is compared to it with total variation distance, and each
cell is reported as over- or under-represented.

Tier: **observed** for occupancy, which is counting. **Estimated** for the
divergence, which carries an interval.

## What is NOT measured, and cannot be

**The dimensions you did not think of.** If your spec has no "adversarial input"
dimension, this reports full coverage of a space that omits it. That is the
unknown-unknown problem and it is not solvable from the data — no amount of
analysis of a file reveals what is absent from the concept behind the file.

This is stated in the output, every time, because a coverage figure is exactly
the kind of number that gets quoted without its caveat.

**Whether an empty cell should be empty.** A chat-only dimension crossed with a
tool-use dimension has combinations that cannot exist. Cells can be marked
impossible in the spec; unmarked ones are reported as gaps, and calling a
deliberate hole a gap is the expected false positive.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import EvalCase, EvalSet
from .base import Check, CheckResult, Finding, Severity
from .separation import wilson_interval

__all__ = [
    "DEFAULT_MIN_CELL",
    "CoverageCell",
    "CoverageCheck",
    "CoverageSpec",
    "Dimension",
    "SpecError",
    "by_expected_type",
    "by_input_length",
    "by_label",
    "by_metadata",
    "by_turn_count",
    "spec_from_config",
]

#: Cases below which a cell cannot support a per-cell rate. A cell of 4 gives an
#: accuracy on a 5-point grid and a Wilson interval roughly 45 points wide, which
#: is not a measurement of anything. A convention, printed with every finding.
DEFAULT_MIN_CELL = 5

#: How far the observed distribution may sit from the reference before it is
#: reported. Total variation distance of 0.10 means "a tenth of the probability
#: mass is in the wrong cells". A convention, and labelled as one.
DEFAULT_MAX_DIVERGENCE = 0.10


class SpecError(ValueError):
    """The coverage spec is unusable. Raised rather than guessed around."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """One axis of the space the eval is meant to span.

    Attributes:
        name: what the axis is called, used in the report.
        values: the levels that OUGHT to be present. Supplied by you — this is
            the reference, and the whole result is relative to it.
        extract: reads the level off a case, or returns None when the case does
            not sit anywhere on this axis. A None is reported as unplaceable
            rather than dropped.
    """

    name: str
    values: tuple[str, ...]
    extract: Callable[[EvalCase], str | None]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SpecError("a dimension needs a name")
        object.__setattr__(self, "values", tuple(dict.fromkeys(self.values)))
        if len(self.values) < 2:
            raise SpecError(
                f"dimension '{self.name}' has {len(self.values)} value(s). An "
                "axis with one level cannot be under- or over-covered, so it "
                "measures nothing."
            )


# ------------------------------------------------------------------ extractors


def by_label(values: Sequence[str]) -> Dimension:
    """Cross-tabulate on `label`. The commonest axis, and the only one already
    present in most eval files."""
    return Dimension("label", tuple(values), lambda case: case.label)


def by_input_length(
    bands: Mapping[str, tuple[int, int]] | None = None,
) -> Dimension:
    """Bands over `len(input)`.

    Character counts, not tokens: tokenisation is model-specific, and importing
    a tokeniser to band an axis would make a core check depend on a model.
    Bands are half-open [low, high) so a case lands in exactly one.
    """
    bands = bands or {
        "short": (0, 120),
        "medium": (120, 600),
        "long": (600, 10**9),
    }
    ordered = tuple(bands)

    def extract(case: EvalCase) -> str | None:
        n = len(case.input)
        for name, (low, high) in bands.items():
            if low <= n < high:
                return name
        return None

    return Dimension("input_length", ordered, extract)


def by_expected_type() -> Dimension:
    """What shape the reference answer takes.

    An eval that is entirely single-string answers has not exercised structured
    output, however many cases it has.
    """

    def extract(case: EvalCase) -> str | None:
        value = case.expected
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "numeric"
        if isinstance(value, dict):
            return "structured"
        if isinstance(value, (list, tuple, set)):
            return "list"
        return "string"

    return Dimension(
        "expected_type",
        ("none", "boolean", "numeric", "string", "list", "structured"),
        extract,
    )


def by_turn_count() -> Dimension:
    """Single-prompt versus multi-turn. Requires schema 2 to be interesting."""

    def extract(case: EvalCase) -> str | None:
        turns = len(case.messages)
        if turns == 0:
            return "single_prompt"
        if turns == 1:
            return "one_turn"
        return "multi_turn" if turns < 5 else "long_conversation"

    return Dimension(
        "turns",
        ("single_prompt", "one_turn", "multi_turn", "long_conversation"),
        extract,
    )


def by_metadata(key: str, values: Sequence[str]) -> Dimension:
    """Any axis you already record in metadata — difficulty, source, locale."""
    return Dimension(
        key,
        tuple(values),
        lambda case: (
            None if case.metadata.get(key) is None else str(case.metadata[key])
        ),
    )


#: Dimension names that map to a built-in extractor. Anything else in a config
#: file is read off `metadata`, which is where a user's own axes live.
_BUILTIN = {
    "label": lambda values: by_label(values),
    "input_length": lambda values: _length_from_values(values),
    "expected_type": lambda values: _restrict(by_expected_type(), values),
    "turns": lambda values: _restrict(by_turn_count(), values),
}


def _restrict(dimension: Dimension, values: Sequence[str]) -> Dimension:
    """Narrow a built-in axis to the levels a spec actually declares.

    A spec listing only `string` and `structured` for expected_type is saying
    those are the two it cares about; cases landing on `numeric` then become
    UNPLACEABLE and are reported as such, rather than silently occupying a cell
    nobody declared.
    """
    unknown = sorted(set(values) - set(dimension.values))
    if unknown:
        raise SpecError(
            f"dimension '{dimension.name}' has no level(s) {unknown}. "
            f"Available: {', '.join(dimension.values)}"
        )
    return Dimension(dimension.name, tuple(values), dimension.extract)


def _length_from_values(values: Sequence[str]) -> Dimension:
    default = by_input_length()
    return _restrict(default, values)


def spec_from_config(section: Mapping[str, Any]) -> CoverageSpec:
    """Build a spec from an `evallint.toml` `[coverage]` table.

    The shape is validated in config.py; the semantics are enforced by
    CoverageSpec, so a hand-built spec is held to identical rules.
    """
    dimensions = []
    for name, values in section["dimensions"].items():
        builder = _BUILTIN.get(name)
        dimensions.append(
            builder(list(values)) if builder else by_metadata(name, list(values))
        )
    return CoverageSpec(
        dimensions=tuple(dimensions),
        reference=section.get("reference"),
        impossible=frozenset(section.get("impossible", ())),
        min_cell=int(section.get("min_cell", DEFAULT_MIN_CELL)),
    )


# ----------------------------------------------------------------------- spec


@dataclass(frozen=True, slots=True)
class CoverageSpec:
    """The reference the coverage figure is relative to.

    Attributes:
        dimensions: the axes. Two or more makes the cross-tabulation meaningful;
            one is allowed and reports marginals only.
        reference: optional target share per cell key, e.g. production traffic.
            Keys are `"|"`-joined dimension values in dimension order. Shares
            need not sum to 1 — they are normalised, and the normalisation is
            reported.
        impossible: cell keys that cannot exist, so an empty one is not a gap.
        min_cell: below this a cell is reported as too thin for a per-cell rate.
    """

    dimensions: tuple[Dimension, ...]
    reference: Mapping[str, float] | None = None
    impossible: frozenset[str] = frozenset()
    min_cell: int = DEFAULT_MIN_CELL

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        if not self.dimensions:
            raise SpecError(
                "a coverage spec needs at least one dimension. Coverage of an "
                "unstated space is not a measurable quantity — see the module "
                "docstring."
            )
        names = [d.name for d in self.dimensions]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise SpecError(f"dimension names must be unique, repeated: {repeated}")
        if self.min_cell < 1:
            raise SpecError(f"min_cell must be at least 1, got {self.min_cell}")

        object.__setattr__(self, "impossible", frozenset(self.impossible))
        keys = {"|".join(k) for k in itertools.product(*(d.values for d in self.dimensions))}
        unknown = sorted(self.impossible - keys)
        if unknown:
            raise SpecError(
                f"'impossible' names {len(unknown)} cell(s) that the dimensions "
                f"cannot produce, e.g. {unknown[:3]}. A typo here would silently "
                "excuse a real gap."
            )
        if self.reference is not None:
            stray = sorted(set(self.reference) - keys)
            if stray:
                raise SpecError(
                    f"reference names {len(stray)} cell(s) outside the "
                    f"dimensions, e.g. {stray[:3]}"
                )
            if any(v < 0 for v in self.reference.values()):
                raise SpecError("reference shares must not be negative")
            if sum(self.reference.values()) <= 0:
                raise SpecError("reference shares sum to zero")

    @property
    def cell_keys(self) -> tuple[str, ...]:
        return tuple(
            "|".join(k)
            for k in itertools.product(*(d.values for d in self.dimensions))
        )

    def normalised_reference(self) -> dict[str, float] | None:
        """Reference shares over EVERY cell, summing to 1.

        The raw numbers you wrote are not modified anywhere else; this is
        derived, so the file and the report cannot disagree.
        """
        if self.reference is None:
            return None
        total = sum(self.reference.values())
        return {key: self.reference.get(key, 0.0) / total for key in self.cell_keys}


@dataclass(frozen=True, slots=True)
class CoverageCell:
    """One cell of the cross-tabulation."""

    key: str
    n_cases: int
    case_ids: tuple[str, ...]
    observed_share: float
    #: Wilson interval on this cell's SHARE of the eval. Quoted instead of a
    #: bare percentage because a cell of 2 has an enormous one.
    share_interval: tuple[float, float]
    reference_share: float | None
    impossible: bool

    @property
    def empty(self) -> bool:
        return self.n_cases == 0

    @property
    def distinct_accuracies(self) -> int:
        """How many values a per-cell accuracy could take. n cases -> n+1."""
        return self.n_cases + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.key,
            "n_cases": self.n_cases,
            "case_ids": list(self.case_ids),
            "observed_share": round(self.observed_share, 4),
            "share_interval": [round(x, 4) for x in self.share_interval],
            "reference_share": (
                None if self.reference_share is None else round(self.reference_share, 4)
            ),
            "impossible": self.impossible,
            "distinct_accuracies": self.distinct_accuracies,
        }


LIMITATIONS = (
    "COVERAGE IS RELATIVE TO THE SPEC YOU SUPPLIED. A dimension you did not "
    "declare is invisible here, so full coverage of a two-dimension spec says "
    "nothing about the axes that spec omits. That is the unknown-unknown "
    "problem and it cannot be solved by analysing the file — nothing in the "
    "data reveals what is absent from the concept behind it.",
    "An empty cell is not necessarily a gap. Some combinations cannot exist; "
    "mark those in `impossible` and they stop being reported. Calling a "
    "deliberate hole a gap is the expected false positive.",
    "Cell counts are a census and exact. The divergence from a reference is an "
    "estimate over a finite sample, which is why every cell share carries a "
    "Wilson interval rather than a bare percentage.",
    "min_cell and the divergence threshold are CONVENTIONS, not statistics. "
    "Both are printed with the findings that use them.",
    "Input length is banded on CHARACTERS, not tokens. Tokenisation is "
    "model-specific and importing a tokeniser would make this check depend on "
    "a model.",
    "This says nothing about whether the cases inside a cell are any good. A "
    "well-covered cell of leaky, ambiguous cases is still well covered.",
)


class CoverageCheck(Check):
    """Cross-tabulate an eval against a declared space."""

    name = "coverage"
    description = "occupancy of a user-declared coverage space"

    def __init__(
        self,
        spec: CoverageSpec,
        *,
        max_divergence: float = DEFAULT_MAX_DIVERGENCE,
    ) -> None:
        if not 0 < max_divergence <= 1:
            raise SpecError("max_divergence must be in (0, 1]")
        self.spec = spec
        self.max_divergence = max_divergence

    def run(self, eval_set: EvalSet) -> CheckResult:
        spec = self.spec
        n = len(eval_set)
        reference = spec.normalised_reference()

        placed: dict[str, list[str]] = {key: [] for key in spec.cell_keys}
        unplaceable: list[str] = []
        missing_axis: dict[str, list[str]] = {d.name: [] for d in spec.dimensions}

        for case in eval_set:
            levels: list[str] = []
            for dimension in spec.dimensions:
                value = dimension.extract(case)
                if value is None or value not in dimension.values:
                    missing_axis[dimension.name].append(case.id)
                    levels = []
                    break
                levels.append(value)
            if levels:
                placed["|".join(levels)].append(case.id)
            else:
                unplaceable.append(case.id)

        n_placed = sum(len(v) for v in placed.values())
        cells = tuple(
            CoverageCell(
                key=key,
                n_cases=len(ids),
                case_ids=tuple(ids),
                observed_share=(len(ids) / n_placed) if n_placed else 0.0,
                share_interval=wilson_interval(len(ids), n_placed) if n_placed else (0.0, 0.0),
                reference_share=None if reference is None else reference[key],
                impossible=key in spec.impossible,
            )
            for key, ids in placed.items()
        )

        possible = [c for c in cells if not c.impossible]
        empty = [c for c in possible if c.empty]
        thin = [c for c in possible if 0 < c.n_cases < spec.min_cell]
        occupied = [c for c in possible if not c.empty]

        divergence = None
        over: list[CoverageCell] = []
        under: list[CoverageCell] = []
        if reference is not None:
            # Total variation distance: half the L1 gap between the two
            # distributions, so it reads as "this share of the probability mass
            # is in the wrong cells".
            divergence = 0.5 * sum(
                abs(c.observed_share - (c.reference_share or 0.0)) for c in cells
            )
            for cell in possible:
                target = cell.reference_share or 0.0
                if target == 0.0 and cell.observed_share == 0.0:
                    continue
                if cell.observed_share > target * 1.5 and cell.observed_share - target > 0.02:
                    over.append(cell)
                elif cell.observed_share < target * 0.5 and target - cell.observed_share > 0.02:
                    under.append(cell)

        stats: dict[str, Any] = {
            "n_cases": n,
            "n_placed": n_placed,
            "n_unplaceable": len(unplaceable),
            "unplaceable_case_ids": unplaceable,
            "cases_missing_a_dimension": {
                name: ids for name, ids in missing_axis.items() if ids
            },
            "dimensions": [
                {"name": d.name, "values": list(d.values)} for d in spec.dimensions
            ],
            "cells_possible": len(possible),
            "cells_impossible": len(cells) - len(possible),
            "cells_occupied": len(occupied),
            "cells_empty": len(empty),
            "cells_thin": len(thin),
            "min_cell": spec.min_cell,
            "occupancy": (len(occupied) / len(possible)) if possible else 0.0,
            "empty_cells": [c.key for c in empty],
            "thin_cells": {c.key: c.n_cases for c in thin},
            "reference_supplied": reference is not None,
            "total_variation_distance": (
                None if divergence is None else round(divergence, 4)
            ),
            "max_divergence": self.max_divergence,
            "over_represented": [c.key for c in over],
            "under_represented": [c.key for c in under],
            "cells": [c.as_dict() for c in cells],
        }

        findings = self._findings(
            spec, possible, empty, thin, unplaceable, missing_axis, divergence,
            over, under, n,
        )

        occupancy = stats["occupancy"]
        summary = (
            f"{len(occupied)}/{len(possible)} declared cells occupied "
            f"({occupancy:.0%}) across "
            f"{len(spec.dimensions)} dimension(s)"
        )
        if thin:
            summary += f"; {len(thin)} too thin for a per-cell rate"
        if divergence is not None:
            summary += f"; total variation distance from reference {divergence:.3f}"
        summary += " — relative to the spec you supplied"

        return CheckResult(
            check=self.name,
            summary=summary,
            findings=tuple(findings),
            stats=stats,
            limitations=LIMITATIONS,
        )

    def _findings(
        self,
        spec: CoverageSpec,
        possible: list[CoverageCell],
        empty: list[CoverageCell],
        thin: list[CoverageCell],
        unplaceable: list[str],
        missing_axis: dict[str, list[str]],
        divergence: float | None,
        over: list[CoverageCell],
        under: list[CoverageCell],
        n: int,
    ) -> list[Finding]:
        findings: list[Finding] = []

        if empty:
            findings.append(
                Finding(
                    f"{len(empty)} of {len(possible)} declared cells have no cases "
                    f"at all: {', '.join(c.key for c in sorted(empty, key=lambda c: c.key)[:8])}"
                    + (" ..." if len(empty) > 8 else "")
                    + ". No claim about those combinations is supported by this "
                    "eval. If a combination cannot exist, list it in the spec's "
                    "`impossible` set so it stops being reported",
                    severity=Severity.WARNING,
                )
            )

        if thin:
            worst = sorted(thin, key=lambda c: c.n_cases)
            findings.append(
                Finding(
                    f"{len(thin)} cell(s) hold fewer than {spec.min_cell} cases, "
                    f"so a per-cell rate there can only land on a few values — "
                    + ", ".join(
                        f"{c.key} ({c.n_cases} case"
                        f"{'s' if c.n_cases != 1 else ''}, "
                        f"{c.distinct_accuracies} distinct accuracies possible)"
                        for c in worst[:5]
                    )
                    + (" ..." if len(thin) > 5 else ""),
                    severity=Severity.WARNING,
                    case_ids=tuple(cid for c in thin for cid in c.case_ids),
                )
            )

        if unplaceable:
            findings.append(
                Finding(
                    f"{len(unplaceable)} of {n} cases could not be placed on "
                    "every declared dimension, so they are excluded from the "
                    "occupancy figure. Per dimension: "
                    + "; ".join(
                        f"{name} missing on {len(ids)}"
                        for name, ids in missing_axis.items()
                        if ids
                    )
                    + ". Occupancy is therefore a statement about the cases that "
                    "COULD be placed, not about the whole eval",
                    severity=Severity.WARNING,
                    case_ids=tuple(unplaceable),
                )
            )

        if divergence is not None and divergence > self.max_divergence:
            findings.append(
                Finding(
                    f"the eval's distribution sits {divergence:.1%} "
                    f"(total variation distance) from the reference you "
                    f"supplied, above the {self.max_divergence:.0%} threshold. "
                    "An aggregate score over this eval is weighted differently "
                    "from the population the reference describes",
                    severity=Severity.WARNING,
                )
            )

        for cells, direction in ((over, "over"), (under, "under")):
            if not cells:
                continue
            findings.append(
                Finding
                (
                    f"{len(cells)} cell(s) are {direction}-represented relative "
                    "to the reference: "
                    + ", ".join(
                        f"{c.key} ({c.observed_share:.0%} vs "
                        f"{(c.reference_share or 0.0):.0%} expected)"
                        for c in sorted(
                            cells, key=lambda c: -abs(
                                c.observed_share - (c.reference_share or 0.0)
                            )
                        )[:5]
                    )
                    + (" ..." if len(cells) > 5 else ""),
                    severity=Severity.INFO,
                    case_ids=tuple(cid for c in cells for cid in c.case_ids),
                )
            )

        # Always emitted, whatever else was found. A coverage percentage is
        # exactly the kind of figure that gets quoted without its caveat, so the
        # caveat is a finding rather than only a limitation.
        findings.append(
            Finding(
                f"occupancy is measured against the {len(spec.dimensions)} "
                f"dimension(s) you declared "
                f"({', '.join(d.name for d in spec.dimensions)}). A dimension "
                "you did not declare is invisible to this check, so this figure "
                "cannot tell you whether the space itself is the right one",
                severity=Severity.INFO,
            )
        )
        return findings
