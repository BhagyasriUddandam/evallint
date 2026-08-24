"""Tests for the coverage check.

The property that matters most is a refusal: **there is no coverage figure
without a declared spec.** A percentage that comes from nowhere is the arbitrary
score this package exists to avoid, so the spec cannot be constructed empty and
the caveat naming the declared dimensions is emitted on every run — including a
run with nothing wrong.

Everything else is arithmetic over a cross-tabulation, checked against
hand-computed values rather than against the implementation.
"""

from __future__ import annotations

import pytest

from evallint.checks.base import Severity
from evallint.checks.coverage import (
    DEFAULT_MIN_CELL,
    CoverageCheck,
    CoverageSpec,
    SpecError,
    by_expected_type,
    by_input_length,
    by_label,
    by_metadata,
    by_turn_count,
)
from evallint.schema import EvalCase, EvalSet, Message, Role

SHORT = "x" * 40
MEDIUM = "x" * 200
LONG = "x" * 900


def make(*specs: tuple[str, str, str]) -> EvalSet:
    """(id, label, padding) triples."""
    return EvalSet(
        cases=tuple(
            EvalCase(id=cid, input=f"Question {cid} {pad}", expected="a", label=label)
            for cid, label, pad in specs
        ),
        source="t.jsonl",
    )


def two_axis(labels: list[str]) -> CoverageSpec:
    return CoverageSpec(dimensions=(by_label(labels), by_input_length()))


# ==========================================================================
# 1. Coverage is undefined without a reference, and refuses to pretend
# ==========================================================================


def test_a_spec_cannot_be_empty() -> None:
    """The central design decision. An unstated space has no coverage."""
    with pytest.raises(SpecError, match="not a measurable quantity"):
        CoverageSpec(dimensions=())


def test_an_axis_with_one_level_is_refused() -> None:
    with pytest.raises(SpecError, match="cannot be under- or over-covered"):
        CoverageSpec(dimensions=(by_label(["only"]),))


def test_duplicate_dimension_names_are_refused() -> None:
    with pytest.raises(SpecError, match="must be unique"):
        CoverageSpec(dimensions=(by_label(["a", "b"]), by_label(["c", "d"])))


def test_a_typo_in_impossible_is_refused_not_ignored() -> None:
    """Silently accepting an unknown cell key would excuse a real gap."""
    with pytest.raises(SpecError, match="cannot produce"):
        CoverageSpec(
            dimensions=(by_label(["a", "b"]),), impossible=frozenset({"typo"})
        )


def test_a_reference_outside_the_spec_is_refused() -> None:
    with pytest.raises(SpecError, match="outside the dimensions"):
        CoverageSpec(dimensions=(by_label(["a", "b"]),), reference={"z": 1.0})


@pytest.mark.parametrize(
    "reference", [{"a": -0.5, "b": 1.0}, {"a": 0.0, "b": 0.0}]
)
def test_a_degenerate_reference_is_refused(reference: dict) -> None:
    with pytest.raises(SpecError):
        CoverageSpec(dimensions=(by_label(["a", "b"]),), reference=reference)


def test_the_caveat_is_emitted_even_when_nothing_is_wrong() -> None:
    """A coverage percentage is exactly the figure that gets quoted without its
    caveat, so the caveat is a FINDING rather than only a limitation."""
    eval_set = make(*[(f"c{i}", "a" if i < 6 else "b", SHORT) for i in range(12)])
    spec = CoverageSpec(
        dimensions=(by_label(["a", "b"]),),
    )
    result = CoverageCheck(spec).run(eval_set)

    assert not result.warnings, "this eval has no gaps"
    caveat = [f for f in result.findings if "did not declare" in f.message]
    assert len(caveat) == 1
    assert caveat[0].severity is Severity.INFO
    assert "label" in caveat[0].message


def test_the_summary_always_states_what_the_figure_is_relative_to() -> None:
    eval_set = make(("c1", "a", SHORT), ("c2", "b", SHORT))
    result = CoverageCheck(CoverageSpec(dimensions=(by_label(["a", "b"]),))).run(
        eval_set
    )
    assert "relative to the spec you supplied" in result.summary


# ==========================================================================
# 2. Occupancy arithmetic, checked by hand
# ==========================================================================


def test_occupancy_counts_cells_not_cases() -> None:
    """4 labels x 3 length bands = 12 cells. Two occupied, whatever the case
    count in them."""
    eval_set = make(
        *[(f"b{i}", "billing", SHORT) for i in range(14)],
        *[(f"s{i}", "shipping", MEDIUM) for i in range(6)],
    )
    stats = CoverageCheck(
        two_axis(["billing", "shipping", "returns", "account"])
    ).run(eval_set).stats

    assert stats["cells_possible"] == 12
    assert stats["cells_occupied"] == 2
    assert stats["cells_empty"] == 10
    assert stats["occupancy"] == pytest.approx(2 / 12)
    assert stats["n_placed"] == 20


def test_impossible_cells_leave_the_denominator() -> None:
    """Marking a combination impossible must shrink the space, not just hide the
    finding -- otherwise occupancy is computed against cells that cannot exist.
    """
    eval_set = make(("c1", "a", SHORT), ("c2", "b", SHORT))
    spec = CoverageSpec(
        dimensions=(by_label(["a", "b"]), by_input_length()),
        impossible=frozenset({"a|long", "b|long", "a|medium", "b|medium"}),
    )
    stats = CoverageCheck(spec).run(eval_set).stats

    assert stats["cells_possible"] == 2
    assert stats["cells_impossible"] == 4
    assert stats["occupancy"] == 1.0
    assert stats["cells_empty"] == 0


def test_empty_cells_are_named_and_warned() -> None:
    eval_set = make(("c1", "a", SHORT), ("c2", "a", SHORT))
    result = CoverageCheck(
        CoverageSpec(dimensions=(by_label(["a", "b", "c"]),))
    ).run(eval_set)

    gaps = [f for f in result.findings if "no cases at all" in f.message]
    assert len(gaps) == 1
    assert gaps[0].severity is Severity.WARNING
    assert "b" in gaps[0].message and "c" in gaps[0].message
    assert set(result.stats["empty_cells"]) == {"b", "c"}
    # And it says what to do about a legitimate hole.
    assert "impossible" in gaps[0].message


def test_a_thin_cell_reports_how_few_accuracies_are_possible() -> None:
    """Same argument the imbalance check makes: n cases -> n+1 distinct rates."""
    eval_set = make(
        *[(f"a{i}", "a", SHORT) for i in range(10)],
        ("b1", "b", SHORT),
        ("b2", "b", SHORT),
    )
    result = CoverageCheck(CoverageSpec(dimensions=(by_label(["a", "b"]),))).run(
        eval_set
    )

    thin = [f for f in result.findings if "fewer than" in f.message]
    assert len(thin) == 1
    assert "b (2 cases, 3 distinct accuracies possible)" in thin[0].message
    assert result.stats["thin_cells"] == {"b": 2}
    assert set(thin[0].case_ids) == {"b1", "b2"}


def test_min_cell_is_configurable_and_printed() -> None:
    eval_set = make(*[(f"c{i}", "a" if i < 3 else "b", SHORT) for i in range(6)])
    spec = CoverageSpec(dimensions=(by_label(["a", "b"]),), min_cell=2)
    result = CoverageCheck(spec).run(eval_set)

    assert result.stats["min_cell"] == 2
    assert result.stats["cells_thin"] == 0  # 3 cases each, above 2

    default = CoverageCheck(
        CoverageSpec(dimensions=(by_label(["a", "b"]),))
    ).run(eval_set)
    assert default.stats["min_cell"] == DEFAULT_MIN_CELL
    assert default.stats["cells_thin"] == 2  # 3 cases each, below 5


def test_every_cell_share_carries_a_wilson_interval() -> None:
    """A bare percentage on a cell of 1 would imply a precision that is absent."""
    eval_set = make(*[(f"c{i}", "a" if i else "b", SHORT) for i in range(10)])
    cells = {
        c["cell"]: c
        for c in CoverageCheck(
            CoverageSpec(dimensions=(by_label(["a", "b"]),))
        ).run(eval_set).stats["cells"]
    }
    low, high = cells["b"]["share_interval"]
    assert low < cells["b"]["observed_share"] < high
    assert high - low > 0.2, "a 1-of-10 cell should have a wide interval"


# ==========================================================================
# 3. Cases that do not fit are reported, never dropped
# ==========================================================================


def test_unplaceable_cases_are_reported_and_excluded_from_occupancy() -> None:
    """A case whose label is not in the spec cannot be placed. Silently dropping
    it would make occupancy a statement about a subset nobody was told about."""
    eval_set = make(
        ("c1", "a", SHORT), ("c2", "a", SHORT), ("rogue", "unlisted", SHORT)
    )
    result = CoverageCheck(CoverageSpec(dimensions=(by_label(["a", "b"]),))).run(
        eval_set
    )

    assert result.stats["n_cases"] == 3
    assert result.stats["n_placed"] == 2
    assert result.stats["unplaceable_case_ids"] == ["rogue"]
    assert result.stats["cases_missing_a_dimension"] == {"label": ["rogue"]}

    note = [f for f in result.findings if "could not be placed" in f.message]
    assert len(note) == 1
    assert note[0].severity is Severity.WARNING
    assert note[0].case_ids == ("rogue",)
    assert "not about the whole eval" in note[0].message


def test_an_unlabelled_case_is_unplaceable_not_an_error() -> None:
    eval_set = EvalSet(
        cases=(
            EvalCase(id="c1", input=f"Q {SHORT}", label="a"),
            EvalCase(id="c2", input=f"Q {SHORT}"),  # no label
        )
    )
    stats = CoverageCheck(
        CoverageSpec(dimensions=(by_label(["a", "b"]),))
    ).run(eval_set).stats
    assert stats["unplaceable_case_ids"] == ["c2"]


# ==========================================================================
# 4. Divergence from a supplied reference
# ==========================================================================


def test_total_variation_distance_matches_the_hand_calculation() -> None:
    """TVD is half the L1 gap. Six 'a' and two 'b' against a 50/50 reference:
    observed (0.75, 0.25), so TVD = (|0.75-0.5| + |0.25-0.5|) / 2 = 0.25.
    """
    eval_set = make(*[(f"c{i}", "a" if i < 6 else "b", SHORT) for i in range(8)])
    spec = CoverageSpec(
        dimensions=(by_label(["a", "b"]),), reference={"a": 0.5, "b": 0.5}
    )
    stats = CoverageCheck(spec).run(eval_set).stats
    assert stats["total_variation_distance"] == pytest.approx(0.25)


def test_reference_shares_are_normalised_not_required_to_sum_to_one() -> None:
    """Counts are a natural way to express a target, so 30/10 must mean the same
    as 0.75/0.25."""
    eval_set = make(*[(f"c{i}", "a" if i < 6 else "b", SHORT) for i in range(8)])
    as_counts = CoverageCheck(
        CoverageSpec(
            dimensions=(by_label(["a", "b"]),), reference={"a": 30, "b": 10}
        )
    ).run(eval_set).stats
    as_shares = CoverageCheck(
        CoverageSpec(
            dimensions=(by_label(["a", "b"]),), reference={"a": 0.75, "b": 0.25}
        )
    ).run(eval_set).stats
    assert (
        as_counts["total_variation_distance"]
        == as_shares["total_variation_distance"]
        == pytest.approx(0.0)
    )


def test_the_raw_reference_is_not_mutated() -> None:
    """Normalisation is derived, so the spec and the report cannot disagree."""
    reference = {"a": 30, "b": 10}
    spec = CoverageSpec(dimensions=(by_label(["a", "b"]),), reference=reference)
    spec.normalised_reference()
    assert reference == {"a": 30, "b": 10}
    assert spec.normalised_reference() == {"a": 0.75, "b": 0.25}


def test_divergence_above_the_threshold_warns_about_weighting() -> None:
    eval_set = make(*[(f"c{i}", "a" if i < 19 else "b", SHORT) for i in range(20)])
    spec = CoverageSpec(
        dimensions=(by_label(["a", "b"]),), reference={"a": 0.5, "b": 0.5}
    )
    result = CoverageCheck(spec).run(eval_set)

    warn = [f for f in result.findings if "total variation distance" in f.message]
    assert len(warn) == 1
    assert "weighted differently" in warn[0].message


def test_a_matching_distribution_produces_no_divergence_finding() -> None:
    """The quiet half."""
    eval_set = make(*[(f"c{i}", "a" if i < 6 else "b", SHORT) for i in range(8)])
    spec = CoverageSpec(
        dimensions=(by_label(["a", "b"]),), reference={"a": 0.75, "b": 0.25}
    )
    result = CoverageCheck(spec).run(eval_set)

    assert result.stats["total_variation_distance"] == pytest.approx(0.0)
    assert not [f for f in result.findings if "total variation" in f.message]
    assert not result.stats["over_represented"]
    assert not result.stats["under_represented"]


def test_over_and_under_representation_are_named_per_cell() -> None:
    eval_set = make(*[(f"c{i}", "a" if i < 18 else "b", SHORT) for i in range(20)])
    spec = CoverageSpec(
        dimensions=(by_label(["a", "b"]),), reference={"a": 0.5, "b": 0.5}
    )
    stats = CoverageCheck(spec).run(eval_set).stats
    assert stats["over_represented"] == ["a"]
    assert stats["under_represented"] == ["b"]


def test_no_reference_means_no_divergence_reported() -> None:
    """Not 0.0, which would read as 'matches the reference'."""
    eval_set = make(("c1", "a", SHORT), ("c2", "b", SHORT))
    stats = CoverageCheck(
        CoverageSpec(dimensions=(by_label(["a", "b"]),))
    ).run(eval_set).stats
    assert stats["reference_supplied"] is False
    assert stats["total_variation_distance"] is None


def test_max_divergence_must_be_a_share() -> None:
    with pytest.raises(SpecError, match="max_divergence"):
        CoverageCheck(CoverageSpec(dimensions=(by_label(["a", "b"]),)), max_divergence=0)


# ==========================================================================
# 5. The built-in extractors
# ==========================================================================


def test_input_length_bands_are_half_open() -> None:
    """A case must land in exactly one band. The default boundary is 120, so a
    119-character input is short and a 120-character one is medium."""
    dimension = by_input_length()
    assert dimension.extract(EvalCase(id="a", input="x" * 119)) == "short"
    assert dimension.extract(EvalCase(id="b", input="x" * 120)) == "medium"
    assert dimension.extract(EvalCase(id="c", input="x" * 599)) == "medium"
    assert dimension.extract(EvalCase(id="d", input="x" * 600)) == "long"


def test_custom_length_bands() -> None:
    dimension = by_input_length({"tiny": (0, 10), "big": (10, 10**9)})
    assert dimension.values == ("tiny", "big")
    assert dimension.extract(EvalCase(id="a", input="x" * 5)) == "tiny"
    assert dimension.extract(EvalCase(id="b", input="x" * 50)) == "big"


@pytest.mark.parametrize(
    ("expected", "band"),
    [
        (None, "none"),
        (True, "boolean"),
        (42, "numeric"),
        (4.2, "numeric"),
        ("Paris", "string"),
        (["a", "b"], "list"),
        ({"total": 1}, "structured"),
    ],
)
def test_expected_type_bands(expected, band: str) -> None:
    """bool before int: `isinstance(True, int)` is True in Python, so checking
    numeric first would classify every boolean as numeric."""
    dimension = by_expected_type()
    assert dimension.extract(EvalCase(id="a", input="q", expected=expected)) == band


def test_turn_count_bands() -> None:
    dimension = by_turn_count()
    assert dimension.extract(EvalCase(id="a", input="q")) == "single_prompt"
    chat = EvalCase(
        id="b",
        input="derived",
        messages=(
            Message(role=Role.USER, content="hi"),
            Message(role=Role.ASSISTANT, content="hello"),
        ),
        input_is_derived=True,
    )
    assert dimension.extract(chat) == "multi_turn"


def test_metadata_dimension() -> None:
    dimension = by_metadata("difficulty", ["easy", "hard"])
    case = EvalCase(id="a", input="q", metadata={"difficulty": "hard"})
    assert dimension.extract(case) == "hard"
    assert dimension.extract(EvalCase(id="b", input="q")) is None


def test_metadata_values_are_stringified() -> None:
    """A CSV writes 1 as text and JSON as a number; those must be one level."""
    dimension = by_metadata("tier", ["1", "2"])
    assert dimension.extract(EvalCase(id="a", input="q", metadata={"tier": 1})) == "1"


# ==========================================================================
# 6. Integration
# ==========================================================================


def test_three_dimensions_cross_tabulate() -> None:
    eval_set = make(*[(f"c{i}", "a" if i < 5 else "b", SHORT) for i in range(10)])
    spec = CoverageSpec(
        dimensions=(by_label(["a", "b"]), by_input_length(), by_expected_type())
    )
    stats = CoverageCheck(spec).run(eval_set).stats
    assert stats["cells_possible"] == 2 * 3 * 6
    assert stats["cells_occupied"] == 2  # a|short|string and b|short|string


def test_the_check_states_its_limitations() -> None:
    """CheckResult refuses to exist without them; this pins the important one."""
    result = CoverageCheck(CoverageSpec(dimensions=(by_label(["a", "b"]),))).run(
        make(("c1", "a", SHORT))
    )
    joined = " ".join(result.limitations)
    assert "COVERAGE IS RELATIVE TO THE SPEC YOU SUPPLIED" in joined
    assert "unknown-unknown" in joined
    assert "CONVENTIONS, not statistics" in joined


def test_an_eval_covering_everything_still_reports_the_caveat() -> None:
    eval_set = make(
        *[(f"a{i}", "a", SHORT) for i in range(6)],
        *[(f"b{i}", "b", MEDIUM) for i in range(6)],
        *[(f"c{i}", "a", LONG) for i in range(6)],
        *[(f"d{i}", "b", SHORT) for i in range(6)],
        *[(f"e{i}", "a", MEDIUM) for i in range(6)],
        *[(f"f{i}", "b", LONG) for i in range(6)],
    )
    result = CoverageCheck(two_axis(["a", "b"])).run(eval_set)

    assert result.stats["occupancy"] == 1.0
    assert result.stats["cells_empty"] == 0
    assert result.stats["cells_thin"] == 0
    assert not result.warnings
    # The one finding that survives a perfect result.
    assert len(result.findings) == 1
    assert "cannot tell you whether the space itself is the right one" in (
        result.findings[0].message
    )
