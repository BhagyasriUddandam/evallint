"""Tests for the imbalance check.

The rule this file exists to enforce: the check must FIRE on a known-bad set
and stay SILENT on a known-good one. A check that warns about everything is
worth exactly as little as one that warns about nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalcheck.checks import CheckResult, Finding, ImbalanceCheck, Severity
from evalcheck.io import load
from evalcheck.schema import EvalCase, EvalSet

EXAMPLE_SET = Path(__file__).resolve().parents[1] / "examples" / "sample_evalset.jsonl"


def make_set(labels: list[str | None], *, expected: object = "ok") -> EvalSet:
    """Build an EvalSet with one case per label. Inputs are distinct filler."""
    return EvalSet(
        cases=tuple(
            EvalCase(
                id=f"c{i}",
                input=f"question number {i}",
                expected=expected,
                label=label,
            )
            for i, label in enumerate(labels, start=1)
        )
    )


def messages(result: CheckResult) -> str:
    return " | ".join(f.message for f in result.findings)


# --------------------------------------------------------------------------
# Stays quiet on known-good input
# --------------------------------------------------------------------------


def test_balanced_set_produces_no_findings() -> None:
    """5/5/5 across three classes is healthy. Nothing should be reported."""
    result = ImbalanceCheck().run(make_set(["a"] * 5 + ["b"] * 5 + ["c"] * 5))

    assert result.findings == ()
    assert not result.has_warnings
    assert result.stats["imbalance_ratio"] == 1.0
    assert result.summary == "15 cases across 3 classes, imbalance ratio 1.0:1"


def test_mild_imbalance_within_thresholds_is_not_flagged() -> None:
    """8/6/6 is uneven but not distorting. The check must tolerate real data."""
    result = ImbalanceCheck().run(make_set(["a"] * 8 + ["b"] * 6 + ["c"] * 6))

    assert result.findings == ()


# --------------------------------------------------------------------------
# Fires on known-bad input
# --------------------------------------------------------------------------


def test_ninety_ten_split_triggers_a_warning() -> None:
    result = ImbalanceCheck().run(make_set(["a"] * 18 + ["b"] * 2))

    assert result.has_warnings
    assert "imbalance ratio is 9.0:1" in messages(result)
    assert "class 'b' is under-represented" in messages(result)
    assert result.stats["imbalance_ratio"] == 9.0


def test_small_share_is_flagged_even_when_the_count_is_adequate() -> None:
    """95/5 of 100: class 'b' has 5 cases (enough to measure) but only 5% of
    the set, so it cannot move the aggregate number. Share and count are
    separate failure modes and this isolates the share one."""
    result = ImbalanceCheck().run(make_set(["a"] * 95 + ["b"] * 5))

    rare = [f for f in result.findings if "under-represented" in f.message]
    assert len(rare) == 1
    assert "5.0% of labelled cases (below 10%)" in rare[0].message
    assert "only 5 cases" not in rare[0].message


def test_small_count_is_flagged_even_when_the_share_is_adequate() -> None:
    """3/3/3 of 9: every class is 33% of the set, but three cases each means
    per-class accuracy can only be 0, 33, 67 or 100%. This isolates count."""
    result = ImbalanceCheck().run(make_set(["a"] * 3 + ["b"] * 3 + ["c"] * 3))

    rare = [f for f in result.findings if "under-represented" in f.message]
    assert len(rare) == 3
    assert "only 3 cases" in rare[0].message
    assert "distinct values" in rare[0].message


def test_finding_lists_the_offending_case_ids() -> None:
    result = ImbalanceCheck().run(make_set(["a"] * 18 + ["b"] * 2))

    rare = next(f for f in result.findings if "class 'b'" in f.message)
    assert rare.case_ids == ("c19", "c20")
    assert rare.severity is Severity.WARNING


# --------------------------------------------------------------------------
# Degenerate inputs must not silently pass
# --------------------------------------------------------------------------


def test_unlabelled_set_reports_that_it_could_not_run() -> None:
    """No labels is not a clean bill of health. It must say so."""
    result = ImbalanceCheck().run(make_set([None] * 10))

    assert len(result.findings) == 1
    assert result.findings[0].severity is Severity.INFO
    assert "no case has a label" in result.findings[0].message
    assert "not assessed" in result.summary
    assert result.stats["imbalance_ratio"] is None
    assert not result.has_warnings


def test_partially_labelled_set_warns_about_the_gap() -> None:
    result = ImbalanceCheck().run(make_set(["a"] * 5 + ["b"] * 5 + [None] * 4))

    assert "4 of 14 cases have no label" in messages(result)
    assert result.stats["n_unlabelled"] == 4
    assert result.stats["n_labelled"] == 10


def test_single_class_is_reported_as_info_not_a_warning() -> None:
    """One class may be intentional for a single-purpose eval, so it is worth
    knowing but is not evidence of a mistake."""
    result = ImbalanceCheck().run(make_set(["a"] * 10))

    assert len(result.findings) == 1
    assert result.findings[0].severity is Severity.INFO
    assert "only measure one scenario" in result.findings[0].message


def test_cases_without_expected_are_flagged_as_unscoreable() -> None:
    result = ImbalanceCheck().run(make_set(["a"] * 5 + ["b"] * 5, expected=None))

    assert "10 of 10 cases have no 'expected' value" in messages(result)


# --------------------------------------------------------------------------
# Stats and configuration
# --------------------------------------------------------------------------


def test_stats_carry_the_raw_numbers() -> None:
    result = ImbalanceCheck().run(make_set(["a"] * 18 + ["b"] * 2))

    assert result.stats["n_cases"] == 20
    assert result.stats["n_classes"] == 2
    assert result.stats["class_counts"] == {"a": 18, "b": 2}
    assert result.stats["class_shares"]["b"] == pytest.approx(0.1)
    assert result.stats["input_length_chars"]["min"] > 0
    assert (
        result.stats["input_length_chars"]["min"]
        <= result.stats["input_length_chars"]["max"]
    )


def test_thresholds_are_configurable() -> None:
    """The defaults are conventions, so a user must be able to override them."""
    eval_set = make_set(["a"] * 18 + ["b"] * 2)

    assert ImbalanceCheck().run(eval_set).has_warnings
    relaxed = ImbalanceCheck(
        min_class_share=0.01, min_class_count=1, max_imbalance_ratio=20.0
    )
    assert not relaxed.run(eval_set).has_warnings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_class_share": 0},
        {"min_class_share": 1.5},
        {"min_class_count": 0},
        {"max_imbalance_ratio": 0.5},
    ],
)
def test_nonsense_thresholds_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ImbalanceCheck(**kwargs)


# --------------------------------------------------------------------------
# The honesty contract from CLAUDE.md, enforced in code
# --------------------------------------------------------------------------


def test_every_result_states_its_limitations() -> None:
    for eval_set in [make_set(["a"] * 5 + ["b"] * 5), make_set([None] * 3)]:
        result = ImbalanceCheck().run(eval_set)
        assert result.limitations
        assert any("not whether" in lim for lim in result.limitations)


def test_check_result_refuses_to_exist_without_limitations() -> None:
    with pytest.raises(ValueError, match="must state what it cannot tell you"):
        CheckResult(check="imbalance", summary="fine")


def test_finding_requires_a_message() -> None:
    with pytest.raises(ValueError, match="must have a message"):
        Finding(message="   ")


# --------------------------------------------------------------------------
# The shipped example: the planted imbalance must actually be caught
# --------------------------------------------------------------------------


def test_example_evalset_imbalance_is_caught() -> None:
    result = ImbalanceCheck().run(load(EXAMPLE_SET))

    assert result.stats["class_counts"] == {
        "billing": 14,
        "password_reset": 4,
        "refund": 1,
        "bug_report": 1,
    }
    assert result.stats["imbalance_ratio"] == 14.0
    assert "imbalance ratio is 14.0:1" in messages(result)

    flagged = {
        f.message.split("'")[1]
        for f in result.findings
        if "under-represented" in f.message
    }
    assert flagged == {"password_reset", "refund", "bug_report"}
