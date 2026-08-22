"""Tests for the label-leakage check.

Each one asserts both directions: the check must FIRE on a planted leak and
stay QUIET on legitimate data. The second half matters more here than for the
other checks, because the ways a clean eval set looks like a leaky one are
numerous and were measured rather than imagined — see REAL_SHAPES below.
"""

from __future__ import annotations

import pytest

from evallint.checks import LeakageCheck, Severity
from evallint.schema import EvalCase, EvalSet


def make_set(pairs: list[tuple[str, str, object]]) -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(id=cid, input=inp, expected=exp) for cid, inp, exp in pairs
        ),
        source="test",
    )


def filler(n: int, prefix: str = "c") -> list[tuple[str, str, object]]:
    """Cases long enough not to be skipped, with no containment."""
    return [
        (
            f"{prefix}{i}",
            f"Question number {i} about an unrelated topic entirely",
            f"A distinct free-text answer for case {i} with enough length",
        )
        for i in range(n)
    ]


# --- fires on known-bad ----------------------------------------------------


def test_fires_when_the_answer_is_pasted_into_the_input() -> None:
    cases = filler(24)
    cases.append(
        (
            "leak_1",
            "Classify this ticket. Hint: the correct label is billing_refund_request.",
            "billing_refund_request",
        )
    )
    result = LeakageCheck().run(make_set(cases))

    assert result.stats["n_contained"] == 1
    assert result.stats["leaking_case_ids"] == ["leak_1"]
    assert len(result.warnings) == 1
    assert "verbatim" in result.warnings[0].message


def test_containment_ignores_case_and_whitespace() -> None:
    cases = filler(24)
    cases.append(
        (
            "leak_1",
            "The   Answer Is:   THE CAPITAL CITY OF FRANCE",
            "the capital city of france",
        )
    )
    result = LeakageCheck().run(make_set(cases))
    assert result.stats["leaking_case_ids"] == ["leak_1"]


def test_reports_every_leaking_case_not_just_the_first() -> None:
    cases = filler(24)
    for i in range(3):
        answer = f"a leaked answer value number {i}"
        cases.append((f"leak_{i}", f"Question ... {answer} ...", answer))
    result = LeakageCheck().run(make_set(cases))
    assert result.stats["n_contained"] == 3
    assert len(result.warnings) == 3


# --- stays quiet on known-good --------------------------------------------


def test_quiet_when_nothing_is_contained() -> None:
    result = LeakageCheck().run(make_set(filler(25)))
    assert result.stats["n_contained"] == 0
    assert result.warnings == ()
    assert "none contain their own answer" in result.summary


def test_short_answers_are_skipped_rather_than_flagged() -> None:
    """A short answer appearing in its own input is coincidence, not a leak.

    The answers here are DISTINCT on purpose. Identical short answers would
    trip the closed-vocabulary guard first and never reach the length check, so
    this isolates the length guard rather than accidentally testing the other
    one — which is what the first version of this test did.
    """
    cases = [
        (f"c{i}", f"Pick a code for item {i}: option k{i} or option z{i}", f"k{i}")
        for i in range(25)
    ]
    result = LeakageCheck().run(make_set(cases))
    assert result.stats["closed_vocabulary"] is False, "wrong guard under test"
    assert result.stats["n_contained"] == 0
    assert result.stats["n_skipped_short_answer"] == 25
    assert result.warnings == ()


def test_a_closed_answer_vocabulary_is_treated_as_classification() -> None:
    """Four distinct answers over 100 cases is a label scheme, not free text.
    Containment there is coincidence — measured at 46/100 on MMLU."""
    cases = [
        (f"c{i}", f"Some question mentioning {'ABCD'[i % 4]} in passing", "ABCD"[i % 4])
        for i in range(100)
    ]
    result = LeakageCheck().run(make_set(cases))
    assert result.stats["closed_vocabulary"] is True
    assert result.stats["n_contained"] == 0
    assert result.warnings == ()
    assert result.findings[0].severity is Severity.INFO


def test_widespread_containment_is_one_note_not_hundreds_of_warnings() -> None:
    """Extractive QA legitimately contains its answers. A check that emits 500
    warnings for that is a check people turn off."""
    cases = [
        (
            f"c{i}",
            f"Passage {i}: the treaty was signed in the year of our reckoning "
            f"{i}. Question: when was it signed?",
            f"the year of our reckoning {i}",
        )
        for i in range(30)
    ]
    result = LeakageCheck().run(make_set(cases))

    assert result.stats["widespread"] is True
    assert result.stats["n_contained"] == 30
    assert result.warnings == ()  # not 30 warnings
    assert len(result.findings) == 1
    assert result.findings[0].severity is Severity.INFO
    assert "intended task" in result.findings[0].message


def test_small_sets_do_not_trigger_the_closed_vocabulary_guard() -> None:
    """Under 20 cases, a handful of distinct answers is indistinguishable from
    free text, so the guard must not fire and silence a real leak."""
    cases = [("leak", "the answer is a long enough phrase", "a long enough phrase")]
    cases += filler(5, "f")
    result = LeakageCheck().run(make_set(cases))
    assert result.stats["closed_vocabulary"] is False
    assert result.stats["leaking_case_ids"] == ["leak"]


# --- degenerate input -----------------------------------------------------


def test_no_expected_values_reports_that_it_cannot_assess() -> None:
    eval_set = EvalSet(
        cases=tuple(EvalCase(id=f"c{i}", input="q") for i in range(5)), source="t"
    )
    result = LeakageCheck().run(eval_set)
    assert result.stats["n_with_expected"] == 0
    assert "cannot be assessed" in result.summary
    assert result.findings[0].severity is Severity.INFO


def test_non_string_expected_is_handled() -> None:
    """`expected` is deliberately untyped; a number must not crash the check."""
    cases = filler(24)
    cases.append(("num", "the total came to 123456789012 in the end", 123456789012))
    result = LeakageCheck().run(make_set(cases))
    assert result.stats["leaking_case_ids"] == ["num"]


def test_limitations_are_always_present() -> None:
    result = LeakageCheck().run(make_set(filler(25)))
    assert result.limitations
    joined = " ".join(result.limitations)
    assert "extractive" in joined.lower()
    assert "token-overlap" in joined.lower()  # the discarded signal is disclosed


def test_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="min_answer_chars"):
        LeakageCheck(min_answer_chars=0)
    with pytest.raises(ValueError, match="widespread_share"):
        LeakageCheck(widespread_share=0)


# --- the measured guarantee ----------------------------------------------

# Shapes taken from the four real datasets this check was tuned against. Each
# produced ZERO findings, and that is the property worth pinning: a leakage
# check with false positives is one that gets switched off.
REAL_SHAPES = {
    # GSM8K: expected is a worked solution restating the question's numbers.
    # Token overlap here reaches 1.00 with no leak, which is why that signal
    # was discarded.
    "gsm8k-like": [
        (
            f"c{i}",
            f"Janet has {i} eggs and sells them for 2 dollars each. How much?",
            f"She has {i} eggs. {i} times 2 equals the total.\n#### {i * 2}",
        )
        for i in range(25)
    ],
    # MMLU: single-character answers from a closed set.
    "mmlu-like": [
        (f"c{i}", f"Which is correct? A B C D. Consider option {'ABCD'[i % 4]}.",
         "ABCD"[i % 4])
        for i in range(25)
    ],
    # TruthfulQA: free-text answers that share vocabulary with the question but
    # never contain it verbatim.
    "truthfulqa-like": [
        (
            f"c{i}",
            f"What happens if you break a mirror number {i}?",
            f"Nothing in particular happens if you break a mirror number {i}, "
            f"despite the superstition",
        )
        for i in range(25)
    ],
}


@pytest.mark.parametrize("shape", sorted(REAL_SHAPES))
def test_no_warnings_on_real_dataset_shapes(shape: str) -> None:
    result = LeakageCheck().run(make_set(REAL_SHAPES[shape]))
    assert result.warnings == (), (
        f"{shape} produced {len(result.warnings)} false positive(s): "
        f"{[w.message for w in result.warnings]}"
    )
