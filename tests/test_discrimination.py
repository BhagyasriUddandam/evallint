"""Tests for the discrimination check.

Every test here uses a lookup-table scorer. No model is ever called, no API key
is needed, and the results are exactly reproducible — which is the whole payoff
of the injected-scorer design. What is under test is the verdict TAXONOMY
(ceiling / floor / inverted / discriminating), not anyone's model.
"""

from __future__ import annotations

import numpy as np
import pytest

from evalcheck.checks import DiscriminationCheck, ScorerError, Severity
from evalcheck.schema import EvalCase, EvalSet

WEAK, STRONG = "weak-model", "strong-model"


def make_set(case_ids: list[str]) -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(id=cid, input=f"prompt for {cid}", expected="ok", label="x")
            for cid in case_ids
        )
    )


def table_scorer(table: dict[str, dict[str, bool | float]]):
    """A scorer that reads verdicts out of a dict instead of calling a model."""

    def scorer(case: EvalCase, model: str) -> bool | float:
        return table[case.id][model]

    return scorer


def from_pattern(pattern: dict[str, tuple[bool, ...]], models=(WEAK, STRONG)):
    """Build a set and scorer from {case_id: (weak_verdict, strong_verdict)}."""
    eval_set = make_set(list(pattern))
    table = {
        cid: dict(zip(models, verdicts, strict=True))
        for cid, verdicts in pattern.items()
    }
    return eval_set, table_scorer(table)


def messages(result) -> str:
    return " | ".join(f.message for f in result.findings)


# --------------------------------------------------------------------------
# Stays quiet on a healthy eval
# --------------------------------------------------------------------------


def test_healthy_eval_raises_no_warnings() -> None:
    """6 of 10 cases separate the models and nothing looks broken.

    Two ceiling cases remain, and they are reported as INFO rather than a
    warning: easy cases are normal and may be deliberate regression guards.
    """
    pattern = {f"c{i}": (False, True) for i in range(1, 7)}
    pattern.update({"easy1": (True, True), "easy2": (True, True)})
    pattern.update({"c9": (False, True), "c10": (False, True)})
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert not result.has_warnings
    assert result.stats["n_discriminating"] == 8
    assert result.stats["non_discriminating_share"] == pytest.approx(0.2)
    assert all(f.severity is Severity.INFO for f in result.findings)


def test_ceiling_cases_are_info_not_warnings() -> None:
    """Explicit: this must not contradict the check's own limitation that a
    non-discriminating case is not automatically a bad case."""
    pattern = {f"c{i}": (False, True) for i in range(1, 9)}
    pattern.update({"easy1": (True, True), "easy2": (True, True)})
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)
    ceiling = next(f for f in result.findings if "every model passes" in f.message)

    assert ceiling.severity is Severity.INFO
    assert ceiling.case_ids == ("easy1", "easy2")


# --------------------------------------------------------------------------
# Fires on known-bad input
# --------------------------------------------------------------------------


def test_eval_where_nothing_discriminates_is_flagged() -> None:
    """The headline failure: every case gives both models the same verdict."""
    pattern = {f"c{i}": (True, True) for i in range(1, 9)}
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert result.has_warnings
    assert "100% of cases (8/8)" in messages(result)
    assert "discriminates like a 0-case set" in messages(result)
    assert result.stats["effective_n_cases"] == 0
    assert "no case separated these models" in messages(result)


def test_majority_non_discriminating_is_flagged() -> None:
    pattern = {f"easy{i}": (True, True) for i in range(1, 7)}
    pattern.update({f"c{i}": (False, True) for i in range(1, 5)})
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert "60% of cases (6/10)" in messages(result)
    assert "discriminates like a 4-case set" in messages(result)


def test_floor_cases_are_warned_about_and_blame_the_case_first() -> None:
    """A case no model can pass is more often a broken reference answer than a
    genuinely hard case, and the message must say so."""
    pattern = {f"c{i}": (False, True) for i in range(1, 9)}
    pattern.update({"broken1": (False, False), "broken2": (False, False)})
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)
    floor = next(f for f in result.findings if "every model fails" in f.message)

    assert floor.severity is Severity.WARNING
    assert floor.case_ids == ("broken1", "broken2")
    assert "Check the reference answer and the grader" in floor.message
    assert result.stats["n_floor"] == 2


def test_inverted_cases_are_flagged() -> None:
    """The weak model passed where the strong model failed."""
    pattern = {f"c{i}": (False, True) for i in range(1, 9)}
    pattern.update({"suspicious": (True, False)})
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)
    inverted = next(f for f in result.findings if "a weaker model passed" in f.message)

    assert inverted.severity is Severity.WARNING
    assert inverted.case_ids == ("suspicious",)
    assert "wrong reference answer or a grader artefact" in inverted.message
    assert "in 1 case a weaker model passed" in inverted.message  # not "1 cases"


def test_counts_read_correctly_in_the_singular() -> None:
    """A report that says "1 cases" looks broken and undermines the rest."""
    eval_set, scorer = from_pattern(
        {"a": (True, True), "b": (False, False), "c": (False, True)}
    )

    text = messages(DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set))

    assert "every model passes 1 case," in text
    assert "every model fails 1 case," in text
    assert "1 cases" not in text


def test_inverted_cases_still_count_as_discriminating() -> None:
    """They do separate the models — just in the wrong direction. Counting
    them as non-discriminating would double-punish and distort the headline."""
    eval_set, scorer = from_pattern({"a": (True, False), "b": (False, True)})

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert result.stats["n_discriminating"] == 2
    assert result.stats["n_non_discriminating"] == 0
    assert result.stats["n_inverted"] == 1


# --------------------------------------------------------------------------
# Checks on the setup, not the eval set
# --------------------------------------------------------------------------


def test_backwards_model_ordering_is_detected() -> None:
    """If the 'weak' model outperforms the 'strong' one, every other number in
    this result is suspect, so it has to be said out loud."""
    pattern = {f"c{i}": (True, False) for i in range(1, 5)}
    pattern.update({"c5": (False, True)})
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert "listed as the weakest model but passed 80%" in messages(result)
    assert "wrong order or the scorer is inverted" in messages(result)


def test_per_model_pass_rates_are_reported() -> None:
    pattern = {f"c{i}": (False, True) for i in range(1, 5)}
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert result.stats["per_model_pass_rate"] == {WEAK: 0.0, STRONG: 1.0}


def test_three_models_are_supported() -> None:
    models = ["tiny", "medium", "large"]
    pattern = {
        "a": (False, False, True),
        "b": (True, True, True),
        "c": (False, True, True),
    }
    eval_set, scorer = from_pattern(pattern, models=tuple(models))

    result = DiscriminationCheck(scorer, models).run(eval_set)

    assert result.stats["n_ceiling"] == 1
    assert result.stats["n_discriminating"] == 2
    assert result.stats["n_scorer_calls"] == 9


def test_inversion_is_detected_across_non_adjacent_models() -> None:
    """tiny passes, large fails: an inversion even though medium sits between."""
    models = ["tiny", "medium", "large"]
    eval_set, scorer = from_pattern(
        {"a": (True, True, False), "b": (False, False, True)}, models=tuple(models)
    )

    result = DiscriminationCheck(scorer, models).run(eval_set)

    assert result.stats["inverted_case_ids"] == ["a"]


# --------------------------------------------------------------------------
# Scorer contract
# --------------------------------------------------------------------------


def test_float_scores_are_compared_against_the_threshold() -> None:
    table = {"a": {WEAK: 0.2, STRONG: 0.9}, "b": {WEAK: 0.6, STRONG: 0.95}}
    eval_set = make_set(["a", "b"])

    result = DiscriminationCheck(table_scorer(table), [WEAK, STRONG]).run(eval_set)

    assert result.stats["n_discriminating"] == 1  # 'a' separates, 'b' is ceiling
    assert result.stats["ceiling_case_ids"] == ["b"]


def test_pass_threshold_is_configurable() -> None:
    """The documented consequence of binarising: moving the threshold changes
    which cases count as discriminating."""
    table = {"a": {WEAK: 0.55, STRONG: 0.95}}
    eval_set = make_set(["a"])

    lenient = DiscriminationCheck(table_scorer(table), [WEAK, STRONG])
    strict = DiscriminationCheck(table_scorer(table), [WEAK, STRONG], pass_threshold=0.8)

    assert lenient.run(eval_set).stats["n_discriminating"] == 0
    assert strict.run(eval_set).stats["n_discriminating"] == 1


def test_scorer_is_called_once_per_case_per_model() -> None:
    """Each call is real money on an API-backed scorer, so no hidden repeats."""
    calls = []

    def counting_scorer(case: EvalCase, model: str) -> bool:
        calls.append((case.id, model))
        return model == STRONG

    eval_set = make_set(["a", "b", "c"])
    result = DiscriminationCheck(counting_scorer, [WEAK, STRONG]).run(eval_set)

    assert len(calls) == 6
    assert len(set(calls)) == 6
    assert result.stats["n_scorer_calls"] == 6


def test_scorer_exception_names_the_case_and_model() -> None:
    def broken_scorer(case: EvalCase, model: str) -> bool:
        if case.id == "b":
            raise RuntimeError("rate limited")
        return True

    with pytest.raises(ScorerError, match="case 'b' with model 'weak-model'"):
        DiscriminationCheck(broken_scorer, [WEAK, STRONG]).run(make_set(["a", "b"]))


def test_scorer_exception_keeps_the_original_cause() -> None:
    def broken_scorer(case: EvalCase, model: str) -> bool:
        raise RuntimeError("rate limited")

    with pytest.raises(ScorerError) as excinfo:
        DiscriminationCheck(broken_scorer, [WEAK, STRONG]).run(make_set(["a"]))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "rate limited" in str(excinfo.value)


def test_unusable_scorer_return_is_rejected() -> None:
    def stringy_scorer(case: EvalCase, model: str) -> bool:
        return "pass"  # type: ignore[return-value]

    with pytest.raises(ScorerError, match="returned str for case 'a'"):
        DiscriminationCheck(stringy_scorer, [WEAK, STRONG]).run(make_set(["a"]))


@pytest.mark.parametrize(
    ("value", "expected_verdict"),
    [
        (np.True_, True),
        (np.False_, False),
        (np.float64(0.9), True),
        (np.float64(0.1), False),
        (np.int64(1), True),
        (np.int32(0), False),
    ],
)
def test_numpy_scalars_from_a_scorer_are_accepted(value, expected_verdict) -> None:
    """A scorer built on array maths returns np.bool_, not bool.

    That is the normal shape of a real scorer (`y_pred == y_true`), and the
    model-agnostic promise is empty if it only accepts Python builtins. np.bool_
    is especially nasty: its type name is "bool", so rejecting it produced the
    message "returned bool ... expected bool or float".
    """
    check = DiscriminationCheck(lambda case, model: value, [WEAK, STRONG])
    result = check.run(make_set(["a"]))

    assert result.stats["per_model_pass_rate"][WEAK] == float(expected_verdict)


def test_an_array_return_is_still_rejected() -> None:
    """Unwrapping numpy scalars must not quietly accept a whole array."""
    with pytest.raises(ScorerError, match="returned ndarray for case 'a'"):
        DiscriminationCheck(
            lambda case, model: np.array([1.0, 0.0]), [WEAK, STRONG]
        ).run(make_set(["a"]))


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_a_single_model_is_rejected() -> None:
    """The entire premise is comparing models of differing capability."""
    with pytest.raises(ValueError, match="at least two models"):
        DiscriminationCheck(lambda case, model: True, [STRONG])


def test_duplicate_model_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="model names must be unique"):
        DiscriminationCheck(lambda case, model: True, [STRONG, STRONG])


@pytest.mark.parametrize("share", [0, -0.1, 1.5])
def test_nonsense_share_threshold_is_rejected(share: float) -> None:
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        DiscriminationCheck(
            lambda case, model: True,
            [WEAK, STRONG],
            max_non_discriminating_share=share,
        )


# --------------------------------------------------------------------------
# The honesty contract
# --------------------------------------------------------------------------


def test_result_states_the_cost_and_the_model_pair_caveat() -> None:
    eval_set, scorer = from_pattern({"a": (False, True)})
    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert any("costs time and money" in lim for lim in result.limitations)
    assert any("model pair you chose" in lim for lim in result.limitations)
    assert any("not automatically a bad case" in lim for lim in result.limitations)
