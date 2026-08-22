"""Tests for the discrimination check.

Every test here uses a lookup-table scorer. No model is ever called, no API key
is needed, and the results are exactly reproducible — which is the whole payoff
of the injected-scorer design. What is under test is the verdict TAXONOMY
(ceiling / floor / inverted / discriminating), not anyone's model.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from evallint.checks import DiscriminationCheck, ScorerError, Severity
from evallint.schema import EvalCase, EvalSet

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


def test_result_warns_that_a_single_run_may_not_reproduce() -> None:
    """The check emits a single-run number. Measured on the bundled example set,
    40% of per-case verdicts flip across identical repeats — so a result that
    did not say so would be exactly the kind of quiet lie this tool exists to
    catch."""
    eval_set, scorer = from_pattern({"a": (False, True)})
    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    text = " ".join(result.limitations)
    assert "SINGLE-RUN MEASUREMENT" in text
    assert "three times" in text
    assert "sampling noise, not a finding" in text


def test_limitations_do_not_claim_temperature_zero_is_available() -> None:
    """An earlier version advised 'score at temperature 0'. That is impossible
    on Claude Opus 5 / Sonnet 5, which reject the parameter."""
    eval_set, scorer = from_pattern({"a": (False, True)})
    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    text = " ".join(result.limitations)
    assert "reject the temperature parameter" in text
    assert "Score at temperature 0, or repeat" not in text


# --------------------------------------------------------------------------
# repeats: detecting verdicts that are not reproducible
# --------------------------------------------------------------------------


def flaky_scorer(schedule: dict[tuple[str, str], list[bool]]):
    """A scorer returning a different answer on each successive call.

    schedule maps (case_id, model) -> the verdicts to return in order.
    """
    calls: dict[tuple[str, str], int] = {}

    def scorer(case: EvalCase, model: str) -> bool:
        key = (case.id, model)
        i = calls.get(key, 0)
        calls[key] = i + 1
        seq = schedule[key]
        return seq[min(i, len(seq) - 1)]

    return scorer


def test_repeats_defaults_to_one_and_changes_nothing() -> None:
    """Backwards compatibility: the old single-run numbers are unchanged."""
    eval_set, scorer = from_pattern(
        {"a": (False, True), "b": (True, True), "c": (False, False)}
    )
    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert result.stats["repeats"] == 1
    assert result.stats["n_unstable"] == 0
    assert result.stats["n_measured"] == 3
    assert result.stats["n_scorer_calls"] == 6


def test_a_flipping_case_is_reported_unstable_not_inverted() -> None:
    """The whole point: a verdict that changes between identical runs must not
    be counted as a finding."""
    eval_set = make_set(["steady", "flaky"])
    scorer = flaky_scorer(
        {
            ("steady", WEAK): [False],
            ("steady", STRONG): [True],
            ("flaky", WEAK): [True, False, True],  # flips
            ("flaky", STRONG): [False],
        }
    )

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=3).run(eval_set)

    assert result.stats["unstable_case_ids"] == ["flaky"]
    assert result.stats["n_unstable"] == 1
    assert result.stats["n_measured"] == 1
    assert result.stats["inverted_case_ids"] == [], (
        "a flipping case must not be reported as an inversion"
    )
    assert "flaky" not in result.stats["ceiling_case_ids"]
    assert "flaky" not in result.stats["floor_case_ids"]


def test_unstable_cases_are_warned_about_first() -> None:
    """If verdicts are not reproducible, every other number is provisional, so
    that has to be the first thing the reader sees."""
    eval_set = make_set(["flaky"])
    scorer = flaky_scorer(
        {("flaky", WEAK): [True, False], ("flaky", STRONG): [True]}
    )

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=2).run(eval_set)

    assert "gave a DIFFERENT verdict" in result.findings[0].message
    assert result.findings[0].case_ids == ("flaky",)
    assert result.findings[0].severity is Severity.WARNING


def test_a_stable_case_survives_repeats() -> None:
    """Repeats must not invent instability where there is none."""
    eval_set, scorer = from_pattern({"a": (False, True), "b": (True, True)})

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=5).run(eval_set)

    assert result.stats["n_unstable"] == 0
    assert result.stats["n_measured"] == 2
    assert result.stats["ceiling_case_ids"] == ["b"]
    assert result.stats["n_discriminating"] == 1


def test_share_denominator_excludes_unstable_cases() -> None:
    """An unstable case is unknown, not discriminating. Counting it either way
    would flatter or slander the eval set."""
    eval_set = make_set(["easy", "flaky"])
    scorer = flaky_scorer(
        {
            ("easy", WEAK): [True],
            ("easy", STRONG): [True],           # stable ceiling
            ("flaky", WEAK): [True, False],     # flips
            ("flaky", STRONG): [True],
        }
    )

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=2).run(eval_set)

    # 1 of 1 MEASURED case is non-discriminating -> 100%, not 50% of all cases.
    assert result.stats["n_measured"] == 1
    assert result.stats["non_discriminating_share"] == 1.0
    assert "1/1" in messages(result)
    assert "measured cases" in messages(result)


def test_scorer_call_count_multiplies_by_repeats() -> None:
    """Cost scales with repeats, and the stat must say so honestly."""
    calls = []

    def counting(case: EvalCase, model: str) -> bool:
        calls.append((case.id, model))
        return model == STRONG

    result = DiscriminationCheck(counting, [WEAK, STRONG], repeats=3).run(
        make_set(["a", "b"])
    )

    assert len(calls) == 12  # 2 cases x 2 models x 3 repeats
    assert result.stats["n_scorer_calls"] == 12


def test_summary_mentions_repeats_and_instability() -> None:
    eval_set = make_set(["flaky"])
    scorer = flaky_scorer(
        {("flaky", WEAK): [True, False], ("flaky", STRONG): [True]}
    )

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=2).run(eval_set)

    assert "x 2 repeats" in result.summary
    assert "1 not reproducible" in result.summary


def test_everything_unstable_says_it_measured_nothing() -> None:
    eval_set = make_set(["flaky"])
    scorer = flaky_scorer(
        {("flaky", WEAK): [True, False], ("flaky", STRONG): [False, True]}
    )

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=2).run(eval_set)

    assert result.stats["n_measured"] == 0
    assert result.stats["non_discriminating_share"] == 0.0
    assert "measured nothing" in messages(result)


@pytest.mark.parametrize("repeats", [0, -1])
def test_nonsense_repeats_is_rejected(repeats: int) -> None:
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        DiscriminationCheck(lambda c, m: True, [WEAK, STRONG], repeats=repeats)


def test_pass_rate_averages_across_repeats() -> None:
    """With disagreeing repeats, a mean is the only honest summary."""
    eval_set = make_set(["flaky"])
    scorer = flaky_scorer(
        {("flaky", WEAK): [True, False], ("flaky", STRONG): [True]}
    )

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=2).run(eval_set)

    assert result.stats["per_model_pass_rate"][WEAK] == 0.5   # 1 of 2 calls
    assert result.stats["per_model_pass_rate"][STRONG] == 1.0


# --------------------------------------------------------------------------
# max_workers: concurrency for an I/O-bound scorer
# --------------------------------------------------------------------------


def test_max_workers_defaults_to_serial() -> None:
    eval_set, scorer = from_pattern({"a": (False, True)})
    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert result.stats["max_workers"] == 1


def test_concurrency_actually_runs_calls_in_parallel() -> None:
    """Prove the pool overlaps scorer calls by OBSERVING the overlap.

    This previously asserted `elapsed < serial_total / 3` with a sleeping
    scorer. That measures the CI runner's scheduling as much as our
    concurrency, and it failed on GitHub's macOS runners in 2 of its first 4
    executions -- once by 16ms, 0.549s against a 0.533s bound -- while passing
    on Ubuntu every time and never reproducing locally in 25 runs. A
    wall-clock margin on a shared runner is a coin flip, and a test that fails
    3% of the time for no reason trains you to re-run CI instead of reading it.

    A barrier asserts the same property without a clock: it releases only once
    `workers` calls are inside the scorer simultaneously, so a serial
    implementation times out and fails EVERY time rather than occasionally.
    The speed-up is a consequence of that overlap, and is not separately
    asserted because it cannot be measured reliably on shared hardware.
    """
    import threading

    workers = 4
    # cases x models must be an exact multiple of `workers`, or the final
    # barrier group would never fill and would wait out its timeout.
    eval_set = make_set([f"c{i}" for i in range(8)])

    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    never_overlapped = False

    def concurrent_only(case: EvalCase, model: str) -> bool:
        nonlocal in_flight, max_in_flight, never_overlapped
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError:
            never_overlapped = True
        with lock:
            in_flight -= 1
        return model == STRONG

    DiscriminationCheck(
        concurrent_only, [WEAK, STRONG], max_workers=workers
    ).run(eval_set)

    assert not never_overlapped, (
        f"the barrier timed out: {workers} scorer calls were never inside the "
        "scorer at the same time, so the calls ran serially"
    )
    # Exactly `workers`, not merely more than one: the barrier guarantees at
    # least that many overlap, and ThreadPoolExecutor guarantees no more. So
    # this also pins that max_workers is respected rather than exceeded.
    assert max_in_flight == workers, (
        f"expected exactly {workers} calls in flight at once, saw {max_in_flight}"
    )


def test_concurrent_and_serial_produce_identical_results() -> None:
    """Concurrency must not change a single number, only the wall clock."""
    pattern = {
        "a": (False, True),
        "b": (True, True),
        "c": (False, False),
        "d": (True, False),
        "e": (False, True),
    }
    eval_set, scorer = from_pattern(pattern)

    serial = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)
    eval_set2, scorer2 = from_pattern(pattern)
    concurrent = DiscriminationCheck(scorer2, [WEAK, STRONG], max_workers=8).run(
        eval_set2
    )

    ignore = {"max_workers"}
    assert {k: v for k, v in serial.stats.items() if k not in ignore} == {
        k: v for k, v in concurrent.stats.items() if k not in ignore
    }
    assert serial.summary == concurrent.summary


def test_verdicts_are_assembled_in_order_not_completion_order() -> None:
    """Workers finish out of order. If results were collected as they arrived,
    a case's verdicts would be attached to the wrong model."""
    import time

    eval_set = make_set(["a", "b", "c", "d"])

    def scorer(case: EvalCase, model: str) -> bool:
        # The WEAK model is slower, so its results land last despite being
        # submitted first. Ordering must come from submission, not completion.
        time.sleep(0.03 if model == WEAK else 0.0)
        return model == STRONG

    result = DiscriminationCheck(scorer, [WEAK, STRONG], max_workers=8).run(eval_set)

    assert result.stats["per_model_pass_rate"] == {WEAK: 0.0, STRONG: 1.0}
    assert result.stats["n_discriminating"] == 4


def test_concurrent_failure_reports_the_same_case_as_serial_would() -> None:
    """Whichever thread fails first, the reported case must be deterministic —
    otherwise the error message depends on scheduling and is unreproducible."""
    import time

    eval_set = make_set(["a", "b", "c", "d"])

    def scorer(case: EvalCase, model: str) -> bool:
        if case.id == "d":
            raise RuntimeError("late failure")
        if case.id == "b":
            time.sleep(0.05)  # b finishes after d has already blown up
            raise RuntimeError("early-submitted failure")
        return True

    for workers in (1, 8):
        with pytest.raises(ScorerError) as excinfo:
            DiscriminationCheck(scorer, [WEAK, STRONG], max_workers=workers).run(
                eval_set
            )
        assert "case 'b'" in str(excinfo.value), (
            f"max_workers={workers} reported the wrong case"
        )


def test_repeats_and_workers_compose() -> None:
    eval_set, scorer = from_pattern({"a": (False, True), "b": (True, True)})

    result = DiscriminationCheck(
        scorer, [WEAK, STRONG], repeats=3, max_workers=4
    ).run(eval_set)

    assert result.stats["n_scorer_calls"] == 12  # 2 cases x 2 models x 3 repeats
    assert result.stats["n_unstable"] == 0
    assert result.stats["max_workers"] == 4


@pytest.mark.parametrize("workers", [0, -1])
def test_nonsense_max_workers_is_rejected(workers: int) -> None:
    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        DiscriminationCheck(lambda c, m: True, [WEAK, STRONG], max_workers=workers)


def test_thread_safety_requirement_is_stated_in_the_limitations() -> None:
    """evallint cannot verify the user's scorer is thread-safe, so it must say
    so rather than let an unsafe scorer corrupt results quietly."""
    eval_set, scorer = from_pattern({"a": (False, True)})
    result = DiscriminationCheck(scorer, [WEAK, STRONG], max_workers=4).run(eval_set)

    text = " ".join(result.limitations)
    assert "thread-safe" in text
    assert "corrupts results quietly" in text


# --------------------------------------------------------------------------
# sample: bounding the cost of the only expensive check
# --------------------------------------------------------------------------


def test_sample_bounds_the_call_count_exactly() -> None:
    """The whole point of sampling is a cost ceiling you can predict, so the
    call count is asserted exactly rather than approximately."""
    eval_set = make_set([f"c{i}" for i in range(100)])
    calls = []

    def counting(case: EvalCase, model: str) -> bool:
        calls.append((case.id, model))
        return model == STRONG

    result = DiscriminationCheck(
        counting, [WEAK, STRONG], sample=10, repeats=3
    ).run(eval_set)

    # 10 cases x 2 models x 3 repeats, and not one call more.
    assert len(calls) == 60
    assert result.stats["n_scorer_calls"] == 60
    assert len({c for c, _ in calls}) == 10


def test_sample_is_reproducible_for_a_given_seed() -> None:
    eval_set = make_set([f"c{i}" for i in range(50)])

    def which(seed: int) -> list[str]:
        seen: list[str] = []
        DiscriminationCheck(
            lambda c, m: (seen.append(c.id) if m == WEAK else None) or m == STRONG,
            [WEAK, STRONG],
            sample=8,
            sample_seed=seed,
        ).run(eval_set)
        return seen

    assert which(0) == which(0)          # same seed -> same cases
    assert which(0) != which(1)          # different seed -> different cases


def test_sample_keeps_file_order_so_output_is_stable() -> None:
    eval_set = make_set([f"c{i}" for i in range(30)])
    seen: list[str] = []
    DiscriminationCheck(
        lambda c, m: (seen.append(c.id) if m == WEAK else None) or True,
        [WEAK, STRONG],
        sample=6,
    ).run(eval_set)
    order = [int(c[1:]) for c in seen]
    assert order == sorted(order), "sampled cases must stay in file order"


def test_sample_larger_than_the_set_scores_everything_without_erroring() -> None:
    """A script that asks for 200 of 100 cases is being reasonable, not wrong."""
    eval_set = make_set([f"c{i}" for i in range(5)])
    result = DiscriminationCheck(
        lambda c, m: m == STRONG, [WEAK, STRONG], sample=200
    ).run(eval_set)
    assert result.stats["n_cases"] == 5
    assert result.stats["sampled"] is False


def test_sample_none_is_unchanged() -> None:
    eval_set = make_set([f"c{i}" for i in range(7)])
    a = DiscriminationCheck(lambda c, m: m == STRONG, [WEAK, STRONG]).run(eval_set)
    b = DiscriminationCheck(
        lambda c, m: m == STRONG, [WEAK, STRONG], sample=None
    ).run(eval_set)
    assert a.stats == b.stats
    assert a.summary == b.summary
    assert a.limitations == b.limitations


def test_a_sampled_run_can_never_look_like_a_full_audit() -> None:
    """The honesty requirement. A sampled result that reads like a complete one
    is the exact overclaim this project exists to report, so the sample must be
    visible in the summary, the stats AND the limitations."""
    eval_set = make_set([f"c{i}" for i in range(100)])
    result = DiscriminationCheck(
        lambda c, m: m == STRONG, [WEAK, STRONG], sample=10
    ).run(eval_set)

    assert result.summary.startswith("SAMPLE of 10 of 100 cases")
    assert result.stats["sampled"] is True
    assert result.stats["n_cases"] == 10
    assert result.stats["n_cases_in_set"] == 100
    assert result.stats["sample_seed"] == 0
    first = result.limitations[0]
    assert "10 of 100" in first
    assert "not evidence" in first


def test_sample_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sample must be at least 1"):
        DiscriminationCheck(lambda c, m: True, [WEAK, STRONG], sample=0)


def test_sample_composes_with_workers_and_repeats() -> None:
    eval_set = make_set([f"c{i}" for i in range(40)])
    lock = threading.Lock()
    calls = []

    def counting(case: EvalCase, model: str) -> bool:
        with lock:
            calls.append(case.id)
        return model == STRONG

    result = DiscriminationCheck(
        counting, [WEAK, STRONG], sample=12, repeats=2, max_workers=4
    ).run(eval_set)
    assert len(calls) == 12 * 2 * 2
    assert result.stats["n_cases"] == 12
    assert result.stats["n_measured"] == 12
