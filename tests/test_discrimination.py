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
    # Wording is deliberate: "provide limited evidence for model separation",
    # not "bad cases". Every model agreeing is a fact about evidence, not a
    # defect -- these eight may be intentional regression guards.
    assert "100% of cases (8/8" in messages(result)
    assert "provide limited evidence for model separation" in messages(result)
    assert "separates these models about as well as a 0-case set" in messages(result)
    assert result.stats["effective_n_cases"] == 0
    assert "no case separated these models" in messages(result)


def test_majority_non_discriminating_is_flagged() -> None:
    pattern = {f"easy{i}": (True, True) for i in range(1, 7)}
    pattern.update({f"c{i}": (False, True) for i in range(1, 5)})
    eval_set, scorer = from_pattern(pattern)

    result = DiscriminationCheck(scorer, [WEAK, STRONG]).run(eval_set)

    assert "60% of cases (6/10, 95% CI" in messages(result)
    assert "separates these models about as well as a 4-case set" in messages(result)


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

    # CHARACTERISATION of the legacy keys. `n_measured` and
    # `non_discriminating_share` still exclude the unstable case, so a consumer
    # reading those keys sees exactly what it saw before -- 1 of 1 measured,
    # 100%. They are retained for compatibility and carry the old selection
    # bias, which is why they must not be compared across repeat counts.
    assert result.stats["n_measured"] == 1
    assert result.stats["non_discriminating_share"] == 1.0

    # The HUMAN-FACING figure uses the unbiased denominator: 1 of 2 cases, not
    # 1 of 1. Nothing is discarded from the separation analysis, so the flaky
    # case is classified from its majority verdict instead of vanishing.
    assert result.stats["n_cases"] == 2
    assert result.stats["limited_evidence_share"] == 0.5
    assert sum(result.stats["n_by_class"].values()) == 2
    # The old message said "measured cases" to flag its shrunken denominator.
    # That qualifier is gone because the denominator is no longer shrunken --
    # the message now reports 1 of 2, all cases classified.
    assert "measured cases" not in messages(result)


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


# ==========================================================================
# Separation analysis: the multi-model taxonomy
#
# Every test here asserts on `n_by_class` / `model_pairs`, the unbiased
# analysis, rather than on the legacy `n_measured` keys. The legacy keys are
# still covered above, as characterisation.
# ==========================================================================

LADDER = ("small", "medium", "large")


def ladder_set(n: int = 12) -> EvalSet:
    return make_set([f"c{i}" for i in range(n)])


def by_class(result) -> dict[str, int]:
    return result.stats["n_by_class"]


def pair(result, weak: str, strong: str) -> dict:
    return next(
        p for p in result.stats["model_pairs"] if p["pair"] == [weak, strong]
    )


# --- required scenario 1: all models pass ---------------------------------


def test_all_models_passing_is_limited_evidence_not_a_defect() -> None:
    """Requirement: do NOT call a case bad merely because models agree."""
    result = DiscriminationCheck(lambda c, m: True, list(LADDER)).run(ladder_set())

    assert by_class(result)["ceiling"] == 12
    assert result.stats["limited_evidence_share"] == 1.0
    text = messages(result)
    assert "provide limited evidence for model separation" in text
    # The wording must not editorialise about quality.
    for banned in ("bad case", "too easy", "useless", "worthless"):
        assert banned not in text.lower()
    # And it must say the cases may be worth keeping.
    assert "regression guards" in text


def test_all_models_passing_yields_no_information_per_pair() -> None:
    """No pair ever disagreed, so the eval says nothing about any of them —
    which is different from saying they are equal."""
    result = DiscriminationCheck(lambda c, m: True, list(LADDER)).run(ladder_set())

    for p in result.stats["model_pairs"]:
        assert p["verdict"] == "no_information"
        assert p["n_discordant"] == 0
        # None, not 0.0: a zero would read as "can detect anything".
        assert p["minimum_detectable_effect"] is None
    assert "never disagreed" in messages(result)
    assert "absence of evidence" in messages(result)


# --- required scenario 2: all models fail --------------------------------


def test_all_models_failing_points_at_the_reference_answer() -> None:
    result = DiscriminationCheck(lambda c, m: False, list(LADDER)).run(ladder_set())

    assert by_class(result)["floor"] == 12
    assert result.stats["limited_evidence_share"] == 1.0
    text = messages(result)
    assert "provide limited evidence for model separation" in text
    # Floor differs from ceiling in what it implies, and the message must say so.
    assert "more often wrong than hard" in text


# --- required scenario 3: monotonic improvement --------------------------


def test_monotonic_improvement_separates_adjacent_levels() -> None:
    """Each rung of the ladder passes strictly more cases than the one below."""
    thresholds = {"small": 0, "medium": 6, "large": 12}

    def scorer(case, model):
        return int(case.id[1:]) < thresholds[model]

    result = DiscriminationCheck(scorer, list(LADDER)).run(ladder_set())

    assert by_class(result)["non_monotonic"] == 0
    assert by_class(result)["separating"] == 12
    assert pair(result, "small", "large")["verdict"] == "separated"
    assert pair(result, "small", "large")["delta"] == pytest.approx(1.0)
    # Every pair ordered as declared: no negative deltas anywhere.
    assert all(p["delta"] >= 0 for p in result.stats["model_pairs"])


def test_monotonic_improvement_reports_which_pairs_are_separated() -> None:
    """Requirement 8. A share alone does not say BETWEEN WHOM."""
    thresholds = {"small": 0, "medium": 6, "large": 12}
    result = DiscriminationCheck(
        lambda c, m: int(c.id[1:]) < thresholds[m], list(LADDER)
    ).run(ladder_set())

    assert ["small", "large"] in result.stats["separated_pairs"]
    assert result.stats["contradicted_pairs"] == []
    assert "does" in messages(result) and "separate that pair" in messages(result)


def test_a_ladder_can_separate_the_ends_but_not_the_middle() -> None:
    """The case that makes per-pair reporting necessary: the eval resolves
    small-vs-large while saying nothing about medium."""

    def scorer(case, model):
        i = int(case.id[1:])
        if model == "small":
            return False
        if model == "large":
            return True
        return i < 1  # medium differs from small on exactly one case

    result = DiscriminationCheck(scorer, list(LADDER)).run(ladder_set(20))

    assert pair(result, "small", "large")["verdict"] == "separated"
    assert pair(result, "small", "medium")["verdict"] == "unresolved"
    assert ["small", "medium"] in result.stats["unresolved_pairs"]
    assert "UNRESOLVED" in messages(result)


# --- required scenario 4: non-monotonic outcomes -------------------------


def test_non_monotonic_cases_are_classified_separately() -> None:
    """A weaker model passing where a stronger one fails is evidence about the
    reference or the ordering, not a capability gap."""

    def scorer(case, model):
        return model == "small"  # only the weakest model ever passes

    result = DiscriminationCheck(scorer, list(LADDER)).run(ladder_set())

    assert by_class(result)["non_monotonic"] == 12
    assert by_class(result)["separating"] == 0
    assert pair(result, "small", "large")["verdict"] == "ordering_contradicted"
    assert ["small", "large"] in result.stats["contradicted_pairs"]


def test_non_monotonic_is_still_separation_just_in_the_wrong_direction() -> None:
    """These cases DO distinguish the models. Counting them as
    non-discriminating would be wrong twice over."""
    result = DiscriminationCheck(
        lambda c, m: m == "small", list(LADDER)
    ).run(ladder_set())

    assert result.stats["n_limited_evidence"] == 0
    assert result.stats["limited_evidence_share"] == 0.0


def test_a_single_non_monotonic_case_among_many_is_reported() -> None:
    def scorer(case, model):
        if case.id == "c0":
            return model == "small"
        return model != "small"

    result = DiscriminationCheck(scorer, list(LADDER)).run(ladder_set())

    assert by_class(result)["non_monotonic"] == 1
    assert result.stats["case_ids_by_class"]["non_monotonic"] == ["c0"]
    assert result.stats["case_classes"]["c0"] == "non_monotonic"


# --- required scenario 5: identical model performance --------------------


def test_identical_performance_reports_no_separation_without_blame() -> None:
    """Two models with exactly the same per-case verdicts. The eval separates
    nothing, and that is a fact about this model pair, not about the cases."""

    def scorer(case, model):
        return int(case.id[1:]) % 2 == 0

    result = DiscriminationCheck(scorer, ["a", "b"]).run(ladder_set())

    assert by_class(result)["ceiling"] == 6
    assert by_class(result)["floor"] == 6
    assert by_class(result)["separating"] == 0
    p = pair(result, "a", "b")
    assert p["verdict"] == "no_information"
    assert p["delta"] == 0.0


def test_identical_performance_does_not_claim_the_models_are_equal() -> None:
    """The distinction the tool must not blur: no evidence of a difference is
    not evidence of no difference."""
    result = DiscriminationCheck(
        lambda c, m: int(c.id[1:]) % 2 == 0, ["a", "b"]
    ).run(ladder_set())

    text = messages(result)
    assert "not evidence they are equally capable" in text
    assert "equally capable" in text  # stated, and stated as a negation


# --- required scenario 6: stochastic model outputs ----------------------


def test_stochastic_outputs_are_classified_by_majority_not_discarded() -> None:
    """The central fix. Under unanimity these cases would be thrown away; the
    denominator must stay at n_cases."""
    import random

    rng = random.Random(0)

    def scorer(case, model):
        return rng.random() < (0.5 if model == "weak" else 0.95)

    result = DiscriminationCheck(
        scorer, ["weak", "strong"], repeats=5
    ).run(ladder_set())

    assert sum(by_class(result).values()) == 12
    assert result.stats["n_cases"] == 12
    assert by_class(result)["indeterminate"] == 0  # odd repeats -> no ties


def test_the_denominator_does_not_move_with_repeats() -> None:
    """REGRESSION, and the reason this analysis exists.

    The legacy `non_discriminating_share` divides by the number of UNANIMOUS
    cases, and unanimity is more likely when a model's success probability is
    extreme. So raising `repeats` enriches the survivors for ceiling cases and
    inflates the share -- measured at 0.52 -> 0.89 going from 1 to 5 repeats on
    a noisy weak model, with the data unchanged.

    `limited_evidence_share` divides by n_cases, which cannot move.
    """
    import random

    cases = make_set([f"c{i}" for i in range(100)])

    def build(seed):
        rng = random.Random(seed)
        return lambda case, model: True if model == "strong" else rng.random() < 0.6

    legacy_denominators = []
    for repeats in (1, 3, 5):
        stats = DiscriminationCheck(
            build(0), ["weak", "strong"], repeats=repeats
        ).run(cases).stats
        legacy_denominators.append(stats["n_measured"])
        # The new denominator is always every scored case.
        assert stats["n_cases"] == 100
        assert sum(stats["n_by_class"].values()) == 100

    # Characterisation: the legacy denominator really does collapse.
    assert legacy_denominators[0] == 100
    assert legacy_denominators[-1] < 20, legacy_denominators


def test_per_case_support_requires_enough_repeats() -> None:
    """At repeats=1 a per-case verdict cannot be statistically supported: one
    observation does not bound a probability. That is the honest answer, and it
    must be reported rather than quietly assumed away."""
    scorer = lambda c, m: m == "strong"  # noqa: E731

    one = DiscriminationCheck(scorer, ["weak", "strong"], repeats=1).run(ladder_set())
    assert one.stats["n_statistically_supported"] == 0

    many = DiscriminationCheck(scorer, ["weak", "strong"], repeats=15).run(ladder_set())
    assert many.stats["n_statistically_supported"] == 12


def test_even_repeats_can_produce_indeterminate_cases() -> None:
    """A 1-1 split has no majority. Reported as INDETERMINATE rather than
    resolved by an arbitrary tie-break, with advice to use an odd count."""
    eval_set = make_set(["tied"])
    scorer = flaky_scorer(
        {("tied", WEAK): [True, False], ("tied", STRONG): [True, True]}
    )

    result = DiscriminationCheck(scorer, [WEAK, STRONG], repeats=2).run(eval_set)

    assert by_class(result)["indeterminate"] == 1
    assert result.stats["case_classes"]["tied"] == "indeterminate"
    assert "use an odd number" in messages(result)


# --- required scenario 7: multiple models --------------------------------


def test_four_model_ladder_reports_every_pair() -> None:
    models = ["a", "b", "c", "d"]

    def scorer(case, model):
        return models.index(model) >= int(case.id[1:]) % 4

    result = DiscriminationCheck(scorer, models).run(ladder_set(40))

    # 4 choose 2 = 6 ordered pairs, all reported, not just adjacent ones.
    assert len(result.stats["model_pairs"]) == 6
    assert sum(1 for p in result.stats["model_pairs"] if p["adjacent"]) == 3
    assert pair(result, "a", "d")["delta"] > pair(result, "a", "b")["delta"]


def test_more_models_can_only_help_or_leave_alone_the_classification() -> None:
    """Adding a middle rung must not turn a separating case into a ceiling one:
    the extremes still disagree."""

    def two(case, model):
        return model == "large"

    def three(case, model):
        return model in ("medium", "large")

    small = DiscriminationCheck(two, ["small", "large"]).run(ladder_set())
    big = DiscriminationCheck(three, list(LADDER)).run(ladder_set())

    assert by_class(small)["separating"] == 12
    assert by_class(big)["separating"] == 12


def test_two_models_remains_the_minimum() -> None:
    with pytest.raises(ValueError, match="at least two models"):
        DiscriminationCheck(lambda c, m: True, ["only-one"])


# --- reporting contract -------------------------------------------------


def test_every_case_gets_exactly_one_class() -> None:
    def scorer(case, model):
        i = int(case.id[1:])
        return {"small": i < 3, "medium": i < 6, "large": i < 9}[model]

    result = DiscriminationCheck(scorer, list(LADDER)).run(ladder_set(12))

    classes = result.stats["case_classes"]
    assert len(classes) == 12
    assert sum(by_class(result).values()) == 12
    ids_by_class = result.stats["case_ids_by_class"]
    flat = [cid for ids in ids_by_class.values() for cid in ids]
    assert sorted(flat) == sorted(classes)
    assert len(flat) == len(set(flat)), "a case must not appear in two classes"


def test_confidence_level_is_reported_not_assumed() -> None:
    result = DiscriminationCheck(lambda c, m: True, ["a", "b"]).run(ladder_set())
    assert result.stats["confidence_level"] == 0.95


def test_intervals_are_flagged_unreliable_at_small_discordant_counts() -> None:
    """The Wald interval on the paired difference is not trustworthy below ten
    discordant cases, and the report must not present it as though it were."""

    def scorer(case, model):
        return model == "strong" and int(case.id[1:]) < 3

    result = DiscriminationCheck(scorer, ["weak", "strong"]).run(ladder_set(40))
    p = pair(result, "weak", "strong")

    assert p["n_discordant"] == 3
    assert p["interval_reliable"] is False
    assert "indicative" in messages(result)
