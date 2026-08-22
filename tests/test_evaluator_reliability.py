"""Tests for the evaluator reliability module.

Built around judges with KNOWN behaviour, so every expected value is derivable
rather than recorded:

  a deterministic judge          intra-consistency must be 1.0
  a coin-flip judge at 5 repeats ~1 of 20 items unanimous, P = 2*(0.5^5) = 0.0625
  a judge that always picks first position bias 1.0, order sensitivity 1.0
  two offset scoring judges      HIGH correlation with only MODERATE agreement

That last one is the point of the module: correlation and agreement are not the
same thing, and a judge suite can look excellent on one while failing the other.
"""

from __future__ import annotations

import random

import pytest

from evallint.checks.evaluator_reliability import (
    EvaluatorReliability,
    JudgeObservation,
    collect_choices,
    collect_verdicts,
)
from evallint.checks.reliability_stats import (
    MIN_BOOTSTRAP_SAMPLE,
    MIN_CORRELATION_SAMPLE,
    bootstrap_interval,
    krippendorff_alpha,
    pearson_r,
    spearman_rho,
)
from evallint.schema import EvalCase


def cases(n: int = 20) -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", input=f"question {i}", expected="a") for i in range(n)]


# =========================================================================
# Krippendorff's alpha
# =========================================================================


def test_alpha_matches_a_hand_computed_case() -> None:
    """4 units, 2 raters, binary, two agreeing and two disagreeing.

    Coincidence matrix (each ordered pair weighted 1/(m-1) = 1):
        o[0][0] = 2, o[1][1] = 2, o[0][1] = 2, o[1][0] = 2,  n = 8
        n_0 = n_1 = 4
        D_o = (1/8)(2 + 2)                = 0.5
        D_e = (1/(8*7))(4*4 + 4*4)        = 32/56 = 0.571428...
        alpha = 1 - 0.5/0.571428          = 0.125
    """
    result = krippendorff_alpha([[0, 0], [1, 1], [0, 1], [1, 0]])
    assert result.value == pytest.approx(0.125)
    assert result.n == 4
    assert result.detail["observed_disagreement"] == pytest.approx(0.5)
    assert result.detail["expected_disagreement"] == pytest.approx(32 / 56)


def test_alpha_is_one_for_perfect_agreement() -> None:
    assert krippendorff_alpha([[1, 1], [0, 0], [1, 1], [0, 0]]).value == pytest.approx(1.0)


def test_alpha_is_refused_when_one_category_is_used_throughout() -> None:
    """Expected disagreement is zero, so alpha is 0/0. Refused, not 1.0: raters
    who never made a distinction have not demonstrated reliability."""
    result = krippendorff_alpha([[1, 1]] * 10)
    assert result.value is None
    assert "0/0" in result.unmet_reason
    assert "Undefined, not zero and not one" in result.unmet_reason


def test_alpha_can_be_negative_for_systematic_disagreement() -> None:
    result = krippendorff_alpha([[1.0, 5.0], [5.0, 1.0]], level="interval")
    assert result.value is not None and result.value < 0
    assert "NEGATIVE" in result.interpretation


def test_ordinal_alpha_is_refused_rather_than_approximated() -> None:
    result = krippendorff_alpha([[1, 2], [2, 3]], level="ordinal")
    assert result.value is None
    assert "not implemented" in result.unmet_reason
    assert "rather than accepting an approximation" in result.unmet_reason


def test_alpha_tolerates_ragged_units() -> None:
    """The reason alpha is used instead of Fleiss' kappa: a judge that errored
    on some cases leaves a different number of ratings per item."""
    result = krippendorff_alpha([[1, 1, 1], [0, 0], [1, 0, 1], [0, 0, 0]])
    assert result.computed


# =========================================================================
# Correlation, and its assumption gates
# =========================================================================


def test_pearson_refuses_a_constant_series() -> None:
    """r is 0/0. Reporting 0.0 would claim the judges are uncorrelated when in
    fact one never varied."""
    result = pearson_r([1.0] * 10, list(range(10)))
    assert result.value is None
    assert "constant" in result.unmet_reason
    assert "Undefined, not zero" in result.unmet_reason


def test_pearson_refuses_too_few_pairs() -> None:
    n = MIN_CORRELATION_SAMPLE - 1
    result = pearson_r(list(range(n)), list(range(n)))
    assert result.value is None
    assert str(MIN_CORRELATION_SAMPLE) in result.unmet_reason


def test_pearson_is_one_for_a_perfect_linear_relation() -> None:
    xs = list(range(10))
    result = pearson_r(xs, [2 * x + 5 for x in xs])
    assert result.value == pytest.approx(1.0)
    # And it says what it does not mean.
    assert "not agreement" in " ".join(result.limitations)


def test_spearman_refuses_when_ties_dominate() -> None:
    """Binary verdicts are all ties. A rank correlation there reflects the
    tie-handling rule, not the data."""
    result = spearman_rho([1, 0] * 6, [1, 0] * 6)
    assert result.value is None
    assert "tied" in result.unmet_reason
    assert "use a kappa instead" in result.unmet_reason


def test_spearman_finds_a_monotonic_but_nonlinear_relation() -> None:
    xs = [float(x) for x in range(1, 11)]
    ys = [x**3 for x in xs]
    assert spearman_rho(xs, ys).value == pytest.approx(1.0)


# =========================================================================
# Bootstrap
# =========================================================================


def test_bootstrap_refuses_a_tiny_sample() -> None:
    result = bootstrap_interval([1.0, 2.0, 3.0])
    assert result.value is None
    assert str(MIN_BOOTSTRAP_SAMPLE) in result.unmet_reason


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    values = [0.0] * 10 + [1.0] * 10
    result = bootstrap_interval(values, seed=0)
    assert result.value == pytest.approx(0.5)
    assert result.interval is not None
    lo, hi = result.interval
    assert lo <= result.value <= hi
    assert result.level == 0.95


def test_bootstrap_is_reproducible_for_a_seed() -> None:
    values = [random.Random(7).random() for _ in range(30)]
    a = bootstrap_interval(values, seed=3)
    b = bootstrap_interval(values, seed=3)
    assert a.interval == b.interval


def test_bootstrap_states_the_independence_assumption() -> None:
    """Repeated verdicts on the same case are NOT independent, and the interval
    is therefore narrower than the truth. That has to be said."""
    result = bootstrap_interval([0.0, 1.0] * 10)
    joined = " ".join(result.limitations)
    assert "independent" in joined
    assert "NOT independent" in joined


# =========================================================================
# A STABLE judge
# =========================================================================


def deterministic(case: EvalCase) -> bool:
    return int(case.id[1:]) % 2 == 0


def test_a_stable_judge_scores_perfectly_on_every_applicable_measure() -> None:
    observations = collect_verdicts(
        cases(), {"a": deterministic, "b": deterministic}, repeats=3
    )
    report = EvaluatorReliability().analyse(observations)

    assert report.per_judge["a"]["intra_judge_consistency"].value == 1.0
    assert report.per_judge["a"]["decision_stability"].value == 1.0
    assert report.metrics["inter_judge_agreement"].value == 1.0
    assert report.metrics["inter_judge_alpha"].value == pytest.approx(1.0)
    assert report.metrics["inter_judge_cohens_kappa"].value == pytest.approx(1.0)


def test_a_stable_judge_still_gets_the_caveat_that_stable_is_not_correct() -> None:
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(), {"a": deterministic}, repeats=3)
    )
    consistency = report.per_judge["a"]["intra_judge_consistency"]
    assert "Self-consistency is not accuracy" in " ".join(consistency.limitations)
    stability = report.per_judge["a"]["decision_stability"]
    assert "not a correct verdict" in " ".join(stability.limitations)


# =========================================================================
# An INTENTIONALLY UNSTABLE judge
# =========================================================================


def test_a_coin_flip_judge_is_caught_as_inconsistent() -> None:
    """With 5 repeats, P(all identical) = 2 * 0.5^5 = 0.0625, so about 1 item in
    20 should look consistent by chance. A judge scoring near 1.0 here would
    mean the harness is caching, not that the judge is stable."""
    rng = random.Random(0)
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(20), {"flaky": lambda c: rng.random() < 0.5}, repeats=5)
    )
    consistency = report.per_judge["flaky"]["intra_judge_consistency"]
    assert consistency.value is not None
    assert consistency.value < 0.25, consistency.value
    assert report.per_judge["flaky"]["decision_stability"].value < 0.25


def test_two_judges_that_disagree_show_low_chance_corrected_agreement() -> None:
    report = EvaluatorReliability().analyse(
        collect_verdicts(
            cases(20),
            {
                "yes": lambda c: True,
                "alternating": lambda c: int(c.id[1:]) % 2 == 0,
            },
            repeats=1,
        )
    )
    # Raw agreement is 50% and looks mediocre; kappa is 0 or negative, because
    # a judge that always says yes agrees with anything by construction.
    raw = report.metrics["inter_judge_agreement"].value
    assert raw == pytest.approx(0.5)
    kappa = report.metrics["inter_judge_cohens_kappa"]
    assert kappa.value is None or kappa.value <= 0.05


def test_correlation_can_be_high_while_agreement_is_only_moderate() -> None:
    """THE POINT OF THE MODULE. Two judges whose scores differ by a constant
    correlate almost perfectly and still disagree on the verdict for every case
    near the threshold. A suite reporting only correlation would call this
    excellent.
    """
    base = {f"c{i}": i / 19 for i in range(20)}
    report = EvaluatorReliability().analyse(
        collect_verdicts(
            cases(20),
            {
                "lenient": lambda c: min(1.0, base[c.id] + 0.25),
                "strict": lambda c: max(0.0, base[c.id] - 0.25),
            },
            repeats=2,
        )
    )
    correlation = report.metrics["judge_correlation:lenient|strict"]
    # Not exactly 1.0: the scores are clipped at 0 and 1, which is realistic
    # (scores are bounded) and makes the relation piecewise rather than
    # perfectly linear. 0.94 is still "these judges track each other closely".
    assert correlation.value is not None
    assert correlation.value > 0.9

    kappa = report.metrics["inter_judge_cohens_kappa"]
    assert kappa.value is not None
    assert kappa.value < 0.8, "offset judges must not look like agreement"
    assert "not agreement" in " ".join(correlation.limitations)


# =========================================================================
# Position bias and order sensitivity
# =========================================================================


def options_for(items: list[EvalCase]) -> dict[str, tuple[str, str]]:
    return {c.id: (f"{c.id}_A", f"{c.id}_B") for c in items}


def test_a_positional_judge_is_caught_on_both_measures() -> None:
    items = cases(20)
    report = EvaluatorReliability().analyse(
        collect_choices(items, {"positional": lambda c, order: order[0]}, options_for(items))
    )
    bias = report.metrics["position_bias"]
    assert bias.value == 1.0
    assert bias.detail["p_value"] < 0.001
    assert "SIGNIFICANT preference for position" in bias.interpretation
    assert report.metrics["order_sensitivity"].value == 1.0


def test_a_merit_based_judge_shows_no_position_bias() -> None:
    items = cases(20)
    report = EvaluatorReliability().analyse(
        collect_choices(
            items, {"fair": lambda c, order: f"{c.id}_A"}, options_for(items)
        )
    )
    bias = report.metrics["position_bias"]
    assert bias.value == pytest.approx(0.5)
    assert bias.detail["p_value"] == pytest.approx(1.0)
    assert "Consistent with no position preference" in bias.interpretation
    assert report.metrics["order_sensitivity"].value == 0.0


def test_position_bias_is_refused_without_pairwise_data() -> None:
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(10), {"a": deterministic}, repeats=2)
    )
    for name in ("position_bias", "order_sensitivity"):
        metric = report.metrics[name]
        assert metric.value is None
        assert metric.unmet_reason


def test_order_sensitivity_is_refused_when_only_one_order_was_used() -> None:
    items = cases(10)
    report = EvaluatorReliability().analyse(
        collect_choices(
            items,
            {"a": lambda c, order: order[0]},
            options_for(items),
            both_orders=False,
        )
    )
    metric = report.metrics["order_sensitivity"]
    assert metric.value is None
    assert "one order only" in metric.unmet_reason
    assert "not evidence it is stable" in " ".join(metric.limitations)


# =========================================================================
# Refusals, and the reporting contract
# =========================================================================


def test_single_repeat_refuses_intra_judge_consistency() -> None:
    """One evaluation per item cannot show self-disagreement, and that must not
    be reported as perfect consistency."""
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(10), {"a": deterministic}, repeats=1)
    )
    metric = report.per_judge["a"]["intra_judge_consistency"]
    assert metric.value is None
    assert metric.unmet_reason == "repeats < 2"
    assert "NOT evidence of perfect consistency" in " ".join(metric.limitations)


def test_one_judge_refuses_every_cross_judge_measure() -> None:
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(10), {"solo": deterministic}, repeats=3)
    )
    assert report.n_judges == 1
    assert report.metrics["inter_judge_agreement"].value is None
    assert any("Only one judge" in note for note in report.notes)


def test_binary_judges_refuse_score_variance_and_correlation() -> None:
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(10), {"a": deterministic, "b": deterministic}, repeats=2)
    )
    assert report.per_judge["a"]["score_variance"].value is None
    assert report.metrics["judge_correlation:a|b"].value is None


def test_every_metric_carries_the_required_reporting_fields() -> None:
    """Required: sample size, metric, interval where appropriate,
    interpretation, limitations."""
    items = cases(20)
    observations = collect_verdicts(
        items, {"a": lambda c: 0.7, "b": lambda c: 0.3}, repeats=3
    ) + collect_choices(items, {"a": lambda c, o: o[0]}, options_for(items))
    report = EvaluatorReliability().analyse(observations)

    everything = list(report.metrics.values())
    for measures in report.per_judge.values():
        everything.extend(measures.values())

    for metric in everything:
        payload = metric.as_dict()
        assert isinstance(payload["n"], int)
        assert payload["metric"]
        assert payload["interpretation"]
        if metric.computed:
            # A computed metric must either carry an interval or be one for
            # which an interval is not meaningful; either way it must explain
            # itself.
            assert payload["interpretation"]
        else:
            assert payload["unmet_reason"], metric.metric


def test_the_report_lists_what_it_refused_and_why() -> None:
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(10), {"solo": deterministic}, repeats=1)
    )
    payload = report.as_dict()
    assert payload["refused"]
    assert any("assumptions were not met" in note for note in report.notes)


def test_notes_state_that_agreement_is_not_correctness() -> None:
    report = EvaluatorReliability().analyse(
        collect_verdicts(cases(10), {"a": deterministic, "b": deterministic}, repeats=2)
    )
    joined = " ".join(report.notes)
    assert "AGREEMENT IS NOT CORRECTNESS" in joined
    assert "blind spots" in joined


def test_observations_can_be_supplied_directly_without_a_collector() -> None:
    """evallint never has to call a judge: the analysis takes plain records."""
    observations = [
        JudgeObservation(judge="a", item="i1", repeat=0, verdict=True),
        JudgeObservation(judge="a", item="i1", repeat=1, verdict=False),
        JudgeObservation(judge="a", item="i2", repeat=0, verdict=True),
        JudgeObservation(judge="a", item="i2", repeat=1, verdict=True),
    ]
    report = EvaluatorReliability().analyse(observations)
    assert report.n_observations == 4
    assert report.per_judge["a"]["intra_judge_consistency"].value == pytest.approx(0.5)


def test_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="level"):
        EvaluatorReliability(level=0)
    with pytest.raises(ValueError, match="repeats"):
        collect_verdicts(cases(2), {"a": deterministic}, repeats=0)
