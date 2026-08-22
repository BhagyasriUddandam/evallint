"""Tests for statistical model comparison.

Contingency tables are constructed exactly, so every expected value is
derivable. The five required scenarios each have a section: equal models, small
samples, large differences, borderline differences, and highly imbalanced
datasets.

The borderline case is the one that matters most. It reproduces the example from
the brief — 87.1% vs 88.4%, +1.3pp — where the difference IS statistically
significant (p = 0.041) and is NOT practically meaningful against a 2pp
threshold. A p-value-only report would have said "significant, ship it".
"""

from __future__ import annotations

import pytest

from evallint.checks.model_comparison import (
    DEFAULT_PRACTICAL_THRESHOLD,
    compare_models,
)
from evallint.checks.separation import (
    cohens_g,
    holm_adjust,
    mcnemar_odds_ratio,
    paired_bootstrap_difference,
)


def table(
    n: int, both_pass: int, only_a: int, only_b: int
) -> dict[str, dict[str, bool]]:
    """Build outcome maps with an exact contingency table."""
    assert both_pass + only_a + only_b <= n
    a: dict[str, bool] = {}
    b: dict[str, bool] = {}
    index = 0
    for _ in range(both_pass):
        a[f"c{index}"], b[f"c{index}"] = True, True
        index += 1
    for _ in range(only_a):
        a[f"c{index}"], b[f"c{index}"] = True, False
        index += 1
    for _ in range(only_b):
        a[f"c{index}"], b[f"c{index}"] = False, True
        index += 1
    while index < n:
        a[f"c{index}"], b[f"c{index}"] = False, False
        index += 1
    return {"A": a, "B": b}


def only_pair(**kwargs):
    return compare_models(table(**kwargs["t"]), **kwargs.get("opts", {})).pairs[0]


# =========================================================================
# 1. EQUAL MODELS
# =========================================================================


def test_identical_models_report_no_information_not_equivalence() -> None:
    """Two models that never disagree provide NO evidence about which is
    better. That is the absence of evidence, not evidence of equivalence, and
    the distinction has to survive into the output."""
    pair = compare_models(table(n=200, both_pass=170, only_a=0, only_b=0)).pairs[0]

    assert pair.difference == 0.0
    assert pair.discordant == 0
    assert pair.mcnemar_p == 1.0
    assert pair.verdict == "no_information"
    # Undefined, not 1.0 and not 0.0.
    assert pair.odds_ratio is None
    assert pair.cohens_g is None
    assert pair.minimum_detectable_effect is None
    assert "absence of evidence" in pair.summary
    assert "not evidence of" in pair.summary


def test_equal_pass_rates_via_offsetting_disagreements_is_not_no_information() -> None:
    """Same pass rate, but the models disagree on 20 cases in both directions.
    That is real information — they are differently wrong — and must not be
    collapsed into "identical"."""
    pair = compare_models(table(n=200, both_pass=100, only_a=10, only_b=10)).pairs[0]

    assert pair.rate_a == pair.rate_b
    assert pair.difference == 0.0
    assert pair.discordant == 20
    assert pair.verdict == "no_meaningful_difference"
    assert pair.cohens_g == pytest.approx(0.0)
    assert pair.minimum_detectable_effect is not None


# =========================================================================
# 2. SMALL SAMPLES
# =========================================================================


def test_a_small_sample_is_underpowered_not_conclusive() -> None:
    """8 cases, 3 discordant all favouring B. The observed difference is huge
    (37.5pp) and the exact p is 0.25 — the eval simply cannot resolve it."""
    pair = compare_models(table(n=8, both_pass=4, only_a=0, only_b=3)).pairs[0]

    assert pair.difference == pytest.approx(0.375)
    assert pair.mcnemar_p == pytest.approx(0.25)
    assert not pair.statistically_significant
    assert pair.practically_significant
    assert pair.verdict == "underpowered"
    assert "more cases, NOT for concluding" in pair.summary


def test_a_small_sample_gets_no_bootstrap_and_says_so() -> None:
    pair = compare_models(table(n=8, both_pass=4, only_a=0, only_b=3)).pairs[0]
    assert pair.bootstrap_interval is None
    assert "too few cases to bootstrap" in pair.summary


def test_the_wald_interval_is_flagged_unreliable_at_low_discordance() -> None:
    """The interval and the exact test can disagree, and the test is right.

    3 discordant cases out of 12: delta = 0.25, SE = 0.125, so the Wald interval
    is [+0.005, +0.495] and EXCLUDES zero -- while the exact p is 0.25. The
    verdict follows the test, and the interval carries a reliability flag.

    Note the n matters. At n = 40 with the same 3 discordant cases, delta falls
    to 0.075 and the interval includes zero again; an earlier version of this
    test used 40 and did not exhibit the phenomenon it claimed to check.
    """
    pair = compare_models(table(n=12, both_pass=6, only_a=0, only_b=3)).pairs[0]

    assert pair.discordant == 3
    assert pair.interval_reliable is False
    assert pair.difference_interval[0] > 0, "Wald interval should exclude zero here"
    assert pair.mcnemar_p == pytest.approx(0.25)
    assert not pair.statistically_significant, "the exact test must win"


# =========================================================================
# 3. LARGE DIFFERENCES
# =========================================================================


def test_a_large_difference_is_real_and_meaningful() -> None:
    pair = compare_models(table(n=200, both_pass=100, only_a=5, only_b=55)).pairs[0]

    assert pair.difference == pytest.approx(0.25)
    assert pair.mcnemar_p < 1e-9
    assert pair.statistically_significant
    assert pair.practically_significant
    assert pair.verdict == "real_and_meaningful"
    assert pair.cohens_g is not None and pair.cohens_g > 0.25
    assert pair.cohens_g_band.startswith("large")
    assert pair.odds_ratio == pytest.approx(11.0)


def test_a_large_difference_has_an_interval_excluding_zero() -> None:
    pair = compare_models(table(n=200, both_pass=100, only_a=5, only_b=55)).pairs[0]
    assert pair.bootstrap_interval is not None
    lo, hi = pair.bootstrap_interval
    assert lo > 0, "a real 25pp advantage must not have zero in its interval"
    assert lo <= pair.difference <= hi


def test_direction_is_reported_correctly_when_a_wins() -> None:
    pair = compare_models(table(n=200, both_pass=100, only_a=55, only_b=5)).pairs[0]
    assert pair.difference < 0
    assert pair.verdict == "real_and_meaningful"
    assert pair.odds_ratio is not None and pair.odds_ratio < 1


# =========================================================================
# 4. BORDERLINE DIFFERENCES
# =========================================================================


def test_the_brief_example_is_significant_but_not_meaningful() -> None:
    """87.1% vs 88.4%, +1.3pp, over 1000 cases with 35 discordant.

    p = 0.041, so the difference is distinguishable from zero. It is below a 2pp
    practical threshold, so it is not worth acting on. Both must be reported;
    reporting only the p-value would say "significant" and stop.
    """
    pair = compare_models(table(n=1000, both_pass=860, only_a=11, only_b=24)).pairs[0]

    assert pair.rate_a == pytest.approx(0.871)
    assert pair.rate_b == pytest.approx(0.884)
    assert pair.difference == pytest.approx(0.013)
    assert pair.mcnemar_p < 0.05
    assert pair.statistically_significant is True
    assert pair.practically_significant is False
    assert pair.verdict == "real_but_below_threshold"
    assert "detectable without being worth acting on" in pair.summary


def test_the_practical_threshold_changes_the_verdict_not_the_statistics() -> None:
    """The same data, a different threshold. The p-value and effect size are
    unchanged; only the practical conclusion moves — which is the point of
    separating them."""
    counts = dict(n=1000, both_pass=860, only_a=11, only_b=24)
    strict = compare_models(table(**counts), practical_threshold=0.005).pairs[0]
    lenient = compare_models(table(**counts), practical_threshold=0.05).pairs[0]

    assert strict.mcnemar_p == lenient.mcnemar_p
    assert strict.difference == lenient.difference
    assert strict.verdict == "real_and_meaningful"
    assert lenient.verdict == "real_but_below_threshold"


def test_the_threshold_is_reported_as_a_choice_not_a_statistic() -> None:
    report = compare_models(table(n=100, both_pass=80, only_a=5, only_b=5))
    assert report.practical_threshold == DEFAULT_PRACTICAL_THRESHOLD
    assert any("YOUR choice, not a statistic" in note for note in report.notes)
    assert any("placeholder with no derivation" in lim for lim in report.limitations)


# =========================================================================
# 5. HIGHLY IMBALANCED DATASETS
# =========================================================================


def test_a_highly_concordant_set_reports_its_resolution_limit() -> None:
    """98% of cases pass for both models and only 2 are discordant. The eval
    cannot resolve a small difference, and the MDE says how small."""
    pair = compare_models(table(n=500, both_pass=490, only_a=1, only_b=1)).pairs[0]

    assert pair.discordant == 2
    assert pair.difference == 0.0
    assert pair.minimum_detectable_effect is not None
    assert pair.minimum_detectable_effect < 0.02
    assert "not proof of equivalence" in pair.summary


def test_the_risk_difference_shrinks_with_concordance_but_cohens_g_does_not() -> None:
    """Why both effect sizes are reported.

    The same lopsided disagreement — 2 versus 18 — on an easy set and a hard
    one. The risk difference is four times larger on the smaller set purely
    because there are fewer concordant cases diluting it. Cohen's g is
    identical, because it looks only at the discordant cases.
    """
    easy = compare_models(table(n=400, both_pass=380, only_a=2, only_b=18)).pairs[0]
    hard = compare_models(table(n=100, both_pass=80, only_a=2, only_b=18)).pairs[0]

    assert hard.difference == pytest.approx(4 * easy.difference)
    assert hard.cohens_g == pytest.approx(easy.cohens_g)
    assert hard.odds_ratio == pytest.approx(easy.odds_ratio)


def test_extreme_pass_rates_get_a_wilson_interval_inside_the_unit_range() -> None:
    """At 99.6% the normal approximation crosses 100%. Wilson does not."""
    report = compare_models(table(n=500, both_pass=498, only_a=1, only_b=1))
    for model in report.models:
        lo, hi = model.interval
        assert 0.0 <= lo <= hi <= 1.0
    assert report.models[0].pass_rate == pytest.approx(0.998)


def test_one_sided_discordance_triggers_the_haldane_correction() -> None:
    """b = 0 makes the odds ratio infinite. The correction is applied AND
    disclosed, because it biases the estimate towards 1."""
    pair = compare_models(table(n=200, both_pass=150, only_a=0, only_b=20)).pairs[0]
    assert pair.odds_ratio is not None
    assert pair.odds_ratio_corrected is True


# =========================================================================
# Multiple comparisons
# =========================================================================


def test_two_models_get_no_adjustment() -> None:
    pair = compare_models(table(n=200, both_pass=150, only_a=5, only_b=20)).pairs[0]
    assert pair.mcnemar_p_adjusted is None


def test_more_than_two_models_are_holm_adjusted() -> None:
    outcomes = {
        name: {f"c{i}": (i % 4) >= (3 - k) for i in range(200)}
        for k, name in enumerate("ABCD")
    }
    report = compare_models(outcomes)

    assert len(report.pairs) == 6
    for pair in report.pairs:
        assert pair.mcnemar_p_adjusted is not None
        assert pair.mcnemar_p_adjusted >= pair.mcnemar_p
    assert any("false positive" in note for note in report.notes)


def test_holm_is_monotone_and_never_exceeds_one() -> None:
    adjusted = holm_adjust([0.001, 0.013, 0.021, 0.04, 0.6])
    assert adjusted == [pytest.approx(0.005), pytest.approx(0.052),
                        pytest.approx(0.063), pytest.approx(0.08),
                        pytest.approx(0.6)]
    assert all(0 <= p <= 1 for p in adjusted)


# =========================================================================
# The reporting contract
# =========================================================================


def test_no_conclusion_rests_on_a_p_value_alone() -> None:
    """Every summary must carry an effect size and a practical verdict beside
    the p-value."""
    for counts in (
        dict(n=1000, both_pass=860, only_a=11, only_b=24),
        dict(n=200, both_pass=100, only_a=5, only_b=55),
        dict(n=8, both_pass=4, only_a=0, only_b=3),
    ):
        summary = compare_models(table(**counts)).pairs[0].summary
        assert "Statistical significance:" in summary
        assert "Practical significance:" in summary
        assert "Effect size:" in summary
        assert "percentage points" in summary


def test_assumptions_name_the_pairing_and_the_independence_limit() -> None:
    report = compare_models(table(n=100, both_pass=80, only_a=5, only_b=5))
    joined = " ".join(report.assumptions)
    assert "SAME cases" in joined
    assert "INDEPENDENT" in joined
    assert "redundancy check" in joined
    assert "not a random sample of production traffic" in joined


def test_limitations_reject_the_equivalence_reading() -> None:
    report = compare_models(table(n=100, both_pass=80, only_a=5, only_b=5))
    joined = " ".join(report.limitations)
    assert "NOT SIGNIFICANT IS NOT EQUIVALENCE" in joined
    assert "p-value is not an effect size" in joined


def test_render_produces_the_shape_from_the_brief() -> None:
    text = compare_models(table(n=1000, both_pass=860, only_a=11, only_b=24)).render()
    assert "A: 87.1%" in text
    assert "B: 88.4%" in text
    assert "+1.3 percentage points" in text
    assert "95% CI" in text


def test_as_dict_is_json_shaped() -> None:
    import json

    report = compare_models(table(n=100, both_pass=80, only_a=5, only_b=5))
    payload = report.as_dict()
    json.dumps(payload)  # must not raise
    assert payload["pairs"][0]["effect_size"]["cohens_g"] is not None
    assert "contingency" in payload["pairs"][0]


# =========================================================================
# Input validation
# =========================================================================


def test_needs_at_least_two_models() -> None:
    with pytest.raises(ValueError, match="at least two"):
        compare_models({"A": {"c1": True}})


def test_models_with_no_shared_cases_cannot_be_paired() -> None:
    with pytest.raises(ValueError, match="share no cases"):
        compare_models({"A": {"c1": True}, "B": {"c2": True}})


def test_only_shared_cases_are_paired() -> None:
    """A model with extra cases contributes them to its own pass rate but not
    to the paired comparison, because there is nothing to pair them with."""
    report = compare_models(
        {
            "A": {"c1": True, "c2": False, "extra": True},
            "B": {"c1": True, "c2": True},
        }
    )
    assert report.pairs[0].n_cases == 2
    assert report.models[0].n == 3


def test_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="practical_threshold"):
        compare_models(table(n=10, both_pass=5, only_a=1, only_b=1),
                       practical_threshold=0)
    with pytest.raises(ValueError, match="alpha"):
        compare_models(table(n=10, both_pass=5, only_a=1, only_b=1), alpha=1.5)


# =========================================================================
# The statistical primitives
# =========================================================================


def test_odds_ratio_matches_the_cell_ratio() -> None:
    ratio, interval, corrected = mcnemar_odds_ratio(5, 20)
    assert ratio == pytest.approx(4.0)
    assert corrected is False
    assert interval is not None and interval[0] < 4.0 < interval[1]


def test_odds_ratio_is_undefined_without_discordance() -> None:
    assert mcnemar_odds_ratio(0, 0) == (None, None, False)


def test_cohens_g_is_zero_for_a_symmetric_split() -> None:
    g, band = cohens_g(10, 10)
    assert g == pytest.approx(0.0)
    assert "negligible" in band


def test_cohens_g_is_half_for_total_lopsidedness() -> None:
    g, _ = cohens_g(0, 20)
    assert g == pytest.approx(0.5)


def test_paired_bootstrap_keeps_a_cases_two_outcomes_together() -> None:
    """Resampling the models independently would destroy the pairing and widen
    the interval. Verified by construction: 100 cases with a real 20pp gap must
    produce an interval that excludes zero."""
    outcomes = [(True, True)] * 70 + [(False, True)] * 25 + [(True, False)] * 5
    result = paired_bootstrap_difference(outcomes, seed=0)
    assert result is not None
    delta, (lo, hi) = result
    assert delta == pytest.approx(0.20)
    assert lo > 0


def test_paired_bootstrap_refuses_a_tiny_sample() -> None:
    assert paired_bootstrap_difference([(True, False)] * 5) is None
