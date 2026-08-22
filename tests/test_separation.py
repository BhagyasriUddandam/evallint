"""Tests for the separation statistics.

Every expected value here is hand-computed or taken from a textbook identity,
not from running the code and recording what it said. That distinction matters:
a test that captures current output cannot detect a wrong estimator, and these
functions decide what the tool claims about model differences.
"""

from __future__ import annotations

import math

import pytest

from evallint.checks.separation import (
    ALPHA,
    NORMAL_APPROX_MIN_DISCORDANT,
    Z_80_POWER,
    Z_95,
    mcnemar_exact_p,
    minimum_detectable_effect,
    paired_difference,
    wilson_interval,
)

# --- Wilson interval ------------------------------------------------------


def test_wilson_matches_the_closed_form() -> None:
    """Computed by hand from the definition, for 98 of 100."""
    n, k, z = 100, 98, Z_95
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))

    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(centre - half)
    assert hi == pytest.approx(centre + half)


def test_wilson_stays_inside_zero_one_at_the_extremes() -> None:
    """The reason Wilson is used instead of the normal approximation.

    At 98/100 the normal interval is [0.943, 1.017] — it includes an impossible
    value. At 10/10 it is [1.0, 1.0], claiming certainty from ten observations.
    """
    for k, n in [(0, 10), (10, 10), (98, 100), (1, 3), (0, 1), (1, 1)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0, f"{k}/{n} gave [{lo}, {hi}]"


def test_wilson_never_has_zero_width_at_the_boundaries() -> None:
    """0 successes is not proof the rate is zero. The normal approximation says
    it is, which is how an eval ends up claiming 100% with n=5."""
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi > 0.2, "an upper bound this loose is the honest answer at n=10"

    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert lo < 0.8


def test_three_of_three_cannot_establish_a_majority() -> None:
    """Load-bearing for the per-case support flag: three successes out of three
    do NOT put the success probability above one half at 95%."""
    lo, _ = wilson_interval(3, 3)
    assert lo < 0.5


def test_wilson_narrows_as_n_grows() -> None:
    widths = [
        wilson_interval(int(0.9 * n), n)[1] - wilson_interval(int(0.9 * n), n)[0]
        for n in (10, 100, 1000, 10000)
    ]
    assert widths == sorted(widths, reverse=True)


def test_wilson_with_no_trials_is_unconstrained() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="within"):
        wilson_interval(11, 10)


# --- McNemar exact test ---------------------------------------------------


def test_mcnemar_matches_the_binomial_tail_by_hand() -> None:
    """b=1, c=9: two-sided p = 2 * P(X <= 1) for X ~ Binomial(10, 0.5)
    = 2 * (C(10,0) + C(10,1)) / 2^10 = 2 * 11/1024 = 22/1024."""
    assert mcnemar_exact_p(1, 9) == pytest.approx(22 / 1024)


def test_mcnemar_is_symmetric_in_its_arguments() -> None:
    for b, c in [(0, 5), (2, 7), (3, 3), (1, 40)]:
        assert mcnemar_exact_p(b, c) == pytest.approx(mcnemar_exact_p(c, b))


def test_mcnemar_with_no_discordance_is_not_evidence_of_equality() -> None:
    """Two models that never disagree give p = 1: no evidence of a difference,
    which is NOT the same as evidence there is none."""
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_ignores_concordant_cases_entirely() -> None:
    """Only discordant pairs carry information about which model is better, so
    the function has no parameter for the rest."""
    assert mcnemar_exact_p(0, 5) == mcnemar_exact_p(0, 5)
    assert mcnemar_exact_p(0, 5) == pytest.approx(2 * (1 / 32))


def test_mcnemar_three_discordant_one_sided_is_not_significant() -> None:
    """REGRESSION. This is the case where the Wald interval and the exact test
    disagree: 3 discordant cases all favouring one model give a Wald interval
    excluding zero, but p = 0.25. The pair verdict must follow the test."""
    assert mcnemar_exact_p(0, 3) == pytest.approx(0.25)
    assert mcnemar_exact_p(0, 3) > ALPHA


def test_mcnemar_agrees_across_the_exact_approximation_boundary() -> None:
    """The switch to the normal approximation above EXACT_TEST_MAX_N must not
    produce a discontinuity that changes a decision."""
    n = 1000
    b = 450
    exact = mcnemar_exact_p(b, n - b)
    approx_side = mcnemar_exact_p(b + 1, n - b)  # just over, still same regime
    assert abs(exact - approx_side) < 0.1


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_exact_p(-1, 3)


# --- paired difference ----------------------------------------------------


def test_paired_difference_matches_the_variance_formula() -> None:
    b, c, n = 4, 12, 100
    delta = (c - b) / n
    p_discordant = (b + c) / n
    expected_se = math.sqrt((p_discordant - delta * delta) / n)

    got_delta, (lo, hi), se = paired_difference(b, c, n)
    assert got_delta == pytest.approx(delta)
    assert se == pytest.approx(expected_se)
    assert lo == pytest.approx(delta - Z_95 * expected_se)
    assert hi == pytest.approx(delta + Z_95 * expected_se)


def test_paired_difference_sign_convention() -> None:
    """Positive delta means the SECOND (declared stronger) model did better."""
    delta, _, _ = paired_difference(b=2, c=8, n=100)
    assert delta > 0
    delta, _, _ = paired_difference(b=8, c=2, n=100)
    assert delta < 0


def test_paired_difference_reproduces_the_real_gsm8k_numbers() -> None:
    """From docs/findings-gsm8k.md: 100 cases, 1 weak-only, 1 strong-only."""
    delta, (lo, hi), se = paired_difference(b=1, c=1, n=100)
    assert delta == 0.0
    assert se == pytest.approx(0.0141, abs=1e-4)
    assert lo == pytest.approx(-0.0277, abs=1e-4)
    assert hi == pytest.approx(0.0277, abs=1e-4)


def test_paired_difference_degenerates_when_everything_is_discordant() -> None:
    """CHARACTERISATION of a known weakness. When every case is discordant in
    one direction the Wald interval collapses to zero width and claims
    certainty. This is why the pair verdict uses the exact test and the
    interval carries a reliability flag."""
    delta, (lo, hi), se = paired_difference(b=0, c=12, n=12)
    assert delta == 1.0
    assert se == 0.0
    assert lo == hi == 1.0


def test_paired_difference_with_no_cases() -> None:
    delta, (lo, hi), se = paired_difference(0, 0, 0)
    assert (delta, lo, hi, se) == (0.0, -1.0, 1.0, 0.0)


def test_paired_difference_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="exceed"):
        paired_difference(b=6, c=6, n=10)


# --- minimum detectable effect -------------------------------------------


def test_mde_matches_the_formula() -> None:
    p_discordant, n = 0.02, 100
    expected = (Z_95 + Z_80_POWER) * math.sqrt(p_discordant / n)
    assert minimum_detectable_effect(p_discordant, n) == pytest.approx(expected)


def test_mde_on_the_real_gsm8k_run_is_four_points() -> None:
    """The number that says the GSM8K result cannot support a model ranking:
    2% discordance over 100 cases resolves nothing below ~4 pp, and the
    observed gap was 0.7 pp."""
    assert minimum_detectable_effect(0.02, 100) == pytest.approx(0.040, abs=0.001)


def test_mde_halves_when_n_quadruples() -> None:
    """It scales as 1/sqrt(n), so buying precision is expensive."""
    assert minimum_detectable_effect(0.02, 400) == pytest.approx(
        minimum_detectable_effect(0.02, 100) / 2
    )


def test_mde_shrinks_with_n_and_grows_with_discordance() -> None:
    assert minimum_detectable_effect(0.1, 1000) < minimum_detectable_effect(0.1, 100)
    assert minimum_detectable_effect(0.4, 100) > minimum_detectable_effect(0.1, 100)


def test_mde_is_zero_when_undefined() -> None:
    """No discordance means the quantity is undefined. Callers must render this
    as "no information", never as "can detect any difference"."""
    assert minimum_detectable_effect(0.0, 100) == 0.0
    assert minimum_detectable_effect(0.5, 0) == 0.0


# --- the documented constants --------------------------------------------


def test_alpha_is_tied_to_the_interval_level() -> None:
    """They must not drift apart: a pair called separated at alpha should not
    contradict a 95% interval reported beside it."""
    assert ALPHA == pytest.approx(0.05)


def test_normal_approximation_floor_is_the_conventional_ten() -> None:
    assert NORMAL_APPROX_MIN_DISCORDANT == 10
