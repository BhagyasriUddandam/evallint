"""Statistics for model-separation analysis.

Pure functions, no dependencies beyond the standard library. Deliberately NOT
numpy: `discrimination.py` never imports it, so that a scorer built on any
stack works without evallint having an opinion about array libraries, and these
helpers are called from the same code path.

Everything here is small enough to check against a textbook, and every constant
is named and justified rather than inlined.

WHY THESE ESTIMATORS AND NOT THE OBVIOUS ONES

  * Wilson, not the normal approximation, for a proportion. Eval accuracies live
    at the extremes -- a real run in this repo measured 98.0% on 100 cases -- and
    the normal interval there is [0.943, 1.017], which includes impossible
    values. Wilson stays inside [0, 1] by construction and is well behaved at
    p = 0 and p = 1, where the normal interval collapses to zero width and
    claims certainty from no evidence.

  * McNemar, not two independent-proportion tests, for comparing two models.
    Both models are scored on the SAME cases, so the samples are paired. Testing
    them as independent throws away the pairing, inflates the variance, and
    hides real differences. Only the discordant cases carry information about
    which model is better; the ones both models get right or both get wrong
    cancel.

  * An exact binomial tail rather than the chi-squared approximation, for small
    discordant counts. A 100-case eval comparing two strong models may have two
    discordant cases in total, and the chi-squared approximation is not valid
    there.
"""

from __future__ import annotations

import math

__all__ = [
    "ALPHA",
    "EXACT_TEST_MAX_N",
    "NORMAL_APPROX_MIN_DISCORDANT",
    "Z_80_POWER",
    "Z_95",
    "mcnemar_exact_p",
    "minimum_detectable_effect",
    "paired_difference",
    "wilson_interval",
]

# Two-sided normal quantile for a 95% interval. 95% is a convention, not a
# derived value; it is stated in every result so a reader can substitute their
# own. Callers may pass any z.
Z_95 = 1.959963984540054

# One-sided normal quantile at 80% power, the conventional floor for "this
# comparison was worth running". Used only for the minimum detectable effect.
Z_80_POWER = 0.8416212335729143

# Above this many discordant pairs the exact test is replaced by the normal
# approximation. 2**n and math.comb are exact but grow, and by n = 1000 the
# approximation agrees to far more precision than any decision here needs.
EXACT_TEST_MAX_N = 1000

#: Significance level for the paired test, tied to the 95% interval level so
#: the two cannot drift apart.
ALPHA = 0.05

#: Below this many discordant pairs the normal-approximation interval for the
#: paired difference is not trustworthy, and is reported as indicative only.
#: Ten is the conventional rule of thumb for a binomial normal approximation.
#: It matters here: with 3 discordant cases all favouring one model, the Wald
#: interval excluded zero while the EXACT test gave p = 0.25. The exact test is
#: right. And when every comparable case is discordant in one direction the
#: Wald interval collapses to zero width and claims certainty, which is the
#: same defect that rules out the normal approximation for a proportion.
NORMAL_APPROX_MIN_DISCORDANT = 10


def wilson_interval(
    successes: int, trials: int, z: float = Z_95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/4n^2 )

    Returns (0.0, 1.0) for zero trials: no evidence means no constraint, which
    is the honest answer rather than a zero-width interval at 0.

    The interval is clamped to [0, 1]. That clamping is cosmetic -- Wilson does
    not stray outside without it except through floating-point error -- but it
    guarantees a caller never has to render a negative percentage.
    """
    if trials <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > trials:
        raise ValueError(
            f"successes must be within [0, trials], got {successes} of {trials}"
        )
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denominator
    half = (z / denominator) * math.sqrt(
        p * (1.0 - p) / trials + z2 / (4 * trials * trials)
    )
    lo, hi = centre - half, centre + half

    # Snap the boundaries, which are EXACT in the algebra and only inexact in
    # floating point. At p = 1 the variance term vanishes, so
    #   centre + half = (1 + z^2/2n + z^2/2n) / (1 + z^2/n) = 1
    # identically, and symmetrically centre - half = 0 at p = 0. Without this
    # the upper bound comes back as 0.9999999999999999, which is not wrong
    # enough to matter in a report but is wrong enough to fail an exact
    # comparison and to mislead anyone testing against the closed form.
    if successes == trials:
        hi = 1.0
    if successes == 0:
        lo = 0.0
    return (max(0.0, lo), min(1.0, hi))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided p-value for a paired binary comparison.

    Args:
        b: cases the FIRST model passed and the second failed.
        c: cases the SECOND model passed and the first failed.

    Under the null hypothesis that the two models are equally likely to be the
    one that succeeds on a discordant case, the count b is Binomial(b + c, 0.5).
    The concordant cases -- both right, both wrong -- carry no information about
    which model is better and correctly do not appear.

    Returns 1.0 when there are no discordant cases: two models that never
    disagree provide no evidence of a difference, which is not the same as
    evidence they are identical.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got {b}, {c}")
    n = b + c
    if n == 0:
        return 1.0
    smaller = min(b, c)
    if n <= EXACT_TEST_MAX_N:
        tail = sum(math.comb(n, i) for i in range(smaller + 1)) / (2.0**n)
    else:
        # Normal approximation with a continuity correction, used only where the
        # exact computation is wasteful and the two agree closely anyway.
        centre = n / 2.0
        sd = math.sqrt(n) / 2.0
        z = (abs(smaller - centre) - 0.5) / sd
        tail = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(1.0, 2.0 * tail)


def paired_difference(
    b: int, c: int, n: int, z: float = Z_95
) -> tuple[float, tuple[float, float], float]:
    """Difference in pass rate between two models scored on the same cases.

    Args:
        b: cases the first (declared weaker) model passed and the second failed.
        c: cases the second (declared stronger) model passed and the first failed.
        n: total comparable cases, including the concordant ones.

    Returns (delta, (lo, hi), standard_error) where delta = (c - b)/n, positive
    meaning the second model did better.

        Var(delta) = [ p_b + p_c - delta^2 ] / n

    which is the standard variance for the difference of paired proportions. The
    concordant cases enter only through n, which is correct: they pin down how
    much of the eval saw no difference at all, and a difference of two cases out
    of ten means something very different from two out of a thousand.
    """
    if n <= 0:
        return (0.0, (-1.0, 1.0), 0.0)
    if b + c > n:
        raise ValueError(f"discordant cases {b + c} exceed comparable cases {n}")
    delta = (c - b) / n
    p_discordant = (b + c) / n
    # Clamped at zero: the expression is non-negative in exact arithmetic, but
    # floating point can produce a tiny negative when delta^2 ~= p_discordant.
    variance = max(0.0, (p_discordant - delta * delta) / n)
    standard_error = math.sqrt(variance)
    return (delta, (delta - z * standard_error, delta + z * standard_error),
            standard_error)


def minimum_detectable_effect(
    p_discordant: float, n: int, z_alpha: float = Z_95, z_beta: float = Z_80_POWER
) -> float:
    """Smallest pass-rate difference this eval could detect, as a proportion.

        MDE ~= (z_alpha + z_beta) * sqrt( p_discordant / n )

    This is the single most useful number in the analysis, because it says what
    the eval CANNOT do. A 100-case eval with 2% discordance has an MDE of about
    4 percentage points, so a reported 0.7 point gap on such a set is inside the
    noise and no amount of interpretation will get it out.

    Note the dependence on observed discordance: two models that agree almost
    everywhere are cheap to distinguish IF they ever disagree, because the
    denominator of the paired test is the discordant set. Returns 0.0 when there
    is no discordance, which should be read as "undefined, this eval saw no
    disagreement at all" rather than "can detect anything".
    """
    if n <= 0 or p_discordant <= 0:
        return 0.0
    return (z_alpha + z_beta) * math.sqrt(p_discordant / n)
