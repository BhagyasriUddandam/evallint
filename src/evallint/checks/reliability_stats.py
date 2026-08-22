"""Statistics for evaluator reliability, with assumption gates.

Standard library only. Every function returns a `MetricResult` rather than a
bare number, because a reliability figure without its sample size, its interval
and its assumptions is not interpretable — and this module exists to judge the
judges, so an unqualified number here is worse than none.

## Refusing is a supported outcome

`MetricResult.value is None` with `unmet_reason` set is the normal way these
functions decline. A metric whose assumptions are violated does not get computed
and rounded — it gets refused, with the reason stated:

  * Pearson on a constant series is 0/0. Reported as undefined, never 0.0.
  * Krippendorff's alpha with only one category in use has zero expected
    disagreement, so alpha is 0/0.
  * Spearman on data that is almost all ties is dominated by the tie correction.
  * A bootstrap interval from six observations is arithmetic, not inference.

Returning 0.0 in any of those cases would claim the judges were no better than
chance; returning 1.0 would claim perfection. Both would be inventions.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MIN_BOOTSTRAP_SAMPLE",
    "MIN_CORRELATION_SAMPLE",
    "MetricResult",
    "bootstrap_interval",
    "krippendorff_alpha",
    "pearson_r",
    "spearman_rho",
]

#: Below this many paired observations a correlation coefficient is dominated by
#: which points happened to be sampled. Three is the arithmetic minimum; eight is
#: the point below which the interval is wider than the [-1, 1] range is useful.
MIN_CORRELATION_SAMPLE = 8

#: Below this many observations a percentile bootstrap is resampling too small a
#: set to approximate the sampling distribution. It would still produce two
#: numbers, which is the danger.
MIN_BOOTSTRAP_SAMPLE = 10

#: Share of tied values above which Spearman's rho is reported as unreliable.
#: Binary judge verdicts are ~100% ties by construction, which is exactly the
#: case that must not silently produce a correlation.
MAX_TIE_SHARE = 0.5


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One statistic, with everything needed to interpret it.

    `value is None` means the metric was NOT computed. Check `unmet_reason`.
    """

    metric: str
    value: float | None
    n: int
    interpretation: str
    limitations: tuple[str, ...] = ()
    interval: tuple[float, float] | None = None
    level: float | None = None
    unmet_reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def computed(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "n": self.n,
            "interval": list(self.interval) if self.interval else None,
            "level": self.level,
            "interpretation": self.interpretation,
            "limitations": list(self.limitations),
            "unmet_reason": self.unmet_reason,
            "detail": dict(self.detail),
        }


def _refused(metric: str, n: int, reason: str, limitations: Sequence[str] = ()) -> MetricResult:
    return MetricResult(
        metric=metric,
        value=None,
        n=n,
        interpretation=f"not computed: {reason}",
        limitations=tuple(limitations),
        unmet_reason=reason,
    )


# --- Krippendorff's alpha ---------------------------------------------------


def krippendorff_alpha(
    units: Sequence[Sequence[Any]], level: str = "nominal"
) -> MetricResult:
    """Krippendorff's alpha for reliability across any number of raters.

        alpha = 1 - D_o / D_e
        D_o = (1/n)       * sum_c sum_k  o_ck * delta(c, k)
        D_e = (1/(n(n-1))) * sum_c sum_k  n_c * n_k * delta(c, k)

    where the coincidence matrix `o` is built by giving every ORDERED pair of
    ratings within a unit a weight of 1/(m_u - 1).

    Chosen over Fleiss' kappa when the data is ragged: alpha tolerates a
    different number of ratings per unit, which is what happens when a judge
    errors on some cases and not others. Fleiss' kappa cannot.

    Args:
        units: one sequence per item, holding that item's ratings. Units with
            fewer than two ratings are dropped — a single rating carries no
            information about agreement.
        level: "nominal" (delta = 0/1) or "interval" (delta = squared
            difference). "ordinal" is refused rather than approximated.
    """
    metric = f"krippendorff_alpha_{level}"
    if level not in ("nominal", "interval"):
        return _refused(
            metric,
            0,
            f"level {level!r} is not implemented. Ordinal alpha needs a "
            "rank-based delta that this module does not provide; use 'nominal' "
            "for categories or 'interval' for numeric scores rather than "
            "accepting an approximation",
        )

    usable = [list(u) for u in units if len([v for v in u if v is not None]) >= 2]
    usable = [[v for v in u if v is not None] for u in usable]
    if len(usable) < 2:
        return _refused(
            metric, len(usable), "fewer than two units have two or more ratings"
        )

    if level == "interval":
        try:
            usable = [[float(v) for v in u] for u in usable]
        except (TypeError, ValueError):
            return _refused(
                metric, len(usable), "interval alpha needs numeric ratings"
            )

        def delta(a: Any, b: Any) -> float:
            return (a - b) ** 2
    else:

        def delta(a: Any, b: Any) -> float:
            return 0.0 if a == b else 1.0

    # Coincidence matrix. Every ordered pair inside a unit contributes
    # 1/(m_u - 1), which is what makes units with more raters count once rather
    # than quadratically.
    coincidence: dict[tuple[Any, Any], float] = {}
    for unit in usable:
        m = len(unit)
        weight = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                key = (unit[i], unit[j])
                coincidence[key] = coincidence.get(key, 0.0) + weight

    total = sum(coincidence.values())
    if total <= 1:
        return _refused(metric, len(usable), "too few paired ratings")

    marginals: dict[Any, float] = {}
    for (a, _), weight in coincidence.items():
        marginals[a] = marginals.get(a, 0.0) + weight

    observed = sum(w * delta(a, b) for (a, b), w in coincidence.items()) / total
    expected = sum(
        na * nb * delta(a, b)
        for a, na in marginals.items()
        for b, nb in marginals.items()
    ) / (total * (total - 1))

    if expected <= 0:
        return _refused(
            metric,
            len(usable),
            "expected disagreement is zero — every rating is in one category, "
            "so alpha is 0/0. Undefined, not zero and not one: raters who never "
            "made a distinction have not demonstrated reliability",
            limitations=(
                "Raw agreement is 100% here and is the only figure that can be "
                "reported.",
            ),
        )

    alpha = 1.0 - observed / expected
    return MetricResult(
        metric=metric,
        value=alpha,
        n=len(usable),
        interpretation=_interpret_alpha(alpha),
        limitations=(
            "Alpha measures CONSENSUS, not correctness. Raters sharing a bias "
            "agree with each other and are wrong together, which is "
            "indistinguishable from reliability without an independent "
            "reference.",
            "Conventional thresholds (0.8 good, 0.667 tentative) are field "
            "conventions with no derivation, and are reported as such.",
        ),
        detail={
            "level": level,
            "observed_disagreement": observed,
            "expected_disagreement": expected,
            "n_categories": len(marginals),
        },
    )


def _interpret_alpha(alpha: float) -> str:
    if alpha >= 0.8:
        band = "at or above the conventional 0.8 threshold for reliable data"
    elif alpha >= 0.667:
        band = "in the 0.667-0.8 band, conventionally 'tentative conclusions only'"
    elif alpha > 0:
        band = "below 0.667: agreement exceeds chance but not by enough to rely on"
    elif alpha == 0:
        band = "exactly chance-level agreement"
    else:
        band = "NEGATIVE: raters disagree more than chance would predict, which "
        band += "usually means a systematic difference in how they read the task"
    return f"alpha = {alpha:.3f}, {band}. This is consensus, not correctness"


# --- correlation -----------------------------------------------------------


def pearson_r(x: Sequence[float], y: Sequence[float]) -> MetricResult:
    """Pearson product-moment correlation, with assumption gates.

    Refuses on: fewer than MIN_CORRELATION_SAMPLE pairs, or zero variance in
    either series. A constant series makes r = 0/0, and reporting 0.0 there
    would claim the two judges are uncorrelated when in fact one of them never
    varied.
    """
    metric = "pearson_r"
    if len(x) != len(y):
        return _refused(metric, 0, f"unequal lengths {len(x)} and {len(y)}")
    n = len(x)
    if n < MIN_CORRELATION_SAMPLE:
        return _refused(
            metric,
            n,
            f"{n} pairs is below the {MIN_CORRELATION_SAMPLE}-pair minimum; a "
            "correlation from this few points is determined by which points "
            "were sampled",
        )

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [a - mean_x for a in x]
    dy = [b - mean_y for b in y]
    var_x = sum(a * a for a in dx)
    var_y = sum(b * b for b in dy)
    if var_x == 0 or var_y == 0:
        which = "first" if var_x == 0 else "second"
        return _refused(
            metric,
            n,
            f"the {which} series is constant, so r is 0/0. Undefined, not zero: "
            "a judge that gave every case the same score has not been shown to "
            "disagree with anyone",
        )

    r = sum(a * b for a, b in zip(dx, dy, strict=True)) / math.sqrt(var_x * var_y)
    return MetricResult(
        metric=metric,
        value=r,
        n=n,
        interpretation=(
            f"r = {r:.3f}. Measures LINEAR association only; two judges can "
            "rank identically and still show a low r if one is nonlinear in the "
            "other"
        ),
        limitations=(
            "Correlation is not agreement. A judge scoring consistently two "
            "points below another correlates at 1.0 while never agreeing on a "
            "single value — use a kappa or alpha for agreement.",
            "Assumes interval-scale scores. Applied to ordinal ratings it "
            "over-claims.",
        ),
    )


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> MetricResult:
    """Spearman rank correlation, refusing when ties dominate.

    Binary verdicts are almost entirely ties, and a rank correlation there is
    not meaningful. That case is refused rather than reported.
    """
    metric = "spearman_rho"
    if len(x) != len(y):
        return _refused(metric, 0, f"unequal lengths {len(x)} and {len(y)}")
    n = len(x)
    if n < MIN_CORRELATION_SAMPLE:
        return _refused(
            metric, n, f"{n} pairs is below the {MIN_CORRELATION_SAMPLE}-pair minimum"
        )

    for name, series in (("first", x), ("second", y)):
        tied = sum(c for c in Counter(series).values() if c > 1)
        if tied / n > MAX_TIE_SHARE:
            return _refused(
                metric,
                n,
                f"{tied / n:.0%} of the {name} series is tied, above the "
                f"{MAX_TIE_SHARE:.0%} limit. Rank correlation on mostly-tied "
                "data reflects the tie-handling rule more than the data — for "
                "binary verdicts use a kappa instead",
            )

    result = pearson_r(_rank(x), _rank(y))
    if not result.computed:
        return _refused(metric, n, result.unmet_reason or "rank transform failed")
    return MetricResult(
        metric=metric,
        value=result.value,
        n=n,
        interpretation=(
            f"rho = {result.value:.3f}. Monotonic association between the two "
            "rankings, insensitive to scale"
        ),
        limitations=(
            "Rank correlation is not agreement: two judges can rank cases "
            "identically while never assigning the same score.",
            "Ties are averaged, which biases rho towards zero as the tie share "
            "grows.",
        ),
    )


def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks, so ties share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        mean_rank = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = mean_rank
        position = end + 1
    return ranks


# --- bootstrap -------------------------------------------------------------


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    *,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> MetricResult:
    """Percentile bootstrap interval for a statistic of `values`.

    Non-parametric, so it makes no distributional assumption — but it DOES
    assume the observations are independent and that the sample represents the
    population. Neither holds for judge verdicts on a curated eval set, and both
    are stated in the limitations rather than assumed away.

    Refuses below MIN_BOOTSTRAP_SAMPLE observations. Resampling six numbers
    produces two more numbers and no inference.
    """
    metric = "bootstrap_interval"
    n = len(values)
    if n < MIN_BOOTSTRAP_SAMPLE:
        return _refused(
            metric,
            n,
            f"{n} observations is below the {MIN_BOOTSTRAP_SAMPLE} minimum; a "
            "bootstrap from this few points resamples the same handful of "
            "values and reports its own sampling noise",
        )
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")

    statistic = statistic or (lambda v: sum(v) / len(v))
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistic(sample))
    estimates.sort()

    tail = (1.0 - level) / 2.0
    lo = estimates[int(tail * resamples)]
    hi = estimates[min(resamples - 1, int((1.0 - tail) * resamples))]
    point = statistic(list(values))

    return MetricResult(
        metric=metric,
        value=point,
        n=n,
        interval=(lo, hi),
        level=level,
        interpretation=(
            f"{point:.3f}, {level:.0%} percentile bootstrap interval "
            f"[{lo:.3f}, {hi:.3f}] from {resamples} resamples"
        ),
        limitations=(
            "Assumes the observations are independent. Repeated verdicts on the "
            "same case are NOT independent of each other, so an interval "
            "computed across repeats is narrower than the truth.",
            "Assumes this sample represents the population you care about. A "
            "curated eval set is not a random sample of production traffic, so "
            "the interval describes re-running on THIS set.",
            "Percentile method, which is biased for skewed statistics; BCa "
            "would be better and is not implemented.",
        ),
        detail={"resamples": resamples, "seed": seed},
    )
