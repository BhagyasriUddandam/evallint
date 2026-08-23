"""Inter-rater agreement statistics.

Pure functions, standard library only. Used to report how much judges disagree,
which is the only defensible use of an LLM judge in this project: the judge is
the OBJECT of measurement rather than a trusted oracle.

## Raw agreement alone is misleading, and must never be reported alone

If 95% of cases are unambiguous and two judges both say "not ambiguous" almost
always, they will agree about 90% of the time by chance. Reporting "90%
agreement" then implies a reliability that does not exist. Chance-corrected
measures divide out that baseline:

    kappa = (p_observed - p_expected) / (1 - p_expected)

## The kappa paradox is handled, not hidden

When every rater assigns every item to the same category, p_expected = 1 and
kappa is 0/0 — undefined, not zero and not one. That happens routinely here: a
clean eval set where all judges say "not ambiguous" for every case. Raw
agreement is then 1.0 and genuinely informative, while kappa cannot be computed
at all. These functions return None in that case, and the caller is expected to
say "undefined" rather than substitute a number.

High agreement also does NOT imply correctness. Judges drawn from one model
family share their training and their blind spots, so correlated error is
indistinguishable from reliability. Only a human-labelled sample can separate
them, and that is stated wherever a kappa is reported without one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

__all__ = [
    "cohens_kappa",
    "kendalls_w",
    "fleiss_kappa",
    "pairwise_agreement",
    "raw_agreement",
]


def raw_agreement(ratings: Sequence[Sequence[object]]) -> float:
    """Share of items on which every rater agreed.

    Args:
        ratings: one sequence per ITEM, each holding that item's ratings.

    The blunt measure. Reported only beside a chance-corrected one, because on
    a skewed distribution it is high whatever the raters do.
    """
    items = [r for r in ratings if r]
    if not items:
        return 1.0
    unanimous = sum(1 for r in items if len(set(r)) == 1)
    return unanimous / len(items)


def pairwise_agreement(ratings: Sequence[Sequence[object]]) -> float:
    """Share of rater PAIRS that agreed, averaged over items.

    More informative than `raw_agreement` with three or more raters: two of
    three agreeing is not the same as a three-way split, and unanimity alone
    cannot tell them apart.
    """
    totals = 0
    matches = 0
    for item in ratings:
        n = len(item)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                totals += 1
                matches += item[i] == item[j]
    return matches / totals if totals else 1.0


def cohens_kappa(a: Sequence[object], b: Sequence[object]) -> float | None:
    """Cohen's kappa for exactly two raters.

    Returns None when expected agreement is 1, i.e. both raters used a single
    category throughout. Kappa is 0/0 there — undefined rather than perfect.
    """
    if len(a) != len(b):
        raise ValueError(f"raters must rate the same items, got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        return None

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        (count_a[category] / n) * (count_b[category] / n)
        for category in set(count_a) | set(count_b)
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(ratings: Sequence[Sequence[object]]) -> float | None:
    """Fleiss' kappa for a fixed number of raters per item.

        P_i     = [ sum_j n_ij^2 - n ] / [ n (n - 1) ]
        P_bar   = mean_i P_i
        p_j     = sum_i n_ij / (N n)
        P_e     = sum_j p_j^2
        kappa   = (P_bar - P_e) / (1 - P_e)

    Args:
        ratings: one sequence per item, each of the SAME length (the number of
            raters). Items with fewer than two ratings are dropped, since a
            single rating carries no agreement information.

    Returns None when P_e >= 1 — every rater used one category throughout — or
    when fewer than two items remain. Undefined, not zero.
    """
    items = [list(r) for r in ratings if len(r) >= 2]
    if len(items) < 2:
        return None
    n_raters = len(items[0])
    if any(len(item) != n_raters for item in items):
        raise ValueError(
            "Fleiss' kappa needs the same number of raters on every item; use "
            "Krippendorff's alpha for ragged data"
        )

    categories = sorted({str(v) for item in items for v in item})
    per_item_agreement = []
    category_totals = {category: 0 for category in categories}
    for item in items:
        counts = Counter(str(v) for v in item)
        for category, count in counts.items():
            category_totals[category] += count
        squared = sum(count * count for count in counts.values())
        per_item_agreement.append(
            (squared - n_raters) / (n_raters * (n_raters - 1))
        )

    p_bar = sum(per_item_agreement) / len(items)
    total_ratings = len(items) * n_raters
    p_expected = sum(
        (total / total_ratings) ** 2 for total in category_totals.values()
    )
    if p_expected >= 1.0:
        return None
    return (p_bar - p_expected) / (1.0 - p_expected)


def kendalls_w(rankings: Sequence[Sequence[float]]) -> float | None:
    """Kendall's coefficient of concordance, for agreement between rankings.

        R_i  = sum of ranks given to item i across all raters
        Rbar = m(n + 1) / 2
        S    = sum_i (R_i - Rbar)^2
        W    = 12 S / ( m^2 (n^3 - n) )

    Args:
        rankings: one sequence per RATER (here, per run), each holding that
            rater's rank for every item in a consistent order.

    W is 1 for perfect agreement and 0 when the rank sums are all equal, i.e.
    the rankings are as inconsistent as they can be. Used for model ranking
    stability: each run ranks the models, and W says how much those rankings
    agree.

    Returns None when there are fewer than two raters or fewer than two items,
    because concordance between one ranking and nothing is undefined rather
    than perfect.

    NO TIE CORRECTION is applied. Tied ranks deflate W, so a reported value is a
    lower bound when ties are present -- conservative, and stated wherever it is
    used.
    """
    raters = [list(r) for r in rankings]
    m = len(raters)
    if m < 2:
        return None
    n = len(raters[0])
    if n < 2:
        return None
    if any(len(r) != n for r in raters):
        raise ValueError("every rater must rank the same number of items")

    rank_sums = [sum(rater[i] for rater in raters) for i in range(n)]
    mean_rank_sum = m * (n + 1) / 2.0
    s = sum((total - mean_rank_sum) ** 2 for total in rank_sums)
    denominator = m * m * (n**3 - n)
    if denominator == 0:
        return None
    return 12.0 * s / denominator
