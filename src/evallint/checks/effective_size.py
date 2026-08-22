"""Redundancy-adjusted coverage — how many distinct scenarios an eval contains.

## What this is, and what it is deliberately not

An eval advertising 500 cases may contain 312 distinct scenarios. The gap
matters because an averaged score gives every case equal weight, so a scenario
appearing eight times counts eight times.

**This is NOT a statistical effective sample size, and it is not presented as
one.** The terminology used throughout is `effective scenario count`,
`redundancy-adjusted coverage` and `independent-case estimate`, because that is
what the method supports.

Why the stronger claim would be wrong. Kish's effective sample size,

    n_eff = (sum_i w_i)^2 / sum_i w_i^2

reduces to exactly the cluster count when each case is weighted by the inverse
of its cluster size. That reduction is not a bonus: it holds precisely because
the weighting ASSUMES within-cluster correlation of 1. Two paraphrases of the
same question are not perfectly redundant — a model can get one right and the
other wrong — so treating them as one observation is the most pessimistic
reading available, not a measurement.

A genuine effective sample size needs a sampling design and an estimated
intra-cluster correlation. Neither exists here: the clusters come from a
similarity threshold, which is a heuristic. So instead of one number this
reports a BRACKET:

    lower bound  = cluster count   assumes cases in a cluster are perfectly
                                   redundant (rho = 1)
    upper bound  = raw case count  assumes every case is independent (rho = 0)

The truth lies between. The lower bound is quoted as the headline figure because
it is the one that cannot flatter the eval, and the bracket is always shown
beside it.

## Threshold sensitivity is part of the answer

The cluster count depends on the similarity threshold, so the estimate is
computed at several and the spread reported. If the count is stable across
0.80-0.95, the threshold is not load-bearing for your data. If it swings, the
estimate is threshold-dependent and you have been told rather than left to
assume.

## What to do with it

The design effect, `raw / effective`, is the factor by which confidence
intervals computed on the raw count are too narrow. An interval from
`compare_models` should be widened by roughly its square root when the eval has
clusters — cases in a cluster are not independent observations, and every
interval in this library assumes they are.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..schema import EvalSet

__all__ = [
    "DEFAULT_SENSITIVITY_THRESHOLDS",
    "LARGE_CLUSTER_MIN",
    "EffectiveSizeEstimate",
    "estimate_effective_size",
    "estimate_from_clusters",
]

#: A group of this many or more cases is reported as a "large cluster". A PAIR
#: is the smallest possible redundancy and is usually benign — two phrasings of
#: one question barely move an average. Three or more is where a single scenario
#: starts to carry visible weight in an aggregate. A convention, and labelled as
#: one in the output.
LARGE_CLUSTER_MIN = 3

#: Thresholds at which the estimate is recomputed, to show whether the choice of
#: threshold is load-bearing.
DEFAULT_SENSITIVITY_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


@dataclass(frozen=True, slots=True)
class EffectiveSizeEstimate:
    """A redundancy-adjusted coverage estimate. An ESTIMATE, not a measurement."""

    raw_cases: int
    scenario_clusters: int
    large_clusters: int
    largest_cluster: int
    redundant_cases: int
    potential_redundancy: float
    #: The bracket. Lower assumes perfect within-cluster redundancy, upper
    #: assumes full independence. The truth is between.
    independent_case_estimate: tuple[int, int]
    design_effect: float
    interval_inflation: float
    cluster_size_distribution: dict[int, int]
    threshold: float | None
    threshold_sensitivity: dict[str, int]
    threshold_is_load_bearing: bool | None
    method: str
    caveats: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_cases": self.raw_cases,
            "scenario_clusters": self.scenario_clusters,
            "large_clusters": self.large_clusters,
            "largest_cluster": self.largest_cluster,
            "redundant_cases": self.redundant_cases,
            "potential_redundancy": self.potential_redundancy,
            "independent_case_estimate": list(self.independent_case_estimate),
            "design_effect": self.design_effect,
            "interval_inflation": self.interval_inflation,
            "cluster_size_distribution": {
                str(k): v for k, v in sorted(self.cluster_size_distribution.items())
            },
            "threshold": self.threshold,
            "threshold_sensitivity": dict(self.threshold_sensitivity),
            "threshold_is_load_bearing": self.threshold_is_load_bearing,
            "method": self.method,
            "caveats": list(self.caveats),
        }

    def render(self) -> str:
        lines = [
            f"Raw cases: {self.raw_cases}",
            f"Semantic clusters: {self.scenario_clusters}",
            f"Large clusters (>= {LARGE_CLUSTER_MIN} cases): {self.large_clusters}",
            f"Potential redundancy: {self.potential_redundancy:.0%}",
        ]
        low, high = self.independent_case_estimate
        lines.append(
            f"Redundancy-adjusted scenario coverage: ~{low} distinct scenarios "
            f"(estimate; plausible range {low}-{high})"
        )
        lines.append(
            f"Design effect: {self.design_effect:.2f} — intervals computed on "
            f"{self.raw_cases} cases are roughly {self.interval_inflation:.2f}x "
            f"too narrow"
        )
        if self.threshold_sensitivity:
            spread = ", ".join(
                f"{t}: {c}" for t, c in sorted(self.threshold_sensitivity.items())
            )
            verdict = (
                "threshold-dependent, so treat the figure above as one reading"
                if self.threshold_is_load_bearing
                else "stable, so the threshold is not load-bearing here"
            )
            lines.append(f"Clusters by threshold — {spread}. That is {verdict}")
        lines.append("")
        lines.append("THIS IS AN ESTIMATE, NOT A STATISTICAL EFFECTIVE SAMPLE SIZE.")
        lines.extend(f"  · {c}" for c in self.caveats)
        return "\n".join(lines)


CAVEATS = (
    "The lower bound assumes cases in a cluster are PERFECTLY redundant. They "
    "are not: a model can pass one paraphrase and fail another. So the true "
    "independent-information count is higher than the headline figure, and the "
    "figure is the conservative end of a range on purpose.",
    "Clustering comes from a similarity threshold, which is a heuristic and not "
    "a sampling design. Kish's effective-sample-size formula reduces to the "
    "cluster count under inverse-cluster weights, but only because those "
    "weights assume within-cluster correlation of 1 — the reduction is an "
    "assumption, not a derivation.",
    "Redundancy is not a defect. Consistency tests reuse an input deliberately, "
    "template families cover an operation across many operands, and regression "
    "suites repeat a scenario that once broke. This counts scenarios; it does "
    "not recommend deleting any case.",
    "Clusters are connected components, so a chain A~B~C groups all three even "
    "when A and C are not themselves similar. That inflates redundancy and "
    "therefore makes the estimate MORE conservative, not less.",
    "Only the fields the redundancy check compared are considered. Two cases "
    "testing genuinely different things through near-identical text are counted "
    "as one scenario.",
)


def estimate_from_clusters(
    multi_case_cluster_sizes: Sequence[int],
    raw_cases: int,
    *,
    large_cluster_min: int = LARGE_CLUSTER_MIN,
    threshold: float | None = None,
    threshold_sensitivity: dict[str, int] | None = None,
    method: str = "connected components over similarity and exact matching",
) -> EffectiveSizeEstimate:
    """Build the estimate from cluster sizes. Pure, so it needs no embedder.

    Args:
        multi_case_cluster_sizes: sizes of the groups with two or more members.
            Singletons are derived from `raw_cases`, so they must not be
            included.
        raw_cases: total number of cases in the eval.
    """
    if raw_cases < 0:
        raise ValueError(f"raw_cases must be non-negative, got {raw_cases}")
    sizes = [int(s) for s in multi_case_cluster_sizes]
    if any(s < 2 for s in sizes):
        raise ValueError(
            "multi_case_cluster_sizes must contain only groups of 2 or more; "
            "singletons are derived from raw_cases"
        )
    clustered = sum(sizes)
    if clustered > raw_cases:
        raise ValueError(
            f"clusters cover {clustered} cases but the eval has {raw_cases}"
        )

    singletons = raw_cases - clustered
    scenario_clusters = singletons + len(sizes)
    redundant = raw_cases - scenario_clusters
    redundancy = redundant / raw_cases if raw_cases else 0.0

    distribution: dict[int, int] = {1: singletons} if singletons else {}
    for size in sizes:
        distribution[size] = distribution.get(size, 0) + 1

    # Design effect. Guard the degenerate case: with no cases there is nothing
    # to inflate, and 1.0 is the honest neutral value.
    design_effect = raw_cases / scenario_clusters if scenario_clusters else 1.0

    load_bearing: bool | None = None
    if threshold_sensitivity:
        counts = list(threshold_sensitivity.values())
        spread = max(counts) - min(counts)
        # "Load-bearing" if moving the threshold across the tested range changes
        # the cluster count by more than a tenth of the raw case count. A
        # convention, chosen so that a handful of borderline pairs does not get
        # called instability.
        load_bearing = spread > max(1, raw_cases // 10)

    return EffectiveSizeEstimate(
        raw_cases=raw_cases,
        scenario_clusters=scenario_clusters,
        large_clusters=sum(1 for s in sizes if s >= large_cluster_min),
        largest_cluster=max(sizes) if sizes else 1,
        redundant_cases=redundant,
        potential_redundancy=redundancy,
        independent_case_estimate=(scenario_clusters, raw_cases),
        design_effect=design_effect,
        interval_inflation=math.sqrt(design_effect),
        cluster_size_distribution=distribution,
        threshold=threshold,
        threshold_sensitivity=dict(threshold_sensitivity or {}),
        threshold_is_load_bearing=load_bearing,
        method=method,
        caveats=CAVEATS,
    )


def estimate_effective_size(
    eval_set: EvalSet,
    *,
    threshold: float = 0.85,
    embedder: Callable[..., Any] | None = None,
    semantic: bool = True,
    compare: str = "input+expected",
    sensitivity: bool = True,
    sensitivity_thresholds: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS,
) -> EffectiveSizeEstimate:
    """Run the redundancy check and turn its clusters into an estimate.

    The embedder is memoised across thresholds, so computing the sensitivity
    curve costs one embedding pass rather than one per threshold.
    """
    from .redundancy import RedundancyCheck

    memo: dict[tuple[str, ...], Any] = {}

    def cached(texts):
        key = tuple(texts)
        if key not in memo:
            base = embedder
            if base is None:
                # Build the default embedder once, through the check's own lazy
                # loader, so the optional-extra error message is reused.
                base = RedundancyCheck(
                    threshold=threshold, compare=compare
                )._get_embedder()
            memo[key] = base(texts)
        return memo[key]

    def clusters_at(tau: float) -> tuple[list[int], str]:
        check = RedundancyCheck(
            threshold=tau,
            compare=compare,
            embedder=cached if semantic else None,
            semantic=semantic,
        )
        stats = check.run(eval_set).stats
        return (
            [c["size"] for c in stats["clusters"]],
            ", ".join(stats["levels_run"]),
        )

    sizes, levels = clusters_at(threshold)

    curve: dict[str, int] = {}
    if sensitivity and semantic:
        for tau in sensitivity_thresholds:
            tau_sizes, _ = clusters_at(tau)
            covered = sum(tau_sizes)
            curve[f"{tau:.2f}"] = (len(eval_set) - covered) + len(tau_sizes)

    return estimate_from_clusters(
        sizes,
        len(eval_set),
        threshold=threshold,
        threshold_sensitivity=curve,
        method=f"connected components over: {levels}",
    )
