"""Evaluator reliability — is the grader itself trustworthy?

Every other check in evallint audits the DATASET. This one audits the
EVALUATOR, which is the one place an LLM judge is legitimately involved: the
judge is the object of measurement, not a trusted oracle.

## evallint does not call your judges

You run them and hand over the observations, exactly as with the injected
scorer. `collect_verdicts` and `collect_choices` are conveniences that call
callables YOU supply; the analysis itself takes a plain list of
`JudgeObservation` and never touches a provider.

## Seven measures

    intra_judge_consistency   does one judge agree with ITSELF across repeats
    inter_judge_agreement     do different judges agree, chance-corrected
    position_bias             does a judge favour whichever option came first
    order_sensitivity         does reversing the order flip the verdict
    score_variance            how much do continuous scores move across repeats
    decision_stability        how often does a binary verdict flip
    judge_correlation         are two judges' scores associated at all

## Nothing is computed when its assumptions fail

Each measure returns a `MetricResult` whose `value` is None when the data cannot
support it, with the reason stated. A correlation from four points, a
rank correlation over binary verdicts, an alpha where every rating is identical
— all refused rather than rounded. See `reliability_stats`.

## High agreement is not correctness

Judges from one model family share their training and their blind spots, so
correlated error is indistinguishable from reliability. Nothing here can tell
them apart, and every agreement figure says so. Two findings in this project
were retracted because a model graded its own work; this module is built on the
assumption that a judge is a measuring instrument of unknown calibration.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import EvalCase
from .agreement import cohens_kappa, pairwise_agreement, raw_agreement
from .reliability_stats import (
    MetricResult,
    bootstrap_interval,
    krippendorff_alpha,
    pearson_r,
    spearman_rho,
)
from .separation import mcnemar_exact_p

__all__ = [
    "EvaluatorReliability",
    "JudgeObservation",
    "ReliabilityReport",
    "collect_choices",
    "collect_verdicts",
]


@dataclass(frozen=True, slots=True)
class JudgeObservation:
    """One judgement by one judge on one item.

    `verdict` and `score` are both optional so the same record type covers a
    binary grader and a scoring grader. `chosen` and `presented_order` are for
    pairwise comparisons and drive the position-bias and order-sensitivity
    measures; leave them None for ordinary grading.
    """

    judge: str
    item: str
    repeat: int = 0
    verdict: bool | None = None
    score: float | None = None
    chosen: str | None = None
    presented_order: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ReliabilityReport:
    """Every measure that could be computed, and why the rest could not."""

    n_observations: int
    n_judges: int
    n_items: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    per_judge: dict[str, dict[str, MetricResult]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "n_judges": self.n_judges,
            "n_items": self.n_items,
            "metrics": {k: v.as_dict() for k, v in self.metrics.items()},
            "per_judge": {
                judge: {k: v.as_dict() for k, v in measures.items()}
                for judge, measures in self.per_judge.items()
            },
            "notes": list(self.notes),
            "refused": sorted(
                k for k, v in self.metrics.items() if not v.computed
            ),
        }


# --- collectors (they call YOUR callables) ---------------------------------

VerdictJudge = Callable[[EvalCase], bool | float]
ChoiceJudge = Callable[[EvalCase, Sequence[str]], str]


def collect_verdicts(
    cases: Sequence[EvalCase],
    judges: Mapping[str, VerdictJudge],
    *,
    repeats: int = 3,
) -> list[JudgeObservation]:
    """Call each judge `repeats` times on each case.

    Repeats are what make intra-judge consistency measurable at all: with
    repeats=1 a judge cannot be shown to disagree with itself, and the
    corresponding metric is refused rather than reported as perfect.

    A bool return is recorded as a verdict, a float as a score. A float is ALSO
    recorded as a verdict at the 0.5 boundary, so decision stability is
    available either way — and that boundary is stated in the metric's
    limitations rather than hidden.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    observations: list[JudgeObservation] = []
    for name, judge in judges.items():
        for case in cases:
            for repeat in range(repeats):
                raw = judge(case)
                if isinstance(raw, bool):
                    observations.append(
                        JudgeObservation(
                            judge=name, item=case.id, repeat=repeat, verdict=raw
                        )
                    )
                else:
                    value = float(raw)
                    observations.append(
                        JudgeObservation(
                            judge=name,
                            item=case.id,
                            repeat=repeat,
                            score=value,
                            verdict=value >= 0.5,
                        )
                    )
    return observations


def collect_choices(
    cases: Sequence[EvalCase],
    judges: Mapping[str, ChoiceJudge],
    options: Mapping[str, tuple[str, str]],
    *,
    both_orders: bool = True,
) -> list[JudgeObservation]:
    """Ask each judge to choose between two options, in both presentation orders.

    Both orders is the point: a judge asked once cannot reveal position bias,
    and a judge that picks whichever option came first will look perfectly
    consistent under a single ordering.
    """
    observations: list[JudgeObservation] = []
    for name, judge in judges.items():
        for case in cases:
            pair = options.get(case.id)
            if pair is None:
                continue
            orders = [pair, (pair[1], pair[0])] if both_orders else [pair]
            for repeat, order in enumerate(orders):
                observations.append(
                    JudgeObservation(
                        judge=name,
                        item=case.id,
                        repeat=repeat,
                        chosen=judge(case, order),
                        presented_order=tuple(order),
                    )
                )
    return observations


class EvaluatorReliability:
    """Computes the seven reliability measures from judge observations."""

    def __init__(self, *, level: float = 0.95, bootstrap_resamples: int = 2000) -> None:
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), got {level}")
        self.level = level
        self.bootstrap_resamples = bootstrap_resamples

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _group(
        observations: Sequence[JudgeObservation],
    ) -> dict[tuple[str, str], list[JudgeObservation]]:
        grouped: dict[tuple[str, str], list[JudgeObservation]] = {}
        for observation in observations:
            grouped.setdefault((observation.judge, observation.item), []).append(
                observation
            )
        return grouped

    def _bootstrap(self, values: Sequence[float]) -> MetricResult:
        return bootstrap_interval(
            values, resamples=self.bootstrap_resamples, level=self.level
        )

    # -- 1. intra-judge consistency -------------------------------------
    def _intra_judge(
        self, judge: str, grouped: dict[tuple[str, str], list[JudgeObservation]]
    ) -> MetricResult:
        """Does this judge agree with ITSELF when asked the same thing twice?"""
        units = [
            [o.verdict for o in obs if o.verdict is not None]
            for (name, _), obs in grouped.items()
            if name == judge
        ]
        repeated = [u for u in units if len(u) >= 2]
        if not repeated:
            return MetricResult(
                metric="intra_judge_consistency",
                value=None,
                n=len(units),
                interpretation=(
                    "not computed: every item was judged once, so this judge "
                    "cannot be shown to disagree with itself"
                ),
                unmet_reason="repeats < 2",
                limitations=(
                    "A single evaluation per item makes intra-judge consistency "
                    "unmeasurable. This is NOT evidence of perfect consistency.",
                ),
            )
        unanimous = sum(1 for u in repeated if len(set(u)) == 1)
        share = unanimous / len(repeated)
        alpha = krippendorff_alpha(repeated, level="nominal")
        return MetricResult(
            metric="intra_judge_consistency",
            value=share,
            n=len(repeated),
            interval=None,
            interpretation=(
                f"{share:.0%} of {len(repeated)} items received an identical "
                f"verdict on every repeat. Krippendorff alpha across repeats: "
                + (f"{alpha.value:.3f}" if alpha.computed else "undefined "
                   f"({alpha.unmet_reason})")
            ),
            limitations=(
                "Self-consistency is not accuracy. A judge that always returns "
                "the same wrong answer scores 100% here.",
                "Repeats of the same prompt may be correlated by caching or by "
                "a deterministic decoding setting, which inflates this figure. "
                "If your harness caches responses, this measures the cache.",
            ),
            detail={
                "unanimous_items": unanimous,
                "alpha": alpha.as_dict(),
            },
        )

    # -- 2. inter-judge agreement ---------------------------------------
    def _inter_judge(
        self, judges: list[str], grouped: dict[tuple[str, str], list[JudgeObservation]]
    ) -> dict[str, MetricResult]:
        items = sorted({item for _, item in grouped})
        matrix: list[list[bool]] = []
        for item in items:
            row = []
            for judge in judges:
                obs = [
                    o.verdict
                    for o in grouped.get((judge, item), [])
                    if o.verdict is not None
                ]
                if obs:
                    # Majority across this judge's repeats, so one judge
                    # contributes one rating rather than `repeats` of them.
                    row.append(sum(obs) * 2 > len(obs))
            if len(row) >= 2:
                matrix.append(row)

        out: dict[str, MetricResult] = {}
        if len(judges) < 2 or not matrix:
            out["inter_judge_agreement"] = MetricResult(
                metric="inter_judge_agreement",
                value=None,
                n=len(matrix),
                interpretation="not computed: fewer than two judges rated a common item",
                unmet_reason="need >= 2 judges on shared items",
                limitations=(
                    "With one judge there is nothing to compare. A single "
                    "judge's output is an opinion, not a measurement.",
                ),
            )
            return out

        out["inter_judge_agreement"] = MetricResult(
            metric="inter_judge_agreement",
            value=raw_agreement(matrix),
            n=len(matrix),
            interpretation=(
                f"raw unanimity {raw_agreement(matrix):.0%}, pairwise "
                f"{pairwise_agreement(matrix):.0%} over {len(matrix)} items. "
                "Raw agreement is NOT sufficient: on a skewed set judges agree "
                "most of the time by accident, which is why the alpha below is "
                "the figure to read"
            ),
            limitations=(
                "Raw agreement is inflated by class imbalance and must be read "
                "beside the chance-corrected figure.",
            ),
            detail={"pairwise": pairwise_agreement(matrix)},
        )
        out["inter_judge_alpha"] = krippendorff_alpha(matrix, level="nominal")

        if len(judges) == 2:
            first = [row[0] for row in matrix]
            second = [row[1] for row in matrix]
            kappa = cohens_kappa(first, second)
            out["inter_judge_cohens_kappa"] = MetricResult(
                metric="inter_judge_cohens_kappa",
                value=kappa,
                n=len(matrix),
                interpretation=(
                    f"Cohen's kappa = {kappa:.3f}" if kappa is not None
                    else "undefined: both judges used a single category "
                    "throughout, so expected agreement is 1 and kappa is 0/0"
                ),
                unmet_reason=None if kappa is not None else "expected agreement = 1",
                limitations=(
                    "Cohen's kappa uses per-rater marginals, so a lenient judge "
                    "and a strict one can show a low kappa while ranking cases "
                    "identically.",
                    "Two judges from the same model family share their blind "
                    "spots. Agreement between them measures consensus only.",
                ),
            )
        return out

    # -- 3 & 4. position bias and order sensitivity ---------------------
    def _position(
        self, observations: Sequence[JudgeObservation]
    ) -> dict[str, MetricResult]:
        pairwise = [
            o
            for o in observations
            if o.chosen is not None and o.presented_order is not None
        ]
        out: dict[str, MetricResult] = {}
        if not pairwise:
            for name in ("position_bias", "order_sensitivity"):
                out[name] = MetricResult(
                    metric=name,
                    value=None,
                    n=0,
                    interpretation=(
                        "not computed: no pairwise observations were supplied. "
                        "Position bias needs a judge choosing between options "
                        "with a recorded presentation order"
                    ),
                    unmet_reason="no pairwise observations",
                )
            return out

        first = sum(1 for o in pairwise if o.chosen == o.presented_order[0])
        second = len(pairwise) - first
        p_value = mcnemar_exact_p(first, second)
        share = first / len(pairwise)
        out["position_bias"] = MetricResult(
            metric="position_bias",
            value=share,
            n=len(pairwise),
            interpretation=(
                f"the first-presented option was chosen {share:.0%} of the time "
                f"({first}/{len(pairwise)}), exact two-sided p={p_value:.3g}. "
                + (
                    "Consistent with no position preference"
                    if p_value >= 0.05
                    else "SIGNIFICANT preference for position: the judge is "
                    "partly reading order rather than content"
                )
            ),
            detail={"chose_first": first, "chose_second": second, "p_value": p_value},
            limitations=(
                "Assumes options were assigned to positions independently of "
                "their quality. If the better answer was usually shown first, "
                "this measures the data, not the judge.",
                "50% is the null only for two options. Three or more needs a "
                "different test.",
            ),
        )

        # Order sensitivity: items seen under BOTH orders.
        by_item: dict[tuple[str, str], set[str]] = {}
        for o in pairwise:
            by_item.setdefault((o.judge, o.item), set()).add(o.chosen or "")
        both = {
            key: choices
            for key, choices in by_item.items()
            if len(
                {
                    o.presented_order
                    for o in pairwise
                    if (o.judge, o.item) == key
                }
            )
            > 1
        }
        if not both:
            out["order_sensitivity"] = MetricResult(
                metric="order_sensitivity",
                value=None,
                n=0,
                interpretation=(
                    "not computed: no item was presented in more than one order, "
                    "so a flip cannot be observed"
                ),
                unmet_reason="each item seen in one order only",
                limitations=(
                    "A judge evaluated under a single ordering cannot be shown "
                    "to be order-sensitive. This is not evidence it is stable.",
                ),
            )
            return out

        flipped = sum(1 for choices in both.values() if len(choices) > 1)
        rate = flipped / len(both)
        interval = self._bootstrap(
            [1.0 if len(c) > 1 else 0.0 for c in both.values()]
        )
        out["order_sensitivity"] = MetricResult(
            metric="order_sensitivity",
            value=rate,
            n=len(both),
            interval=interval.interval,
            level=interval.level if interval.computed else None,
            interpretation=(
                f"{rate:.0%} of {len(both)} items changed the chosen option when "
                f"the presentation order was reversed. A judge insensitive to "
                f"order would score 0%"
                + (
                    ""
                    if interval.computed
                    else f" (no interval: {interval.unmet_reason})"
                )
            ),
            limitations=(
                "A flip can also mean the two options are genuinely close, so a "
                "high rate on near-identical options is expected.",
                "Measured on the items supplied. Order sensitivity may differ by "
                "domain and by how long the options are.",
            ),
        )
        return out

    # -- 5 & 6. score variance and decision stability -------------------
    def _variance_and_stability(
        self, judge: str, grouped: dict[tuple[str, str], list[JudgeObservation]]
    ) -> dict[str, MetricResult]:
        out: dict[str, MetricResult] = {}

        score_groups = [
            [o.score for o in obs if o.score is not None]
            for (name, _), obs in grouped.items()
            if name == judge
        ]
        repeated_scores = [g for g in score_groups if len(g) >= 2]
        if not repeated_scores:
            out["score_variance"] = MetricResult(
                metric="score_variance",
                value=None,
                n=0,
                interpretation=(
                    "not computed: no item has two or more numeric scores from "
                    "this judge. A binary grader has no score variance"
                ),
                unmet_reason="no repeated numeric scores",
            )
        else:
            variances = [statistics.pvariance(g) for g in repeated_scores]
            interval = self._bootstrap(variances)
            mean_variance = sum(variances) / len(variances)
            out["score_variance"] = MetricResult(
                metric="score_variance",
                value=mean_variance,
                n=len(repeated_scores),
                interval=interval.interval,
                level=interval.level if interval.computed else None,
                interpretation=(
                    f"mean within-item score variance {mean_variance:.4f} across "
                    f"{len(repeated_scores)} items (sd "
                    f"{mean_variance ** 0.5:.4f} on the score scale)"
                    + ("" if interval.computed else f"; no interval: {interval.unmet_reason}")
                ),
                limitations=(
                    "Variance is on YOUR score scale, so it is not comparable "
                    "between graders that use different ranges.",
                    "Zero variance means the judge is deterministic, not that it "
                    "is correct.",
                ),
            )

        verdict_groups = [
            [o.verdict for o in obs if o.verdict is not None]
            for (name, _), obs in grouped.items()
            if name == judge
        ]
        repeated_verdicts = [g for g in verdict_groups if len(g) >= 2]
        if not repeated_verdicts:
            out["decision_stability"] = MetricResult(
                metric="decision_stability",
                value=None,
                n=0,
                interpretation=(
                    "not computed: no item has two or more verdicts from this "
                    "judge, so a flip cannot be observed"
                ),
                unmet_reason="repeats < 2",
                limitations=(
                    "One evaluation per item cannot show instability. This is "
                    "not evidence of stability.",
                ),
            )
            return out

        flips = [0.0 if len(set(g)) == 1 else 1.0 for g in repeated_verdicts]
        rate = sum(flips) / len(flips)
        interval = self._bootstrap(flips)
        out["decision_stability"] = MetricResult(
            metric="decision_stability",
            value=1.0 - rate,
            n=len(repeated_verdicts),
            interval=(
                (1.0 - interval.interval[1], 1.0 - interval.interval[0])
                if interval.computed and interval.interval
                else None
            ),
            level=interval.level if interval.computed else None,
            interpretation=(
                f"{1 - rate:.0%} of {len(repeated_verdicts)} items kept the same "
                f"binary verdict across every repeat; {rate:.0%} flipped"
                + ("" if interval.computed else f". No interval: {interval.unmet_reason}")
            ),
            limitations=(
                "For a scoring judge the verdict is derived at the 0.5 "
                "boundary, so a score wobbling around 0.5 flips while one "
                "wobbling around 0.9 does not. Stability here is partly a "
                "property of where your threshold sits.",
                "A stable verdict is not a correct verdict.",
            ),
        )
        return out

    # -- 7. correlation between judges ---------------------------------
    def _correlation(
        self, judges: list[str], grouped: dict[tuple[str, str], list[JudgeObservation]]
    ) -> dict[str, MetricResult]:
        out: dict[str, MetricResult] = {}
        items = sorted({item for _, item in grouped})
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                a, b = judges[i], judges[j]
                xs, ys = [], []
                for item in items:
                    left = [
                        o.score for o in grouped.get((a, item), []) if o.score is not None
                    ]
                    right = [
                        o.score for o in grouped.get((b, item), []) if o.score is not None
                    ]
                    if left and right:
                        xs.append(sum(left) / len(left))
                        ys.append(sum(right) / len(right))
                key = f"judge_correlation:{a}|{b}"
                if len(xs) < 2:
                    out[key] = MetricResult(
                        metric=key,
                        value=None,
                        n=len(xs),
                        interpretation=(
                            "not computed: these judges share fewer than two "
                            "items with numeric scores. A binary grader has no "
                            "scores to correlate"
                        ),
                        unmet_reason="insufficient shared numeric scores",
                    )
                    continue
                out[key] = pearson_r(xs, ys)
                out[f"{key}:spearman"] = spearman_rho(xs, ys)
        return out

    # -- entry point -----------------------------------------------------
    def analyse(
        self, observations: Sequence[JudgeObservation]
    ) -> ReliabilityReport:
        grouped = self._group(observations)
        judges = sorted({o.judge for o in observations})
        items = sorted({o.item for o in observations})

        metrics: dict[str, MetricResult] = {}
        metrics.update(self._inter_judge(judges, grouped))
        metrics.update(self._position(observations))
        metrics.update(self._correlation(judges, grouped))

        per_judge: dict[str, dict[str, MetricResult]] = {}
        for judge in judges:
            measures = {
                "intra_judge_consistency": self._intra_judge(judge, grouped)
            }
            measures.update(self._variance_and_stability(judge, grouped))
            per_judge[judge] = measures

        notes = [
            "Every figure here measures the JUDGE, not the dataset and not the "
            "model being graded.",
            "AGREEMENT IS NOT CORRECTNESS. Judges from one model family share "
            "their blind spots, so correlated error is indistinguishable from "
            "reliability. Only a human-labelled sample separates them, and this "
            "module cannot obtain one.",
        ]
        if len(judges) == 1:
            notes.append(
                "Only one judge was supplied, so inter-judge agreement, "
                "correlation and every cross-judge figure are unavailable. A "
                "single judge cannot be shown to be unreliable by comparison "
                "with itself alone."
            )
        refused = sorted(k for k, v in metrics.items() if not v.computed)
        if refused:
            notes.append(
                "Not computed because their assumptions were not met: "
                + ", ".join(refused)
                + ". Each states its own reason rather than reporting a number."
            )

        return ReliabilityReport(
            n_observations=len(observations),
            n_judges=len(judges),
            n_items=len(items),
            metrics=metrics,
            per_judge=per_judge,
            notes=tuple(notes),
        )
