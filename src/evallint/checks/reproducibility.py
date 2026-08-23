"""Reproducibility — does the eval return the same answer twice, and why not?

## Three sources of variance, never one number

An unstable score has at least three separable causes, and collapsing them into
a single "variance" figure destroys the only actionable information in the
measurement. They need different fixes:

    model stochasticity      the model samples differently between runs
                             -> fix by lowering temperature (if your model
                                accepts it), or by averaging more repeats
    evaluator stochasticity  the grader disagrees with itself or with another
                             grader on identical model output
                             -> fix the grader; averaging model runs will not
                                help at all
    dataset sampling         the score would differ on a different sample of
                             cases from the same population
                             -> fix by adding cases; re-running will not help

This analyser reports each separately, with its own sample size and its own
assumptions, and **refuses to sum them**. They are additive only under a
variance-components model with independent effects, which the designs seen here
do not guarantee. A single total would be a number nobody could act on.

## The design determines what is identifiable

  * Model stochasticity needs two or more RUNS with the judge held fixed.
  * Evaluator stochasticity needs two or more JUDGES scoring the same run, so
    the model output is identical and only the grader differs.
  * Dataset sampling variability is estimated by bootstrapping CASES within one
    run, which is a different question from run-to-run variance and is
    computable from a single run.

When the design cannot separate two sources, the analyser says so rather than
attributing the variance to one of them. Runs that regenerate model output AND
change the judge confound the first two irrecoverably.

## Aggregate stability and per-case stability are different things

They diverge, and the divergence is the interesting finding. An eval can flip
40% of its individual verdicts while its headline score barely moves, because
independent flips cancel. That is not a reproducible eval — it is a coincidence
that will not survive a new model. Both figures are always reported, and the
combination is called out when it occurs.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .agreement import kendalls_w
from .reliability_stats import MetricResult, bootstrap_interval
from .separation import wilson_interval

__all__ = [
    "FLIP_PRONE_SHARE",
    "ReproducibilityReport",
    "RunOutcome",
    "analyse_reproducibility",
]

#: A case flipping on at least this share of run-pairs is reported individually
#: as flip-prone. A convention: at 0.25 a case that changes verdict in one run
#: out of four is named. Stated as a convention in the output.
FLIP_PRONE_SHARE = 0.25

#: Minimum runs and judges needed before the corresponding variance component is
#: computed at all. Two is the arithmetic minimum; the report says that a
#: variance estimated from two observations is itself extremely uncertain.
MIN_RUNS = 2
MIN_JUDGES = 2


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """One verdict: model M on case C in run R, optionally by judge J."""

    run: int
    model: str
    case: str
    passed: bool
    score: float | None = None
    judge: str | None = None


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    n_outcomes: int
    n_runs: int
    n_models: int
    n_cases: int
    n_judges: int
    design: str
    #: Deliberately three separate entries. There is no "total".
    variance_sources: dict[str, MetricResult] = field(default_factory=dict)
    aggregate_stability: dict[str, MetricResult] = field(default_factory=dict)
    case_stability: dict[str, Any] = field(default_factory=dict)
    ranking_stability: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_outcomes": self.n_outcomes,
            "n_runs": self.n_runs,
            "n_models": self.n_models,
            "n_cases": self.n_cases,
            "n_judges": self.n_judges,
            "design": self.design,
            "variance_sources": {
                k: v.as_dict() for k, v in self.variance_sources.items()
            },
            "aggregate_stability": {
                k: v.as_dict() for k, v in self.aggregate_stability.items()
            },
            "case_stability": dict(self.case_stability),
            "ranking_stability": dict(self.ranking_stability),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = [f"Design: {self.design}", ""]
        lines.append("VARIANCE SOURCES (reported separately, never summed)")
        for name, metric in self.variance_sources.items():
            value = "not identifiable" if not metric.computed else f"sd {metric.value:.4f}"
            lines.append(f"  {name:<26} {value}   n={metric.n}")
            lines.append(f"    {metric.interpretation}")
        lines.append("")
        lines.append("AGGREGATE STABILITY")
        for name, metric in self.aggregate_stability.items():
            lines.append(f"  {name}: {metric.interpretation}")
        lines.append("")
        stability = self.case_stability
        if stability:
            lines.append(
                f"PER-CASE STABILITY: {stability['stable_share']:.0%} of "
                f"{stability['n_cases_assessed']} cases kept the same verdict in "
                f"every run; {stability['n_flip_prone']} are flip-prone"
            )
            if stability["flip_prone_case_ids"]:
                shown = ", ".join(stability["flip_prone_case_ids"][:8])
                lines.append(f"  worst: {shown}")
        ranking = self.ranking_stability
        if ranking.get("assessed"):
            lines.append("")
            lines.append(
                f"MODEL RANKING STABILITY: the modal ranking held in "
                f"{ranking['modal_ranking_share']:.0%} of runs; Kendall's W "
                + (
                    f"{ranking['kendalls_w']:.3f}"
                    if ranking["kendalls_w"] is not None
                    else "undefined"
                )
            )
            if ranking["unstable_pairs"]:
                lines.append(
                    "  order flipped between runs for: "
                    + ", ".join(f"{a} vs {b}" for a, b in ranking["unstable_pairs"])
                )
        if self.notes:
            lines.append("")
            lines.extend(f"  · {note}" for note in self.notes)
        return "\n".join(lines)


def _score(outcomes: Sequence[RunOutcome]) -> float:
    return sum(1 for o in outcomes if o.passed) / len(outcomes) if outcomes else 0.0


def _refused(metric: str, n: int, reason: str, limitations: tuple[str, ...]) -> MetricResult:
    return MetricResult(
        metric=metric,
        value=None,
        n=n,
        interpretation=f"not identifiable from this design: {reason}",
        limitations=limitations,
        unmet_reason=reason,
    )


def analyse_reproducibility(
    outcomes: Sequence[RunOutcome],
    *,
    flip_prone_share: float = FLIP_PRONE_SHARE,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> ReproducibilityReport:
    """Decompose the instability of a repeated evaluation.

    Args:
        outcomes: every (run, model, case) verdict, optionally tagged with the
            judge that produced it.
    """
    if not outcomes:
        raise ValueError("no outcomes supplied")
    if not 0 < flip_prone_share <= 1:
        raise ValueError("flip_prone_share must be between 0 and 1")

    runs = sorted({o.run for o in outcomes})
    models = sorted({o.model for o in outcomes})
    cases = sorted({o.case for o in outcomes})
    judges = sorted({o.judge for o in outcomes if o.judge is not None})

    design_bits = [f"{len(runs)} run(s)", f"{len(models)} model(s)", f"{len(cases)} case(s)"]
    design_bits.append(f"{len(judges)} judge(s)" if judges else "judge not recorded")
    design = ", ".join(design_bits)

    variance_sources = {
        "model_stochasticity": _model_variance(outcomes, runs, models, judges),
        "evaluator_stochasticity": _evaluator_variance(outcomes, runs, models, judges),
        "dataset_sampling": _sampling_variance(
            outcomes, runs, models, bootstrap_resamples, seed
        ),
    }
    aggregate = _aggregate_stability(outcomes, runs, models)
    case_stability = _case_stability(outcomes, runs, flip_prone_share)
    ranking = _ranking_stability(outcomes, runs, models)

    notes = [
        "The three variance sources above are NOT summed. They are additive only "
        "under a variance-components model with independent effects, which this "
        "design does not guarantee, and a single total would not tell you which "
        "fix to apply.",
    ]
    if len(runs) < MIN_RUNS:
        notes.append(
            "Only one run was supplied. Nothing here can show run-to-run "
            "instability, and that is NOT evidence the eval is reproducible."
        )
    if not judges:
        notes.append(
            "No judge was recorded on any outcome, so evaluator stochasticity "
            "cannot be separated from model stochasticity. Tag outcomes with the "
            "judge that produced them to separate the two."
        )
    if (
        case_stability.get("stable_share") is not None
        and case_stability["stable_share"] < 0.8
        and aggregate.get("score_sd")
        and aggregate["score_sd"].computed
        and aggregate["score_sd"].value is not None
        and aggregate["score_sd"].value < 0.02
    ):
        notes.append(
            f"WARNING: the headline score is stable (sd "
            f"{aggregate['score_sd'].value:.3f}) while "
            f"{1 - case_stability['stable_share']:.0%} of individual verdicts "
            f"flip between runs. Independent flips cancel in an average, so this "
            f"is a coincidence rather than reproducibility, and it will not "
            f"survive a change of model."
        )

    return ReproducibilityReport(
        n_outcomes=len(outcomes),
        n_runs=len(runs),
        n_models=len(models),
        n_cases=len(cases),
        n_judges=len(judges),
        design=design,
        variance_sources=variance_sources,
        aggregate_stability=aggregate,
        case_stability=case_stability,
        ranking_stability=ranking,
        notes=tuple(notes),
    )


# --- 1. model stochasticity -------------------------------------------------


def _model_variance(outcomes, runs, models, judges) -> MetricResult:
    """Variance of a model's score across runs, with the judge held fixed."""
    limitations = (
        "Isolated by holding the judge fixed, so it captures variation in the "
        "MODEL's output only. If your harness caches model responses between "
        "runs, this measures the cache and will read as zero.",
        "A variance estimated from a handful of runs is itself very uncertain. "
        "Three runs give a point estimate with almost no precision.",
    )
    if len(runs) < MIN_RUNS:
        return _refused(
            "model_stochasticity",
            len(runs),
            f"{len(runs)} run(s); needs at least {MIN_RUNS} to see run-to-run "
            "variation",
            limitations,
        )
    if judges and len(judges) > 1:
        fixed = judges[0]
        subset = [o for o in outcomes if o.judge == fixed]
        note = f" with judge {fixed!r} held fixed"
    else:
        subset = list(outcomes)
        note = "" if not judges else f" (single judge {judges[0]!r})"

    per_model_sd = []
    for model in models:
        scores = [
            _score([o for o in subset if o.model == model and o.run == run])
            for run in runs
        ]
        scores = [s for s in scores if s is not None]
        if len(scores) >= MIN_RUNS:
            per_model_sd.append(statistics.stdev(scores))
    if not per_model_sd:
        return _refused(
            "model_stochasticity", 0, "no model appears in two or more runs",
            limitations,
        )
    mean_sd = sum(per_model_sd) / len(per_model_sd)
    return MetricResult(
        metric="model_stochasticity",
        value=mean_sd,
        n=len(runs),
        interpretation=(
            f"mean across-run standard deviation of the model score is "
            f"{mean_sd:.4f} ({mean_sd * 100:.2f} percentage points) over "
            f"{len(runs)} runs{note}. "
            + (
                "Zero, so either the model is deterministic or its responses "
                "were cached between runs — those look identical from here"
                if mean_sd == 0
                else "Lower the sampling temperature if your model accepts one, "
                "or average more repeats"
            )
        ),
        limitations=limitations,
        detail={"per_model_sd": per_model_sd},
    )


# --- 2. evaluator stochasticity --------------------------------------------


def _evaluator_variance(outcomes, runs, models, judges) -> MetricResult:
    """Variance across judges scoring the SAME run, so model output is fixed."""
    limitations = (
        "Isolated by comparing judges WITHIN a run, so the model output is "
        "identical and only the grader differs. That is the only way to separate "
        "grader noise from model noise.",
        "Averaging more model runs does not reduce this. A disagreeing grader is "
        "fixed by fixing the grader.",
        "Judges from one model family share their blind spots, so LOW variance "
        "here is consensus rather than correctness.",
    )
    if len(judges) < MIN_JUDGES:
        return _refused(
            "evaluator_stochasticity",
            len(judges),
            f"{len(judges)} judge(s) recorded; needs at least {MIN_JUDGES} "
            "scoring the same run to separate grader noise from model noise",
            limitations,
        )
    spreads = []
    for run in runs:
        for model in models:
            scores = [
                _score(
                    [
                        o
                        for o in outcomes
                        if o.run == run and o.model == model and o.judge == judge
                    ]
                )
                for judge in judges
            ]
            present = [s for s in scores if s is not None]
            if len(present) >= MIN_JUDGES:
                spreads.append(statistics.stdev(present))
    if not spreads:
        return _refused(
            "evaluator_stochasticity",
            0,
            "no run has two or more judges scoring the same model",
            limitations,
        )
    mean_sd = sum(spreads) / len(spreads)
    return MetricResult(
        metric="evaluator_stochasticity",
        value=mean_sd,
        n=len(spreads),
        interpretation=(
            f"mean within-run standard deviation ACROSS JUDGES is {mean_sd:.4f} "
            f"({mean_sd * 100:.2f} percentage points) over {len(spreads)} "
            f"run-model combinations. "
            + (
                "Zero, so the judges agreed exactly on every combination"
                if mean_sd == 0
                else "This is grader disagreement on identical model output, and "
                "no amount of re-running the model will reduce it"
            )
        ),
        limitations=limitations,
        detail={"n_combinations": len(spreads)},
    )


# --- 3. dataset sampling variability ---------------------------------------


def _sampling_variance(outcomes, runs, models, resamples, seed) -> MetricResult:
    """How much the score would move on a different sample of cases.

    Bootstrapped over CASES within one run. A different question from run-to-run
    variance, and answerable from a single run.
    """
    limitations = (
        "Estimated by resampling CASES within one run, so it answers 'what if "
        "the eval had drawn different cases' rather than 'what if we ran again'.",
        "Assumes the cases are an independent sample of the population you care "
        "about. A curated eval set is neither independent (near-duplicates) nor "
        "a random sample of production traffic, so this is a lower bound on the "
        "real sampling variability.",
        "Adding cases reduces this. Re-running the same cases does not.",
    )
    run = runs[0]
    per_model: dict[str, MetricResult] = {}
    for model in models:
        indicators = [
            1.0 if o.passed else 0.0
            for o in outcomes
            if o.run == run and o.model == model
        ]
        if len(indicators) >= 10:
            per_model[model] = bootstrap_interval(
                indicators, resamples=resamples, seed=seed
            )
    if not per_model:
        return _refused(
            "dataset_sampling",
            0,
            "fewer than 10 cases in the first run, too few to bootstrap",
            limitations,
        )
    widths = [
        (m.interval[1] - m.interval[0]) / 2
        for m in per_model.values()
        if m.interval is not None
    ]
    half_width = sum(widths) / len(widths) if widths else 0.0
    return MetricResult(
        metric="dataset_sampling",
        value=half_width,
        # n is the number of CASES bootstrapped, which is the sample size that
        # governs this interval. An earlier version tried to read it out of the
        # bootstrap's `resamples` detail and called len() on an int.
        n=max(m.n for m in per_model.values()),
        interpretation=(
            f"a 95% bootstrap interval over cases is about +-{half_width:.4f} "
            f"({half_width * 100:.2f} percentage points) wide. That is how much "
            f"the score could move on a different sample of cases from the same "
            f"population — reduced by ADDING cases, not by re-running"
        ),
        limitations=limitations,
        detail={
            model: m.interval for model, m in per_model.items() if m.interval
        },
    )


# --- aggregate stability ---------------------------------------------------


def _aggregate_stability(outcomes, runs, models) -> dict[str, MetricResult]:
    scores = [
        _score([o for o in outcomes if o.run == run]) for run in runs
    ]
    limitations = (
        "A stable aggregate does NOT imply stable per-case verdicts. "
        "Independent flips cancel in an average, so read the per-case figure "
        "beside this one.",
    )
    if len(scores) < MIN_RUNS:
        return {
            "score_sd": _refused(
                "aggregate_score_sd",
                len(scores),
                f"{len(scores)} run(s); needs at least {MIN_RUNS}",
                limitations,
            )
        }
    sd = statistics.stdev(scores)
    lo, hi = min(scores), max(scores)
    passes = sum(1 for o in outcomes if o.passed)
    return {
        "score_sd": MetricResult(
            metric="aggregate_score_sd",
            value=sd,
            n=len(scores),
            interval=(lo, hi),
            interpretation=(
                f"across {len(scores)} runs the aggregate score was "
                f"{', '.join(f'{s:.1%}' for s in scores)} — sd {sd:.4f} "
                f"({sd * 100:.2f} pp), range {lo:.1%} to {hi:.1%}"
            ),
            limitations=limitations,
        ),
        "pooled_interval": MetricResult(
            metric="pooled_wilson_interval",
            value=passes / len(outcomes),
            n=len(outcomes),
            interval=wilson_interval(passes, len(outcomes)),
            level=0.95,
            interpretation=(
                "Wilson interval over ALL outcomes pooled across runs. Narrower "
                "than the truth, because repeated verdicts on the same case are "
                "not independent observations — shown for reference, not for "
                "inference"
            ),
            limitations=(
                "Pooling runs treats correlated observations as independent, so "
                "this interval is too narrow. Use the across-run sd instead.",
            ),
        ),
    }


# --- per-case stability ----------------------------------------------------


def _case_stability(outcomes, runs, flip_prone_share) -> dict[str, Any]:
    if len(runs) < MIN_RUNS:
        return {
            "assessed": False,
            "reason": f"{len(runs)} run(s); needs at least {MIN_RUNS}",
            "stable_share": None,
            "n_cases_assessed": 0,
            "n_flip_prone": 0,
            "flip_prone_case_ids": [],
        }
    by_key: dict[tuple[str, str], list[bool]] = {}
    for o in outcomes:
        by_key.setdefault((o.model, o.case), []).append(o.passed)

    assessed = {k: v for k, v in by_key.items() if len(v) >= MIN_RUNS}
    stable = sum(1 for v in assessed.values() if len(set(v)) == 1)
    flip_rates = {
        key: min(sum(v), len(v) - sum(v)) / len(v) for key, v in assessed.items()
    }
    flip_prone = sorted(
        (key for key, rate in flip_rates.items() if rate >= flip_prone_share),
        key=lambda k: -flip_rates[k],
    )
    return {
        "assessed": True,
        "n_cases_assessed": len(assessed),
        "n_stable": stable,
        "stable_share": stable / len(assessed) if assessed else None,
        "n_flip_prone": len(flip_prone),
        "flip_prone_case_ids": [f"{model}:{case}" for model, case in flip_prone],
        "flip_prone_threshold": flip_prone_share,
        "threshold_note": (
            f"'flip-prone' means the minority verdict occurred on at least "
            f"{flip_prone_share:.0%} of runs. A convention, not a statistic."
        ),
    }


# --- model ranking stability -----------------------------------------------


def _ranking_stability(outcomes, runs, models) -> dict[str, Any]:
    if len(models) < 2 or len(runs) < MIN_RUNS:
        return {
            "assessed": False,
            "reason": (
                f"{len(models)} model(s) and {len(runs)} run(s); ranking "
                f"stability needs at least 2 of each"
            ),
        }
    orderings: list[tuple[str, ...]] = []
    rank_rows: list[list[float]] = []
    for run in runs:
        scored = [
            (model, _score([o for o in outcomes if o.run == run and o.model == model]))
            for model in models
        ]
        ordered = sorted(scored, key=lambda pair: -pair[1])
        orderings.append(tuple(model for model, _ in ordered))
        position = {model: index + 1 for index, (model, _) in enumerate(ordered)}
        rank_rows.append([position[model] for model in models])

    modal = max(set(orderings), key=orderings.count)
    unstable_pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            order = {ordering.index(a) < ordering.index(b) for ordering in orderings}
            if len(order) > 1:
                unstable_pairs.append((a, b))

    return {
        "assessed": True,
        "n_runs": len(runs),
        "modal_ranking": list(modal),
        "modal_ranking_share": orderings.count(modal) / len(orderings),
        "distinct_rankings": len(set(orderings)),
        "top_model_stable": len({ordering[0] for ordering in orderings}) == 1,
        "kendalls_w": kendalls_w(rank_rows),
        "unstable_pairs": unstable_pairs,
        "note": (
            "Kendall's W is computed without a tie correction, so ties deflate "
            "it and a reported value is a lower bound on agreement."
        ),
    }
