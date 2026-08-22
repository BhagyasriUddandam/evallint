"""Statistical comparison of models on a shared eval set.

## Cases are paired, so the tests are paired

Both models are scored on the SAME cases. Two independent-sample tests would
throw away the pairing, inflate the variance and hide real differences. Only the
DISCORDANT cases — where the models disagree — carry information about which is
better; cases both models get right, or both get wrong, cancel exactly.

That is why the primary test here is McNemar's, computed exactly. On a 100-case
eval comparing two strong models the discordant count can be two, and the
chi-squared approximation is not valid there.

## Statistical significance and practical significance are different questions

A p-value answers "is this difference distinguishable from zero?". It does not
answer "is this difference big enough to care about?", and the two have different
answers often enough that reporting only the first misleads. So every comparison
reports both, in a 2x2:

                        |  effect >= threshold   |  effect < threshold
    ------------------- | ---------------------- | ---------------------
    significant         |  REAL AND MEANINGFUL   |  real but too small
    not significant     |  UNDERPOWERED          |  no meaningful difference

`UNDERPOWERED` is the cell people misread as "no difference". It means the eval
saw an effect worth caring about and could not resolve it — the answer is more
cases, not a conclusion.

**Only you can set the threshold.** What counts as a meaningful difference
depends on what the model is for: half a percentage point matters for a billing
classifier and is noise for a creative-writing grader. The default of 2
percentage points is a placeholder, is labelled as one, and is reported in every
result so nobody mistakes it for a derived quantity.

## Effect size is reported three ways

  risk difference   the raw gap in pass rate, in percentage points. Intuitive,
                    but it SHRINKS as concordance grows: the same disagreement
                    on an easy set looks smaller.
  odds ratio        c/b on the discordant cells, with a log-scale interval.
  Cohen's g         how lopsided the discordant cases are. Independent of how
                    many cases both models got right, so it is comparable
                    between an easy eval and a hard one.

Reporting only the risk difference makes an eval look less discriminating as it
gets easier, which is the opposite of informative.

## Multiple comparisons are adjusted

Four models is six pairwise tests. At alpha = 0.05, six independent tests carry
roughly a 26% chance of at least one false positive. Holm-adjusted p-values are
reported alongside the raw ones whenever more than one pair is compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .separation import (
    ALPHA,
    NORMAL_APPROX_MIN_DISCORDANT,
    cohens_g,
    holm_adjust,
    mcnemar_exact_p,
    mcnemar_odds_ratio,
    minimum_detectable_effect,
    paired_bootstrap_difference,
    paired_difference,
    wilson_interval,
)

__all__ = [
    "DEFAULT_PRACTICAL_THRESHOLD",
    "ComparisonReport",
    "ModelSummary",
    "PairComparison",
    "compare_models",
]

#: Default smallest difference treated as practically meaningful, in proportion
#: units (0.02 = 2 percentage points). A PLACEHOLDER, not a derived value: what
#: counts as meaningful depends entirely on what the model is for. Reported in
#: every result so it is never mistaken for a statistic.
DEFAULT_PRACTICAL_THRESHOLD = 0.02


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """One model's pass rate with an interval."""

    model: str
    n: int
    passes: int
    pass_rate: float
    interval: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n": self.n,
            "passes": self.passes,
            "pass_rate": self.pass_rate,
            "interval": list(self.interval),
        }


@dataclass(frozen=True, slots=True)
class PairComparison:
    """One paired comparison, with both kinds of significance."""

    model_a: str
    model_b: str
    n_cases: int
    both_pass: int
    both_fail: int
    only_a: int
    only_b: int
    rate_a: float
    rate_b: float
    difference: float
    difference_interval: tuple[float, float]
    interval_reliable: bool
    bootstrap_interval: tuple[float, float] | None
    mcnemar_p: float
    mcnemar_p_adjusted: float | None
    odds_ratio: float | None
    odds_ratio_interval: tuple[float, float] | None
    odds_ratio_corrected: bool
    cohens_g: float | None
    cohens_g_band: str
    minimum_detectable_effect: float | None
    practical_threshold: float
    statistically_significant: bool
    practically_significant: bool
    verdict: str
    summary: str

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [self.model_a, self.model_b],
            "n_cases": self.n_cases,
            "contingency": {
                "both_pass": self.both_pass,
                "both_fail": self.both_fail,
                f"only_{self.model_a}": self.only_a,
                f"only_{self.model_b}": self.only_b,
                "discordant": self.discordant,
            },
            "pass_rates": {self.model_a: self.rate_a, self.model_b: self.rate_b},
            "difference": self.difference,
            "difference_interval": list(self.difference_interval),
            "interval_reliable": self.interval_reliable,
            "bootstrap_interval": (
                list(self.bootstrap_interval) if self.bootstrap_interval else None
            ),
            "mcnemar_p": self.mcnemar_p,
            "mcnemar_p_adjusted": self.mcnemar_p_adjusted,
            "effect_size": {
                "risk_difference": self.difference,
                "odds_ratio": self.odds_ratio,
                "odds_ratio_interval": (
                    list(self.odds_ratio_interval)
                    if self.odds_ratio_interval
                    else None
                ),
                "odds_ratio_haldane_corrected": self.odds_ratio_corrected,
                "cohens_g": self.cohens_g,
                "cohens_g_band": self.cohens_g_band,
            },
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "practical_threshold": self.practical_threshold,
            "statistically_significant": self.statistically_significant,
            "practically_significant": self.practically_significant,
            "verdict": self.verdict,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    models: tuple[ModelSummary, ...]
    pairs: tuple[PairComparison, ...]
    practical_threshold: float
    alpha: float
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [m.as_dict() for m in self.models],
            "pairs": [p.as_dict() for p in self.pairs],
            "practical_threshold": self.practical_threshold,
            "alpha": self.alpha,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Human-readable report. Never a p-value on its own."""
        lines: list[str] = []
        for model in self.models:
            lo, hi = model.interval
            lines.append(
                f"{model.model}: {model.pass_rate:.1%} "
                f"({model.passes}/{model.n}, 95% CI {lo:.1%}-{hi:.1%})"
            )
        for pair in self.pairs:
            lines.append("")
            lines.append(pair.summary)
        if self.notes:
            lines.append("")
            lines.extend(self.notes)
        return "\n".join(lines)


ASSUMPTIONS = (
    "The two models were scored on the SAME cases, and a case's two outcomes "
    "belong together. Every test and interval here depends on that pairing; "
    "applied to results from different case sets they are wrong.",
    "Cases are treated as INDEPENDENT of each other. Near-duplicate cases "
    "violate this and make every interval narrower than the truth — run the "
    "redundancy check and divide by the effective case count if it finds "
    "clusters.",
    "Each case's outcome is treated as fixed. If your scorer is stochastic, a "
    "single run's verdicts carry noise these intervals do not include; score "
    "each case several times and use the majority.",
    "McNemar's null hypothesis is that among cases where the models disagree, "
    "either model is equally likely to be the one that succeeds. It says "
    "nothing about the cases they agree on.",
    "The eval set is treated as the population of interest. A curated set is "
    "not a random sample of production traffic, so these intervals describe "
    "re-running on THIS set, not deployment.",
)

LIMITATIONS = (
    "A p-value is not an effect size and is never reported alone here. "
    "Significance says a difference is distinguishable from zero; only the "
    "effect size and your own threshold say whether it matters.",
    "NOT SIGNIFICANT IS NOT EQUIVALENCE. It means this eval could not resolve a "
    "difference, which on a small or highly concordant set is entirely "
    "expected. The minimum detectable effect states what it could have "
    "resolved.",
    "The practical threshold is YOURS. The default is a placeholder with no "
    "derivation, and a conclusion of 'not meaningful' is only as good as the "
    "threshold it was measured against.",
    "The risk difference shrinks as the models agree more, so the same "
    "disagreement looks smaller on an easy eval. Cohen's g does not, which is "
    "why both are reported.",
    "With more than two models, Holm-adjusted p-values control the family-wise "
    "error rate for the comparisons MADE HERE. Comparisons you ran elsewhere "
    "and did not report are not accounted for.",
)


def compare_models(
    outcomes: Mapping[str, Mapping[str, bool]],
    *,
    practical_threshold: float = DEFAULT_PRACTICAL_THRESHOLD,
    alpha: float = ALPHA,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> ComparisonReport:
    """Compare models pairwise on their shared cases.

    Args:
        outcomes: ``{model: {case_id: passed}}``. Pairs are formed from the
            case ids the two models have in COMMON, and a model with extra
            cases contributes them only to its own pass rate.
        practical_threshold: smallest difference in pass rate you would act on,
            as a proportion. There is no correct default; the one used is
            reported in the result.
        alpha: significance level for the paired test.

    Raises:
        ValueError: fewer than two models, or no shared cases between a pair.
    """
    if not 0 < practical_threshold < 1:
        raise ValueError(
            f"practical_threshold must be in (0, 1), got {practical_threshold}"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    names = list(outcomes)
    if len(names) < 2:
        raise ValueError(
            f"comparing models needs at least two, got {len(names)}"
        )

    summaries = tuple(
        ModelSummary(
            model=name,
            n=len(outcomes[name]),
            passes=sum(1 for v in outcomes[name].values() if v),
            pass_rate=(
                sum(1 for v in outcomes[name].values() if v) / len(outcomes[name])
                if outcomes[name]
                else 0.0
            ),
            interval=wilson_interval(
                sum(1 for v in outcomes[name].values() if v), len(outcomes[name])
            ),
        )
        for name in names
    )

    raw: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = sorted(set(outcomes[a]) & set(outcomes[b]))
            if not shared:
                raise ValueError(
                    f"models {a!r} and {b!r} share no cases, so they cannot be "
                    "compared as paired data"
                )
            pairs = [(outcomes[a][cid], outcomes[b][cid]) for cid in shared]
            raw.append({"a": a, "b": b, "pairs": pairs})

    adjusted = (
        holm_adjust([mcnemar_exact_p(*_counts(r["pairs"])[2:4]) for r in raw])
        if len(raw) > 1
        else [None] * len(raw)
    )

    comparisons = tuple(
        _compare_pair(
            r["a"],
            r["b"],
            r["pairs"],
            practical_threshold=practical_threshold,
            alpha=alpha,
            adjusted_p=adjusted[index],
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        for index, r in enumerate(raw)
    )

    notes = [
        f"Practical threshold: {practical_threshold:.1%}. This is YOUR choice, "
        f"not a statistic — a 'not meaningful' verdict is only as good as it.",
    ]
    if len(raw) > 1:
        notes.append(
            f"{len(raw)} pairwise comparisons, so Holm-adjusted p-values are "
            f"reported. At alpha={alpha} the chance of at least one false "
            f"positive across {len(raw)} unadjusted tests is roughly "
            f"{1 - (1 - alpha) ** len(raw):.0%}."
        )
    underpowered = [p for p in comparisons if p.verdict == "underpowered"]
    if underpowered:
        notes.append(
            f"{len(underpowered)} comparison(s) are UNDERPOWERED: an effect "
            f"worth caring about was observed and could not be resolved. That "
            f"calls for more cases, not for a conclusion."
        )

    return ComparisonReport(
        models=summaries,
        pairs=comparisons,
        practical_threshold=practical_threshold,
        alpha=alpha,
        assumptions=ASSUMPTIONS,
        limitations=LIMITATIONS,
        notes=tuple(notes),
    )


def _counts(pairs: list[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    """(both_pass, both_fail, only_a, only_b)."""
    both_pass = sum(1 for a, b in pairs if a and b)
    both_fail = sum(1 for a, b in pairs if not a and not b)
    only_a = sum(1 for a, b in pairs if a and not b)
    only_b = sum(1 for a, b in pairs if b and not a)
    return both_pass, both_fail, only_a, only_b


def _compare_pair(
    model_a: str,
    model_b: str,
    pairs: list[tuple[bool, bool]],
    *,
    practical_threshold: float,
    alpha: float,
    adjusted_p: float | None,
    bootstrap_resamples: int,
    seed: int,
) -> PairComparison:
    both_pass, both_fail, only_a, only_b = _counts(pairs)
    n = len(pairs)
    discordant = only_a + only_b

    difference, interval, _ = paired_difference(only_a, only_b, n)
    p_value = mcnemar_exact_p(only_a, only_b)
    odds_ratio, odds_interval, corrected = mcnemar_odds_ratio(only_a, only_b)
    g, g_band = cohens_g(only_a, only_b)
    mde = (
        minimum_detectable_effect(discordant / n, n) if discordant and n else None
    )
    bootstrap = paired_bootstrap_difference(
        pairs, resamples=bootstrap_resamples, seed=seed
    )

    # The decision comes from the EXACT test, not from whether the Wald
    # interval excludes zero. They disagree at small discordant counts and the
    # interval is the one that is wrong there.
    significant = p_value < alpha
    practical = abs(difference) >= practical_threshold

    if significant and practical:
        verdict = "real_and_meaningful"
    elif significant and not practical:
        verdict = "real_but_below_threshold"
    elif not significant and practical:
        verdict = "underpowered"
    elif discordant == 0:
        verdict = "no_information"
    else:
        verdict = "no_meaningful_difference"

    rate_a = (both_pass + only_a) / n
    rate_b = (both_pass + only_b) / n
    summary = _summarise(
        model_a,
        model_b,
        rate_a,
        rate_b,
        difference,
        interval,
        bootstrap[1] if bootstrap else None,
        p_value,
        adjusted_p,
        g,
        g_band,
        odds_ratio,
        mde,
        practical_threshold,
        verdict,
        discordant,
        n,
    )

    return PairComparison(
        model_a=model_a,
        model_b=model_b,
        n_cases=n,
        both_pass=both_pass,
        both_fail=both_fail,
        only_a=only_a,
        only_b=only_b,
        rate_a=rate_a,
        rate_b=rate_b,
        difference=difference,
        difference_interval=interval,
        interval_reliable=discordant >= NORMAL_APPROX_MIN_DISCORDANT,
        bootstrap_interval=bootstrap[1] if bootstrap else None,
        mcnemar_p=p_value,
        mcnemar_p_adjusted=adjusted_p,
        odds_ratio=odds_ratio,
        odds_ratio_interval=odds_interval,
        odds_ratio_corrected=corrected,
        cohens_g=g,
        cohens_g_band=g_band,
        minimum_detectable_effect=mde,
        practical_threshold=practical_threshold,
        statistically_significant=significant,
        practically_significant=practical,
        verdict=verdict,
        summary=summary,
    )


def _summarise(
    a: str,
    b: str,
    rate_a: float,
    rate_b: float,
    difference: float,
    interval: tuple[float, float],
    bootstrap: tuple[float, float] | None,
    p_value: float,
    adjusted_p: float | None,
    g: float | None,
    g_band: str,
    odds_ratio: float | None,
    mde: float | None,
    threshold: float,
    verdict: str,
    discordant: int,
    n: int,
) -> str:
    parts = [
        f"{a} {rate_a:.1%} vs {b} {rate_b:.1%} — difference "
        f"{difference * 100:+.1f} percentage points"
    ]
    if bootstrap is not None:
        parts.append(
            f"95% CI (paired bootstrap) [{bootstrap[0] * 100:+.1f}, "
            f"{bootstrap[1] * 100:+.1f}] pp"
        )
    else:
        parts.append(
            f"95% CI [{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}] pp "
            f"(normal approximation; too few cases to bootstrap)"
        )

    p_text = f"McNemar exact p={p_value:.3g}"
    if adjusted_p is not None:
        p_text += f" (Holm-adjusted p={adjusted_p:.3g})"
    p_text += f", on {discordant} discordant of {n} cases"
    parts.append(f"Statistical significance: {p_text}")

    if g is None:
        effect = "Effect size: undefined — the models never disagreed"
    else:
        effect = f"Effect size: Cohen's g {g:.2f} ({g_band})"
        if odds_ratio is not None:
            effect += f", odds ratio {odds_ratio:.2f}"
    parts.append(effect)

    explanations = {
        "real_and_meaningful": (
            f"Practical significance: YES. The difference is distinguishable "
            f"from zero AND at or above your {threshold:.1%} threshold."
        ),
        "real_but_below_threshold": (
            f"Practical significance: NO. The difference is real but smaller "
            f"than your {threshold:.1%} threshold, so it is detectable without "
            f"being worth acting on."
        ),
        "underpowered": (
            f"Practical significance: UNKNOWN — this eval is UNDERPOWERED for "
            f"this comparison. The observed difference is above your "
            f"{threshold:.1%} threshold but cannot be distinguished from zero"
            + (
                f"; with this discordance the eval could only resolve "
                f"{mde:.1%} or larger. "
                if mde
                else ". "
            )
            + "That calls for more cases, NOT for concluding the models are "
            "equivalent."
        ),
        "no_meaningful_difference": (
            f"Practical significance: NO. The difference is neither "
            f"distinguishable from zero nor above your {threshold:.1%} "
            f"threshold. This is not proof of equivalence"
            + (f"; the eval could resolve {mde:.1%} or larger." if mde else ".")
        ),
        "no_information": (
            "Practical significance: NO INFORMATION. The two models never "
            "disagreed on any case, so this eval says nothing about which is "
            "better. That is the absence of evidence, not evidence of "
            "equivalence."
        ),
    }
    parts.append(explanations[verdict])
    return "\n".join(parts)
