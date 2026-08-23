"""Check 1 — model separation analysis.

## The question

Not "is this case good?" but "how much evidence about model separation does
this eval carry, and between which capability levels?".

Run every case against two or more models of KNOWN differing capability,
ordered weakest first, and classify what the pattern of verdicts supports.

## The taxonomy

Five outcomes, none of which is a defect label:

  ceiling        every model passed
  floor          every model failed
  separating     verdicts follow the declared order, with at least one change
  non_monotonic  a weaker model passed where a stronger one failed
  indeterminate  the repeats did not agree well enough to assign a verdict

A ceiling case is NOT a bad case. It may be a deliberate regression guard, and
an eval with none is an eval that cannot detect a regression. What ceiling and
floor cases do is **provide limited evidence for model separation** — they say
nothing about which model is better. That is a statement about evidence, not
about quality, and the wording is deliberate throughout.

`floor` and `non_monotonic` are worth more attention than `ceiling`, for a
reason that has nothing to do with difficulty: a case every model fails, or one
where the weaker model wins, is more often a wrong reference answer or a broken
grader than a hard case. They are pointers to the ground truth.

## Majority, not unanimity — the central methodological change

An earlier version of this check classified a case only when every model's
verdict was unanimous across repeats, and discarded the rest. That filter is
not neutral with respect to the outcome it measures.

For a model that passes a case with probability p, a unanimous run is much more
likely when p is extreme: at p = 0.6 and 5 repeats, P(all pass) = 0.078 while
P(all fail) = 0.010. Surviving cases are therefore enriched for "this model
consistently passes" — which is the ceiling category. Measured on a synthetic
100-case set with a weak model at p = 0.6:

    repeats   n_measured   non-discriminating share
       1         100              0.52
       3          27              0.74
       5           9              0.89

The share nearly doubles without the data changing. Worse, the check's own
advice was to raise `repeats` before trusting the figure, so following it made
the number less trustworthy.

The analysis now uses the MAJORITY verdict per model and classifies every
scored case, so the denominator is always `n_cases` and does not move with
`repeats`. Raising `repeats` moves cases out of `indeterminate` into resolved
classes; it no longer deletes them.

`non_discriminating_share` and the `n_measured` / `unstable` keys are retained
unchanged for backward compatibility. They are computed the old way and carry
the old bias, so **do not compare them across different `repeats` values** —
use `limited_evidence_share`, whose denominator is fixed.

## Uncertainty

Per case, a model's success count out of `repeats` gets a Wilson interval.
A classification is marked `supported` only when every model's interval
excludes 0.5, i.e. the repeats are sufficient to resolve that verdict alone.
At `repeats=1` nothing is ever supported, because one observation cannot bound
a probability — that is the correct answer rather than a gap in the code.

Per model pair, the comparison is PAIRED (both models saw the same cases), so
it uses McNemar on the discordant counts rather than two independent-proportion
tests. Each pair reports the difference in pass rate with an interval, an exact
p-value, and a minimum detectable effect: the smallest difference this eval
could have resolved at all. A real run in this repo has an MDE of 4.0
percentage points on 100 cases, against an observed gap of 0.7 — so that eval
cannot support a claim about which model is better, and says so.

## Thresholds, and their justification

  pass_threshold = 0.5   Only used to turn a float score into a verdict.
                         A convention; move it if your scorer's scale differs.
  0.5 (support test)     A per-case verdict is "resolved" when the interval for
                         that model's success probability excludes one half,
                         i.e. it is more likely than not in a way the repeats
                         can demonstrate. Follows from majority voting rather
                         than being chosen independently.
  confidence = 0.95      A convention. Exported as `confidence_level` in the
                         stats so a reader can substitute their own.
  alpha = 0.05           Derived from the confidence level, not chosen
                         separately, so the two cannot drift apart.
  max_non_discriminating_share = 0.5
                         The level above which a warning is emitted. A
                         convention with no statistical content, and stated as
                         such in the limitations.
  10 discordant pairs    Below this the normal-approximation interval on the
                         paired difference is reported as indicative only. The
                         conventional rule of thumb for a binomial normal
                         approximation, and it earns its place here: see below.

A pair is called separated by the EXACT test, not by whether its interval
excludes zero. Those two disagree at small discordant counts, and there the
interval is the one that is wrong. Three discordant cases all favouring the
same model give a Wald interval of [+0.005, +0.495] — excluding zero — while
the exact p-value is 0.25. And when every comparable case is discordant in one
direction the Wald interval collapses to zero width and claims certainty, the
same defect that rules out the normal approximation for a proportion. So the
interval describes magnitude and carries an `interval_reliable` flag; the
decision comes from the test.

## The scorer is injected

The single most important design choice in the tool: evallint never talks to a
model provider, so it works with anyone's stack and the tests need no API key.
The contract is unchanged — ``scorer(case, model) -> bool | float``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..schema import EvalCase, EvalSet
from .base import Check, CheckResult, Finding, Severity
from .separation import (
    ALPHA,
    NORMAL_APPROX_MIN_DISCORDANT,
    Z_95,
    mcnemar_exact_p,
    minimum_detectable_effect,
    paired_difference,
    wilson_interval,
)

#: Interval level used for every reported interval. A convention, stated in the
#: stats so a reader can substitute their own rather than having to guess.
CONFIDENCE_LEVEL = 0.95

__all__ = [
    "CONFIDENCE_LEVEL",
    "CaseClass",
    "CaseSeparation",
    "DiscriminationCheck",
    "Scorer",
    "ScorerError",
]

Scorer = Callable[[EvalCase, Any], bool | float]

LIMITATIONS = (
    "This is the only check that runs models. It calls the scorer once per "
    "case per model, so on a real API it costs time and money proportional to "
    "n_cases x n_models.",
    "The verdict depends entirely on the model pair you chose. Two models that "
    "are closer in capability than you assumed will make a perfectly good eval "
    "look non-discriminating, and the check cannot tell the difference.",
    "A non-discriminating case is not automatically a bad case. Easy cases "
    "every model passes may be deliberate regression guards. This reports how "
    "much of your eval is doing discriminating work; it does not tell you what "
    "to delete.",
    "Continuous scores are reduced to pass/fail at the threshold, so a case "
    "where the strong model scores 0.95 and the weak one 0.55 is recorded as "
    "non-discriminating. Use a binary scorer, or move the threshold, if that "
    "distinction matters to you.",
    "Model outputs are usually sampled, so a case can flip between runs. Note "
    "that temperature=0 is NOT always available: current frontier models "
    "including Claude Opus 5 and Sonnet 5 reject the temperature parameter "
    "outright, so determinism may not be an option for your model pair.",
    "With max_workers above 1 the scorer is called from multiple threads. "
    "evallint cannot verify that your scorer is thread-safe, and an unsafe one "
    "corrupts results quietly rather than crashing. If a run at max_workers>1 "
    "disagrees with a serial run on the same data, suspect the scorer before "
    "the eval set.",
    "THIS IS A SINGLE-RUN MEASUREMENT AND MAY NOT REPRODUCE. Every number "
    "above comes from scoring each case exactly once. On the author's own "
    "20-case example set, repeating an identical run three times changed the "
    "verdict for 8 of 20 cases (40%), and only 1 of the inversions found in a "
    "single run survived all three repeats. Before reporting any figure here "
    "-- especially a specific inverted or floor case -- run the same "
    "configuration at least three times and keep only the cases whose verdict "
    "is identical every time. A case that flips is sampling noise, not a "
    "finding.",
)


class ScorerError(RuntimeError):
    """The user-supplied scorer failed or returned something unusable."""


class DiscriminationCheck(Check):
    """Reports how many cases actually separate models of differing capability."""

    name = "discrimination"
    description = "Cases that give the same verdict for strong and weak models."

    def __init__(
        self,
        scorer: Scorer,
        models: Sequence[Any],
        *,
        pass_threshold: float = 0.5,
        max_non_discriminating_share: float = 0.5,
        repeats: int = 1,
        max_workers: int = 1,
        sample: int | None = None,
        sample_seed: int = 0,
    ) -> None:
        """
        Args:
            scorer: ``scorer(case, model) -> bool | float``. Returns whether
                the model got this case right, or a score compared against
                ``pass_threshold``. evallint never calls a provider itself.
            models: Two or more model identifiers ordered WEAKEST FIRST. The
                order is what makes inversion detectable, so getting it
                backwards inverts that finding — see the limitations.
            pass_threshold: A float score >= this counts as a pass. Ignored
                for bool scorers.
            max_non_discriminating_share: Warn above this share of MEASURED
                cases (unstable cases are excluded from the denominator).
            repeats: How many times to score each (case, model). A case whose
                verdict is not unanimous across repeats is reported as
                UNSTABLE and excluded from the ceiling/floor/inverted counts,
                because a verdict that changes between identical runs is
                sampling noise rather than a property of the eval set.

                Default 1 preserves single-run behaviour and cost, but 1 also
                cannot detect instability at all. On the bundled 20-case
                example, 3 repeats found 40% of verdicts non-reproducible, so
                a single run there was measuring almost nothing reliably.
                Cost multiplies by ``repeats``.
            max_workers: How many scorer calls to run concurrently. Scoring is
                almost always I/O-bound (an API call per case per model), and
                running it serially is the difference between minutes and hours:
                1000 cases x 2 models x 3 repeats at 2s per call is 3.3 hours
                serial, roughly 15 minutes at 16 workers.

                Default 1 runs serially and takes a code path with no threads in
                it at all. That is deliberate: YOUR SCORER MUST BE THREAD-SAFE
                to raise this above 1. evallint cannot check that for you — the
                scorer is your code, and a client object, a shared session, a
                file handle or a non-locking cache inside it will corrupt
                quietly rather than crash. Opt in once you have checked.
            sample: Score only this many cases, chosen at random, instead of
                all of them. This check is the only expensive one — every case
                costs a real API call per model per repeat — so on a large set
                the choice is often between auditing a sample and not auditing
                at all.

                The cost is that findings then describe the sample, NOT the
                set: a clean sample does not mean a clean eval set, and a
                specific bad case that was not drawn will not be reported. That
                is stated in the limitations of every sampled run, and the
                summary says so too, because a sampled audit that reads like a
                full one is exactly the kind of quiet overclaim this project
                exists to catch.

                None (default) scores everything. A value at or above the case
                count also scores everything, and is not an error — asking for
                200 of 100 cases is a reasonable thing for a script to do.
            sample_seed: Seed for the sample, so the same set and seed always
                draw the same cases. Sampling that changed between runs would
                make a re-run non-comparable and, with a response cache, would
                quietly cost money every time.
        """
        if len(models) < 2:
            raise ValueError(
                "discrimination needs at least two models of differing "
                f"capability, got {len(models)}"
            )
        labels = [str(m) for m in models]
        if len(set(labels)) != len(labels):
            raise ValueError(f"model names must be unique, got {labels}")
        if not 0 < max_non_discriminating_share <= 1:
            raise ValueError("max_non_discriminating_share must be between 0 and 1")
        if repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {repeats}")
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        if sample is not None and sample < 1:
            raise ValueError(f"sample must be at least 1 or None, got {sample}")
        self.scorer = scorer
        self.models = tuple(models)
        self.model_names = tuple(labels)
        self.pass_threshold = pass_threshold
        self.max_non_discriminating_share = max_non_discriminating_share
        self.repeats = repeats
        self.max_workers = max_workers
        self.sample = sample
        self.sample_seed = sample_seed

    def _select(self, cases: list) -> tuple[list, bool]:
        """Pick the cases to score. Returns (cases, was_sampled)."""
        if self.sample is None or self.sample >= len(cases):
            return cases, False
        # Sample INDICES and sort them, rather than using the shuffled order
        # random.sample returns. The subset is identical either way, but keeping
        # file order means the report lists cases in the order the user wrote
        # them, and two runs with the same seed produce byte-identical output.
        chosen = sorted(
            random.Random(self.sample_seed).sample(range(len(cases)), self.sample)
        )
        return [cases[i] for i in chosen], True

    def run(self, eval_set: EvalSet) -> CheckResult:
        all_cases = list(eval_set)
        cases, sampled = self._select(all_cases)
        # runs[case.id][model_index] -> tuple of `repeats` verdicts
        runs = self._score_all(cases)

        # A case is only measurable if every model gave the same answer on
        # every repeat. Anything else is a verdict that changes between
        # identical runs, which is noise — folding it into ceiling/floor/
        # inverted would report sampling variance as a property of the data.
        stable: dict[str, tuple[bool, ...]] = {}
        unstable: list[str] = []
        for case in cases:
            per_model = runs[case.id]
            if all(len(set(r)) == 1 for r in per_model):
                stable[case.id] = tuple(r[0] for r in per_model)
            else:
                unstable.append(case.id)

        ceiling = [c.id for c in cases if c.id in stable and all(stable[c.id])]
        floor = [c.id for c in cases if c.id in stable and not any(stable[c.id])]
        inverted = [
            c.id for c in cases if c.id in stable and _is_inverted(stable[c.id])
        ]
        non_discriminating = len(ceiling) + len(floor)
        n_cases = len(cases)
        n_measured = len(stable)

        # ------------------------------------------------------------------
        # Separation analysis. Classifies EVERY scored case, using the majority
        # verdict per model rather than requiring unanimity, so the denominator
        # is n_cases and does not move when `repeats` changes. See the module
        # docstring for why the legacy `non_discriminating_share` above is
        # retained but should not be compared across repeat counts.
        # ------------------------------------------------------------------
        classified = {
            case.id: _classify_case(runs[case.id], self.repeats) for case in cases
        }
        by_class: dict[str, list[str]] = {c.value: [] for c in CaseClass}
        for case in cases:
            by_class[classified[case.id].label.value].append(case.id)
        supported_ids = [c.id for c in cases if classified[c.id].supported]
        pairs = _pair_analysis(cases, classified, self.model_names)
        # Denominator is MEASURED cases, not all cases: an unstable case is
        # unknown, not discriminating, and counting it as either would flatter
        # or slander the eval set. With repeats=1 nothing can be unstable, so
        # this is identical to the old behaviour.
        share = non_discriminating / n_measured if n_measured else 0.0

        stats: dict[str, Any] = {
            # n_cases is the number actually SCORED, so it stays consistent with
            # n_scorer_calls and every derived figure. n_cases_in_set is the
            # whole set, and the two differing is the signal that this was a
            # sample.
            "n_cases": n_cases,
            "n_cases_in_set": len(all_cases),
            "sampled": sampled,
            "sample_seed": self.sample_seed if sampled else None,
            "models": list(self.model_names),
            "repeats": self.repeats,
            "max_workers": self.max_workers,
            "n_scorer_calls": n_cases * len(self.models) * self.repeats,
            "pass_threshold": self.pass_threshold,
            "n_measured": n_measured,
            "n_unstable": len(unstable),
            "n_discriminating": n_measured - non_discriminating,
            "n_non_discriminating": non_discriminating,
            "non_discriminating_share": share,
            "effective_n_cases": n_measured - non_discriminating,
            "n_ceiling": len(ceiling),
            "n_floor": len(floor),
            "n_inverted": len(inverted),
            # Mean pass rate across every call, so it stays meaningful when
            # verdicts disagree between repeats.
            "per_model_pass_rate": {
                name: sum(sum(runs[c.id][i]) for c in cases)
                / (n_cases * self.repeats)
                for i, name in enumerate(self.model_names)
            },
            "ceiling_case_ids": ceiling,
            "floor_case_ids": floor,
            "inverted_case_ids": inverted,
            "unstable_case_ids": unstable,
        }

        # --- separation analysis (denominator is always n_cases) ------------
        n_limited = len(by_class["ceiling"]) + len(by_class["floor"])
        stats.update(
            confidence_level=CONFIDENCE_LEVEL,
            case_classes={cid: cls.label.value for cid, cls in classified.items()},
            n_by_class={k: len(v) for k, v in by_class.items()},
            case_ids_by_class=by_class,
            n_limited_evidence=n_limited,
            limited_evidence_share=n_limited / n_cases if n_cases else 0.0,
            limited_evidence_interval=list(wilson_interval(n_limited, n_cases)),
            n_separating=len(by_class["separating"]),
            separating_share=(
                len(by_class["separating"]) / n_cases if n_cases else 0.0
            ),
            separating_interval=list(
                wilson_interval(len(by_class["separating"]), n_cases)
            ),
            # How many per-case classifications the repeats actually support.
            # With repeats=1 this is 0 by construction: a single observation
            # cannot bound a per-case success probability.
            n_statistically_supported=len(supported_ids),
            statistically_supported_case_ids=supported_ids,
            model_pairs=pairs,
            separated_pairs=[p["pair"] for p in pairs if p["verdict"] == "separated"],
            contradicted_pairs=[
                p["pair"] for p in pairs if p["verdict"] == "ordering_contradicted"
            ],
            unresolved_pairs=[
                p["pair"] for p in pairs if p["verdict"] == "unresolved"
            ],
        )

        findings: list[Finding] = []
        if unstable:
            # First, and a WARNING: if verdicts are not reproducible then every
            # other number here is provisional, so this must be read before them.
            findings.append(
                Finding(
                    f"{len(unstable)} of {n_cases} cases gave a DIFFERENT verdict "
                    f"across {self.repeats} identical runs, so they cannot be "
                    f"classified and are excluded from the counts below. A verdict "
                    f"that changes between runs is sampling noise, not a property "
                    f"of the eval set — the remaining figures describe only the "
                    f"{n_measured} reproducible cases",
                    case_ids=tuple(unstable),
                )
            )
        # stats is dict[str, Any], so these need naming to stay checkable.
        limited_share = float(stats["limited_evidence_share"])
        lo, hi = (float(x) for x in stats["limited_evidence_interval"])
        if limited_share > self.max_non_discriminating_share:
            findings.append(
                Finding(
                    f"{limited_share:.0%} of cases ({n_limited}/{n_cases}, 95% CI "
                    f"{lo:.0%}-{hi:.0%}) provide limited evidence for model "
                    f"separation: every model returned the same verdict. That is "
                    f"not a defect in those cases — ceiling cases can be "
                    f"deliberate regression guards — but it means this eval "
                    f"separates these models about as well as a "
                    f"{stats['n_separating']}-case set would"
                )
            )
        if ceiling:
            # INFO, not WARNING. Every eval has easy cases, and some are
            # deliberate regression guards. Warning about them would flatly
            # contradict this check's own third limitation. Floor and inverted
            # cases DO get a warning, because they suggest something is broken
            # rather than merely easy.
            findings.append(
                Finding(
                    f"every model passes {_n_cases(len(ceiling))}, including "
                    f"'{self.model_names[0]}', so these provide limited evidence "
                    f"for model separation. They may still be worth keeping as "
                    f"regression guards",
                    severity=Severity.INFO,
                    case_ids=tuple(ceiling),
                )
            )
        if floor:
            findings.append(
                Finding(
                    f"every model fails {_n_cases(len(floor))}, including "
                    f"'{self.model_names[-1]}', so these provide limited evidence "
                    f"for model separation. Check the reference answer and the "
                    f"grader before assuming difficulty — a case nothing passes "
                    f"is more often wrong than hard",
                    case_ids=tuple(floor),
                )
            )
        if inverted:
            findings.append(
                Finding(
                    f"in {_n_cases(len(inverted))} a weaker model passed where a "
                    f"stronger one failed. That ordering is suspicious: it "
                    f"usually means a wrong reference answer or a grader "
                    f"artefact rather than a real capability gap",
                    case_ids=tuple(inverted),
                )
            )
        non_monotonic = by_class["non_monotonic"]
        if non_monotonic and set(non_monotonic) != set(inverted):
            findings.append(
                Finding(
                    f"{_n_cases(len(non_monotonic))} are NON-MONOTONIC on the "
                    f"majority verdict: a weaker model passed where a stronger "
                    f"one failed. These do separate the models, but in the "
                    f"direction the declared ordering says is impossible, so "
                    f"suspect the reference answer, the grader, or the ordering "
                    f"itself before a real capability gap",
                    case_ids=tuple(non_monotonic),
                )
            )
        indeterminate = by_class["indeterminate"]
        if indeterminate:
            findings.append(
                Finding(
                    f"{_n_cases(len(indeterminate))} could not be assigned a "
                    f"verdict: the {self.repeats} repeats split exactly evenly for "
                    f"at least one model. An even repeat count makes ties "
                    f"reachable — use an odd number",
                    severity=Severity.INFO,
                    case_ids=tuple(indeterminate),
                )
            )
        findings.extend(self._pair_findings(stats))
        findings.extend(self._sanity_findings(stats, non_discriminating, n_measured))

        summary = (
            # Lead with the sample, before any number it qualifies. A reader who
            # stops after the first clause must still know this is a subset.
            (f"SAMPLE of {n_cases} of {len(all_cases)} cases" if sampled
             else f"{n_cases} cases")
            + f" x {len(self.models)} models"
            + (f" x {self.repeats} repeats" if self.repeats > 1 else "")
            + f": {stats['n_separating']} separate models, "
            f"{n_limited} provide limited evidence ({limited_share:.0%})"
            + (f", {len(unstable)} not reproducible" if unstable else "")
        )
        limitations: tuple[str, ...] = LIMITATIONS
        if sampled:
            limitations = (
                f"Only {n_cases} of {len(all_cases)} cases were scored "
                f"(random sample, seed {self.sample_seed}). Every figure above "
                f"describes THAT SAMPLE, not your eval set. A clean sample does "
                f"not mean a clean set, and a broken case that was not drawn is "
                f"not reported here — absence of a finding is not evidence.",
            ) + LIMITATIONS
        return CheckResult(
            check=self.name,
            summary=summary,
            findings=tuple(findings),
            stats=stats,
            limitations=limitations,
        )

    def _pair_findings(self, stats: dict) -> list[Finding]:
        """Report which capability levels this eval actually separates.

        The headline share says how much of the eval does separating work; it
        does not say BETWEEN WHOM. An eval can separate the weakest model from
        the strongest while resolving nothing in between, which changes what you
        may conclude from it entirely.
        """
        findings: list[Finding] = []
        for pair in stats["model_pairs"]:
            weak, strong = pair["pair"]
            lo, hi = pair["delta_interval"]
            if pair["verdict"] == "separated":
                findings.append(
                    Finding(
                        f"'{strong}' outperforms '{weak}' by "
                        f"{pair['delta']:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}, "
                        f"McNemar p={pair['mcnemar_p']:.3g}) — this eval does "
                        f"separate that pair"
                        + ("" if pair["interval_reliable"] else
                           f"; only {pair['n_discordant']} discordant cases, so "
                           f"the interval is indicative"),
                        severity=Severity.INFO,
                    )
                )
            elif pair["verdict"] == "ordering_contradicted":
                findings.append(
                    Finding(
                        f"'{weak}' outperformed '{strong}' by "
                        f"{-pair['delta']:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}, "
                        f"McNemar p={pair['mcnemar_p']:.3g}), contradicting the "
                        f"capability order you declared. Either the ordering is "
                        f"wrong or the references favour the weaker model"
                    )
                )
            elif pair["verdict"] == "no_information":
                findings.append(
                    Finding(
                        f"'{weak}' and '{strong}' never disagreed on any of the "
                        f"{pair['n_comparable']} cases, so this eval provides NO "
                        f"information about that pair. That is not evidence they "
                        f"are equally capable — it is the absence of evidence "
                        f"either way",
                        severity=Severity.INFO,
                    )
                )
            elif pair["verdict"] == "unresolved":
                mde = pair["minimum_detectable_effect"]
                caveat = (
                    "" if pair["interval_reliable"]
                    else f" (only {pair['n_discordant']} discordant cases, so "
                         f"treat the interval as indicative)"
                )
                findings.append(
                    Finding(
                        f"'{weak}' vs '{strong}' is UNRESOLVED: observed "
                        f"difference {pair['delta']:+.1%}, McNemar "
                        f"p={pair['mcnemar_p']:.3g}. With {pair['n_discordant']} "
                        f"discordant of {pair['n_comparable']} cases this eval "
                        f"could only have detected a difference of about "
                        f"{mde:.1%} or larger, so it cannot support a claim "
                        f"either way about this pair{caveat}",
                        severity=Severity.INFO,
                    )
                )
        return findings

    def _sanity_findings(
        self, stats: dict, non_discriminating: int, n_cases: int
    ) -> list[Finding]:
        """Checks on the SETUP rather than the eval set.

        If these fire, the numbers above are probably not measuring what the
        user thinks they are, so it matters more that they are surfaced than
        that they fit the check's stated job.
        """
        findings = []
        rates = stats["per_model_pass_rate"]
        weakest, strongest = self.model_names[0], self.model_names[-1]
        if rates[weakest] > rates[strongest]:
            findings.append(
                Finding(
                    f"'{weakest}' is listed as the weakest model but passed "
                    f"{rates[weakest]:.0%} of cases against "
                    f"{rates[strongest]:.0%} for '{strongest}'. Either the "
                    f"models are in the wrong order or the scorer is inverted",
                    case_ids=(),
                )
            )
        if n_cases and non_discriminating == n_cases:
            # Same rule as the share finding: name the narrowed scope only when
            # the scope was actually narrowed.
            scope = "no measured case" if stats["n_unstable"] else "no case"
            findings.append(
                Finding(
                    f"{scope} separated these models at all. They may be closer "
                    "in capability than assumed, or the scorer may not be "
                    "sensitive enough to tell them apart",
                    severity=Severity.INFO,
                )
            )
        if not n_cases:
            findings.append(
                Finding(
                    "not one case produced a reproducible verdict, so this run "
                    "measured nothing. Either the scorer is highly "
                    "non-deterministic or the two models are indistinguishable "
                    "on this data",
                    severity=Severity.INFO,
                )
            )
        return findings

    def _score_all(
        self, cases: list[EvalCase]
    ) -> dict[str, tuple[tuple[bool, ...], ...]]:
        """Every (case, model, repeat) verdict, keyed by case id.

        The unit of work is one scorer call, not one case, because a case with
        two models and three repeats is six independent I/O-bound calls and
        there is no reason for them to wait on each other.

        ``max_workers == 1`` takes a plainly serial path rather than a pool of
        one: it keeps the default free of thread-pool semantics entirely, so a
        user whose scorer is not thread-safe is unaffected unless they opt in.
        """
        tasks = [
            (case, model_index, model, name, repeat)
            for case in cases
            for model_index, (model, name) in enumerate(
                zip(self.models, self.model_names, strict=True)
            )
            for repeat in range(self.repeats)
        ]

        if self.max_workers == 1:
            outcomes: list[bool | BaseException] = []
            for case, _, model, name, _ in tasks:
                # Serial path fails fast, exactly as it always has.
                outcomes.append(self._score_one(case, model, name))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [
                    pool.submit(self._score_one, case, model, name)
                    for case, _, model, name, _ in tasks
                ]
                # Collect everything, including failures, before deciding what
                # to raise. Raising from whichever thread happened to finish
                # first would make the error message depend on scheduling.
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except BaseException as exc:  # noqa: BLE001 - re-raised below
                        outcomes.append(exc)

            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    # Task order is submission order, so the reported failure is
                    # the same one the serial path would have hit.
                    raise outcome

        runs: dict[str, list[list[bool]]] = {
            case.id: [[] for _ in self.models] for case in cases
        }
        for (case, model_index, _, _, _), outcome in zip(
            tasks, outcomes, strict=True
        ):
            runs[case.id][model_index].append(bool(outcome))

        return {
            case_id: tuple(tuple(per_model) for per_model in per_case)
            for case_id, per_case in runs.items()
        }

    def _verdicts_for(self, case: EvalCase) -> tuple[bool, ...]:
        return tuple(
            self._score_one(case, model, name)
            for model, name in zip(self.models, self.model_names, strict=True)
        )

    def _score_one(self, case: EvalCase, model: Any, name: str) -> bool:
        try:
            raw = self.scorer(case, model)
        except Exception as exc:
            # Scoring is the expensive part of this tool. If it breaks on case
            # 400 of 500, the user needs to know exactly where without digging
            # through a traceback from their own scorer.
            raise ScorerError(
                f"scorer raised on case '{case.id}' with model '{name}': {exc}"
            ) from exc

        # numpy / pandas scalars are the NORMAL return type for a scorer built
        # on array maths: `y_pred == y_true` gives np.bool_, not bool. Those are
        # not Python bool or int, so without this they were rejected — and with
        # a baffling message, since type(np.True_).__name__ is itself "bool".
        # Unwrapping via .item() keeps this module dependency-free: it never
        # imports numpy, which matters for the check that is meant to work with
        # anyone's stack. A real array fails .item() and falls through to the
        # error below, which is what should happen.
        if not isinstance(raw, (bool, int, float)) and hasattr(raw, "item"):
            try:
                raw = raw.item()
            except (ValueError, AttributeError):
                pass

        value = _unwrap_scalar(raw)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) >= self.pass_threshold
        raise ScorerError(
            f"scorer returned {type(raw).__name__} for case '{case.id}' with "
            f"model '{name}': expected bool or float"
        )


def _unwrap_scalar(value: Any) -> Any:
    """Return the Python equivalent of a 0-dimensional array scalar.

    numpy scalars are not Python builtins: np.bool_ is not a bool and np.int64
    is not an int. A scorer written with array maths — `y_pred == y_true` is
    the obvious way to write one — returns exactly those, so rejecting them
    would make "works with any provider" hollow. np.bool_ is the nastiest of
    them because its type name is literally "bool", which produced the
    baffling error "returned bool ... expected bool or float".

    Duck-typing on ndim instead of importing numpy means a torch or jax scalar
    works too, while a real array (ndim >= 1) still falls through and is
    rejected.
    """
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        return value.item()
    return value


def _n_cases(n: int) -> str:
    return "1 case" if n == 1 else f"{n} cases"


class CaseClass(str, Enum):
    """What a case tells you about the models, not whether the case is good.

    None of these is a defect label. A ceiling case is not a bad case: it may be
    a deliberate regression guard, and every eval needs some. What the taxonomy
    records is how much EVIDENCE ABOUT MODEL SEPARATION a case carries, which is
    a different question from whether it belongs in the set.
    """

    #: Every model passed. Provides limited evidence for model separation.
    CEILING = "ceiling"
    #: Every model failed. Provides limited evidence for model separation, and
    #: may indicate a wrong reference answer rather than a hard case.
    FLOOR = "floor"
    #: Verdicts follow the declared capability order with at least one change.
    SEPARATING = "separating"
    #: A weaker model passed where a stronger one failed, contradicting the
    #: declared order. Evidence about the reference or the ordering, not
    #: necessarily about capability.
    NON_MONOTONIC = "non_monotonic"
    #: The repeats did not agree well enough to assign a verdict at all.
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class CaseSeparation:
    """One case's classification, with the verdicts it was derived from."""

    label: CaseClass
    verdicts: tuple[bool | None, ...]
    successes: tuple[int, ...]
    #: True when every model's per-case success interval excludes 0.5, i.e. the
    #: repeats are sufficient to resolve each verdict on their own.
    supported: bool


def _majority(verdicts: tuple[bool, ...]) -> bool | None:
    """Majority verdict, or None on an exact tie.

    Majority rather than unanimity, and this is the central methodological
    change. Requiring unanimity and discarding the rest biases the result: for a
    model passing with probability p, a unanimous run is far more likely when p
    is extreme, so the surviving cases are enriched for ceiling and floor -- the
    very categories being counted. Raising `repeats` then inflates the
    non-discriminating share rather than measuring it better.

    A tie is only reachable with an even number of repeats, and returns None so
    the case is reported INDETERMINATE rather than broken arbitrarily. Odd
    repeat counts are therefore preferable, which the check says in its
    limitations.
    """
    if not verdicts:
        return None
    successes = sum(verdicts)
    trials = len(verdicts)
    if 2 * successes == trials:
        return None
    return 2 * successes > trials


def _classify_case(per_model: Sequence[tuple[bool, ...]], repeats: int) -> CaseSeparation:
    """Classify one case from its per-model repeat verdicts."""
    verdicts = tuple(_majority(r) for r in per_model)
    successes = tuple(sum(r) for r in per_model)

    # A per-case verdict is statistically supported only if the interval for
    # that model's success probability on this case excludes 0.5. At repeats=1
    # the interval is [0.21, 1.0] or [0.0, 0.79], so nothing is ever supported
    # -- which is the correct answer, not a limitation of the code.
    supported = all(
        _excludes(wilson_interval(k, repeats), 0.5) for k in successes
    ) and None not in verdicts

    if None in verdicts:
        label = CaseClass.INDETERMINATE
    elif all(verdicts):
        label = CaseClass.CEILING
    elif not any(verdicts):
        label = CaseClass.FLOOR
    elif _is_inverted([bool(v) for v in verdicts]):
        label = CaseClass.NON_MONOTONIC
    else:
        label = CaseClass.SEPARATING

    return CaseSeparation(
        label=label, verdicts=verdicts, successes=successes, supported=supported
    )


def _excludes(interval: tuple[float, float], value: float) -> bool:
    lo, hi = interval
    return value < lo or value > hi


def _pair_analysis(
    cases: Sequence[EvalCase],
    classified: dict[str, CaseSeparation],
    model_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Which model pairs this eval actually separates, with uncertainty.

    Every ordered pair is reported, not only adjacent ones: an eval can separate
    the weakest from the strongest while resolving nothing in between, and that
    is exactly the kind of thing a user needs told.

    The comparison is PAIRED -- both models saw the same cases -- so it uses
    McNemar rather than two independent-proportion tests. Only discordant cases
    carry information about which model is better.
    """
    out: list[dict[str, Any]] = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            weaker_passed = stronger_passed = comparable = 0
            for case in cases:
                v = classified[case.id].verdicts
                if v[i] is None or v[j] is None:
                    continue
                comparable += 1
                if v[i] and not v[j]:
                    weaker_passed += 1
                elif v[j] and not v[i]:
                    stronger_passed += 1

            delta, interval, standard_error = paired_difference(
                weaker_passed, stronger_passed, comparable
            )
            p_value = mcnemar_exact_p(weaker_passed, stronger_passed)
            p_discordant = (
                (weaker_passed + stronger_passed) / comparable if comparable else 0.0
            )
            mde = minimum_detectable_effect(p_discordant, comparable)

            # The DECISION comes from the exact test, not from the interval.
            # They disagree at small discordant counts, and there the exact test
            # is the correct one: 3 discordant cases all favouring one model
            # give a Wald interval excluding zero but an exact p of 0.25.
            # The interval is retained to describe MAGNITUDE, flagged when the
            # normal approximation behind it is not trustworthy.
            discordant = weaker_passed + stronger_passed
            if comparable == 0:
                verdict = "not_comparable"
            elif discordant == 0:
                # Not merely unresolved: the two models never once disagreed, so
                # this eval carries no information about the pair at all.
                verdict = "no_information"
            elif p_value < ALPHA and delta > 0:
                verdict = "separated"
            elif p_value < ALPHA and delta < 0:
                verdict = "ordering_contradicted"
            else:
                verdict = "unresolved"

            out.append(
                {
                    "pair": [model_names[i], model_names[j]],
                    "adjacent": j == i + 1,
                    "n_comparable": comparable,
                    "n_stronger_only": stronger_passed,
                    "n_weaker_only": weaker_passed,
                    "n_discordant": weaker_passed + stronger_passed,
                    "delta": delta,
                    "delta_interval": list(interval),
                    "standard_error": standard_error,
                    "mcnemar_p": p_value,
                    # None, not 0.0, when nothing was discordant. A zero would
                    # read as "can detect any difference", the exact opposite of
                    # what no disagreement means.
                    "minimum_detectable_effect": mde if discordant else None,
                    "interval_reliable": discordant >= NORMAL_APPROX_MIN_DISCORDANT,
                    "verdict": verdict,
                }
            )
    return out


def _is_inverted(verdicts: Sequence[bool]) -> bool:
    """True if a weaker model passed where a stronger one failed.

    Verdicts are ordered weakest to strongest, so a correct pattern never has
    a True before a False. Anything else contradicts the stated ordering.
    """
    return any(
        verdicts[i] and not verdicts[j]
        for i in range(len(verdicts))
        for j in range(i + 1, len(verdicts))
    )
