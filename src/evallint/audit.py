"""The unified audit report.

## What replaces the single score

There is no overall number. A test asserts there is no field that could be
mistaken for one, because the temptation to add "Eval Trust Score: 78/100" is
the strongest bad idea in this problem space: it would need weights across
incommensurable evidence (is 12% redundancy worth more or less than an unmeasured
judge?), those weights would be invented, and a reader would act on the number
without reading the evidence under it.

What replaces it is the **evidence tier**, designed in `docs/architecture-v2.md`
and enforced here for the first time:

    OBSERVED     A census of the data. Counting, not inference. Exact.
                 "3 of 20 cases carry the label 'refund'."
    HEURISTIC    A signal from a chosen threshold with no sampling theory
                 behind it. Carries the source's own confidence, and never
                 upgrades. "cosine >= 0.85, so these two look redundant."
    ESTIMATED    A statistical estimate with an interval and stated
                 assumptions. "58% pass rate, 95% CI [0.41, 0.73]."
    RECOMMENDED  An action. Inherits the WEAKEST tier it cites.

Two rules, both enforced in `__post_init__` rather than left to discipline:

  1. **No promotion.** An `OBSERVED` finding may not carry a probabilistic
     confidence, because it is certain by construction and attaching "HIGH"
     to it would imply a doubt that does not exist. A `HEURISTIC` finding
     MUST carry one, because a threshold-derived signal without a confidence
     reads as a fact.
  2. **No silent pass.** A section that could not run is `NOT_RUN` with the
     reason, never an empty `RAN`. "No findings" and "not looked at" are
     opposite states and collapsing them is the exact failure this package
     reports in eval sets.

## Sections 1, 2, 3 and 11 are views, not analyses

Executive summary, critical findings, warnings and recommendations are derived
from the findings in sections 4-10. They are computed, not written, so a finding
cannot appear in "critical findings" phrased differently from how it appears in
its own section.

## What this report does not say

It never says an eval is good or bad. It reports what was observed, what was
inferred and on what threshold, what could not be assessed and why, and what
each of those implies for a conclusion drawn from this eval. Whether that is
acceptable is the reader's call, and the tool does not have the information to
make it — an eval with 40% redundancy may be a deliberate consistency suite.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .checks.base import CheckResult, Finding, Severity
from .schema import EvalSet

__all__ = [
    "SECTION_ORDER",
    "AuditFinding",
    "AuditReport",
    "AuditSection",
    "SectionStatus",
    "Tier",
    "render_html",
    "render_terminal",
    "run_audit",
    "to_json",
]


class Tier(StrEnum):
    """How much weight a claim can bear. See the module docstring."""

    OBSERVED = "observed"
    HEURISTIC = "heuristic"
    ESTIMATED = "estimated"
    RECOMMENDED = "recommended"


#: Weakest first, for tier inheritance on recommendations.
_TIER_STRENGTH = {
    Tier.HEURISTIC: 0,
    Tier.RECOMMENDED: 1,
    Tier.ESTIMATED: 2,
    Tier.OBSERVED: 3,
}


class SectionStatus(StrEnum):
    """Whether a section's analysis actually happened.

    ``NOT_RUN`` is not a mild version of ``RAN`` with nothing found. It means
    nobody looked, and the report says so wherever the section appears.
    """

    RAN = "ran"
    PARTIAL = "partial"
    NOT_RUN = "not_run"


#: The eleven sections, in report order. Four of them are views over the others.
SECTION_ORDER = (
    ("executive_summary", "Executive summary"),
    ("critical_findings", "Critical findings"),
    ("warnings", "Warnings"),
    ("dataset_statistics", "Dataset statistics"),
    ("redundancy", "Redundancy analysis"),
    ("discrimination", "Discrimination analysis"),
    ("ground_truth", "Ground-truth analysis"),
    ("evaluator_reliability", "Evaluator reliability"),
    ("statistical_reliability", "Statistical reliability"),
    ("reproducibility", "Reproducibility"),
    ("recommendations", "Recommendations"),
)

#: Sections that hold their own findings. The other four are derived views.
ANALYSIS_SECTIONS = (
    "dataset_statistics",
    "redundancy",
    "discrimination",
    "ground_truth",
    "evaluator_reliability",
    "statistical_reliability",
    "reproducibility",
)


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One thing found, with everything needed to judge whether to act on it.

    Every field is required. A finding without a stated limitation invites a
    reader to treat a heuristic as a fact, and a finding without a recommended
    action leaves them to guess -- both of which this package exists to avoid.
    """

    section: str
    severity: Severity
    tier: Tier
    evidence: str
    explanation: str
    limitation: str
    recommended_action: str
    case_ids: tuple[str, ...] = ()
    #: The source analyser's own confidence, e.g. leakage's LOW/MEDIUM/HIGH.
    #: None for OBSERVED findings, which are certain by construction.
    confidence: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_ids", tuple(self.case_ids))
        for name in ("evidence", "explanation", "limitation", "recommended_action"):
            if not str(getattr(self, name)).strip():
                raise ValueError(
                    f"finding in section '{self.section}' has no {name}; every "
                    "finding must carry all seven fields"
                )
        # Rule 1, enforced rather than documented. An OBSERVED claim is a count:
        # attaching "HIGH confidence" to it implies a doubt that does not exist,
        # and a HEURISTIC claim without a confidence reads as a fact.
        if self.tier is Tier.OBSERVED and self.confidence is not None:
            raise ValueError(
                f"OBSERVED finding in '{self.section}' carries confidence "
                f"{self.confidence!r}. A census is exact; a confidence on it "
                "implies a doubt that does not exist."
            )
        if self.tier is Tier.HEURISTIC and self.confidence is None:
            raise ValueError(
                f"HEURISTIC finding in '{self.section}' carries no confidence. "
                "A threshold-derived signal without one reads as a fact."
            )

    @property
    def n_cases(self) -> int:
        return len(self.case_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "severity": str(self.severity),
            "tier": str(self.tier),
            "confidence": self.confidence,
            "evidence": self.evidence,
            "affected_cases": list(self.case_ids),
            "n_affected_cases": len(self.case_ids),
            "explanation": self.explanation,
            "limitation": self.limitation,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True, slots=True)
class AuditSection:
    """One analysis, its findings, and whether it ran at all."""

    key: str
    title: str
    status: SectionStatus
    summary: str
    findings: tuple[AuditFinding, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    #: Why the analysis did not run, and what it would take. Required when the
    #: status is NOT_RUN or PARTIAL: "not run" without a reason is unactionable.
    not_run_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if self.status is not SectionStatus.RAN and not self.not_run_reason:
            raise ValueError(
                f"section '{self.key}' is {self.status} with no reason; the "
                "report must say what it would take to run it"
            )
        if self.status is SectionStatus.NOT_RUN and self.findings:
            raise ValueError(
                f"section '{self.key}' did not run but reports "
                f"{len(self.findings)} finding(s)"
            )

    @property
    def short_reason(self) -> str | None:
        """First sentence of `not_run_reason`, for the terminal view.

        The full reason names the exact call to make, which is what someone
        needs and also three lines long. The terminal stays scannable; the
        complete text is in `--format json` and `--format html`.
        """
        if not self.not_run_reason:
            return None
        head, _, rest = self.not_run_reason.partition(". ")
        return f"{head}." if rest else self.not_run_reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": str(self.status),
            "summary": self.summary,
            "not_run_reason": self.not_run_reason,
            "findings": [f.as_dict() for f in self.findings],
            "stats": _plain(self.stats),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Everything the audit has to say. Deliberately without an overall score.

    ``evidence_coverage`` is what a score would have been asked to summarise,
    and it refuses to compress: it lists which analyses produced evidence and
    which did not. Four of seven analyses run from a file alone; the other three
    need model runs, judges or repeated runs, and their absence is a gap in the
    evidence rather than a point deduction.
    """

    source: str
    n_cases: int
    schema_version: int
    sections: tuple[AuditSection, ...]
    evallint_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))

    def section(self, key: str) -> AuditSection:
        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(key)

    @property
    def findings(self) -> tuple[AuditFinding, ...]:
        return tuple(f for s in self.sections for f in s.findings)

    @property
    def critical(self) -> tuple[AuditFinding, ...]:
        """WARNING-severity findings, most-cases-affected first.

        "Critical" here means "would distort a number you are about to quote",
        which is what Severity.WARNING has always meant in this package. It does
        NOT mean "your eval is broken", and the report never says that.
        """
        return tuple(
            sorted(
                (f for f in self.findings if f.severity is Severity.WARNING),
                key=lambda f: (-f.n_cases, f.section),
            )
        )

    @property
    def informational(self) -> tuple[AuditFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.INFO)

    @property
    def ran(self) -> tuple[AuditSection, ...]:
        return tuple(
            s
            for s in self.sections
            if s.key in ANALYSIS_SECTIONS and s.status is not SectionStatus.NOT_RUN
        )

    @property
    def not_run(self) -> tuple[AuditSection, ...]:
        return tuple(
            s
            for s in self.sections
            if s.key in ANALYSIS_SECTIONS and s.status is SectionStatus.NOT_RUN
        )

    @property
    def evidence_coverage(self) -> dict[str, Any]:
        return {
            "analyses_available": len(ANALYSIS_SECTIONS),
            "analyses_with_evidence": [s.key for s in self.ran],
            "analyses_without_evidence": {
                s.key: s.not_run_reason for s in self.not_run
            },
        }

    @property
    def recommendations(self) -> tuple[AuditFinding, ...]:
        """Deduplicated actions, weakest-tier first.

        Tier inheritance: a recommendation is only as strong as the finding it
        rests on, so the tier shown is the cited finding's own. Sorting weakest
        first is deliberate -- it puts "this rests on a threshold" above "this
        is a count", so nobody acts on a heuristic thinking it was a census.
        """
        seen: dict[str, AuditFinding] = {}
        for finding in self.findings:
            seen.setdefault(finding.recommended_action, finding)
        return tuple(
            sorted(seen.values(), key=lambda f: _TIER_STRENGTH[f.tier])
        )

    def executive_summary(self) -> tuple[str, ...]:
        """Facts and gaps. No grade, and nothing that could be read as one."""
        lines = [
            f"{self.n_cases} cases from {self.source} (schema version "
            f"{self.schema_version}).",
            f"{len(self.ran)} of {len(ANALYSIS_SECTIONS)} analyses produced "
            f"evidence; {len(self.not_run)} could not run.",
        ]
        n_warn, n_info = len(self.critical), len(self.informational)
        if n_warn or n_info:
            lines.append(
                f"{n_warn} finding(s) that could distort a reported number, "
                f"{n_info} informational."
            )
        else:
            lines.append(
                "No findings from the analyses that ran. That is evidence about "
                "those analyses only, not a clean bill of health."
            )
        if self.not_run:
            lines.append(
                "NOT ASSESSED: "
                + "; ".join(f"{s.title.lower()}" for s in self.not_run)
                + ". A conclusion drawn from this eval is unsupported in those "
                "respects — not supported-with-caveats."
            )
        lines.append(
            "This report contains no overall quality score, deliberately. "
            "Weighing redundancy against an unmeasured judge would need invented "
            "weights, and a reader would act on the number instead of the "
            "evidence."
        )
        return tuple(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evallint_version": self.evallint_version,
            "source": self.source,
            "n_cases": self.n_cases,
            "schema_version": self.schema_version,
            "executive_summary": list(self.executive_summary()),
            "evidence_coverage": self.evidence_coverage,
            "critical_findings": [f.as_dict() for f in self.critical],
            "warnings": [f.as_dict() for f in self.informational],
            "recommendations": [
                {
                    "action": f.recommended_action,
                    "tier": str(f.tier),
                    "because": f.evidence,
                    "section": f.section,
                }
                for f in self.recommendations
            ],
            "sections": [s.as_dict() for s in self.sections],
        }


# ==========================================================================
# Adapting existing analysers into the unified finding shape
# ==========================================================================

#: Which tier each check's plain findings belong to, and why. Recorded here
#: rather than guessed per finding, because the tier is a property of the METHOD
#: and getting it wrong in either direction misleads: calling a count heuristic
#: invites doubt about arithmetic, calling a threshold observed hides that a
#: number was chosen.
_CHECK_TIER = {
    # Counts and ratios over the file. Arithmetic, not inference.
    "imbalance": Tier.OBSERVED,
    # Cell occupancy against a user-declared space. A census of THEIR space, so
    # observed -- the choice of space is theirs and is reported as a finding.
    "coverage": Tier.OBSERVED,
    # Substring and word-boundary matching against a chosen marker list.
    "leakage": Tier.HEURISTIC,
    # Pattern matching plus a chosen share threshold.
    "ground_truth": Tier.HEURISTIC,
    # Exact and normalised matching are OBSERVED; the semantic level is a cosine
    # threshold. The mixed case is resolved per finding, below.
    "redundancy": Tier.HEURISTIC,
    "discrimination": Tier.ESTIMATED,
}

#: Confidence assigned to a HEURISTIC finding whose source did not compute one.
#: Named rather than inlined so it cannot be mistaken for a measurement.
_UNSTATED = "not stated by the source analyser"


def _adapt(
    result: CheckResult,
    section: str,
    *,
    tier: Tier | None = None,
    action: str | None = None,
) -> tuple[AuditFinding, ...]:
    """Lift a CheckResult's findings into the unified shape.

    The limitation attached to each finding is the check's OWN stated
    limitation, joined -- not a new one written here. A limitation invented at
    the reporting layer could contradict the analyser that produced the finding.
    """
    base_tier = tier or _CHECK_TIER.get(result.check, Tier.HEURISTIC)
    limitation = " ".join(result.limitations) or "none stated"
    out = []
    for finding in result.findings:
        finding_tier = _finding_tier(result.check, finding, base_tier)
        out.append(
            AuditFinding(
                section=section,
                severity=finding.severity,
                tier=finding_tier,
                confidence=(
                    None if finding_tier is Tier.OBSERVED else _UNSTATED
                ),
                evidence=finding.message,
                explanation=result.summary,
                limitation=limitation,
                recommended_action=action or _default_action(result.check),
                case_ids=finding.case_ids,
            )
        )
    return tuple(out)


def _finding_tier(check: str, finding: Finding, default: Tier) -> Tier:
    """Resolve the mixed cases, where one check produces both kinds of claim."""
    message = finding.message.lower()
    if check == "redundancy":
        # An exact or normalised duplicate is a fact about bytes. Only the
        # semantic level involves a threshold.
        if "identical" in message or "exact" in message:
            return Tier.OBSERVED
    if check in ("imbalance", "coverage"):
        return Tier.OBSERVED
    return default


def _default_action(check: str) -> str:
    return {
        "imbalance": "Read the class counts before quoting a per-class "
        "accuracy; a class with few cases cannot produce a stable one.",
        "coverage": "Fill the empty cells, mark the impossible ones in the "
        "spec, or state which combinations your result does not cover.",
        "leakage": "Inspect the listed cases by hand. A confirmed leak means "
        "the case measures retrieval from the prompt rather than the ability "
        "you meant to test.",
        "ground_truth": "Have a human read the listed cases. An "
        "under-specified reference makes a correct answer scoreable as wrong.",
        "redundancy": "Decide whether the repetition is deliberate. If it is "
        "not, note that the aggregate score weights the repeated scenario "
        "once per copy.",
        "discrimination": "Check whether the non-separating cases are meant to "
        "be easy. If not, they consume budget without informing the comparison.",
    }.get(check, "Review the listed cases.")


# ==========================================================================
# Running the audit
# ==========================================================================


def run_audit(
    eval_set: EvalSet,
    *,
    imbalance: CheckResult | None = None,
    coverage: CheckResult | None = None,
    redundancy: CheckResult | None = None,
    leakage: CheckResult | None = None,
    ground_truth: CheckResult | None = None,
    discrimination: CheckResult | None = None,
    effective_size: Any | None = None,
    model_comparison: Any | None = None,
    evaluator_reliability: Any | None = None,
    reproducibility: Any | None = None,
    source: str | None = None,
) -> AuditReport:
    """Assemble a report from analyses that have ALREADY been run.

    Deliberately a pure composer: it runs no analyser and calls no model, so it
    cannot decide on your behalf to spend money or download a model. The CLI
    runs the file-only checks and hands them here; the three that need a scorer,
    judges or repeated runs are passed in by a caller who has them, and reported
    as NOT RUN otherwise.

    Every argument being None is a valid call and produces a report that says
    nothing was assessed -- which is the honest output for that input.
    """
    from . import __version__

    sections: list[AuditSection] = []

    sections.append(_stats_section(eval_set, imbalance, coverage))
    sections.append(
        _from_check(
            "redundancy",
            "Redundancy analysis",
            redundancy,
            "needs no model for the exact and normalised levels; the semantic "
            "level needs `pip install evallint[embeddings]`",
        )
    )
    sections.append(
        _from_check(
            "discrimination",
            "Discrimination analysis",
            discrimination,
            "needs a scoring function that runs your models against at least "
            "two of them. Pass DiscriminationCheck(scorer, models).run(...) "
            "into run_audit(discrimination=...)",
        )
    )
    sections.append(_ground_truth_section(ground_truth, leakage))
    sections.append(_reliability_section(evaluator_reliability))
    sections.append(_statistical_section(effective_size, model_comparison))
    sections.append(_reproducibility_section(reproducibility))

    report = AuditReport(
        source=source or eval_set.source or "(in memory)",
        n_cases=len(eval_set),
        schema_version=eval_set.schema_version,
        sections=tuple(sections),
        evallint_version=__version__,
    )
    return report


def _stats_section(
    eval_set: EvalSet,
    imbalance: CheckResult | None,
    coverage: CheckResult | None = None,
) -> AuditSection:
    """Dataset statistics: a census, so OBSERVED throughout.

    Class distribution and declared-space coverage share this section for the
    same reason leakage shares the ground-truth one: both are censuses of the
    file's composition, and the eleven requested sections have no separate slot.
    The summary names both so the grouping is not left to be inferred.
    """
    labelled = sum(1 for c in eval_set if c.label is not None)
    scoreable = sum(1 for c in eval_set if c.expected is not None)
    stats: dict[str, Any] = {
        "n_cases": len(eval_set),
        "n_labelled": labelled,
        "n_with_expected": scoreable,
        "n_chat_cases": eval_set.chat_cases,
        "n_with_system_prompt": sum(1 for c in eval_set if c.system is not None),
        "n_with_rubric": sum(1 for c in eval_set if c.rubric is not None),
        "n_with_tools": sum(1 for c in eval_set if c.tools),
        "n_with_multiple_acceptable": sum(1 for c in eval_set if c.acceptable),
        "n_derived_inputs": sum(1 for c in eval_set if c.input_is_derived),
        "schema_version": eval_set.schema_version,
    }
    summary = (
        f"{len(eval_set)} cases; {labelled} labelled, {scoreable} with a "
        f"reference answer, {eval_set.chat_cases} multi-turn"
    )
    findings: tuple[AuditFinding, ...] = ()
    limitations: tuple[str, ...] = (
        "These are counts of what the file contains. They say nothing about "
        "whether the cases are correct, well-chosen, or hard.",
    )
    if imbalance is not None:
        stats.update(imbalance.stats)
        findings = _adapt(imbalance, "dataset_statistics")
        limitations = imbalance.limitations
        summary = f"{summary}. {imbalance.summary}"
    if coverage is not None:
        stats["coverage"] = coverage.stats
        findings = findings + _adapt(coverage, "dataset_statistics")
        limitations = limitations + coverage.limitations
        summary = f"{summary}. Coverage: {coverage.summary}"
    return AuditSection(
        key="dataset_statistics",
        title="Dataset statistics",
        status=SectionStatus.RAN,
        summary=summary,
        findings=findings,
        stats=stats,
        limitations=limitations,
    )


def _from_check(
    key: str, title: str, result: CheckResult | None, reason: str
) -> AuditSection:
    if result is None:
        return AuditSection(
            key=key, title=title, status=SectionStatus.NOT_RUN,
            summary="Not assessed.", not_run_reason=reason,
        )
    return AuditSection(
        key=key,
        title=title,
        status=SectionStatus.PARTIAL if result.partial else SectionStatus.RAN,
        summary=result.summary,
        findings=_adapt(result, key),
        stats=result.stats,
        limitations=result.limitations,
        not_run_reason="; ".join(result.partial) if result.partial else None,
    )


def _ground_truth_section(
    ground_truth: CheckResult | None, leakage: CheckResult | None
) -> AuditSection:
    """Ground truth AND leakage.

    The eleven requested sections have no "leakage" entry, and dropping the
    leakage analyser would lose information the CLI already reports. They belong
    together because they answer the same question -- can this case be scored as
    intended? An under-specified reference and an answer sitting in the prompt
    are two different ways for the answer to fail to test what you meant. The
    grouping is stated in the summary so nobody has to infer it.
    """
    if ground_truth is None and leakage is None:
        return AuditSection(
            key="ground_truth",
            title="Ground-truth analysis",
            status=SectionStatus.NOT_RUN,
            summary="Not assessed.",
            not_run_reason=(
                "runs deterministically from the file, so this only happens if "
                "neither GroundTruthCheck nor LeakageCheck was passed in"
            ),
        )

    findings: list[AuditFinding] = []
    stats: dict[str, Any] = {}
    limitations: list[str] = []
    summaries: list[str] = []
    partial: list[str] = []

    for result, label in ((ground_truth, "references"), (leakage, "leakage")):
        if result is None:
            partial.append(f"{label} was not analysed")
            continue
        findings.extend(_adapt(result, "ground_truth"))
        stats[result.check] = result.stats
        limitations.extend(result.limitations)
        summaries.append(result.summary)
        partial.extend(result.partial)

    return AuditSection(
        key="ground_truth",
        title="Ground-truth analysis",
        status=SectionStatus.PARTIAL if partial else SectionStatus.RAN,
        summary="Under-specified references and answer leakage — "
        + "; ".join(summaries),
        findings=tuple(findings),
        stats=stats,
        limitations=tuple(limitations) or ("none stated",),
        not_run_reason="; ".join(partial) if partial else None,
    )


def _reliability_section(report: Any | None) -> AuditSection:
    """Evaluator reliability.

    The findings here are deliberately about the ABSENCE of evidence rather
    than about low agreement. A metric this module refused to compute --
    kappa undefined because every judge used one category, a correlation on
    too few items -- is the finding, because it means there is no
    chance-corrected evidence that the judge is reliable, and a score produced
    by that judge inherits that gap.

    No threshold is invented for "agreement too low". Every computed metric
    goes into `stats` with its own value, interval and interpretation, which
    the analyser wrote; inventing a pass mark here would be exactly the
    arbitrary cutoff this report exists to avoid.
    """
    if report is None:
        return AuditSection(
            key="evaluator_reliability",
            title="Evaluator reliability",
            status=SectionStatus.NOT_RUN,
            summary="Not assessed.",
            not_run_reason=(
                "needs repeated judge verdicts on the same cases. Collect them "
                "with collect_verdicts(...) and pass "
                "EvaluatorReliability().analyse(observations) into "
                "run_audit(evaluator_reliability=...). Without it, a score from "
                "an LLM judge has no measured reproducibility."
            ),
        )

    findings: list[AuditFinding] = []
    metrics = getattr(report, "metrics", {}) or {}
    limitations: list[str] = []
    for name, metric in metrics.items():
        limitations.extend(getattr(metric, "limitations", ()))
        reason = getattr(metric, "unmet_reason", None)
        if not reason:
            continue
        findings.append(
            AuditFinding(
                section="evaluator_reliability",
                severity=Severity.WARNING,
                # A refusal is a FACT about the design, not an estimate. So the
                # tier is OBSERVED and it carries no confidence: there is no
                # doubt that the metric was not computed.
                tier=Tier.OBSERVED,
                evidence=f"{name} was not computed: {reason}",
                explanation=(
                    "This metric was refused because its assumptions were not "
                    "met, so it provides no evidence either way. An absent "
                    "reliability measure is not a passing one."
                ),
                limitation=(
                    "Refusing to compute a metric says nothing about whether "
                    "the judge is reliable. It says only that this design "
                    "cannot tell you."
                ),
                recommended_action=(
                    "Change the design so the metric becomes computable, or "
                    "state in your results that judge reliability is unmeasured."
                ),
            )
        )

    for note in getattr(report, "notes", ()):
        if note.upper().startswith("WARNING"):
            findings.append(
                AuditFinding(
                    section="evaluator_reliability",
                    severity=Severity.WARNING,
                    tier=Tier.ESTIMATED,
                    confidence="see the interval on the metric named in it",
                    evidence=note,
                    explanation="Reported by the reliability analysis itself.",
                    limitation="Agreement is not correctness: judges from one "
                    "model family share their blind spots, so correlated error "
                    "is indistinguishable from reliability.",
                    recommended_action="Read the metric this refers to in the "
                    "section's statistics before quoting any judge-derived "
                    "number.",
                )
            )
        else:
            limitations.append(note)

    computed = [
        f"{name}={metric.value:.3f}"
        for name, metric in metrics.items()
        if getattr(metric, "value", None) is not None
    ]
    summary = (
        f"{getattr(report, 'n_judges', '?')} judges over "
        f"{getattr(report, 'n_items', '?')} items; "
        + (", ".join(computed) if computed else "no metric was computable")
    )
    return AuditSection(
        key="evaluator_reliability",
        title="Evaluator reliability",
        status=SectionStatus.RAN,
        summary=summary,
        findings=tuple(findings),
        stats=report.as_dict() if hasattr(report, "as_dict") else {},
        limitations=tuple(dict.fromkeys(limitations)) or ("none stated",),
    )


def _statistical_section(
    effective_size: Any | None, comparison: Any | None
) -> AuditSection:
    if effective_size is None and comparison is None:
        return AuditSection(
            key="statistical_reliability",
            title="Statistical reliability",
            status=SectionStatus.NOT_RUN,
            summary="Not assessed.",
            not_run_reason=(
                "needs either a redundancy estimate or per-case outcomes for "
                "two or more models. Pass "
                "effective_size=estimate_effective_size(eval_set) for "
                "redundancy-adjusted coverage, and/or "
                "model_comparison=compare_models(outcomes) for a paired "
                "comparison."
            ),
        )

    findings: list[AuditFinding] = []
    stats: dict[str, Any] = {}
    limitations: list[str] = []

    if effective_size is not None:
        stats["effective_size"] = effective_size.as_dict()
        low, high = effective_size.independent_case_estimate
        limitations.extend(effective_size.caveats)
        if effective_size.design_effect > 1.0:
            findings.append(
                AuditFinding(
                    section="statistical_reliability",
                    severity=Severity.WARNING,
                    tier=Tier.HEURISTIC,
                    confidence=(
                        "threshold-dependent"
                        if effective_size.threshold_is_load_bearing
                        else "stable across the thresholds tested"
                    ),
                    evidence=(
                        f"{effective_size.raw_cases} cases contain an estimated "
                        f"{low} distinct scenarios (plausible range {low}-{high}); "
                        f"design effect {effective_size.design_effect:.2f}"
                    ),
                    explanation=(
                        "Cases in a redundancy cluster are not independent "
                        "observations, but every confidence interval in this "
                        "package assumes they are. So an interval computed on "
                        f"{effective_size.raw_cases} cases is roughly "
                        f"{effective_size.interval_inflation:.2f}x too narrow."
                    ),
                    limitation=(
                        "The lower bound assumes cases in a cluster are "
                        "PERFECTLY redundant, which they are not: a model can "
                        "pass one paraphrase and fail another. This is a "
                        "bracket, not a statistical effective sample size."
                    ),
                    recommended_action=(
                        f"Widen any interval computed on this eval by roughly "
                        f"{effective_size.interval_inflation:.2f}x, or report "
                        f"the scenario count ({low}) alongside the case count."
                    ),
                )
            )

    if comparison is not None:
        stats["model_comparison"] = (
            comparison.as_dict() if hasattr(comparison, "as_dict") else {}
        )
        limitations.extend(getattr(comparison, "limitations", ()))
        limitations.extend(getattr(comparison, "assumptions", ()))
        # The verdict is the ANALYSER'S OWN label, so no threshold is invented
        # here. Only the two verdicts that mean "this eval cannot answer the
        # question" become findings; a real difference is a result, not a defect.
        for pair in getattr(comparison, "pairs", ()):
            verdict = getattr(pair, "verdict", "")
            if verdict not in ("underpowered", "no_information"):
                continue
            findings.append(
                AuditFinding(
                    section="statistical_reliability",
                    severity=Severity.WARNING,
                    tier=Tier.ESTIMATED,
                    confidence=(
                        f"95% CI {pair.difference_interval}"
                        if getattr(pair, "interval_reliable", False)
                        else "interval unreliable at this discordant count"
                    ),
                    evidence=f"{pair.model_a} vs {pair.model_b}: {pair.summary}",
                    explanation=(
                        f"{pair.n_cases} paired cases, "
                        f"{pair.only_a + pair.only_b} of them discordant. The "
                        f"smallest difference this many cases could detect is "
                        f"{pair.minimum_detectable_effect:.1%}, so a smaller "
                        "true difference would be invisible here regardless of "
                        "which model is better."
                    ),
                    limitation=(
                        "An underpowered comparison is not evidence that the "
                        "models are equal. It is the absence of evidence about "
                        "their difference."
                    ),
                    recommended_action=(
                        "Add cases, or state that this eval cannot separate "
                        "these two models. A smaller p-value is not available "
                        "from the same data."
                    ),
                )
            )

    return AuditSection(
        key="statistical_reliability",
        title="Statistical reliability",
        status=SectionStatus.RAN,
        summary="; ".join(
            part
            for part in (
                effective_size.render().splitlines()[0]
                if effective_size is not None
                else None,
                getattr(comparison, "summary", None),
            )
            if part
        ),
        findings=tuple(findings),
        stats=stats,
        limitations=tuple(limitations) or ("none stated",),
    )


def _reproducibility_section(report: Any | None) -> AuditSection:
    if report is None:
        return AuditSection(
            key="reproducibility",
            title="Reproducibility",
            status=SectionStatus.NOT_RUN,
            summary="Not assessed.",
            not_run_reason=(
                "needs the SAME eval run more than once. Collect RunOutcome "
                "records across runs and pass "
                "analyse_reproducibility(outcomes) into "
                "run_audit(reproducibility=...). A single run cannot show "
                "whether its own numbers are stable."
            ),
        )
    findings: list[AuditFinding] = []
    limitations: list[str] = []

    for note in getattr(report, "notes", ()):
        if not note.upper().startswith("WARNING"):
            limitations.append(note)
            continue
        findings.append(
            AuditFinding(
                section="reproducibility",
                severity=Severity.WARNING,
                tier=Tier.ESTIMATED,
                confidence="see the interval on each variance component",
                evidence=note,
                explanation="Reported by the reproducibility analysis itself.",
                limitation=(
                    "The three variance sources are never summed: they are "
                    "additive only under a variance-components model this "
                    "design does not guarantee."
                ),
                recommended_action=(
                    "Match the fix to the SOURCE: model stochasticity responds "
                    "to a lower temperature or more repeats, evaluator "
                    "stochasticity needs the grader fixed, sampling "
                    "variability needs more cases. Re-running cannot fix the "
                    "last two."
                ),
            )
        )

    # An unidentifiable variance source is a finding for the same reason a
    # refused metric is: it means a whole cause of instability is unmeasured,
    # and unmeasured is not zero.
    for name, metric in (getattr(report, "variance_sources", {}) or {}).items():
        reason = getattr(metric, "unmet_reason", None)
        if not reason:
            continue
        findings.append(
            AuditFinding(
                section="reproducibility",
                severity=Severity.WARNING,
                tier=Tier.OBSERVED,
                evidence=f"{name} is not identifiable: {reason}",
                explanation=(
                    "This design cannot separate that source of variance, so "
                    "its contribution is unknown rather than zero. Reporting it "
                    "as 0.0 would claim a stability that was never measured."
                ),
                limitation=(
                    "Not identifiable says nothing about the size of this "
                    "source. It says the design cannot see it."
                ),
                recommended_action=(
                    "Change the design if you need this number: two judges on "
                    "the same run separate grader noise, two runs separate "
                    "model noise, ten or more cases allow a bootstrap."
                ),
            )
        )

    ranking = getattr(report, "ranking_stability", {}) or {}
    if ranking.get("assessed") and ranking.get("unstable_pairs"):
        findings.append(
            AuditFinding(
                section="reproducibility",
                severity=Severity.WARNING,
                tier=Tier.ESTIMATED,
                confidence=(
                    f"modal ranking held in "
                    f"{ranking.get('modal_ranking_share', 0):.0%} of runs"
                ),
                evidence=(
                    f"{len(ranking['unstable_pairs'])} model pair(s) changed "
                    f"order across runs: "
                    + ", ".join(
                        " vs ".join(pair) for pair in ranking["unstable_pairs"]
                    )
                ),
                explanation=(
                    "The ranking reported from a single run is one of "
                    f"{ranking.get('distinct_rankings', '?')} that these runs "
                    "produced, so quoting it as the ranking overstates what the "
                    "eval showed."
                ),
                limitation=(
                    "Kendall's W here carries no tie correction, so the "
                    "reported concordance is a lower bound when ties are "
                    "present."
                ),
                recommended_action=(
                    "Report the ranking's stability alongside the ranking, or "
                    "add cases until the pairs in question separate."
                ),
            )
        )
    resolved = tuple(findings)
    return AuditSection(
        key="reproducibility",
        title="Reproducibility",
        status=SectionStatus.RAN,
        summary=(
            f"{getattr(report, 'n_runs', '?')} runs, "
            f"{getattr(report, 'n_models', '?')} model(s), "
            f"{getattr(report, 'n_cases', '?')} cases; "
            f"{len(resolved)} instability finding(s)"
        ),
        findings=resolved,
        stats=report.as_dict() if hasattr(report, "as_dict") else {},
        limitations=tuple(dict.fromkeys(limitations)) or ("none stated",),
    )


# ==========================================================================
# Renderers
# ==========================================================================

_SEVERITY_STYLE = {Severity.WARNING: "bold yellow", Severity.INFO: "bold cyan"}
_MAX_TERMINAL_CASES = 6


def render_terminal(
    report: AuditReport,
    *,
    console: Any | None = None,
    max_cases: int = _MAX_TERMINAL_CASES,
    detail: bool = False,
) -> None:
    """Print the concise view.

    Concise means: the summary, the findings one line each with their tier, the
    coverage gaps, and the recommendations. Per-case ids are truncated and the
    full stats are not printed at all -- `--format json` and `--format html`
    exist for those. What is NEVER truncated is the not-run list, because that
    is the part a reader is most likely to mistake for a pass.
    """
    from rich.console import Console
    from rich.padding import Padding
    from rich.text import Text

    console = console or Console()

    console.print()
    console.print(
        Text("evallint audit", style="bold")
        + Text(f"  {report.source}", style="dim")
    )
    console.print()

    console.print(Text("Executive summary", style="bold"))
    for line in report.executive_summary():
        console.print(Padding(Text(f"· {line}"), (0, 0, 0, 2)))

    for label, findings in (
        ("Critical findings", report.critical),
        ("Warnings", report.informational),
    ):
        console.print()
        console.print(Text(label, style="bold"))
        if not findings:
            console.print(Padding(Text("· none", style="dim"), (0, 0, 0, 2)))
            continue
        for finding in findings:
            console.print(
                Padding(
                    Text(f"· [{finding.tier.value}] ", style="dim")
                    + Text(finding.evidence),
                    (0, 0, 0, 2),
                )
            )
            if finding.case_ids:
                shown = ", ".join(finding.case_ids[:max_cases])
                extra = len(finding.case_ids) - max_cases
                if extra > 0:
                    shown += f", ... (+{extra} more)"
                console.print(
                    Padding(Text(shown, style="dim"), (0, 0, 0, 6))
                )
            if detail:
                for name in ("explanation", "limitation", "recommended_action"):
                    console.print(
                        Padding(
                            Text(
                                f"{name.replace('_', ' ')}: "
                                f"{getattr(finding, name)}",
                                style="dim italic",
                            ),
                            (0, 0, 0, 6),
                        )
                    )

    console.print()
    console.print(Text("Analyses", style="bold"))
    for section in report.sections:
        if section.key not in ANALYSIS_SECTIONS:
            continue
        if section.status is SectionStatus.NOT_RUN:
            console.print(
                Padding(
                    Text("· NOT ASSESSED  ", style="bold red")
                    + Text(f"{section.title} — {section.short_reason}"),
                    (0, 0, 0, 2),
                )
            )
        else:
            marker = (
                "· PARTIAL       "
                if section.status is SectionStatus.PARTIAL
                else "· assessed     "
            )
            console.print(
                Padding(
                    Text(marker, style="bold yellow" if section.status
                         is SectionStatus.PARTIAL else "green")
                    + Text(f"{section.title} — {section.summary}"),
                    (0, 0, 0, 2),
                )
            )

    console.print()
    console.print(Text("Recommendations", style="bold"))
    if not report.recommendations:
        console.print(
            Padding(
                Text(
                    "· none from the analyses that ran. See NOT ASSESSED above "
                    "for what was not looked at.",
                    style="dim",
                ),
                (0, 0, 0, 2),
            )
        )
    for finding in report.recommendations:
        console.print(
            Padding(
                Text(f"· [{finding.tier.value}] ", style="dim")
                + Text(finding.recommended_action),
                (0, 0, 0, 2),
            )
        )

    console.print()
    console.print(
        Padding(
            Text(
                "No overall score is reported. This audit describes evidence "
                "and gaps; whether that is acceptable for your purpose is not "
                "something the tool can know.",
                style="dim italic",
            ),
            (0, 0, 0, 2),
        )
    )
    console.print()


def to_json(report: AuditReport, *, indent: int = 2) -> str:
    """The full report, nothing truncated."""
    return json.dumps(report.as_dict(), indent=indent)


def render_html(report: AuditReport) -> str:
    """A single self-contained HTML file.

    No CDN, no framework, no JavaScript: expandable sections use `<details>`,
    which every browser implements natively. A report that silently fails to
    render because a CDN is unreachable is worse than a plain one, and an audit
    artefact should still open in five years.
    """
    d = report.as_dict()
    out: list[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>evallint audit — {html.escape(report.source)}</title>",
        _HTML_STYLE,
        "</head><body><main>",
        f"<h1>evallint audit</h1>",
        f"<p class='sub'>{html.escape(report.source)} · {report.n_cases} cases "
        f"· schema {report.schema_version} · evallint "
        f"{html.escape(report.evallint_version)}</p>",
        "<section><h2>Executive summary</h2><ul>",
    ]
    out.extend(f"<li>{html.escape(line)}</li>" for line in d["executive_summary"])
    out.append("</ul></section>")

    coverage = d["evidence_coverage"]
    out.append("<section><h2>Evidence coverage</h2>")
    out.append(
        "<p class='note'>There is no overall score. This is what a score would "
        "have compressed: which analyses produced evidence, and which did not.</p>"
    )
    out.append("<table><tr><th>Analysis</th><th>Status</th></tr>")
    for key in coverage["analyses_with_evidence"]:
        out.append(
            f"<tr><td>{html.escape(key)}</td>"
            "<td class='ok'>evidence produced</td></tr>"
        )
    for key, reason in coverage["analyses_without_evidence"].items():
        out.append(
            f"<tr><td>{html.escape(key)}</td>"
            f"<td class='gap'>NOT ASSESSED — {html.escape(reason or '')}</td></tr>"
        )
    out.append("</table></section>")

    for label, key in (
        ("Critical findings", "critical_findings"),
        ("Warnings", "warnings"),
    ):
        out.append(f"<section><h2>{label}</h2>")
        items = d[key]
        if not items:
            out.append("<p class='note'>None from the analyses that ran.</p>")
        for finding in items:
            out.append(_html_finding(finding))
        out.append("</section>")

    out.append("<section><h2>Recommendations</h2>")
    if not d["recommendations"]:
        out.append("<p class='note'>None from the analyses that ran.</p>")
    out.append("<ol>")
    for rec in d["recommendations"]:
        out.append(
            f"<li><span class='tier tier-{rec['tier']}'>{rec['tier']}</span> "
            f"{html.escape(rec['action'])}"
            f"<div class='because'>rests on: {html.escape(rec['because'])}</div>"
            "</li>"
        )
    out.append("</ol></section>")

    out.append("<section><h2>Sections</h2>")
    for section in d["sections"]:
        status = section["status"]
        badge = {
            "ran": "<span class='ok'>assessed</span>",
            "partial": "<span class='gap'>partial</span>",
            "not_run": "<span class='gap'>NOT ASSESSED</span>",
        }[status]
        out.append(
            f"<details><summary>{html.escape(section['title'])} — {badge}"
            f"</summary>"
        )
        out.append(f"<p>{html.escape(section['summary'])}</p>")
        if section["not_run_reason"]:
            out.append(
                f"<p class='gap'>{html.escape(section['not_run_reason'])}</p>"
            )
        for finding in section["findings"]:
            out.append(_html_finding(finding))
        if section["limitations"]:
            out.append("<details><summary>What this cannot tell you</summary><ul>")
            out.extend(
                f"<li>{html.escape(str(x))}</li>" for x in section["limitations"]
            )
            out.append("</ul></details>")
        if section["stats"]:
            out.append(
                "<details><summary>Raw statistics</summary><pre>"
                + html.escape(json.dumps(section["stats"], indent=2))
                + "</pre></details>"
            )
        out.append("</details>")
    out.append("</section>")

    out.append(
        "<footer><p>evallint reports evidence and gaps. It does not grade an "
        "eval, and contains no overall quality score: weighing incommensurable "
        "evidence would need invented weights, and a single number invites "
        "action without reading what is under it.</p></footer>"
    )
    out.append("</main></body></html>")
    return "\n".join(out)


def _html_finding(finding: Mapping[str, Any]) -> str:
    cases = finding["affected_cases"]
    parts = [
        "<div class='finding'>",
        f"<div class='head'><span class='sev sev-{finding['severity']}'>"
        f"{finding['severity']}</span> "
        f"<span class='tier tier-{finding['tier']}'>{finding['tier']}</span>",
    ]
    if finding["confidence"]:
        parts.append(
            f" <span class='conf'>confidence: "
            f"{html.escape(str(finding['confidence']))}</span>"
        )
    parts.append("</div>")
    parts.append(f"<p class='ev'>{html.escape(finding['evidence'])}</p>")
    parts.append(
        "<dl>"
        f"<dt>Explanation</dt><dd>{html.escape(finding['explanation'])}</dd>"
        f"<dt>Limitation</dt><dd>{html.escape(finding['limitation'])}</dd>"
        f"<dt>Recommended action</dt>"
        f"<dd>{html.escape(finding['recommended_action'])}</dd>"
        "</dl>"
    )
    if cases:
        parts.append(
            f"<details><summary>{len(cases)} affected case"
            f"{'s' if len(cases) != 1 else ''}</summary><ul class='cases'>"
        )
        parts.extend(f"<li><code>{html.escape(str(c))}</code></li>" for c in cases)
        parts.append("</ul></details>")
    parts.append("</div>")
    return "".join(parts)


_HTML_STYLE = """<style>
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 2rem 1rem; }
main { max-width: 60rem; margin: 0 auto; }
h1 { margin: 0 0 .2rem; font-size: 1.6rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .6rem; border-bottom: 1px solid #8884;
     padding-bottom: .3rem; }
.sub { color: #7a7a7a; margin: 0 0 1.5rem; font-size: .9rem; }
.note { color: #7a7a7a; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #8883; }
.ok { color: #1a7f37; }
.gap { color: #b3261e; font-weight: 600; }
details { margin: .4rem 0; }
summary { cursor: pointer; padding: .25rem 0; }
.finding { border-left: 3px solid #8886; padding: .5rem .8rem; margin: .7rem 0;
           background: #8881; border-radius: 0 4px 4px 0; }
.finding .head { font-size: .75rem; text-transform: uppercase;
                 letter-spacing: .04em; margin-bottom: .3rem; }
.ev { margin: .2rem 0 .5rem; font-weight: 500; }
.sev, .tier { padding: .1rem .4rem; border-radius: 3px; font-weight: 700; }
.sev-warning { background: #f5c04222; color: #8a6100; }
.sev-info { background: #2a7fff22; color: #1a5fbf; }
.tier-observed { background: #1a7f3722; color: #1a7f37; }
.tier-heuristic { background: #b3261e22; color: #b3261e; }
.tier-estimated { background: #6f42c122; color: #6f42c1; }
.conf { color: #7a7a7a; text-transform: none; letter-spacing: 0; }
dl { margin: .3rem 0 0; font-size: .88rem; }
dt { font-weight: 600; color: #7a7a7a; margin-top: .35rem; }
dd { margin: 0 0 0 0; }
.cases { columns: 3; font-size: .85rem; }
.because { color: #7a7a7a; font-size: .85rem; margin-top: .2rem; }
pre { overflow-x: auto; background: #8881; padding: .6rem; border-radius: 4px;
      font-size: .8rem; }
footer { margin-top: 3rem; color: #7a7a7a; font-size: .85rem;
         border-top: 1px solid #8884; padding-top: 1rem; }
</style>"""


def _plain(value: Any) -> Any:
    """Convert numpy scalars to built-ins so json.dumps cannot choke.

    Same reasoning as report._plain: the checks use numpy internally, and a
    stray np.float64 in stats would turn --format json into a crash at the very
    end of a long run.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError):  # pragma: no cover
            return value
    return value
