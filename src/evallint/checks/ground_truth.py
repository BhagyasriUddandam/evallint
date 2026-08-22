"""Check 5 — ground-truth quality.

Identifies cases whose expected answer may not be well enough defined for a
score against it to mean anything.

## Deterministic first, judges only if you ask

Seven detectors run with no model, no API key and no cost. An LLM-assisted
analyser exists but is OFF unless judges are passed explicitly, and even then
its findings are subordinate to the deterministic ones.

## Never one judge as truth

This is a hard constraint in the design, not a suggestion:

  * With ONE judge, its findings are capped at LOW confidence and labelled
    `single_judge_uncorroborated`. One model's opinion about whether a question
    is ambiguous is an opinion.
  * With TWO OR MORE, only cases the judges AGREE on are reported at MODERATE.
    Cases they disagree about are not reported as ambiguous at all — they are
    routed to human review, because judge disagreement is evidence that the
    question is hard to adjudicate, not evidence about the answer.
  * Judge agreement is reported with a chance-corrected statistic, never raw
    agreement alone. On a skewed set two judges agree most of the time by
    accident.
  * High agreement is NOT correctness. Judges from one model family share their
    blind spots, so correlated error is indistinguishable from reliability. That
    is stated in the limitations whenever no human-labelled sample is supplied.

Two findings in this project were previously retracted because a model graded
its own work, which is why the judge path is built to distrust itself.

## A subjective case is not an invalid case

"Is this response helpful?" has no single right answer, and an eval of such
questions measures preference, which is a real thing to measure. The finding
says the score will be grader-dependent. Nothing here says a case is invalid,
and a test asserts the absence of that wording.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..schema import EvalCase, EvalSet
from .agreement import fleiss_kappa, pairwise_agreement, raw_agreement
from .ambiguity_detectors import (
    AmbiguityFinding,
    AmbiguityType,
    Confidence,
    detect_ambiguous_wording,
    detect_missing_criteria,
    detect_multiple_interpretations,
    detect_multiple_valid_answers,
    detect_reference_inconsistency,
    detect_subjective_question,
    detect_underspecified_reference,
)
from .base import Check, CheckResult, Finding, Severity

__all__ = ["AmbiguityJudge", "GroundTruthCheck"]

#: A judge decides whether ONE case's ground truth is under-specified.
#: evallint never calls a provider: you supply the callable, exactly as with the
#: scorer and the embedder.
AmbiguityJudge = Callable[[EvalCase], bool]

#: When this share of cases share ONE ambiguity type, it is a property of the
#: dataset rather than of individual cases, and is reported once. Measured need:
#: HellaSwag has no reference answer at all, so `missing_criteria` fired on 100
#: of 100 cases and produced 100 warnings. That is one fact about the file, and
#: a hundred findings for it is the noise that gets a check switched off.
WIDESPREAD_TYPE_SHARE = 0.5

#: Severity by confidence. Only HIGH warns. A MODERATE finding about a
#: subjective question is information about how to read the score, not a defect
#: to fix, and warning about it would be the noise that gets a check ignored.
_SEVERITY = {
    Confidence.HIGH: Severity.WARNING,
    Confidence.MODERATE: Severity.INFO,
    Confidence.LOW: Severity.INFO,
}

LIMITATIONS = (
    "A SUBJECTIVE CASE IS NOT AN INVALID CASE. Helpfulness, preference and "
    "tone are legitimate things to evaluate. What a subjective finding says is "
    "that the score for that case depends on the grader, which is something to "
    "report alongside the number rather than a defect to remove.",
    "The deterministic detectors are lexical and structural. They cannot tell "
    "whether a question is ambiguous IN ITS DOMAIN: a term that is vague in "
    "general may be precisely defined in a specification the eval assumes. "
    "Expect to overrule some findings, particularly the LOW-confidence ones.",
    "Absence of a finding is not evidence of a well-defined reference. Every "
    "detector here matches surface features, so an ambiguity expressed in "
    "perfectly precise-looking language is invisible.",
    "Reference inconsistency collapses case and whitespace but KEEPS "
    "punctuation, so it finds the same question re-typed and NOT two cases "
    "asking the same thing in different words. Punctuation is kept because "
    "stripping it produced three false contradictions on MMLU, where inputs "
    "differing only by '+' versus ',' versus '*' are mathematically distinct "
    "questions. A missed inconsistency is a gap; a fabricated one is a "
    "confident wrong answer.",
    "When most of the set shares ONE ambiguity type, it is reported once as a "
    "property of the dataset rather than per case. HellaSwag has no reference "
    "answers at all, which is one fact about the file, not 100 findings.",
    "JUDGE AGREEMENT IS NOT CORRECTNESS. Judges drawn from one model family "
    "share their training and their blind spots, so correlated error looks "
    "exactly like reliability. Without a human-labelled sample, agreement "
    "measures consensus only — and this check has no way to obtain one.",
    "With a single judge, findings are capped at LOW confidence and marked "
    "uncorroborated, because one model's opinion about ambiguity is an opinion. "
    "Nothing in this check treats a judge as ground truth.",
    "Cases the judges disagree about are NOT reported as ambiguous. They are "
    "routed to human review, because disagreement is evidence that the case is "
    "hard to adjudicate rather than evidence about its answer.",
)


class GroundTruthCheck(Check):
    """Flags cases whose reference answer may be under-specified."""

    name = "ground_truth"
    description = "Cases whose expected answer may not be well enough defined."

    def __init__(
        self,
        *,
        judges: Mapping[str, AmbiguityJudge] | None = None,
        require_judge_agreement: bool = True,
        max_ambiguous_share: float = 0.25,
    ) -> None:
        """
        Args:
            judges: Optional ``{name: judge}`` mapping, where a judge takes a
                case and returns whether its ground truth looks
                under-specified. evallint never calls a provider — you supply
                the callable, as with the scorer and the embedder. Omitted
                means no LLM analysis happens at all.
            require_judge_agreement: With two or more judges, report only cases
                they all agree on. Disagreements go to human review instead.
                Setting this False reports majority opinion instead, and lowers
                the confidence of those findings accordingly.
            max_ambiguous_share: Share of potentially ambiguous cases above
                which a set-level warning is emitted. A convention, stated as
                one in the limitations.
        """
        if not 0 < max_ambiguous_share <= 1:
            raise ValueError("max_ambiguous_share must be between 0 and 1")
        self.judges = dict(judges or {})
        self.require_judge_agreement = require_judge_agreement
        self.max_ambiguous_share = max_ambiguous_share

    # -- deterministic pass ---------------------------------------------
    def _deterministic(self, cases: Sequence[EvalCase]) -> list[AmbiguityFinding]:
        found: list[AmbiguityFinding] = []
        for case in cases:
            for detector in (
                detect_missing_criteria,
                detect_multiple_valid_answers,
                detect_subjective_question,
                detect_underspecified_reference,
                detect_multiple_interpretations,
                detect_ambiguous_wording,
            ):
                finding = detector(case)
                if finding is not None:
                    found.append(finding)
        # Set-level: needs every case at once.
        found.extend(detect_reference_inconsistency(cases))
        return found

    # -- optional judge pass --------------------------------------------
    def _judge_pass(
        self, cases: Sequence[EvalCase]
    ) -> tuple[list[AmbiguityFinding], dict[str, Any]]:
        """Run the configured judges and reconcile them.

        Returns (findings, stats). Cases the judges disagree about produce NO
        finding; they are listed for human review instead.
        """
        names = sorted(self.judges)
        opinions: dict[str, list[bool]] = {}
        errors: list[str] = []
        for case in cases:
            votes: list[bool] = []
            for name in names:
                try:
                    votes.append(bool(self.judges[name](case)))
                except Exception as exc:  # a user-supplied judge failed
                    errors.append(f"judge {name!r} raised on case {case.id!r}: {exc}")
                    votes.append(False)
            opinions[case.id] = votes

        ratings = [opinions[c.id] for c in cases]
        n_judges = len(names)
        disputed = [c.id for c in cases if len(set(opinions[c.id])) > 1]

        stats: dict[str, Any] = {
            "judges": names,
            "n_judges": n_judges,
            "judge_errors": errors,
            "per_judge_flag_rate": {
                name: sum(opinions[c.id][i] for c in cases) / len(cases)
                for i, name in enumerate(names)
            }
            if cases
            else {},
            "judge_raw_agreement": raw_agreement(ratings) if n_judges > 1 else None,
            "judge_pairwise_agreement": (
                pairwise_agreement(ratings) if n_judges > 1 else None
            ),
            # None means UNDEFINED, not zero: when every judge gives every case
            # the same verdict, kappa is 0/0. Raw agreement is 1.0 and kappa
            # cannot be computed, and substituting a number there would invent
            # reliability.
            "judge_fleiss_kappa": fleiss_kappa(ratings) if n_judges > 1 else None,
            "n_judge_disagreements": len(disputed),
            "judge_disagreement_case_ids": disputed,
            "single_judge": n_judges == 1,
        }

        findings: list[AmbiguityFinding] = []
        for case in cases:
            votes = opinions[case.id]
            if not any(votes):
                continue
            unanimous = len(set(votes)) == 1

            if n_judges == 1:
                # One opinion. Capped at LOW and labelled, because a single
                # model's view of ambiguity is a view.
                findings.append(
                    AmbiguityFinding(
                        case_id=case.id,
                        ambiguity_type=AmbiguityType.JUDGE_FLAGGED,
                        confidence=Confidence.LOW,
                        evidence=(
                            f"single_judge_uncorroborated: {names[0]!r} flagged "
                            f"this case, with nothing to corroborate it"
                        ),
                        recommended_action=(
                            "add a second judge, or review by hand. One model's "
                            "opinion about ambiguity is not evidence the "
                            "reference is under-specified"
                        ),
                    )
                )
                continue

            if not unanimous:
                # Disagreement is routed to review, NOT reported as ambiguity.
                continue
            if self.require_judge_agreement or unanimous:
                findings.append(
                    AmbiguityFinding(
                        case_id=case.id,
                        ambiguity_type=AmbiguityType.JUDGE_FLAGGED,
                        confidence=Confidence.MODERATE,
                        evidence=(
                            f"all {n_judges} judges ({', '.join(names)}) flagged "
                            f"this case as under-specified"
                        ),
                        recommended_action=(
                            "review the reference by hand. Agreement between "
                            "judges is consensus, not correctness — judges from "
                            "one model family share their blind spots"
                        ),
                    )
                )
        return findings, stats

    def run(self, eval_set: EvalSet) -> CheckResult:
        cases = list(eval_set)
        n = len(cases)

        findings_data = self._deterministic(cases)
        stats: dict[str, Any] = {
            "n_cases": n,
            "judges_configured": bool(self.judges),
            "detectors_run": [
                t.value for t in AmbiguityType if t is not AmbiguityType.JUDGE_FLAGGED
            ],
        }

        judge_stats: dict[str, Any] = {}
        if self.judges:
            judge_findings, judge_stats = self._judge_pass(cases)
            findings_data.extend(judge_findings)
        stats.update(judge_stats)

        flagged_ids = {f.case_id for f in findings_data}
        by_type: dict[str, int] = {}
        by_confidence: dict[str, int] = {}
        for finding in findings_data:
            by_type[finding.ambiguity_type.value] = (
                by_type.get(finding.ambiguity_type.value, 0) + 1
            )
            by_confidence[finding.confidence.value] = (
                by_confidence.get(finding.confidence.value, 0) + 1
            )

        # Human review: HIGH-confidence structural findings, plus every case the
        # judges could not agree about. Those are different reasons for the same
        # conclusion -- a person has to look.
        review = {
            f.case_id for f in findings_data if f.confidence is Confidence.HIGH
        }
        review.update(judge_stats.get("judge_disagreement_case_ids", []))

        stats.update(
            n_potentially_ambiguous=len(flagged_ids),
            potentially_ambiguous_share=len(flagged_ids) / n if n else 0.0,
            n_findings=len(findings_data),
            n_by_type=by_type,
            n_by_confidence=by_confidence,
            n_requiring_human_review=len(review),
            human_review_case_ids=sorted(review),
            findings=[f.as_dict() for f in findings_data],
        )

        return CheckResult(
            check=self.name,
            summary=self._summary(stats),
            findings=tuple(self._render(findings_data, stats)),
            stats=stats,
            limitations=LIMITATIONS,
        )

    def _summary(self, stats: dict) -> str:
        n = stats["n_cases"]
        flagged = stats["n_potentially_ambiguous"]
        if not flagged:
            base = f"{n} cases, no potentially ambiguous ground truth found"
        else:
            base = (
                f"{n} cases, {flagged} potentially ambiguous "
                f"({stats['potentially_ambiguous_share']:.0%}), "
                f"{stats['n_requiring_human_review']} need human review"
            )
        if not stats["judges_configured"]:
            base += " — deterministic detectors only, no judges configured"
        return base

    def _render(
        self, findings_data: list[AmbiguityFinding], stats: dict
    ) -> list[Finding]:
        rendered: list[Finding] = []

        share = stats["potentially_ambiguous_share"]
        if share > self.max_ambiguous_share:
            rendered.append(
                Finding(
                    f"{share:.0%} of cases ({stats['n_potentially_ambiguous']}/"
                    f"{stats['n_cases']}) have POTENTIALLY under-specified "
                    f"ground truth. That is not a count of broken cases — "
                    f"subjective questions and multi-answer references are "
                    f"legitimate — but it bounds how much of this eval can be "
                    f"scored unambiguously"
                )
            )

        if stats.get("n_judges", 0) > 1:
            kappa = stats["judge_fleiss_kappa"]
            kappa_text = (
                "UNDEFINED (every judge gave every case the same verdict, so "
                "chance-corrected agreement cannot be computed)"
                if kappa is None
                else f"{kappa:.2f}"
            )
            rendered.append(
                Finding(
                    f"judge agreement across {stats['n_judges']} judges: "
                    f"raw {stats['judge_raw_agreement']:.0%}, pairwise "
                    f"{stats['judge_pairwise_agreement']:.0%}, Fleiss kappa "
                    f"{kappa_text}. {stats['n_judge_disagreements']} cases were "
                    f"disputed and are routed to human review rather than "
                    f"reported as ambiguous. Agreement is consensus, not "
                    f"correctness: judges from one model family share their "
                    f"blind spots",
                    severity=Severity.INFO,
                )
            )
        elif stats.get("single_judge"):
            rendered.append(
                Finding(
                    "only ONE judge was configured, so its findings are capped "
                    "at low confidence and marked uncorroborated. A single "
                    "model's opinion about whether a question is ambiguous is "
                    "an opinion — add a second judge to measure disagreement",
                    severity=Severity.INFO,
                )
            )

        for errors in (stats.get("judge_errors") or [],):
            for error in errors[:5]:
                rendered.append(Finding(f"judge error: {error}"))

        # Collapse any type that affects most of the set into one statement.
        n_cases = stats["n_cases"]
        widespread = {
            name
            for name, count in stats["n_by_type"].items()
            if n_cases and count / n_cases >= WIDESPREAD_TYPE_SHARE
        }
        for name in sorted(widespread):
            affected = [
                f.case_id for f in findings_data
                if f.ambiguity_type.value == name
            ]
            rendered.append(
                Finding(
                    f"{len(affected)} of {n_cases} cases "
                    f"({len(affected) / n_cases:.0%}) share the SAME issue: "
                    f"{name}. At that rate this is a property of the dataset "
                    f"rather than of individual cases, so they are reported "
                    f"here once instead of one finding each",
                    severity=Severity.WARNING,
                    case_ids=tuple(affected),
                )
            )

        for finding in findings_data:
            if finding.ambiguity_type.value in widespread:
                continue
            rendered.append(
                Finding(
                    f"POTENTIALLY AMBIGUOUS [{finding.ambiguity_type.value}, "
                    f"confidence {finding.confidence.value}] in case "
                    f"{finding.case_id!r}: {finding.evidence}. Recommended: "
                    f"{finding.recommended_action}",
                    severity=_SEVERITY[finding.confidence],
                    case_ids=(finding.case_id,),
                )
            )
        return rendered
