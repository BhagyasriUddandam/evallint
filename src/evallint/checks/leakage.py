"""Check 4 — label leakage.

The failure this catches: a case whose EXPECTED answer can be read off its
INPUT. The model does not need the capability under test, because the answer is
sitting in the prompt. The eval reports high accuracy and has measured nothing.

Real causes, all of them accidents rather than malice: a template that
interpolated the answer into the prompt, a "context" field pasted in wholesale,
a category name left in the question, an id or slug that encodes the label.

WHY THIS CHECK IS ALMOST ENTIRELY ABOUT FALSE POSITIVES
Legitimate cases overlap with their answers constantly. Extractive QA is
*supposed* to contain the answer — that is the task. Multiple choice repeats
its options. Classification labels occur naturally in text. A leakage check
that fires on those is worse than no check, because it teaches people to
ignore it.

So the signals here were chosen by MEASUREMENT against four real public eval
sets, not by intuition, and one obvious idea was discarded because the data
killed it:

  TOKEN OVERLAP DOES NOT WORK. Sharing tokens between `expected` and `input`
  looks like the natural signal. On GSM8K the median overlap is 0.57 and the
  MAXIMUM IS 1.00 -- with no leakage at all -- because the reference answer is
  a worked solution that restates the question's numbers and nouns. A planted
  leak also scores 1.00. There is no threshold separating them, so token
  overlap is not used as a finding at all.

  VERBATIM CONTAINMENT DOES WORK, once guarded. Measured hits, whole set:

      GSM8K        0 / 100
      TruthfulQA   0 / 100
      MMLU        46 / 100   -> suppressed by the closed-set guard
      bundled      0 / 20

  MMLU is the instructive one. Its answers are single characters drawn from a
  set of four, so 46 of them appear in their own question by coincidence. That
  is a classification scheme, not a leak, and the guards below exist for it.

The check therefore prefers silence. It reports the set-wide pattern once when
containment is widespread, rather than emitting one warning per case, because a
uniformly high rate means "this is extractive QA" and not "your eval is broken
500 times".
"""

from __future__ import annotations

import re

from ..schema import EvalCase, EvalSet
from .base import Check, CheckResult, Finding, Severity
from .leakage_detectors import (
    Confidence,
    LeakageType,
    PotentialLeak,
    detect_answer_in_instructions,
    detect_expected_in_input,
    detect_explicit_reveal,
    detect_fewshot_target,
    detect_high_overlap,
    detect_label_in_input,
    detect_metadata_leak,
)

__all__ = [
    "DEFAULT_MIN_ANSWER_CHARS",
    "DEFAULT_WIDESPREAD_SHARE",
    "LeakageCheck",
]

# Below this many characters an answer is too short for containment to mean
# anything: "yes", "4", "B" and "Paris" turn up in inputs by chance. Measured
# on MMLU, whose single-character answers produced 46 coincidental hits in 100
# cases.
DEFAULT_MIN_ANSWER_CHARS = 12

# When this share of cases contain their own answer, the set is treated as one
# where that is the intended task (extractive QA, span selection) and reported
# once rather than per case.
DEFAULT_WIDESPREAD_SHARE = 0.5

LIMITATIONS = (
    "This check CANNOT distinguish a leak from a task where the answer is "
    "supposed to appear in the input. Extractive QA, span selection and "
    "reading comprehension all legitimately contain their answers. When most "
    "of a set looks like that, this check says so once and stops flagging "
    "individual cases — read that note before treating any finding as a bug.",
    "Containment is literal text matching after whitespace and case "
    "normalisation. A leak that paraphrases the answer, or encodes it as a "
    "synonym, a number in a different format, or a hint, is invisible here. "
    "Absence of a finding is not evidence of absence.",
    "Token-overlap scoring was deliberately NOT used. On GSM8K the overlap "
    "between a reference answer and its question reaches 1.00 with no leakage, "
    "because the answer is a worked solution that restates the question. No "
    "threshold separates that from a real leak, so no such finding is emitted "
    "— this check is quieter than it could be, on purpose.",
    f"Answers shorter than {DEFAULT_MIN_ANSWER_CHARS} characters are skipped, "
    "and sets whose answers come from a small closed vocabulary are treated as "
    "classification and skipped entirely. A genuine leak of a short label will "
    "therefore be missed.",
    "Only the 'input' and 'expected' fields are read. An answer leaked through "
    "metadata, a filename, or an id is not seen.",
)


#: Severity per detector, chosen deliberately rather than derived from
#: confidence. The two answer different questions: confidence is "how sure is
#: the interpretation", severity is "how much should you act on it".
#:
#:   expected_in_input       WARNING  moderate confidence, but the canonical
#:                                    leak and directly actionable -- go read
#:                                    the case
#:   explicit_reveal         WARNING  high confidence
#:   answer_in_instructions  WARNING  high confidence
#:   fewshot_target          WARNING  high confidence
#:   label_in_input          INFO     usually topical; the set-level rate note
#:                                    is the useful signal, not each case
#:   metadata_leak           INFO     depends on whether YOUR harness puts
#:                                    metadata in the prompt, which evallint
#:                                    cannot see
#:   high_overlap            INFO     opt-in, low confidence, poor separation
_SEVERITY_BY_TYPE = {
    LeakageType.EXPECTED_IN_INPUT: Severity.WARNING,
    LeakageType.EXPLICIT_REVEAL: Severity.WARNING,
    LeakageType.ANSWER_IN_INSTRUCTIONS: Severity.WARNING,
    LeakageType.FEWSHOT_TARGET: Severity.WARNING,
    LeakageType.LABEL_IN_INPUT: Severity.INFO,
    LeakageType.METADATA_LEAK: Severity.INFO,
    LeakageType.HIGH_OVERLAP: Severity.INFO,
}


def _render(leak: PotentialLeak) -> Finding:
    """One finding per potential leak, in a single format for every detector.

    "POTENTIAL leakage" always, never a verdict. The evidence is included so a
    reader can check the claim instead of trusting it.
    """
    return Finding(
        f"POTENTIAL LEAKAGE [{leak.type.value}, confidence "
        f"{leak.confidence.value}] in case {leak.case_id!r}: "
        f"{leak.explanation}. Evidence: {leak.evidence}",
        severity=_SEVERITY_BY_TYPE.get(leak.type, Severity.INFO),
        case_ids=(leak.case_id,),
    )


def _normalise(text: object) -> str:
    return " ".join(str(text).lower().split())


def _looks_like_a_closed_vocabulary(answers: list[str], n_cases: int) -> bool:
    """True when the answers look like class labels rather than free text.

    MMLU's answers are one of four letters; a yes/no set has two. Containment
    is meaningless there — the answer occurs in the input by chance — so the
    whole check stands down rather than emitting coincidences as findings.
    """
    if n_cases < 20:
        # Too few cases to tell a closed vocabulary from a small set of
        # distinct free-text answers. Do not guess.
        return False
    distinct = len(set(answers))
    return distinct <= max(10, n_cases // 20)


class LeakageCheck(Check):
    """Finds cases whose expected answer is recoverable from their input."""

    name = "leakage"
    description = "Cases whose expected answer appears in their own input."

    def __init__(
        self,
        *,
        min_answer_chars: int = DEFAULT_MIN_ANSWER_CHARS,
        widespread_share: float = DEFAULT_WIDESPREAD_SHARE,
        overlap: bool = False,
        overlap_threshold: float = 0.9,
    ) -> None:
        """
        Args:
            min_answer_chars: Answers shorter than this are skipped, because a
                short string turns up in an input by chance. Measured: MMLU's
                single-character answers appear in their own question in 46 of
                100 cases.
            widespread_share: Above this share of cases containing their own
                answer, the set is treated as extractive-by-design and reported
                once rather than per case.
            overlap: Enable the token-overlap detector. OFF by default because
                it is measurably unreliable — on GSM8K it reaches 100% overlap
                with no leakage present, since reference answers restate the
                question. Opt in only if you know your answers are short labels.
            overlap_threshold: Overlap at or above which that detector fires.
        """
        if min_answer_chars < 1:
            raise ValueError(
                f"min_answer_chars must be at least 1, got {min_answer_chars}"
            )
        if not 0 < widespread_share <= 1:
            raise ValueError("widespread_share must be between 0 and 1")
        if not 0 < overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be between 0 and 1")
        self.min_answer_chars = min_answer_chars
        self.widespread_share = widespread_share
        self.overlap = overlap
        self.overlap_threshold = overlap_threshold

    def run(self, eval_set: EvalSet) -> CheckResult:
        cases = list(eval_set)
        answered = [c for c in cases if c.expected is not None]
        stats: dict = {
            "n_cases": len(cases),
            "n_with_expected": len(answered),
            "n_skipped_short_answer": 0,
            "closed_vocabulary": False,
            "n_contained": 0,
            "contained_share": 0.0,
            "widespread": False,
            "min_answer_chars": self.min_answer_chars,
            "leaking_case_ids": [],
        }

        if not answered:
            return CheckResult(
                check=self.name,
                summary=(
                    f"{len(cases)} cases, none with an 'expected' value — "
                    "leakage cannot be assessed"
                ),
                findings=(
                    Finding(
                        f"no case has an 'expected' value, so this check has "
                        f"nothing to compare against the input for these "
                        f"{len(cases)} cases",
                        severity=Severity.INFO,
                    ),
                ),
                stats=stats,
                limitations=LIMITATIONS,
            )

        normalised = [(c, _normalise(c.expected), _normalise(c.input)) for c in answered]

        if _looks_like_a_closed_vocabulary(
            [answer for _, answer, _ in normalised], len(answered)
        ):
            stats["closed_vocabulary"] = True
            distinct = len({answer for _, answer, _ in normalised})
            return CheckResult(
                check=self.name,
                summary=(
                    f"{len(answered)} answers drawn from {distinct} distinct "
                    "values — treated as class labels, leakage not assessed"
                ),
                findings=(
                    Finding(
                        f"the {len(answered)} 'expected' values take only "
                        f"{distinct} distinct forms, so this looks like a "
                        f"classification scheme rather than free-text answers. "
                        f"A label appearing in its own input is coincidence at "
                        f"that vocabulary size — on MMLU it happens in 46 of "
                        f"100 cases — so no leakage finding is reported. Check "
                        f"by hand if a label word could give the answer away",
                        severity=Severity.INFO,
                    ),
                ),
                stats=stats,
                limitations=LIMITATIONS,
            )

        contained = []
        for case, answer, prompt in normalised:
            if len(answer) < self.min_answer_chars:
                stats["n_skipped_short_answer"] += 1
                continue
            if answer and answer in prompt:
                contained.append(case)

        # Built once, used for both the findings and the stats, so the rendered
        # report and the JSON payload cannot disagree about what was found.
        contained_leaks = [
            detect_expected_in_input(case, self.min_answer_chars)
            for case in contained
        ]
        contained_leaks = [leak for leak in contained_leaks if leak is not None]

        share = len(contained) / len(answered)
        stats.update(
            n_contained=len(contained),
            contained_share=share,
            widespread=share >= self.widespread_share,
            leaking_case_ids=[c.id for c in contained],
        )

        findings: list[Finding] = []
        if stats["widespread"]:
            # One note, not N warnings. A set where most answers appear in
            # their input is almost certainly extractive by design, and
            # flagging every case would be the noise that gets a check ignored.
            findings.append(
                Finding(
                    f"{share:.0%} of cases ({len(contained)}/{len(answered)}) "
                    f"contain their own expected answer. At that rate this is "
                    f"probably the intended task — extractive QA, span "
                    f"selection or reading comprehension — rather than "
                    f"leakage. No per-case findings are reported. If the answer "
                    f"is NOT meant to be present, every one of these cases is "
                    f"measuring reading rather than the capability you think",
                    severity=Severity.INFO,
                    case_ids=tuple(c.id for c in contained[:20]),
                )
            )
        elif contained:
            # Rendered through the shared path so every detector's findings
            # look alike. Severity stays WARNING for this one -- see
            # _SEVERITY_BY_TYPE for why it is not derived from confidence.
            findings.extend(_render(leak) for leak in contained_leaks)

        # --- the other detectors -------------------------------------------
        # Detector 1 (expected-in-input) is handled above, unchanged, including
        # its closed-vocabulary and short-answer guards. The rest run on every
        # case regardless of those guards, because their evidence does not
        # depend on answer length in the same way: an explicit reveal phrase
        # followed by "B" is still an explicit reveal.
        extra = self._other_detectors(cases)
        leaks = contained_leaks + extra
        by_type: dict[str, int] = {}
        by_confidence: dict[str, int] = {}
        for leak in leaks:
            by_type[leak.type.value] = by_type.get(leak.type.value, 0) + 1
            by_confidence[leak.confidence.value] = (
                by_confidence.get(leak.confidence.value, 0) + 1
            )
        stats.update(
            detectors_run=self._detectors_run(),
            n_potential_leaks=len(leaks),
            n_cases_with_potential_leaks=len({lk.case_id for lk in leaks}),
            n_by_type=by_type,
            n_by_confidence=by_confidence,
            potential_leaks=[lk.as_dict() for lk in leaks],
        )
        findings.extend(self._leak_findings(extra, len(cases)))

        if contained:
            summary = (
                f"{len(answered)} answered cases, {len(contained)} contain their "
                f"own answer ({share:.0%})"
            )
        else:
            summary = (
                f"{len(answered)} answered cases, none contain their own answer"
            )
        if stats["n_skipped_short_answer"]:
            summary += (
                f" — {stats['n_skipped_short_answer']} skipped as too short to judge"
            )

        return CheckResult(
            check=self.name,
            summary=summary,
            findings=tuple(findings),
            stats=stats,
            limitations=LIMITATIONS,
        )

    # -- the additional detectors ---------------------------------------
    def _detectors_run(self) -> list[str]:
        names = [
            LeakageType.EXPECTED_IN_INPUT.value,
            LeakageType.LABEL_IN_INPUT.value,
            LeakageType.EXPLICIT_REVEAL.value,
            LeakageType.ANSWER_IN_INSTRUCTIONS.value,
            LeakageType.METADATA_LEAK.value,
            LeakageType.FEWSHOT_TARGET.value,
        ]
        if self.overlap:
            names.append(LeakageType.HIGH_OVERLAP.value)
        return names

    def _other_detectors(self, cases: list[EvalCase]) -> list[PotentialLeak]:
        """Run detectors 2-7. Deterministic; no model, no judge, no network."""
        found: list[PotentialLeak] = []

        # Label-in-input is handled at SET level first. A label that appears in
        # most of its own inputs is topical -- a billing question containing the
        # word "billing" is not a leak -- so per-case findings would be noise.
        # Only a label present in a MINORITY of inputs is anomalous enough to
        # name individual cases.
        labelled = [c for c in cases if c.label is not None]
        label_hits = [c for c in labelled if detect_label_in_input(c) is not None]
        self._label_rate = (
            len(label_hits) / len(labelled) if labelled else 0.0
        )
        self._label_hits = label_hits
        if labelled and self._label_rate < 0.5:
            found.extend(
                lk
                for lk in (detect_label_in_input(c) for c in label_hits)
                if lk is not None
            )

        for case in cases:
            for detector in (
                detect_explicit_reveal,
                detect_answer_in_instructions,
                detect_fewshot_target,
            ):
                leak = detector(case)
                if leak is not None:
                    found.append(leak)
            leak = detect_metadata_leak(case, self.min_answer_chars)
            if leak is not None:
                found.append(leak)
            if self.overlap:
                leak = detect_high_overlap(case, self.overlap_threshold)
                if leak is not None:
                    found.append(leak)
        return found

    def _leak_findings(
        self, leaks: list[PotentialLeak], n_cases: int
    ) -> list[Finding]:
        """One finding per potential leak, plus the set-level label note.

        Severity follows CONFIDENCE, not leak type: only HIGH-confidence
        evidence warns. A MODERATE finding is an observation whose
        interpretation depends on the task, and warning about those is how a
        check gets switched off.
        """
        findings: list[Finding] = []

        if getattr(self, "_label_rate", 0.0) >= 0.5:
            findings.append(
                Finding(
                    f"{self._label_rate:.0%} of labelled cases contain their own "
                    f"label as a word in the input. At that rate this is "
                    f"topical rather than leakage — a question about billing "
                    f"naturally contains the word billing — so individual cases "
                    f"are not flagged",
                    severity=Severity.INFO,
                    case_ids=tuple(c.id for c in self._label_hits[:20]),
                )
            )

        findings.extend(_render(leak) for leak in leaks)
        return findings

