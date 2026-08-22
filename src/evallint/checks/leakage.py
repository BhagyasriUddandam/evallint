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

from ..schema import EvalSet
from .base import Check, CheckResult, Finding, Severity

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
    ) -> None:
        if min_answer_chars < 1:
            raise ValueError(
                f"min_answer_chars must be at least 1, got {min_answer_chars}"
            )
        if not 0 < widespread_share <= 1:
            raise ValueError("widespread_share must be between 0 and 1")
        self.min_answer_chars = min_answer_chars
        self.widespread_share = widespread_share

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
            for case in contained:
                answer = _normalise(case.expected)
                findings.append(
                    Finding(
                        f"the expected answer appears verbatim in the input "
                        f"({len(answer)} characters), so this case can be "
                        f"answered by copying rather than by the capability "
                        f"being tested",
                        case_ids=(case.id,),
                    )
                )

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
