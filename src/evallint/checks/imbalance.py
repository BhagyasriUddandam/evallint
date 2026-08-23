"""Check 3 — class imbalance and basic dataset statistics.

The failure this catches: an eval reports 85% accuracy, everyone relaxes, and
nobody notices that 70% of the cases are one easy class. The model is failing
the rare classes, but the rare classes are too small to move the average.

Two independent things make a class unmeasurable, so both are checked:
  - its SHARE is small, so it barely affects the aggregate number, and
  - its COUNT is small, so its own accuracy is coarse and noisy.
In a 10,000-case set a 1% class still has 100 cases and is perfectly
measurable; in a 20-case set a 5% class has one. Share alone would miss that.
"""

from __future__ import annotations

from collections import Counter
from statistics import median


from ..schema import EvalSet
from .base import Check, CheckResult, Finding, Severity

__all__ = ["ImbalanceCheck"]

LIMITATIONS = (
    "This check measures whether the class distribution is UNEVEN, not whether "
    "it is WRONG. If production traffic really is 70% billing, a 70% billing "
    "eval may be exactly right. Only you know the target distribution.",
    "Classes are read from the 'label' field only. If your categories live in "
    "metadata or are implied by the text, this check sees nothing.",
    "The thresholds are conventions, not statistics. They are a prompt to look, "
    "not a verdict.",
    "Input length is measured in characters, not tokens, so it is a rough "
    "proxy for cost and complexity.",
)


class ImbalanceCheck(Check):
    """Reports class distribution, imbalance ratio and basic stats."""

    name = "imbalance"
    description = "Class distribution, imbalance ratio, and basic dataset stats."

    def __init__(
        self,
        *,
        min_class_share: float = 0.10,
        min_class_count: int = 5,
        max_imbalance_ratio: float = 3.0,
    ) -> None:
        if not 0 < min_class_share < 1:
            raise ValueError("min_class_share must be between 0 and 1 (exclusive)")
        if min_class_count < 1:
            raise ValueError("min_class_count must be at least 1")
        if max_imbalance_ratio < 1:
            raise ValueError("max_imbalance_ratio must be at least 1")
        self.min_class_share = min_class_share
        self.min_class_count = min_class_count
        self.max_imbalance_ratio = max_imbalance_ratio

    def run(self, eval_set: EvalSet) -> CheckResult:
        n_cases = len(eval_set)
        counts = Counter(case.label for case in eval_set if case.label is not None)
        n_labelled = sum(counts.values())

        findings: list[Finding] = []
        findings.extend(self._missing_field_findings(eval_set, n_cases, n_labelled))

        stats = {
            "n_cases": n_cases,
            "n_labelled": n_labelled,
            "n_unlabelled": n_cases - n_labelled,
            "n_classes": len(counts),
            "class_counts": dict(counts.most_common()),
            "class_shares": {
                label: count / n_labelled for label, count in counts.most_common()
            },
            "imbalance_ratio": None,
            "input_length_chars": _length_stats(eval_set),
        }

        if not counts:
            # No labels is not a pass. Say the check could not run, loudly.
            findings.append(
                Finding(
                    f"no case has a label, so class balance cannot be assessed "
                    f"for these {n_cases} cases",
                    severity=Severity.INFO,
                )
            )
            return CheckResult(
                check=self.name,
                summary=f"{n_cases} cases, no labels — class balance not assessed",
                findings=tuple(findings),
                stats=stats,
                limitations=LIMITATIONS,
            )

        largest, largest_n = counts.most_common()[0]
        smallest, smallest_n = counts.most_common()[-1]
        ratio = largest_n / smallest_n
        stats["imbalance_ratio"] = ratio

        if len(counts) == 1:
            findings.append(
                Finding(
                    f"every labelled case is class '{largest}', so this eval can "
                    "only measure one scenario",
                    severity=Severity.INFO,
                )
            )
        elif ratio > self.max_imbalance_ratio:
            findings.append(
                Finding(
                    f"class imbalance ratio is {ratio:.1f}:1 — '{largest}' has "
                    f"{largest_n} cases, '{smallest}' has {smallest_n}. Aggregate "
                    f"accuracy is dominated by '{largest}'"
                )
            )

        findings.extend(self._rare_class_findings(eval_set, counts, n_labelled))

        summary = (
            f"{n_cases} cases across {len(counts)} classes, "
            f"imbalance ratio {ratio:.1f}:1"
        )
        return CheckResult(
            check=self.name,
            summary=summary,
            findings=tuple(findings),
            stats=stats,
            limitations=LIMITATIONS,
        )

    def _rare_class_findings(
        self, eval_set: EvalSet, counts: Counter, n_labelled: int
    ) -> list[Finding]:
        """One finding per under-represented class, naming every reason it tripped."""
        findings = []
        for label, count in counts.most_common():
            share = count / n_labelled
            reasons = []
            if share < self.min_class_share:
                reasons.append(
                    f"{share:.1%} of labelled cases (below {self.min_class_share:.0%})"
                )
            if count < self.min_class_count:
                plural = "" if count == 1 else "s"
                reasons.append(
                    f"only {count} case{plural}, so its accuracy can only land on "
                    f"{count + 1} distinct values"
                )
            if reasons:
                findings.append(
                    Finding(
                        f"class '{label}' is under-represented: " + "; ".join(reasons),
                        case_ids=tuple(c.id for c in eval_set if c.label == label),
                    )
                )
        return findings

    def _missing_field_findings(
        self, eval_set: EvalSet, n_cases: int, n_labelled: int
    ) -> list[Finding]:
        findings = []
        if 0 < n_labelled < n_cases:
            unlabelled = tuple(c.id for c in eval_set if c.label is None)
            findings.append(
                Finding(
                    f"{len(unlabelled)} of {n_cases} cases have no label; the "
                    f"distribution below covers only the {n_labelled} labelled cases",
                    case_ids=unlabelled,
                )
            )
        unscoreable = tuple(c.id for c in eval_set if c.expected is None)
        if unscoreable:
            findings.append(
                Finding(
                    f"{len(unscoreable)} of {n_cases} cases have no 'expected' "
                    "value, so they cannot be scored automatically",
                    case_ids=unscoreable,
                )
            )
        return findings


def _length_stats(eval_set: EvalSet) -> dict[str, float]:
    """Four scalars from the stdlib.

    This was the ONLY numpy use on the no-embeddings path, and numpy is 25 MiB
    of a 28 MiB core install. `statistics.median` matches numpy's definition for
    both odd and even counts, including the mean-of-two-middles case, so the
    numbers are unchanged -- asserted in tests/test_imbalance.py.
    """
    lengths = sorted(len(case.input) for case in eval_set)
    return {
        "min": lengths[0],
        "median": float(median(lengths)),
        "mean": round(sum(lengths) / len(lengths), 1),
        "max": lengths[-1],
    }
