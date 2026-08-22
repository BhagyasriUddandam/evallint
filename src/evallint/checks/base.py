"""The interface every check implements.

One shape in (EvalSet), one shape out (CheckResult). report.py can then format
any check without knowing what it does, and adding a fourth check later touches
no existing code.

The one opinionated bit: CheckResult refuses to exist without ``limitations``.
"Every check states what it cannot tell you" is a rule this project cares
about, so it is enforced by the type rather than left to discipline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from ..schema import EvalSet

__all__ = ["Check", "CheckResult", "Finding", "Severity"]


class Severity(StrEnum):
    """How much a finding should worry you.

    Two levels on purpose. This tool reports, it does not grade: the user
    decides what is acceptable for their eval. WARNING means "this probably
    distorts your numbers"; INFO means "you should know this, it may be fine".
    """

    INFO = "info"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    """One specific problem found in an eval set.

    ``case_ids`` is the full list of cases involved, never truncated. Deciding
    to print only the first five is the report's job, not the check's — the
    check should not throw away data a JSON consumer might want.
    """

    message: str
    severity: Severity = Severity.WARNING
    case_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("a finding must have a message")
        object.__setattr__(self, "case_ids", tuple(self.case_ids))


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Everything one check has to say about one eval set.

    Attributes:
        check: The check's name, e.g. "imbalance".
        summary: One factual line, printed whether or not anything was found.
        findings: The specific problems. Empty means the check ran and was
            happy — which is different from the check not running.
        stats: The raw numbers behind the summary, for JSON output and for a
            user who wants to apply their own thresholds.
        limitations: What this check cannot tell you. Must not be empty.
        partial: Reasons this check ran but could not run FULLY — e.g. an
            optional detector whose dependency is absent. Empty means the check
            did everything it claims to do.

            This exists so a degraded check cannot report a clean gate. A check
            that raises is already recorded as an incomplete audit; one that
            quietly does less than advertised was not, and "some of the
            analysis silently did not happen" is precisely the failure this
            project reports in eval sets. The CLI treats a non-empty `partial`
            the same as a raised check: exit 3, not 0.
    """

    check: str
    summary: str
    findings: tuple[Finding, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    partial: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "partial", tuple(self.partial))
        if not self.limitations:
            raise ValueError(
                f"check '{self.check}' returned no limitations: every check "
                "must state what it cannot tell you"
            )

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


class Check(ABC):
    """Base class for the three v1 checks.

    Thresholds are constructor arguments, not run() arguments, so a configured
    check can be built once and handed around as a plain callable-ish object.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    def run(self, eval_set: EvalSet) -> CheckResult:
        """Audit ``eval_set`` and report what was found."""
