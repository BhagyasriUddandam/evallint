"""The v1 checks. Each one implements Check and returns a CheckResult."""

from .base import Check, CheckResult, Finding, Severity
from .discrimination import (
    CaseClass,
    CaseSeparation,
    DiscriminationCheck,
    Scorer,
    ScorerError,
)
from .separation import (
    mcnemar_exact_p,
    minimum_detectable_effect,
    paired_difference,
    wilson_interval,
)
from .duplicates import DuplicateCheck, Embedder, MissingEmbeddingsError
from .imbalance import ImbalanceCheck
from .leakage import LeakageCheck

__all__ = [
    "CaseClass",
    "CaseSeparation",
    "Check",
    "CheckResult",
    "DiscriminationCheck",
    "DuplicateCheck",
    "Embedder",
    "Finding",
    "ImbalanceCheck",
    "LeakageCheck",
    "MissingEmbeddingsError",
    "Scorer",
    "ScorerError",
    "Severity",
    # separation statistics, exported so a reader can re-derive any interval
    # or p-value the check reports rather than taking it on trust
    "mcnemar_exact_p",
    "minimum_detectable_effect",
    "paired_difference",
    "wilson_interval",
]
