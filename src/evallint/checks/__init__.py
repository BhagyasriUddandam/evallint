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
from .agreement import cohens_kappa, fleiss_kappa, pairwise_agreement, raw_agreement
from .evaluator_reliability import (
    EvaluatorReliability,
    JudgeObservation,
    ReliabilityReport,
    collect_choices,
    collect_verdicts,
)
from .ground_truth import AmbiguityJudge, GroundTruthCheck
from .reliability_stats import (
    MetricResult,
    bootstrap_interval,
    krippendorff_alpha,
    pearson_r,
    spearman_rho,
)
from .leakage import LeakageCheck
from .redundancy import (
    CompareFields,
    RedundancyCheck,
    RedundancyLevel,
)

__all__ = [
    "AmbiguityJudge",
    "CaseClass",
    "CompareFields",
    "CaseSeparation",
    "Check",
    "CheckResult",
    "DiscriminationCheck",
    "DuplicateCheck",
    "Embedder",
    "Finding",
    "ImbalanceCheck",
    "EvaluatorReliability",
    "GroundTruthCheck",
    "JudgeObservation",
    "MetricResult",
    "ReliabilityReport",
    "LeakageCheck",
    "RedundancyCheck",
    "RedundancyLevel",
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
    # inter-rater agreement, exported so a judge-reliability figure can be
    # re-derived rather than taken on trust
    "cohens_kappa",
    "fleiss_kappa",
    "pairwise_agreement",
    "raw_agreement",
    # evaluator-reliability statistics
    "bootstrap_interval",
    "collect_choices",
    "collect_verdicts",
    "krippendorff_alpha",
    "pearson_r",
    "spearman_rho",
]
