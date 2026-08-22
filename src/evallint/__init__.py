"""evallint — audits LLM eval datasets for flaws that make evaluations lie.

Everyone tests their model. Almost nobody tests whether their test set is any
good. evallint is not an eval runner: it audits the dataset an eval runner
consumes, for three failure modes that make a score misleading.

    >>> from evallint import load, ImbalanceCheck
    >>> result = ImbalanceCheck().run(load("evalset.jsonl"))
    >>> result.summary
    '20 cases across 4 classes, imbalance ratio 14.0:1'
    >>> result.limitations          # every check states what it cannot tell you
    (...)

The four checks:

    ImbalanceCheck        class distribution, imbalance ratio, basic stats
    LeakageCheck          cases whose expected answer is in their own input
    DuplicateCheck        near-duplicate cases via embedding similarity
                          (needs the optional `evallint[embeddings]` extra, or
                          your own embedder)
    DiscriminationCheck   cases that cannot separate a strong model from a weak
                          one. Takes an injected `score(case, model) -> bool`,
                          so evallint never talks to a model provider itself.

`DiscriminationCheck` is imported lazily via module `__getattr__` only in the
sense that everything here is cheap: no check imports a heavy dependency at
module scope, so `import evallint` stays fast and provider-free.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from .checks import (
    Check,
    CheckResult,
    DiscriminationCheck,
    DuplicateCheck,
    Embedder,
    Finding,
    ImbalanceCheck,
    LeakageCheck,
    MissingEmbeddingsError,
    Scorer,
    ScorerError,
    Severity,
)
from .io import LoadError, load, load_with_mapping
from .mapping import (
    AmbiguousMappingError,
    FieldMapping,
    MappingError,
    resolve_mapping,
)
from .report import render_text, to_dict
from .schema import EvalCase, EvalSet, SchemaError

# A library must not configure logging for the application that imports it.
# NullHandler means evallint's log records go nowhere until someone calls
# logging.basicConfig() (or the CLI does it), and nobody gets a
# "No handlers could be found" warning in the meantime.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    # Read from installed metadata so pyproject.toml is the ONE place the
    # version lives. Hardcoding it here too is how a package ends up reporting
    # a version it isn't.
    __version__ = version("evallint")
except PackageNotFoundError:  # pragma: no cover - source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    # data
    "EvalCase",
    "EvalSet",
    "load",
    "load_with_mapping",
    # column mapping (real eval sets do not use evallint's field names)
    "FieldMapping",
    "resolve_mapping",
    # checks
    "Check",
    "CheckResult",
    "DiscriminationCheck",
    "DuplicateCheck",
    "Finding",
    "ImbalanceCheck",
    "LeakageCheck",
    "Severity",
    # scorer / embedder contracts
    "Embedder",
    "Scorer",
    # reporting
    "render_text",
    "to_dict",
    # errors
    "AmbiguousMappingError",
    "LoadError",
    "MappingError",
    "MissingEmbeddingsError",
    "SchemaError",
    "ScorerError",
    "__version__",
]
