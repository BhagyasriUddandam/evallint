"""The public API surface is pinned.

evallint is on 0.x, so it is allowed to break — but not by ACCIDENT. A rename or
a dropped export should be a deliberate act recorded in the changelog, not
something a refactor does quietly to someone's import.

Adding a name is fine and only needs the list below extended. Removing or
renaming one fails here, which is the prompt to write the changelog entry.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import evallint

#: Every name `from evallint import *` provides, as of 0.17.0.
PUBLIC_API = frozenset(
    {
        # data model
        "EvalCase", "EvalSet", "Message", "Role", "Rubric", "Criterion",
        "ToolCall", "ToolSpec", "derive_input",
        "SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS",
        # loading
        "load", "load_with_mapping", "FieldMapping", "resolve_mapping",
        # checks
        "Check", "CheckResult", "Finding", "Severity", "CaseClass",
        "DiscriminationCheck", "DuplicateCheck", "ImbalanceCheck",
        "LeakageCheck", "RedundancyCheck", "RedundancyLevel", "CompareFields",
        "GroundTruthCheck", "EvaluatorReliability", "JudgeObservation",
        "CoverageCheck", "CoverageSpec",
        # analyses
        "RunOutcome", "analyse_reproducibility", "compare_models",
        "estimate_effective_size", "krippendorff_alpha", "mcnemar_exact_p",
        "minimum_detectable_effect", "wilson_interval",
        # unified report
        "AuditFinding", "AuditReport", "AuditSection", "SectionStatus", "Tier",
        "run_audit", "render_terminal", "render_html", "audit_to_json",
        # legacy report
        "render_text", "to_dict",
        # migration
        "MigrationReport", "migrate_file",
        # protocols
        "Embedder", "Scorer",
        # errors
        "AmbiguousMappingError", "LoadError", "MappingError",
        "MigrationError", "MissingEmbeddingsError", "SchemaError",
        "SchemaValidationError", "ScorerError", "ValidationIssue",
        # metadata
        "__version__",
    }
)


def test_no_public_name_disappeared() -> None:
    """The direction that breaks a user's import."""
    missing = sorted(PUBLIC_API - set(evallint.__all__))
    assert not missing, (
        f"{missing} vanished from evallint.__all__. If that is deliberate, note "
        "it as a breaking change in CHANGELOG.md and update PUBLIC_API here."
    )


def test_new_public_names_are_recorded() -> None:
    """Additive changes are fine; this only keeps the pin honest."""
    added = sorted(set(evallint.__all__) - PUBLIC_API)
    assert not added, (
        f"{added} were added to evallint.__all__ but not to PUBLIC_API. Add "
        "them here so the next removal is still caught."
    )


def test_every_exported_name_actually_resolves() -> None:
    """A name in __all__ that does not exist makes `import *` raise."""
    broken = [n for n in evallint.__all__ if not hasattr(evallint, n)]
    assert not broken, f"exported but absent: {broken}"


def test_import_star_works() -> None:
    namespace: dict[str, object] = {}
    exec("from evallint import *", namespace)  # noqa: S102 - that is the test
    assert "EvalCase" in namespace


@pytest.mark.parametrize(
    "module",
    [m.name for m in pkgutil.walk_packages(evallint.__path__, "evallint.")],
)
def test_every_submodule_imports_cleanly(module: str) -> None:
    """No import-time side effects, and no module that only works by accident
    of being imported second."""
    importlib.import_module(module)


def test_every_submodule_declares_all() -> None:
    """__all__ is what makes a module's surface reviewable. A module without one
    exports whatever it happens to have imported."""
    undeclared = []
    for info in pkgutil.walk_packages(evallint.__path__, "evallint."):
        module = importlib.import_module(info.name)
        if not hasattr(module, "__all__"):
            undeclared.append(info.name)
    assert not undeclared, f"no __all__ in: {undeclared}"


def test_the_version_is_a_real_release_number() -> None:
    assert evallint.__version__ != "0.0.0+unknown", (
        "version metadata is missing; the package is not installed properly"
    )
    parts = evallint.__version__.split(".")
    assert len(parts) >= 3 and all(p.isdigit() for p in parts[:3])
