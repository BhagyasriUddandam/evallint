"""Assertions that the documentation matches the code.

Every number in the README is a claim, and a stale one is the same defect this
project reports in eval sets: confident text about the wrong state. The test
count in particular has now gone stale twice -- once caught only by reading the
published PyPI page, which cannot be edited after release.

So the invariant is enforced here rather than remembered.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _collected_test_count() -> int:
    """Count the full suite via a subprocess collection.

    A subprocess rather than the current session's item count, because the
    session count depends on how pytest was invoked -- `-k something` would
    make an in-session check fail for no reason. --collect-only does not
    execute anything, so this cannot recurse.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", result.stdout)
    assert match, f"could not parse collection output:\n{result.stdout[-500:]}"
    return int(match.group(1))


def test_readme_states_the_real_test_count() -> None:
    stated = re.search(r"uv run pytest\s*#\s*([\d,]+)\s+tests", README.read_text())
    assert stated, "README no longer states a test count in the expected form"
    claimed = int(stated.group(1).replace(",", ""))
    actual = _collected_test_count()
    assert claimed == actual, (
        f"README claims {claimed} tests, the suite collects {actual}. "
        "Update README.md — a stale count is a false claim in the package "
        "description published to PyPI, which cannot be edited after release."
    )


def test_readme_documents_every_discrimination_knob() -> None:
    """A public constructor parameter absent from the README is undiscoverable.

    `sample` shipped without a mention while `repeats` and `max_workers` were
    both documented, so this pins the whole set rather than the one that was
    missed.
    """
    import inspect

    from evallint.checks import DiscriminationCheck

    text = README.read_text()
    params = [
        name
        for name, p in inspect.signature(DiscriminationCheck.__init__).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert params, "expected keyword-only parameters to document"
    # Word-boundary match, not substring: a plain `"sample" in text` is
    # satisfied by the filename `sample_evalset.jsonl`, which would have hidden
    # the very parameter that prompted this test.
    missing = [
        p for p in params
        if not re.search(rf"(?<![\w-]){re.escape(p)}(?![\w-])", text)
    ]
    assert not missing, f"README does not mention: {missing}"


# --------------------------------------------------------------------------
# Schema 2 documentation
# --------------------------------------------------------------------------

SCHEMA_DOC = ROOT / "docs" / "schema-v2.md"


def test_every_canonical_field_is_documented() -> None:
    """A field the loader reads but the docs never mention is undiscoverable,
    and the reference table is exactly where someone looks for it."""
    from evallint.validation import CANONICAL_FIELDS

    text = SCHEMA_DOC.read_text()
    missing = [f for f in CANONICAL_FIELDS if f"`{f}`" not in text]
    assert not missing, f"undocumented schema fields: {missing}"


def test_every_role_is_documented() -> None:
    from evallint.schema import Role

    text = SCHEMA_DOC.read_text()
    missing = [r.value for r in Role if r.value not in text]
    assert not missing, f"undocumented roles: {missing}"


def test_the_readme_documents_the_new_cli_flags() -> None:
    """Mirrors the discrimination-knob test: a flag absent from the README
    cannot be found by anyone who is not reading --help."""
    text = README.read_text()
    for flag in ("--migrate-to", "--overwrite"):
        assert flag in text, f"{flag} is not mentioned in the README"


def test_the_supported_versions_claim_matches_the_code() -> None:
    """The docs promise "supported versions: 1, 2" in an error message. If the
    tuple grows, that promise goes stale in two places at once."""
    from evallint.schema import SUPPORTED_SCHEMA_VERSIONS

    assert SUPPORTED_SCHEMA_VERSIONS == (1, 2), (
        "SUPPORTED_SCHEMA_VERSIONS changed; update docs/schema-v2.md and the "
        "README, which both quote the supported set."
    )
    assert "supported versions: 1, 2" in SCHEMA_DOC.read_text()


# --------------------------------------------------------------------------
# The unified audit report
# --------------------------------------------------------------------------

AUDIT_DOC = ROOT / "docs" / "audit-report.md"


def test_the_audit_doc_states_the_real_test_count_for_itself() -> None:
    """The doc claims a number of audit tests. Same reasoning as the README
    count: a stale number is a false claim in published documentation."""
    import re as _re
    import subprocess as _sub

    stated = _re.search(r"Yes — (\d+) tests", AUDIT_DOC.read_text())
    assert stated, "docs/audit-report.md no longer states its test count"
    result = _sub.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         str(ROOT / "tests" / "test_audit.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    actual = int(_re.search(r"(\d+)\s+tests?\s+collected", result.stdout).group(1))
    assert int(stated.group(1)) == actual, (
        f"docs/audit-report.md claims {stated.group(1)} audit tests, "
        f"test_audit.py collects {actual}"
    )


def test_every_tier_is_documented() -> None:
    from evallint.audit import Tier

    text = AUDIT_DOC.read_text()
    missing = [t.value for t in Tier if f"`{t.value.upper()}`" not in text
               and f"`{t.value}`" not in text]
    assert not missing, f"undocumented tiers: {missing}"


def test_every_analysis_section_is_documented() -> None:
    from evallint.audit import ANALYSIS_SECTIONS

    text = AUDIT_DOC.read_text()
    missing = [k for k in ANALYSIS_SECTIONS if k not in text]
    assert not missing, f"undocumented sections: {missing}"


def test_the_readme_documents_the_format_flags() -> None:
    text = README.read_text()
    for flag in ("--format terminal", "--format json", "--format html", "--detail"):
        assert flag in text, f"{flag} is not mentioned in the README"


# --------------------------------------------------------------------------
# The documentation principle: evallint audits the reliability of LLM
# evaluations, and says what it cannot tell you
# --------------------------------------------------------------------------

CHECKS_DOC = ROOT / "docs" / "checks.md"

#: The eight headings every check must document. Required, not encouraged: a
#: check whose assumptions or false negatives are undocumented invites a reader
#: to treat a heuristic as a measurement.
REQUIRED_HEADINGS = (
    "What it measures",
    "Why it matters",
    "Methodology",
    "Assumptions",
    "Limitations",
    "Example",
    "False positives",
    "False negatives",
)

#: Claims no method in this package can support. Each was present in the README
#: before the rewrite; the guard exists so they cannot come back.
BANNED_CLAIMS = (
    "silently lie",
    "silent lie",
    "bad eval set",
    "too easy",
    "proves that",
    "guarantees that",
    "detects all",
    "a good eval",
)


def test_the_readme_states_the_principle() -> None:
    assert "audits the reliability of LLM evaluations" in README.read_text()


def test_the_readme_distinguishes_the_three_qualities() -> None:
    """Conflating dataset, evaluation and model quality is the most common way
    to misread the output, so the distinction must be stated, not implied."""
    text = README.read_text()
    for phrase in ("Dataset quality", "Evaluation quality", "Model quality"):
        assert f"**{phrase}**" in text, f"{phrase} is not called out in the README"
    assert "does not measure it" in text


def test_the_readme_has_a_cannot_tell_you_section() -> None:
    assert "## What evallint cannot tell you" in README.read_text()


def test_the_readme_denies_that_a_non_separating_case_is_bad() -> None:
    """The explicit instruction behind the rewrite, pinned so a future edit
    cannot quietly reintroduce the verdict."""
    text = README.read_text()
    assert "A non-separating case is not a bad case" in text
    assert "provides limited evidence for model separation" in text


@pytest.mark.parametrize("claim", BANNED_CLAIMS)
def test_unsupportable_claims_are_absent(claim: str) -> None:
    for doc in (README, CHECKS_DOC):
        assert claim not in doc.read_text().lower(), (
            f"{doc.name} contains {claim!r}, which no method in this package "
            "can support"
        )


def test_every_check_documents_all_eight_headings() -> None:
    """Each `## n. Name` section must carry every required heading."""
    import re as _re

    text = CHECKS_DOC.read_text()
    sections = _re.split(r"\n## (?=\d+\. )", text)[1:]
    assert len(sections) >= 9, f"expected at least 9 checks, found {len(sections)}"

    problems = []
    for section in sections:
        name = section.split("\n", 1)[0].strip()
        for heading in REQUIRED_HEADINGS:
            if f"### {heading}" not in section:
                # False positives / false negatives may be documented together
                # where the check is not a classifier and neither applies alone.
                if heading.startswith("False") and (
                    "### False positives / false negatives" in section
                ):
                    continue
                problems.append(f"{name}: missing '### {heading}'")
    assert not problems, "\n".join(problems)


def test_the_documented_case_classes_match_the_code() -> None:
    """Regression: the README documented `inverted` for two releases after the
    0.4.0 redesign renamed it `non_monotonic`, and never mentioned `separating`
    or `indeterminate` at all. Names now come from the enum."""
    from evallint.checks import CaseClass

    text = README.read_text() + CHECKS_DOC.read_text()
    missing = [c.value for c in CaseClass if f"`{c.value}`" not in text]
    assert not missing, f"case classes absent from the docs: {missing}"
    assert "`inverted`" not in text, "`inverted` was renamed in 0.4.0"


def test_the_precise_terminology_is_used() -> None:
    text = (README.read_text() + CHECKS_DOC.read_text()).lower()
    for phrase in (
        "potential redundancy",
        "potential leakage",
        "ground-truth ambiguity",
        "limited evidence for model separation",
        "not identifiable",
    ):
        assert phrase in text, f"{phrase!r} is not used anywhere in the docs"


def test_the_checks_doc_is_linked_from_the_readme() -> None:
    assert "docs/checks.md" in README.read_text()


def test_measured_rates_in_the_checks_doc_match_the_benchmark_baseline() -> None:
    """The doc quotes precision and recall figures. A stale number there is a
    false claim about measured performance, which is worse than no number."""
    import json as _json

    baseline = _json.loads(
        (ROOT / "benchmarks" / "results" / "test.json").read_text()
    )
    text = CHECKS_DOC.read_text()
    for detector, metric, claimed in (
        ("leakage", "precision", "0.667"),
        ("ambiguous_references", "recall", "0.667"),
        ("class_imbalance", "recall", "0.667"),
    ):
        actual = baseline["detectors"][detector]["confusion"][metric]
        assert f"{actual:.3f}" == claimed, (
            f"{detector} {metric} is now {actual}, but docs/checks.md quotes "
            f"{claimed}; update the doc"
        )
        assert claimed in text
