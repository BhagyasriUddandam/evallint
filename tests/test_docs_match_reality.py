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
