"""Tests for report rendering.

The report is where honesty either survives or quietly gets dropped, so most
of these assert on what MUST appear: limitations, skipped checks, and the fact
that a clean run says so rather than printing nothing.
"""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from evallint.checks.base import CheckResult, Finding, Severity
from evallint.report import render_text, to_dict

LIMITS = ("this check cannot read minds",)


def result(
    check: str = "imbalance",
    summary: str = "20 cases across 4 classes",
    findings: tuple[Finding, ...] = (),
) -> CheckResult:
    return CheckResult(
        check=check,
        summary=summary,
        findings=findings,
        stats={"n_cases": 20},
        limitations=LIMITS,
    )


def render(results, **kwargs) -> str:
    """Render to a string with a fixed width so wrapping is deterministic."""
    console = Console(width=100, force_terminal=False, no_color=True)
    with console.capture() as capture:
        render_text(results, console=console, **kwargs)
    return capture.get()


# --------------------------------------------------------------------------
# The honesty contract
# --------------------------------------------------------------------------


def test_limitations_are_always_printed() -> None:
    text = render([result()], source="x.jsonl")

    assert "What this audit cannot tell you" in text
    assert "this check cannot read minds" in text


def test_limitations_are_printed_even_when_nothing_was_found() -> None:
    """A clean result is exactly when a reader is most likely to over-trust it."""
    text = render([result()], source="x.jsonl")

    assert "No warnings from 1 check" in text
    assert "this check cannot read minds" in text


def test_skipped_checks_are_named_not_omitted() -> None:
    """A report covering two thirds of the tool must say which third is absent."""
    text = render([result()], source="x.jsonl", notes=["discrimination — not run"])

    assert "Not run" in text
    assert "discrimination — not run" in text


def test_source_is_shown() -> None:
    assert "examples/sample_evalset.jsonl" in render(
        [result()], source="examples/sample_evalset.jsonl"
    )


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


def test_warnings_and_info_are_labelled_differently() -> None:
    findings = (
        Finding("something is wrong", severity=Severity.WARNING),
        Finding("something is worth knowing", severity=Severity.INFO),
    )
    text = render([result(findings=findings)], source="x.jsonl")

    assert "WARN" in text
    assert "INFO" in text
    assert "1 warning across 1 check, plus 1 for information" in text


def test_case_ids_are_truncated_in_the_terminal_view() -> None:
    """The check keeps every id; only the human view shortens the list."""
    finding = Finding("many cases", case_ids=tuple(f"c{i}" for i in range(20)))
    text = render([result(findings=(finding,))], source="x.jsonl", max_case_ids=3)

    assert "c0, c1, c2 (+17 more)" in text
    assert "c19" not in text


def test_short_case_id_lists_are_not_truncated() -> None:
    finding = Finding("two cases", case_ids=("a1", "a2"))
    text = render([result(findings=(finding,))], source="x.jsonl", max_case_ids=6)

    assert "a1, a2" in text
    assert "more)" not in text


def test_long_messages_wrap_with_a_hanging_indent() -> None:
    """Continuation lines must line up under the message, not at the margin,
    or a report of long findings becomes unreadable."""
    finding = Finding("word " * 40)
    text = render([result(findings=(finding,))], source="x.jsonl")

    wrapped = [line for line in text.splitlines() if line.strip().startswith("word")]
    assert wrapped, "expected the message to wrap onto its own lines"
    assert all(line.startswith(" " * 8) for line in wrapped)


def test_markup_in_user_data_is_printed_verbatim() -> None:
    """Case ids and messages come from a user's file and may contain square
    brackets, which rich would read as style tags.

    Text() already renders its content literally, so the id must survive
    unchanged — and must NOT pick up a backslash from being escaped a second
    time on top of that. Protection belongs at exactly one layer.
    """
    finding = Finding("[red]message[/red]", case_ids=("[bold]not-a-style[/bold]",))
    text = render([result(findings=(finding,))], source="x.jsonl")

    assert "[bold]not-a-style[/bold]" in text
    assert "[red]message[/red]" in text
    assert "\\" not in text


def test_empty_results_are_reported_as_such() -> None:
    assert "No checks ran." in render([], source="x.jsonl")


# --------------------------------------------------------------------------
# JSON form
# --------------------------------------------------------------------------


def test_to_dict_is_json_serialisable() -> None:
    finding = Finding("problem", severity=Severity.WARNING, case_ids=("a1",))
    payload = to_dict([result(findings=(finding,))], source="x.jsonl", notes=["n"])

    text = json.dumps(payload)  # must not raise

    assert json.loads(text)["checks"][0]["findings"][0]["severity"] == "warning"


def test_to_dict_keeps_every_case_id() -> None:
    """The JSON consumer is a script, so nothing is truncated for it."""
    finding = Finding("many", case_ids=tuple(f"c{i}" for i in range(20)))
    payload = to_dict([result(findings=(finding,))], source="x.jsonl")

    assert len(payload["checks"][0]["findings"][0]["case_ids"]) == 20


def test_to_dict_carries_limitations_and_notes() -> None:
    payload = to_dict([result()], source="x.jsonl", notes=["discrimination not run"])

    assert payload["checks"][0]["limitations"] == list(LIMITS)
    assert payload["notes"] == ["discrimination not run"]


def test_to_dict_counts_warnings() -> None:
    findings = (
        Finding("a", severity=Severity.WARNING),
        Finding("b", severity=Severity.INFO),
    )
    payload = to_dict([result(findings=findings)], source="x.jsonl")

    assert payload["summary"] == {"n_checks": 1, "n_findings": 2, "n_warnings": 1}


def test_numpy_stats_do_not_break_json() -> None:
    """The checks use numpy, and a stray np.float64 in stats would otherwise
    turn --json into a crash at the very end of a long run."""
    np = pytest.importorskip("numpy")
    check_result = CheckResult(
        check="imbalance",
        summary="s",
        stats={"ratio": np.float64(2.5), "counts": [np.int64(3)], "n": np.int32(1)},
        limitations=LIMITS,
    )

    payload = to_dict([check_result], source="x.jsonl")
    text = json.dumps(payload)  # must not raise

    assert json.loads(text)["checks"][0]["stats"] == {
        "ratio": 2.5,
        "counts": [3],
        "n": 1,
    }
