"""Turn CheckResults into something a person reads or a machine parses.

This module knows nothing about what any individual check does — it renders
whatever CheckResults it is handed. That is the payoff of the common Check
interface: adding a fourth check later never touches this file.

Two deliberate layout choices:
  - Limitations are collected into one section at the end rather than printed
    under each check. They are never suppressible (there is no --brief flag),
    because "every check states what it cannot tell you" is the point of the
    tool, but interleaving 15 paragraphs of caveats with the findings would
    bury the findings.
  - Case id lists are truncated here, not in the checks. A check keeps the
    full list so a JSON consumer gets everything; only the terminal view
    shortens it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from .checks.base import CheckResult, Severity

__all__ = ["MAX_CASE_IDS", "render_text", "to_dict"]

MAX_CASE_IDS = 6
_SEVERITY_STYLE = {Severity.WARNING: "bold yellow", Severity.INFO: "bold cyan"}


def render_text(
    results: Sequence[CheckResult],
    *,
    source: str,
    notes: Sequence[str] = (),
    console: Console | None = None,
    max_case_ids: int = MAX_CASE_IDS,
) -> None:
    """Print a human-readable audit report."""
    console = console or Console()

    console.print()
    console.print(Text("evallint", style="bold") + Text(f"  {source}", style="dim"))

    for result in results:
        console.print()
        console.print(Text(result.check, style="bold magenta"))
        console.print(Padding(Text(result.summary), (0, 0, 0, 2)))
        if result.findings:
            console.print(Padding(_findings_grid(result, max_case_ids), (0, 0, 0, 2)))

    _render_verdict(console, results)

    if notes:
        console.print()
        console.print(Text("Not run", style="bold"))
        console.print(Padding(_bullets(notes), (0, 0, 0, 2)))

    if results:
        console.print()
        console.print(Text("What this audit cannot tell you", style="bold"))
        for result in results:
            console.print(Padding(Text(result.check, style="dim italic"), (0, 0, 0, 2)))
            console.print(Padding(_bullets(result.limitations), (0, 0, 0, 4)))
    console.print()


def _findings_grid(result: CheckResult, max_case_ids: int) -> Table:
    """Lay findings out as a two-column grid.

    A grid rather than plain prints so that a message long enough to wrap has
    its continuation lines aligned under the message, not back at the left
    margin. Long findings are the norm here, so this is not cosmetic.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=4, no_wrap=True)
    grid.add_column(overflow="fold")
    for finding in result.findings:
        label = "WARN" if finding.severity is Severity.WARNING else "INFO"
        grid.add_row(
            Text(label, style=_SEVERITY_STYLE[finding.severity]),
            Text(finding.message),
        )
        if finding.case_ids:
            grid.add_row(
                "", Text(_format_case_ids(finding.case_ids, max_case_ids), style="dim")
            )
    return grid


def _bullets(lines: Sequence[str]) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(overflow="fold")
    for line in lines:
        grid.add_row(Text("·", style="dim"), Text(line, style="dim"))
    return grid


def _render_verdict(console: Console, results: Sequence[CheckResult]) -> None:
    n_warnings = sum(len(r.warnings) for r in results)
    n_info = sum(len(r.findings) - len(r.warnings) for r in results)
    console.print()
    if not results:
        console.print(Text("No checks ran.", style="bold red"))
    elif n_warnings:
        console.print(
            Text(
                f"{n_warnings} warning{'' if n_warnings == 1 else 's'} "
                f"across {len(results)} check{'' if len(results) == 1 else 's'}"
                + (f", plus {n_info} for information" if n_info else ""),
                style="bold yellow",
            )
        )
    else:
        console.print(
            Text(
                f"No warnings from {len(results)} check"
                f"{'' if len(results) == 1 else 's'}"
                + (f" ({n_info} note{'' if n_info == 1 else 's'})" if n_info else ""),
                style="bold green",
            )
        )


def _format_case_ids(case_ids: Sequence[str], limit: int) -> str:
    # No rich.markup.escape() here on purpose. Every string this module renders
    # goes through Text(), which treats its content as literal and never parses
    # markup. Escaping first would double-protect and print the backslash: a
    # case id of "[bold]x" would come out as "\[bold]x", corrupting the user's
    # own data in the report. Protection belongs at exactly one layer.
    shown = ", ".join(case_ids[:limit])
    remaining = len(case_ids) - limit
    return f"{shown} (+{remaining} more)" if remaining > 0 else shown


def to_dict(
    results: Sequence[CheckResult],
    *,
    source: str,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the JSON-serialisable form of a report.

    Unlike the text view this keeps every case id and every stat, because the
    consumer is a script that may want to apply its own thresholds.
    """
    return {
        "source": source,
        "summary": {
            "n_checks": len(results),
            "n_findings": sum(len(r.findings) for r in results),
            "n_warnings": sum(len(r.warnings) for r in results),
        },
        "checks": [
            {
                "check": result.check,
                "summary": result.summary,
                "findings": [
                    {
                        "severity": str(finding.severity),
                        "message": finding.message,
                        "case_ids": list(finding.case_ids),
                    }
                    for finding in result.findings
                ],
                "stats": _plain(result.stats),
                "limitations": list(result.limitations),
            }
            for result in results
        ],
        "notes": list(notes),
    }


def _plain(value: Any) -> Any:
    """Convert numpy scalars to built-ins so json.dumps cannot choke.

    The checks use numpy internally, and a stray np.float64 in stats would
    otherwise turn --json into a crash at the very end of a long run.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value
