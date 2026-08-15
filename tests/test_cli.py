"""Tests for the command line entry point.

These run the CLI in-process with click's CliRunner. Duplicates are skipped in
most of them so the suite never depends on downloading an embedding model; one
test covers the path where loading the model fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from evalcheck.cli import main

EXAMPLE_SET = Path(__file__).resolve().parents[1] / "examples" / "sample_evalset.jsonl"


def run(*args: str):
    return CliRunner().invoke(main, list(args))


# --------------------------------------------------------------------------
# The v1 definition of done
# --------------------------------------------------------------------------


def test_running_the_example_set_prints_a_report() -> None:
    result = run(str(EXAMPLE_SET), "--skip-duplicates")

    assert result.exit_code == 0
    assert "imbalance" in result.output
    assert "imbalance ratio 14.0:1" in result.output
    assert "What this audit cannot tell you" in result.output


def test_discrimination_is_reported_as_not_run() -> None:
    """The CLI cannot run discrimination, and must say so rather than quietly
    producing a report that looks complete."""
    result = run(str(EXAMPLE_SET), "--skip-duplicates")

    assert "Not run" in result.output
    assert "discrimination" in result.output
    assert "needs a scoring function" in result.output


def test_skipping_duplicates_is_disclosed() -> None:
    result = run(str(EXAMPLE_SET), "--skip-duplicates")
    assert "skipped with --skip-duplicates" in result.output


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


def test_json_output_is_parseable() -> None:
    result = run(str(EXAMPLE_SET), "--json", "--skip-duplicates")

    payload = json.loads(result.output)

    assert payload["source"].endswith("sample_evalset.jsonl")
    assert payload["checks"][0]["check"] == "imbalance"
    assert payload["checks"][0]["stats"]["class_counts"]["billing"] == 14
    assert any("discrimination" in note for note in payload["notes"])


def test_imbalance_thresholds_are_honoured() -> None:
    """Relaxing the thresholds must actually silence the warnings."""
    strict = run(str(EXAMPLE_SET), "--json", "--skip-duplicates")
    relaxed = run(
        str(EXAMPLE_SET),
        "--json",
        "--skip-duplicates",
        "--min-class-share",
        "0.01",
        "--min-class-count",
        "1",
    )

    assert json.loads(strict.output)["summary"]["n_warnings"] > 0
    assert json.loads(relaxed.output)["summary"]["n_warnings"] == 1  # ratio only


def test_bad_threshold_is_rejected_by_the_parser() -> None:
    result = run(str(EXAMPLE_SET), "--skip-duplicates", "--duplicate-threshold", "2")

    assert result.exit_code != 0
    assert "2" in result.output


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


def test_missing_file_is_a_clean_error_not_a_traceback() -> None:
    result = run("no_such_file.jsonl")

    assert result.exit_code != 0
    assert "no_such_file.jsonl" in result.output
    assert "Traceback" not in result.output


def test_malformed_file_is_a_clean_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "input": "ok"}\n{not json\n', encoding="utf-8")

    result = run(str(bad), "--skip-duplicates")

    assert result.exit_code == 1
    assert "line 2: invalid JSON" in result.output
    assert "Traceback" not in result.output


def test_duplicate_check_failure_does_not_lose_the_imbalance_result(
    monkeypatch,
) -> None:
    """If the embedding model cannot load, the user should still get the check
    that already succeeded, plus the reason the other one is missing."""

    def explode(self, eval_set):
        raise OSError("no network")

    monkeypatch.setattr("evalcheck.cli.DuplicateCheck.run", explode)
    result = run(str(EXAMPLE_SET))

    assert result.exit_code == 0
    assert "imbalance ratio 14.0:1" in result.output
    assert "duplicates — could not run: no network" in result.output


def test_unsupported_extension_is_a_clean_error(tmp_path: Path) -> None:
    other = tmp_path / "cases.txt"
    other.write_text("hello", encoding="utf-8")

    result = run(str(other))

    assert result.exit_code == 1
    assert "unsupported file type" in result.output
