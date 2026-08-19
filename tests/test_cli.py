"""Tests for the command line entry point.

These run the CLI in-process with click's CliRunner. Duplicates are skipped in
most of them so the suite never depends on downloading an embedding model; one
test covers the path where loading the model fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from evallint.cli import main

EXAMPLE_SET = Path(__file__).resolve().parents[1] / "examples" / "sample_evalset.jsonl"


def run(*args: str):
    """Invoke the CLI with config discovery OFF unless the test asks for it.

    find_config() searches parent directories, so without this a config file
    added anywhere above the repo -- or in a developer's home directory --
    would silently change the behaviour these tests assert. Config resolution
    gets its own tests, which opt in explicitly.
    """
    argv = list(args)
    if "--config" not in argv and "--no-config" not in argv:
        argv.append("--no-config")
    return CliRunner().invoke(main, argv)


def run_raw(*args: str):
    """Invoke without injecting --no-config, for the config tests themselves."""
    return CliRunner().invoke(main, list(args))


# NOTE on click >= 8.2: Result.output is stdout AND stderr combined. Tests that
# parse the --json payload must use result.stdout, or a single line written to
# stderr (the gate explanation) silently breaks the JSON parse.


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

    payload = json.loads(result.stdout)

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

    assert json.loads(strict.stdout)["summary"]["n_warnings"] > 0
    assert json.loads(relaxed.stdout)["summary"]["n_warnings"] == 1  # ratio only


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

    # 3, not 1: an unreadable file is an INCOMPLETE audit, not a failed gate.
    # A pipeline must be able to tell "your eval set has warnings" from
    # "I could not read your eval set".
    assert result.exit_code == 3
    assert "line 2: invalid JSON" in result.output
    assert "Traceback" not in result.output


def test_duplicate_check_failure_does_not_lose_the_imbalance_result(
    monkeypatch,
) -> None:
    """If the embedding model cannot load, the user should still get the check
    that already succeeded, plus the reason the other one is missing."""

    def explode(self, eval_set):
        raise OSError("no network")

    monkeypatch.setattr("evallint.cli.DuplicateCheck.run", explode)
    result = run(str(EXAMPLE_SET))

    # The user still gets the check that succeeded...
    assert "imbalance ratio 14.0:1" in result.output
    assert "duplicates — could not run: no network" in result.output
    # ...but the exit code says the audit was PARTIAL. This used to exit 0,
    # which meant a CI run where the embedding model failed to load went green
    # having actually run one check out of three. A partial audit reported as
    # clean is precisely the silent lie this project exists to catch.
    assert result.exit_code == 3
    assert "audit incomplete" in result.output


def test_unsupported_extension_is_a_clean_error(tmp_path: Path) -> None:
    other = tmp_path / "cases.txt"
    other.write_text("hello", encoding="utf-8")

    result = run(str(other))

    assert result.exit_code == 3  # incomplete, not a gate failure
    assert "unsupported file type" in result.output


# --------------------------------------------------------------------------
# --fail-on: the CI gate
# --------------------------------------------------------------------------


def test_default_exits_zero_even_with_warnings() -> None:
    """The default is deliberately permissive. Every check states that a
    finding is a prompt to look rather than a verdict, so failing a build by
    default would assert something the tool explicitly declines to claim."""
    result = run(str(EXAMPLE_SET), "--skip-duplicates")

    assert result.exit_code == 0
    assert "under-represented" in result.output  # warnings ARE present


def test_fail_on_warning_trips_the_gate() -> None:
    result = run(str(EXAMPLE_SET), "--skip-duplicates", "--fail-on", "warning")

    assert result.exit_code == 1
    assert "gate: FAILED" in result.output
    assert "--fail-on warning" in result.output


def test_fail_on_warning_passes_a_clean_set(tmp_path: Path) -> None:
    """A balanced, non-duplicated set must exit 0 under the gate, or the gate
    is useless — it would fail every build regardless of the data."""
    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        "\n".join(
            f'{{"id": "c{i}", "input": "question number {i}", '
            f'"expected": "answer {i}", "label": "{chr(97 + i % 3)}"}}'
            for i in range(18)
        ),
        encoding="utf-8",
    )

    result = run(str(clean), "--skip-duplicates", "--fail-on", "warning")

    assert result.exit_code == 0, result.output
    assert "gate: passed" in result.output


def test_fail_on_any_also_counts_info_findings(tmp_path: Path) -> None:
    """An unlabelled set produces an INFO finding and no warnings, so it
    separates the two gate levels."""
    unlabelled = tmp_path / "u.jsonl"
    unlabelled.write_text(
        "\n".join(
            f'{{"id": "c{i}", "input": "q{i}", "expected": "a{i}"}}'
            for i in range(6)
        ),
        encoding="utf-8",
    )

    warn_gate = run(str(unlabelled), "--skip-duplicates", "--fail-on", "warning")
    any_gate = run(str(unlabelled), "--skip-duplicates", "--fail-on", "any")

    assert warn_gate.exit_code == 0, "no warnings, so the warning gate passes"
    assert any_gate.exit_code == 1, "an INFO finding must trip --fail-on any"


def test_incomplete_audit_outranks_a_passing_gate(monkeypatch) -> None:
    """Even with --fail-on never, a check that could not run must not report a
    clean pass."""

    def explode(self, eval_set):
        raise OSError("no network")

    monkeypatch.setattr("evallint.cli.DuplicateCheck.run", explode)
    result = run(str(EXAMPLE_SET), "--fail-on", "never")

    assert result.exit_code == 3
    assert "audit incomplete" in result.output


def test_deliberate_skip_is_not_treated_as_incomplete() -> None:
    """--skip-duplicates is the user choosing a narrower scope, not a failure."""
    result = run(str(EXAMPLE_SET), "--skip-duplicates", "--fail-on", "never")

    assert result.exit_code == 0
    assert "audit incomplete" not in result.output


def test_json_output_carries_the_gate_decision() -> None:
    """A CI consumer parsing --json should not have to re-derive the verdict."""
    result = run(str(EXAMPLE_SET), "--json", "--skip-duplicates", "--fail-on", "warning")

    payload = json.loads(result.stdout)

    assert payload["gate"]["fail_on"] == "warning"
    assert payload["gate"]["tripped"] is True
    assert payload["gate"]["exit_code"] == 1
    assert payload["gate"]["incomplete"] == []
    assert result.exit_code == 1


def test_json_stays_parseable_when_the_gate_trips() -> None:
    """The gate explanation goes to stderr so `--json > f` stays clean."""
    # click >= 8.2 separates stdout and stderr on Result by default; the
    # mix_stderr constructor argument was removed.
    result = run(str(EXAMPLE_SET), "--json", "--skip-duplicates", "--fail-on", "warning")

    json.loads(result.stdout)  # must not raise — the gate line went to stderr
    assert "gate: FAILED" in result.stderr
    assert result.exit_code == 1


def test_bad_fail_on_value_is_rejected() -> None:
    result = run(str(EXAMPLE_SET), "--skip-duplicates", "--fail-on", "sometimes")

    assert result.exit_code == 2  # click usage error
    assert "sometimes" in result.output


# --------------------------------------------------------------------------
# Config file: thresholds that live in version control
# --------------------------------------------------------------------------


def _write_set(tmp_path: Path) -> Path:
    """An 18-case set, 6 per class — clean under the default thresholds."""
    p = tmp_path / "set.jsonl"
    p.write_text(
        "\n".join(
            f'{{"id": "c{i}", "input": "question {i}", "expected": "a{i}", '
            f'"label": "{chr(97 + i % 3)}"}}'
            for i in range(18)
        ),
        encoding="utf-8",
    )
    return p


def test_evallint_toml_is_discovered_and_applied(tmp_path: Path) -> None:
    data = _write_set(tmp_path)
    (tmp_path / "evallint.toml").write_text(
        "min_class_count = 10\nskip_duplicates = true\nfail_on = 'warning'\n",
        encoding="utf-8",
    )

    result = run_raw(str(data), "--json")

    # 6 per class < 10, so the config's stricter threshold must bite...
    assert json.loads(result.stdout)["summary"]["n_warnings"] == 3
    # ...and its fail_on must be honoured too.
    assert result.exit_code == 1


def test_pyproject_tool_section_is_discovered(tmp_path: Path) -> None:
    data = _write_set(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'x'\n\n[tool.evallint]\nmin_class_count = 10\n"
        "skip_duplicates = true\n",
        encoding="utf-8",
    )

    result = run_raw(str(data), "--json")

    assert json.loads(result.stdout)["summary"]["n_warnings"] == 3


def test_evallint_toml_wins_over_pyproject(tmp_path: Path) -> None:
    data = _write_set(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.evallint]\nmin_class_count = 99\nskip_duplicates = true\n",
        encoding="utf-8",
    )
    (tmp_path / "evallint.toml").write_text(
        "min_class_count = 1\nskip_duplicates = true\n", encoding="utf-8"
    )

    result = run_raw(str(data), "--json")

    # min_class_count = 1 silences the count warnings; 99 would flag all three.
    assert json.loads(result.stdout)["summary"]["n_warnings"] == 0


def test_config_is_found_in_a_parent_directory(tmp_path: Path) -> None:
    """A config at the repo root should apply to a set in a subdirectory."""
    (tmp_path / "evallint.toml").write_text(
        "min_class_count = 10\nskip_duplicates = true\n", encoding="utf-8"
    )
    nested = tmp_path / "evals" / "deep"
    nested.mkdir(parents=True)
    data = _write_set(nested)

    result = run_raw(str(data), "--json")

    assert json.loads(result.stdout)["summary"]["n_warnings"] == 3


def test_an_explicit_flag_overrides_the_config(tmp_path: Path) -> None:
    """Precedence: CLI flag > config file > default."""
    data = _write_set(tmp_path)
    (tmp_path / "evallint.toml").write_text(
        "min_class_count = 10\nskip_duplicates = true\n", encoding="utf-8"
    )

    result = run_raw(str(data), "--json", "--min-class-count", "1")

    assert json.loads(result.stdout)["summary"]["n_warnings"] == 0


def test_a_flag_typed_at_its_default_value_still_wins(tmp_path: Path) -> None:
    """The subtle one: --min-class-count 5 is the DEFAULT value, so comparing
    values could not tell it from 'not given'. Precedence must come from the
    parameter source, not from the value."""
    data = _write_set(tmp_path)
    (tmp_path / "evallint.toml").write_text(
        "min_class_count = 10\nskip_duplicates = true\n", encoding="utf-8"
    )

    result = run_raw(str(data), "--json", "--min-class-count", "5")

    # 6 per class >= 5, so the explicitly-typed default silences the warnings.
    assert json.loads(result.stdout)["summary"]["n_warnings"] == 0


def test_no_config_ignores_a_present_file(tmp_path: Path) -> None:
    data = _write_set(tmp_path)
    (tmp_path / "evallint.toml").write_text(
        "min_class_count = 10\nskip_duplicates = true\n", encoding="utf-8"
    )

    result = run_raw(str(data), "--json", "--no-config", "--skip-duplicates")

    assert json.loads(result.stdout)["summary"]["n_warnings"] == 0


def test_explicit_config_path_is_used(tmp_path: Path) -> None:
    data = _write_set(tmp_path)
    elsewhere = tmp_path / "custom.toml"
    elsewhere.write_text(
        "min_class_count = 10\nskip_duplicates = true\n", encoding="utf-8"
    )

    result = run_raw(str(data), "--json", "--config", str(elsewhere))

    assert json.loads(result.stdout)["summary"]["n_warnings"] == 3


def test_unknown_config_key_is_a_loud_error(tmp_path: Path) -> None:
    """A silently ignored typo means a team believes a threshold is in force
    when it is not — the same quiet wrongness this tool reports on."""
    data = _write_set(tmp_path)
    (tmp_path / "evallint.toml").write_text("min_class_shrae = 0.5\n", encoding="utf-8")

    result = run_raw(str(data), "--json")

    assert result.exit_code == 3
    assert "unknown setting" in result.stderr
    assert "min_class_shrae" in result.stderr


def test_wrong_type_in_config_is_rejected(tmp_path: Path) -> None:
    data = _write_set(tmp_path)
    (tmp_path / "evallint.toml").write_text("min_class_count = true\n", encoding="utf-8")

    result = run_raw(str(data), "--json")

    assert result.exit_code == 3
    assert "must be int" in result.stderr


def test_bad_fail_on_in_config_is_rejected(tmp_path: Path) -> None:
    data = _write_set(tmp_path)
    (tmp_path / "evallint.toml").write_text("fail_on = 'sometimes'\n", encoding="utf-8")

    result = run_raw(str(data), "--json")

    assert result.exit_code == 3
    assert "fail_on must be one of" in result.stderr


# --------------------------------------------------------------------------
# Verbosity
# --------------------------------------------------------------------------


def test_verbose_logs_to_stderr_not_stdout() -> None:
    """`--json -v > report.json` must still produce parseable JSON."""
    result = run(str(EXAMPLE_SET), "--json", "--skip-duplicates", "-v")

    json.loads(result.stdout)  # must not raise
    assert "loaded 20 cases" in result.stderr


def test_quiet_by_default() -> None:
    result = run(str(EXAMPLE_SET), "--json", "--skip-duplicates")

    assert result.stderr == ""


def test_vv_includes_debug_detail() -> None:
    result = run(str(EXAMPLE_SET), "--json", "--skip-duplicates", "-vv")

    assert "thresholds:" in result.stderr
