"""`evallint path/to/evalset.jsonl` — audit an eval set from the terminal.

The CLI runs the two checks that need nothing but the file: imbalance and
duplicates. Discrimination needs a scoring function that calls your models,
which cannot come from a command line without locking the tool to one
provider — so the CLI reports that it was NOT run, and how to run it, rather
than quietly leaving it out. A report that silently covers two thirds of the
tool would be exactly the kind of quiet lie this project is about.

EXIT CODES (for CI)
    0  the gate passed
    1  the gate tripped -- findings at or above --fail-on
    2  usage error (bad flag or missing file; click's own convention)
    3  the audit could not be completed -- unreadable file, or a check raised

3 is deliberately distinct from 1. "Your eval set has warnings" and "I could
not finish auditing your eval set" need different reactions in a pipeline, and
collapsing them would let an incomplete audit pass as a clean one.
"""

from __future__ import annotations

import json as jsonlib
import logging
import sys
from pathlib import Path

import click
from rich.console import Console

from .checks import (
    CompareFields,
    GroundTruthCheck,
    ImbalanceCheck,
    LeakageCheck,
    RedundancyCheck,
)
from .checks.base import CheckResult
from .checks.duplicates import DEFAULT_MODEL, DEFAULT_THRESHOLD
from .config import ConfigError, find_config, load_config
from .io import LoadError, load_with_mapping
from .report import render_text, to_dict

log = logging.getLogger(__name__)

DISCRIMINATION_NOTE = (
    "discrimination — needs a scoring function that runs your models, so it "
    "cannot run from the CLI. Use the library: "
    "DiscriminationCheck(scorer, ['weak-model', 'strong-model']).run(load(path))"
)

EXIT_OK = 0
EXIT_GATE_TRIPPED = 1
EXIT_INCOMPLETE = 3


def _gate(results: list[CheckResult], fail_on: str) -> tuple[int, str]:
    """Decide the exit code from the findings. Returns (code, explanation).

    Default is `never`, i.e. always exit 0. That is a deliberate choice rather
    than laziness: this tool's own position is that a finding is "a prompt to
    look, not a verdict" -- the checks say so in their limitations. Failing a
    build by default would assert that a warning means "broken", which the tool
    explicitly declines to claim. CI opts in with one flag.
    """
    n_warn = sum(len(r.warnings) for r in results)
    n_all = sum(len(r.findings) for r in results)

    if fail_on == "never":
        return EXIT_OK, f"gate: never (exit 0 regardless; {n_warn} warnings)"
    if fail_on == "warning":
        if n_warn:
            return EXIT_GATE_TRIPPED, f"gate: FAILED — {n_warn} warning(s), --fail-on warning"
        return EXIT_OK, "gate: passed — no warnings"
    # fail_on == "any"
    if n_all:
        return EXIT_GATE_TRIPPED, f"gate: FAILED — {n_all} finding(s), --fail-on any"
    return EXIT_OK, "gate: passed — no findings"


def _emit_audit(
    *,
    eval_set,
    results: list[CheckResult],
    path: Path,
    output_format: str,
    out_path: Path | None,
    detail: bool,
    schema_notes: list[str],
) -> None:
    """Build and write the unified report.

    The audit is composed from the results already computed above, so the
    unified view and the legacy view can never disagree about what was found.
    The three analyses that need a scorer, judges or repeated runs are not
    passed, so they appear as NOT ASSESSED -- which is what they are from a
    command line, and is reported rather than left as an empty section.
    """
    from .audit import render_html, render_terminal, run_audit, to_json

    by_check = {r.check: r for r in results}
    report = run_audit(
        eval_set,
        imbalance=by_check.get("imbalance"),
        leakage=by_check.get("leakage"),
        ground_truth=by_check.get("ground_truth"),
        redundancy=by_check.get("redundancy") or by_check.get("duplicates"),
        source=str(path),
    )

    if output_format == "terminal":
        if out_path is not None:
            with out_path.open("w", encoding="utf-8") as handle:
                render_terminal(
                    report, console=Console(file=handle, width=100), detail=detail
                )
        else:
            render_terminal(report, detail=detail)
        return

    text = to_json(report) if output_format == "json" else render_html(report)
    if out_path is not None:
        out_path.write_text(text, encoding="utf-8")
        click.echo(f"wrote {output_format} report to {out_path}", err=True)
    else:
        click.echo(text)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the full report as JSON on stdout instead of text.",
)
@click.option(
    "--skip-duplicates",
    is_flag=True,
    help=f"Skip the duplicate check and avoid loading {DEFAULT_MODEL}.",
)
@click.option(
    "--duplicate-threshold",
    default=DEFAULT_THRESHOLD,
    show_default=True,
    type=click.FloatRange(0, 1, min_open=True),
    help="Cosine similarity at or above which two cases are near-duplicates.",
)
@click.option(
    "--min-class-share",
    default=0.10,
    show_default=True,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    help="Warn when a class holds less than this share of labelled cases.",
)
@click.option(
    "--min-class-count",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Warn when a class has fewer than this many cases.",
)
@click.option(
    "--fail-on",
    type=click.Choice(["never", "warning", "any"]),
    default="never",
    show_default=True,
    help="Exit non-zero so CI can gate on the result. 'warning' fails on any "
    "WARNING finding; 'any' also fails on INFO. Default 'never' always exits 0, "
    "because a finding is a prompt to look rather than a verdict — see the "
    "limitations each check reports.",
)
@click.option(
    "--leakage-overlap",
    is_flag=True,
    help="Enable the token-overlap leakage detector. OFF by default because it "
    "is measurably unreliable: on GSM8K it reaches 100% overlap with no leakage "
    "present, since reference answers restate the question. Findings from it are "
    "labelled LOW confidence.",
)
@click.option(
    "--compare",
    type=click.Choice([f.value for f in CompareFields]),
    default=CompareFields.INPUT_EXPECTED.value,
    show_default=True,
    help="Which fields the redundancy check compares. 'input' reproduces the "
    "older input-only behaviour; the default also compares the expected answer, "
    "because two cases with the same question and different answers are a "
    "ground-truth contradiction rather than a duplicate.",
)
@click.option(
    "--map",
    "field_map",
    multiple=True,
    metavar="FIELD=COLUMN",
    help="Map one of evallint's fields onto a column in your file, e.g. "
    "--map input=question --map label=subject. Repeatable. Common aliases "
    "(question/prompt/ctx, answer/reference/best_answer, category/subject) are "
    "inferred automatically; use this when inference is ambiguous or wrong.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["terminal", "json", "html"]),
    default=None,
    help="Emit the unified audit report: eleven sections, with each finding "
    "carrying severity, evidence tier, affected cases, explanation, limitation "
    "and recommended action. 'terminal' stays concise; 'json' and 'html' carry "
    "everything. HTML sections are expandable and include case-level detail. "
    "There is deliberately no overall quality score in any format.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the report to this file instead of stdout. Required in "
    "practice for --format html.",
)
@click.option(
    "--detail",
    is_flag=True,
    help="With --format terminal, also print each finding's explanation, "
    "limitation and recommended action rather than just its evidence.",
)
@click.option(
    "--migrate-to",
    "migrate_to",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Convert PATH to the declared schema-2 format and write it here, "
    "instead of auditing. Makes an inferred --map permanent and declares the "
    "version, so later typos become errors rather than silent metadata. Your "
    "original file is never modified, and nothing is written unless the result "
    "reloads to identical cases.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow --migrate-to to replace an existing destination file.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read settings from this file instead of searching for evallint.toml "
    "or a [tool.evallint] section in pyproject.toml.",
)
@click.option(
    "--no-config",
    is_flag=True,
    help="Ignore any config file and use flags and defaults only.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Log what evallint is doing to stderr. -v for progress, -vv for detail.",
)
@click.version_option(package_name="evallint")
@click.pass_context
def main(
    ctx: click.Context,
    path: Path,
    as_json: bool,
    skip_duplicates: bool,
    duplicate_threshold: float,
    min_class_share: float,
    min_class_count: int,
    fail_on: str,
    leakage_overlap: bool,
    compare: str,
    field_map: tuple[str, ...],
    output_format: str | None,
    out_path: Path | None,
    detail: bool,
    migrate_to: Path | None,
    overwrite: bool,
    config_path: Path | None,
    no_config: bool,
    verbose: int,
) -> None:
    """Audit the eval set at PATH for flaws that make evaluations lie."""
    # Logging to stderr, never stdout: `--json > report.json` must stay clean.
    #
    # Deliberately NOT logging.basicConfig(): that is a no-op whenever the root
    # logger already has a handler, which is true under pytest and in any host
    # application that configured logging before calling us. Configuring the
    # `evallint` logger directly works regardless, and — unlike basicConfig —
    # touches nothing outside this package's namespace.
    if verbose:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_log = logging.getLogger("evallint")
        # Assign rather than append: main() can run more than once in a process
        # (tests, or an embedding host), and appending would duplicate output.
        package_log.handlers = [handler]
        package_log.setLevel(logging.DEBUG if verbose > 1 else logging.INFO)
        # Don't also bubble up to the application's root handler and print twice.
        package_log.propagate = False

    # Config file, then flags on top. A flag the user did not type must not
    # override a config value, so the check is the parameter's SOURCE rather
    # than whether it equals the default -- `--min-class-count 5` typed
    # explicitly is indistinguishable from the default by value alone.
    settings: dict[str, object] = {}
    if not no_config:
        found = config_path or find_config(path)
        if found is not None:
            try:
                settings = load_config(found)
            except ConfigError as exc:
                click.echo(f"Error: {exc}", err=True)
                sys.exit(EXIT_INCOMPLETE)
            log.info("using config %s", found)

    def resolve(name: str, flag_value):
        source = ctx.get_parameter_source(name)
        if source is not None and source.name != "DEFAULT":
            return flag_value  # explicitly given on the command line
        return settings.get(name, flag_value)

    skip_duplicates = bool(resolve("skip_duplicates", skip_duplicates))
    duplicate_threshold = float(resolve("duplicate_threshold", duplicate_threshold))
    min_class_share = float(resolve("min_class_share", min_class_share))
    min_class_count = int(resolve("min_class_count", min_class_count))
    fail_on = str(resolve("fail_on", fail_on))

    # Progress goes to stderr so that `evallint --json file > out.json` gives
    # a clean JSON file.
    status_console = Console(stderr=True)

    overrides: dict[str, str] = {}
    for item in field_map:
        if "=" not in item:
            click.echo(
                f"Error: --map expects FIELD=COLUMN, got {item!r}", err=True
            )
            sys.exit(EXIT_INCOMPLETE)
        field, _, column = item.partition("=")
        overrides[field.strip()] = column.strip()

    if migrate_to is not None:
        # Converting is not auditing, so it returns here rather than falling
        # through: running checks on the way past would make a one-line
        # conversion take an embedding pass, and the user asked for neither.
        from .migrate import MigrationError, migrate_file

        try:
            report = migrate_file(
                path, migrate_to, field_map=overrides or None, overwrite=overwrite
            )
        except MigrationError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(EXIT_INCOMPLETE)
        click.echo(report.render())
        sys.exit(EXIT_OK)

    try:
        eval_set, mapping = load_with_mapping(path, overrides or None)
    except LoadError as exc:
        # Exit 3, not 1: an unreadable file is an incomplete audit, not a
        # failed gate. A pipeline should react differently to the two.
        click.echo(f"Error: {exc}", err=True)
        sys.exit(EXIT_INCOMPLETE)

    log.debug(
        "thresholds: min_class_share=%s min_class_count=%s "
        "duplicate_threshold=%s fail_on=%s",
        min_class_share, min_class_count, duplicate_threshold, fail_on,
    )
    log.info("running imbalance and leakage on %d cases", len(eval_set))
    # Both are pure text analysis: no model, no network, no optional extra. They
    # always run, which is why they are the ones a new user actually tries.
    results = [
        ImbalanceCheck(
            min_class_share=min_class_share, min_class_count=min_class_count
        ).run(eval_set),
        LeakageCheck(overlap=leakage_overlap).run(eval_set),
        # No judges from the CLI: an LLM analyser cannot be configured from
        # a command line without locking the tool to one provider, exactly as
        # with the discrimination scorer. The deterministic detectors run.
        GroundTruthCheck().run(eval_set),
    ]
    notes = [DISCRIMINATION_NOTE]
    # Always surface a non-identity mapping. A silently wrong column choice
    # produces a plausible report about the wrong data, so it must be visible
    # without needing -v.
    if not mapping.is_identity() or mapping.alternatives:
        notes.extend(f"field map: {line}" for line in mapping.explain())
    # Same reasoning as the field map: a field that was DEMOTED to metadata, or
    # a conversation shape that had to be interpreted, changes what every check
    # below actually looked at. It must be visible without -v -- and under its
    # own heading rather than "Not run", which would misdescribe it.
    schema_notes = list(eval_set.load_notes)
    if eval_set.schema_version < 2 and eval_set.chat_cases:
        schema_notes.append(
            f"{eval_set.chat_cases} case(s) were read as conversations from a "
            'file that declares no version. Add {"evallint_schema": 2} to have '
            "evallint reject malformed turns instead of keeping them as metadata."
        )
    incomplete: list[str] = []

    if skip_duplicates:
        notes.append("duplicates — skipped with --skip-duplicates")
    else:
        # RedundancyCheck supersedes DuplicateCheck: five levels rather than one
        # cosine threshold, and the three deterministic ones need no model at
        # all. DuplicateCheck remains exported and unchanged for anyone relying
        # on input-only semantic detection.
        check = RedundancyCheck(
            threshold=duplicate_threshold, compare=CompareFields(compare)
        )
        try:
            with status_console.status(
                f"embedding {len(eval_set)} cases with {DEFAULT_MODEL} "
                "(first run downloads about 90 MB)"
            ):
                results.append(check.run(eval_set))
        except Exception as exc:
            # A missing model or no network should not throw away the imbalance
            # result the user already has. Report why it is missing instead.
            log.info("duplicate check could not run: %s", exc)
            notes.append(f"duplicates — could not run: {exc}")
            # But DO record it: a check that failed unexpectedly means the audit
            # is partial, and a partial audit must not report a clean gate.
            # An explicit --skip-duplicates is not recorded, because the user
            # chose that scope deliberately.
            incomplete.append(f"duplicates: {exc}")

    # A check that ran but could not run fully is treated exactly like one that
    # raised: the audit is narrower than advertised, so it must not go green.
    for result in results:
        for reason in result.partial:
            log.info("%s ran partially: %s", result.check, reason)
            incomplete.append(f"{result.check}: {reason}")

    code, explanation = _gate(results, fail_on)

    if output_format is not None:
        _emit_audit(
            eval_set=eval_set,
            results=results,
            path=path,
            output_format=output_format,
            out_path=out_path,
            detail=detail,
            schema_notes=schema_notes,
        )
    elif as_json:
        payload = to_dict(
            results, source=str(path), notes=notes, schema_notes=schema_notes
        )
        payload["gate"] = {
            "fail_on": fail_on,
            "exit_code": EXIT_INCOMPLETE if incomplete else code,
            "tripped": code == EXIT_GATE_TRIPPED,
            "incomplete": incomplete,
        }
        click.echo(jsonlib.dumps(payload, indent=2))
    else:
        render_text(
            results, source=str(path), notes=notes, schema_notes=schema_notes
        )

    # Always on stderr, in both modes: a CI log should say WHY the build failed
    # without anyone having to parse the JSON, and stdout stays clean for
    # `--json > report.json`.
    if fail_on != "never" or incomplete:
        click.echo(explanation, err=True)

    if incomplete:
        # Overrides the gate either way: "I could not finish" outranks
        # "I found nothing", and must never be reported as a pass.
        click.echo(
            "audit incomplete — a check could not run: "
            + "; ".join(incomplete),
            err=True,
        )
        sys.exit(EXIT_INCOMPLETE)
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    main()
