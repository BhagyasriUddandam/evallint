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

from .checks import DuplicateCheck, ImbalanceCheck, LeakageCheck
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
    field_map: tuple[str, ...],
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
        LeakageCheck().run(eval_set),
    ]
    notes = [DISCRIMINATION_NOTE]
    # Always surface a non-identity mapping. A silently wrong column choice
    # produces a plausible report about the wrong data, so it must be visible
    # without needing -v.
    if not mapping.is_identity() or mapping.alternatives:
        notes.extend(f"field map: {line}" for line in mapping.explain())
    incomplete: list[str] = []

    if skip_duplicates:
        notes.append("duplicates — skipped with --skip-duplicates")
    else:
        check = DuplicateCheck(threshold=duplicate_threshold)
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

    code, explanation = _gate(results, fail_on)

    if as_json:
        payload = to_dict(results, source=str(path), notes=notes)
        payload["gate"] = {
            "fail_on": fail_on,
            "exit_code": EXIT_INCOMPLETE if incomplete else code,
            "tripped": code == EXIT_GATE_TRIPPED,
            "incomplete": incomplete,
        }
        click.echo(jsonlib.dumps(payload, indent=2))
    else:
        render_text(results, source=str(path), notes=notes)

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
