"""`evalcheck path/to/evalset.jsonl` — audit an eval set from the terminal.

The CLI runs the two checks that need nothing but the file: imbalance and
duplicates. Discrimination needs a scoring function that calls your models,
which cannot come from a command line without locking the tool to one
provider — so the CLI reports that it was NOT run, and how to run it, rather
than quietly leaving it out. A report that silently covers two thirds of the
tool would be exactly the kind of quiet lie this project is about.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path

import click
from rich.console import Console

from .checks import DuplicateCheck, ImbalanceCheck
from .checks.duplicates import DEFAULT_MODEL, DEFAULT_THRESHOLD
from .io import LoadError, load
from .report import render_text, to_dict

DISCRIMINATION_NOTE = (
    "discrimination — needs a scoring function that runs your models, so it "
    "cannot run from the CLI. Use the library: "
    "DiscriminationCheck(scorer, ['weak-model', 'strong-model']).run(load(path))"
)


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
@click.version_option(package_name="evalcheck")
def main(
    path: Path,
    as_json: bool,
    skip_duplicates: bool,
    duplicate_threshold: float,
    min_class_share: float,
    min_class_count: int,
) -> None:
    """Audit the eval set at PATH for flaws that make evaluations lie."""
    # Progress goes to stderr so that `evalcheck --json file > out.json` gives
    # a clean JSON file.
    status_console = Console(stderr=True)

    try:
        eval_set = load(path)
    except LoadError as exc:
        raise click.ClickException(str(exc)) from exc

    results = [
        ImbalanceCheck(
            min_class_share=min_class_share, min_class_count=min_class_count
        ).run(eval_set)
    ]
    notes = [DISCRIMINATION_NOTE]

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
            notes.append(f"duplicates — could not run: {exc}")

    if as_json:
        click.echo(
            jsonlib.dumps(
                to_dict(results, source=str(path), notes=notes), indent=2
            )
        )
    else:
        render_text(results, source=str(path), notes=notes)


if __name__ == "__main__":  # pragma: no cover
    main()
