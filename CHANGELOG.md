# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-19

First release.

### Added
- Three dataset checks, each of which reports its own limitations:
  - `ImbalanceCheck` — class distribution, imbalance ratio, per-class counts,
    and basic stats. Flags a class on either a small **share** or a small
    **count**, since those are independent ways a class becomes unmeasurable.
  - `DuplicateCheck` — near-duplicate detection via embedding cosine
    similarity, reported as clusters with their pairwise similarity range.
  - `DiscriminationCheck` — flags cases that give models of differing
    capability the same verdict, split into `ceiling` / `floor` / `inverted`.
    Takes an injected `score(case, model) -> bool`, so evallint never talks to
    a model provider itself.
- `evallint PATH` CLI with `--json` for CI, plus threshold flags. The CLI runs
  the two checks that need only the file, and reports the third as *not run*
  rather than silently covering two thirds of the tool.
- Loaders for `.jsonl`, `.ndjson`, `.json` and `.csv`; unrecognised fields are
  preserved rather than dropped.
- `CheckResult` refuses to be constructed without `limitations`, so a future
  check cannot ship without declaring its blind spots.
- `py.typed` marker — the package is fully typed and ships its annotations.
- `--fail-on {never,warning,any}` so CI can gate on the audit. Exit codes are
  `0` pass, `1` gate tripped, `2` usage error, `3` audit incomplete. `3` is
  distinct on purpose: a check that could not run means a partial audit, and a
  partial audit must never report a clean pass — that holds even under
  `--fail-on never`. A deliberate `--skip-duplicates` is not "incomplete".
  With `--json` the decision is included in the payload, and the human-readable
  explanation goes to stderr so redirected JSON stays clean.
- `DiscriminationCheck(..., repeats=N)` scores each case N times and reports any
  case whose verdict is not unanimous as **unstable**, excluding it from the
  ceiling/floor/inverted counts. A verdict that changes between identical runs
  is sampling noise, not a property of the eval set. Measured on the bundled
  20-case example, 3 repeats found 8 of 20 verdicts (40%) non-reproducible and
  only 1 of the single-run inversions survived — so `repeats=1` (the default,
  for cost) cannot detect instability at all, and says so in its limitations.
  New stats: `repeats`, `n_measured`, `n_unstable`, `unstable_case_ids`.
- `DiscriminationCheck(..., max_workers=N)` runs scorer calls concurrently.
  Scoring is I/O-bound, so this is the difference between minutes and hours:
  measured 14.5x at 16 workers, taking a 1000-case × 2-model × 3-repeat audit
  from 3.3 hours to 13 minutes. Default 1 is serial and takes a code path with
  no threads at all, because the scorer is user code and evallint cannot verify
  it is thread-safe — that requirement is stated in the check's limitations.
  Results are assembled in submission order, so a concurrent run produces
  byte-identical stats to a serial one and reports the same case on failure
  regardless of thread scheduling.

- Config file support: `evallint.toml` or `[tool.evallint]` in `pyproject.toml`,
  discovered upward from the eval-set path. Precedence is CLI flag > config >
  default, decided by whether a flag was typed rather than by its value.
  Unknown keys and wrong types are hard errors — a silently ignored typo would
  mean a team believes a threshold is in force when it is not.
- `-v` / `-vv` logging to stderr. As a library, evallint installs a
  `NullHandler` and emits nothing until the host application configures
  logging; the CLI configures only the `evallint` logger, never the root.

### Changed
- The duplicate check's similarity matrix is `float32` rather than `float64`.
  Peak memory at 10,000 cases drops from 1631 MB to 816 MB (measured, exactly
  half) and the check runs ~1.7x faster, with identical clusters and identical
  similarities to four decimal places. Precision was never the constraint: the
  threshold is quoted to two decimals and reported to four.

### Fixed
- Files with a UTF-8 BOM (written by Excel, PowerShell and other Windows tools)
  are now read correctly. Previously a BOM'd CSV loaded **without error** but
  its first column was named `\ufeffid` rather than `id`, so real case ids were
  silently demoted to metadata and every case was renamed `case_1`, `case_2`...
  Findings then referred to ids that did not exist in the user's file. All
  readers now use `utf-8-sig`.

### Notes
- `sentence-transformers` is an optional extra (`evallint[embeddings]`) because
  it pulls in torch. `DuplicateCheck` raises `MissingEmbeddingsError` with
  install instructions if used without it, and accepts any custom
  `embedder=` callable instead.
- Not an eval runner. It audits the dataset that eval runners consume.

[0.1.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.1.0
