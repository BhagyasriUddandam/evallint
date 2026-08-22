# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-22

### Added
- **Fourth check: `LeakageCheck`** — flags cases whose `expected` answer appears
  verbatim in their own `input`. Those cases don't need the capability under
  test, so they inflate the score while measuring reading. Runs in the CLI
  alongside imbalance: no model, no API key, no cost.

  **The design is mostly about what it refuses to do.** Token overlap between
  an answer and its question is the obvious signal, and it was measured and
  **discarded**: on GSM8K the overlap reaches 1.00 with no leakage at all,
  because the reference answer is a worked solution restating the question. A
  planted leak also scores 1.00, so no threshold separates them. Verbatim
  containment does work — measured 0 hits on GSM8K, 0 on TruthfulQA, and 46 on
  MMLU, where the answers are single characters from a set of four and the
  hits are coincidence.

  So: answers under 12 characters are skipped, sets whose answers form a small
  closed vocabulary are treated as classification and skipped entirely, and a
  set where most cases contain their answer is reported ONCE as probable
  extractive QA rather than as hundreds of warnings. A leakage check that fires
  on every reading-comprehension set is worse than no check.

  **Zero false positives across GSM8K, MMLU, TruthfulQA and HellaSwag**, pinned
  by tests.

### Changed
- The CLI now runs three checks rather than two; discrimination is still
  reported under "Not run" because it needs your models.
- README example output regenerated, and the "three checks" claims in the
  package docstring, README, CLAUDE.md and SPEC.md corrected.

## [0.2.0] — 2026-08-22

### Added
- **`DiscriminationCheck(sample=N, sample_seed=0)`** — bound the cost of the only
  expensive check. One API call per case per model per repeat meant a 1000-case
  set was 2000 calls with no way to estimate it beforehand, so on any large set
  the real choice was auditing everything or auditing nothing.

  A sampled run cannot be mistaken for a full one: the summary leads with
  `SAMPLE of 100 of 1319 cases` before any figure it qualifies, and the caveat is
  prepended as the *first* limitation. A sampled audit that reads like a complete
  one is precisely the overclaim this project exists to report.
- `docs/findings-gsm8k.md` — the discrimination check run against a public
  benchmark rather than the author's own example set. **98% of 100 GSM8K cases
  cannot separate Haiku 4.5 from Opus 5**, scored with no LLM judge (GSM8K
  answers end in `#### n`, so grading is numeric). All 800 API responses are
  committed, so the result recomputes for $0.00.
- README: documents `sample`, `sample_seed`, `pass_threshold` and
  `max_non_discriminating_share`, which were public parameters with no public
  documentation.
- `tests/test_docs_match_reality.py` — asserts the README's test count matches
  the suite, and that every keyword-only parameter of `DiscriminationCheck` is
  mentioned in the README. Both invariants had already gone stale once; the
  count was caught only by reading the published PyPI page, which cannot be
  edited after release.

### Fixed
- The README again claimed a stale test count. Now enforced by a test rather
  than by remembering.

## [0.1.1] — 2026-08-22

Documentation only. No code changes, so 0.1.0 remains functionally identical --
but a published PyPI description can never be edited, so correcting it requires
a release.

### Fixed
- The README claimed **124 tests**; there are 269.
- The roadmap listed "Config file for thresholds" as a future v2+ item while the
  same README documented the config file as shipped, two sections earlier. It has
  been implemented since 0.1.0. Replaced with the multi-turn limitation, which is
  real and not implemented: one case is one input string, so a conversational set
  like MT-Bench cannot load at all.
- The core install was described as **~29 MB**; measured from the published wheel
  it is 33 MB.
- "Only `id` and `input` are required" was wrong: `id` is optional and generated
  as `case_1`, `case_2`... when absent. Only `input` is required.
- The source-tree listing omitted `mapping.py` and `config.py`.

## [0.1.0] — 2026-08-21

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
- **Column mapping** (`evallint.mapping`, `--map FIELD=COLUMN`). Five widely-used
  public eval datasets were tried cold and all five failed on line 1, because
  none of them names its input column `input`. Common aliases (`question`,
  `prompt`, `ctx`, `answer`, `best_answer`, `reference`, `category`, `subject`,
  …) are now inferred, and the resolved mapping is always reported.
  - Ambiguity is an error, not a guess: TruthfulQA's `category` *and* `type` are
    both plausible labels, so evallint names both and asks.
  - A canonical column name is used as-is, with passed-over alternatives
    reported. HellaSwag's `label` means the correct-ending index, not a class;
    taking it yields a meaningless "4 classes, 1.2:1" where
    `--map label=activity_label` yields the real "39 classes, 14.0:1".
  - `load_with_mapping()` returns the mapping alongside the eval set.
- Public API: `load_with_mapping`, `FieldMapping`, `resolve_mapping`,
  `MappingError`, `AmbiguousMappingError` (24 exports, up from 19).

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
- A leftover column named `label` could overwrite the value the mapping had just
  taken from another column, silently. `--map label=activity_label` appeared to
  do nothing while reporting a confident, wrong result. The displaced column is
  now moved to metadata and reported rather than dropped or allowed to win.

### Notes
- `sentence-transformers` is an optional extra (`evallint[embeddings]`) because
  it pulls in torch. `DuplicateCheck` raises `MissingEmbeddingsError` with
  install instructions if used without it, and accepts any custom
  `embedder=` callable instead.
- Not an eval runner. It audits the dataset that eval runners consume.

[0.3.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.3.0
[0.2.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.2.0
[0.1.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.1.0
