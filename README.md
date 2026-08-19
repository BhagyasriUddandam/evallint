# evallint

**Audits LLM eval datasets for the flaws that make evaluations silently lie to you.**

Everyone tests their model. Almost nobody tests whether their test set is any good.

## The problem

A bad eval set doesn't fail loudly — it returns a number, and the number looks fine.
Suppose a 20-case support-triage eval reports 85%: if 14 of those cases are billing
questions, three of them are the same "I was charged twice" question reworded, and most
are easy enough that any competent model passes, then that 85% is mostly measuring one
scenario, counted several times, on cases that can't tell a strong model from a weak one.

The model looks fine and ships. Production disagrees. Nothing in the eval was wrong
enough to notice — which is exactly why it needs auditing separately from the model.

## What it checks (v1)

**Discrimination failure.** Runs each case against two or more models of known differing
capability and flags cases where every model produces the same verdict. Those cases carry
no evidence about which model is better, so an eval with 200 cases where 140 don't
discriminate is really a 60-case eval wearing a 200-case costume. Results split into
`ceiling` (every model passes — too easy), `floor` (every model fails — check the
reference answer and grader before blaming difficulty), and `inverted` (a *weaker* model
passed where a stronger one failed — usually a broken reference answer, and worth more
than either).

**Near-duplicate detection.** Embeds each case's input with a local sentence-transformers
model, computes pairwise cosine similarity, and groups linked cases into clusters. Matters
because duplicates make an eval look bigger and broader than it is while silently
multiplying one scenario's weight in the aggregate score — a set of 20 cases with three
phrasings of the same question is really 18 scenarios, and the repeated one counts triple.

**Class imbalance and basic stats.** Reports class distribution, imbalance ratio, and
per-class counts. Matters because aggregate accuracy hides failure on rare classes: a
model that handles the 70% majority class well and everything else badly still scores
~70% overall. A class becomes unmeasurable two independent ways — its **share** is too
small to move the aggregate, or its **count** is too small for its own accuracy to mean
anything — so both are checked. In a 10,000-case set a 1% class still has 100 cases and is
perfectly measurable; in a 20-case set a 5% class has one.

## Install

Requires Python 3.12+.

```bash
pip install evallint                # core: imbalance + stats, ~29 MB installed
pip install 'evallint[embeddings]'  # adds the duplicate check (pulls torch)
```

**Why the split.** The duplicate check needs sentence-transformers, which pulls in torch —
roughly 2 GB. Making that mandatory would mean downloading torch to get class-imbalance
statistics, so it's an optional extra instead. The core install is numpy, click and rich.

Without the extra, the duplicate check fails with instructions rather than a traceback,
and the other checks are unaffected:

```
The duplicate check needs sentence-transformers, which is an optional extra
because it pulls in torch (~2GB).

    pip install 'evallint[embeddings]'

Alternatively, pass your own embedder — DuplicateCheck(embedder=my_fn) takes any
callable mapping a sequence of strings to a 2-D array, so no extra is needed. Or
skip this check entirely: `evallint --skip-duplicates PATH`.
```

With the extra, the first run downloads `all-MiniLM-L6-v2` (**about 90 MB**); every run
after that is offline.

### From source

```bash
git clone https://github.com/BhagyasriUddandam/evallint && cd evallint
uv sync    # exact versions from uv.lock, installs the `evallint` command
```

`uv sync` reproduces the environment from `uv.lock` and installs evallint in editable
mode, so there is no separate install step. It includes the embeddings extra, so the full
test suite runs.

## Usage

```bash
evallint examples/sample_evalset.jsonl
```

| flag | default | what it's for |
|---|---|---|
| `--json` | off | Emit the full report as JSON on stdout for CI. Progress goes to stderr, so `> report.json` gives a clean file. Keeps every case ID, unlike the terminal view. |
| `--skip-duplicates` | off | Skip the embedding check. Avoids the 90 MB model download — use when you want the instant checks only. |
| `--duplicate-threshold` | `0.85` | Cosine similarity at or above which two cases count as near-duplicates. Lower catches looser paraphrases; raise it in domains with formulaic phrasing. |
| `--min-class-share` | `0.10` | Warn when a class holds less than this share of labelled cases (too small to move the aggregate number). |
| `--min-class-count` | `5` | Warn when a class has fewer than this many cases (too few for its own accuracy to mean anything). |
| `--fail-on` | `never` | Exit non-zero so CI can gate on the result: `warning` fails on any warning, `any` also fails on info. |

```bash
evallint --json evalset.jsonl > report.json      # machine-readable, for CI
evallint --skip-duplicates evalset.jsonl         # no model download
evallint --duplicate-threshold 0.92 set.jsonl    # stricter: only very close pairs
evallint --min-class-share 0.05 set.jsonl        # tolerate smaller classes
evallint --min-class-count 20 set.jsonl          # demand more cases per class
```

### Configuration file

Thresholds can live in version control instead of a CI step, so they get reviewed
like any other decision. `evallint.toml`, or a `[tool.evallint]` section in
`pyproject.toml`:

```toml
# evallint.toml
skip_duplicates  = true
min_class_count  = 10
min_class_share  = 0.15
duplicate_threshold = 0.90
fail_on          = "warning"
```

Discovery searches upward from the eval-set file, so a config at the repository root
applies to a set in a subdirectory. `evallint.toml` wins over `pyproject.toml` in the same
directory. Precedence is **CLI flag > config file > default**, and that is decided by
whether a flag was *typed*, not by its value — `--min-class-count 5` overrides a config
even though 5 is also the default.

`--config PATH` points at a specific file; `--no-config` ignores any file.

An unknown key or a wrong type is a **hard error, not a warning**:

```
Error: evallint.toml: unknown setting(s) min_class_shrae.
Valid settings: duplicate_threshold, fail_on, min_class_count, min_class_share, skip_duplicates
```

A silently ignored typo means a team believes a threshold is in force when it is not —
the same quiet wrongness this tool exists to report.

### Logging

`-v` logs progress to stderr, `-vv` adds detail. Quiet by default, and always stderr so
`--json > report.json` stays clean.

```
INFO evallint.cli: using config /repo/evallint.toml
INFO evallint.io: loaded 20 cases from evalset.jsonl (jsonl) in 0.002s
INFO evallint.cli: running imbalance on 20 cases
```

As a library, evallint attaches a `NullHandler` and emits nothing until your application
configures logging — it never configures logging on your behalf.

### Using it as a CI gate

```yaml
- run: evallint evalset.jsonl --fail-on warning
```

| exit | meaning |
|---|---|
| `0` | the gate passed |
| `1` | the gate tripped — findings at or above `--fail-on` |
| `2` | usage error (bad flag or missing file) |
| `3` | **the audit could not be completed** — unreadable file, or a check raised |

`3` is deliberately separate from `1`. *"Your eval set has warnings"* and *"I could not
finish auditing your eval set"* need different reactions in a pipeline, and collapsing
them would let an incomplete audit pass as a clean one. So a run where the embedding model
fails to download exits `3` even under `--fail-on never` — you still get the report for the
checks that did succeed, but the build does not go green on a one-third audit. An explicit
`--skip-duplicates` is not treated as incomplete, because that is you choosing the scope.

The default is `never`, which always exits `0`. That is deliberate rather than lazy: every
check reports that a finding is *a prompt to look, not a verdict*, so failing a build by
default would assert something this tool explicitly declines to claim. CI opts in.

With `--json`, the decision is in the payload too, so a consumer never has to re-derive it:

```json
"gate": {"fail_on": "warning", "exit_code": 1, "tripped": true, "incomplete": []}
```

The explanation goes to **stderr**, so `evallint --json ... > report.json` still produces a
clean file.

Accepts `.jsonl`, `.ndjson`, `.json`, and `.csv`. Only `id` and `input` are required;
`expected` and `label` are optional, and unrecognised fields are preserved rather than
dropped.

```json
{"id": "billing_001", "input": "I was charged twice.", "expected": "Refund the extra.", "label": "billing"}
```

Discrimination needs model verdicts, so it runs from the library rather than the CLI:

```python
from evallint.io import load
from evallint.checks import DiscriminationCheck

def score(case, model) -> bool:
    """Your code, your provider. True if `model` got `case` right."""
    return grade(my_client.complete(model=model, prompt=case.input), case.expected)

# models ordered WEAKEST FIRST — that ordering is what makes inversion detectable
result = DiscriminationCheck(score, ["small-model", "large-model"]).run(load("evalset.jsonl"))
print(result.summary, result.stats["non_discriminating_share"])
```

The scorer is injected so evallint never talks to a provider: it works with any stack,
you define what "correct" means, and the test suite needs no API key.

**Repeat your runs.** Model sampling makes per-case verdicts unstable, and `temperature=0`
is not universally available — current frontier models including Claude Opus 5 and Sonnet 5
reject the parameter outright. Pass `repeats=N` to score each case N times; any case whose
verdict is not unanimous is reported as *unstable* and excluded from the counts:

```python
result = DiscriminationCheck(score, ["small-model", "large-model"], repeats=3).run(eval_set)
result.stats["n_unstable"]          # cases whose verdict changed between runs
result.stats["unstable_case_ids"]   # ...and which ones
result.stats["n_measured"]          # the denominator the other figures actually use
```

On the bundled 20-case example, three repeats found **8 of 20 verdicts (40%)
non-reproducible**, and only one of the inversions a single run reported survived all
three. The default `repeats=1` keeps cost down but cannot detect this at all — which is
stated in the check's own limitations.

**Run it concurrently.** Scoring is I/O-bound — one API call per case per model per repeat
— so serial execution is the difference between minutes and hours. Measured on a 40 ms
stand-in call:

| `max_workers` | 300 calls | 1000 cases × 2 models × 3 repeats @ 2s |
|---|---|---|
| `1` (default) | 12.99 s | **3.3 hours** |
| `8` | 1.65 s (7.3×) | 25 min |
| `16` | 0.83 s (14.5×) | **13 min** |

```python
DiscriminationCheck(score, models, repeats=3, max_workers=16).run(eval_set)
```

> ⚠️ **Your scorer must be thread-safe** to raise this above 1. evallint cannot check that
> for you — the scorer is your code, and a shared client, session, file handle or
> non-locking cache inside it will corrupt results *quietly* rather than crash. The default
> is `1`, and it takes a code path with no threads in it at all, so you are unaffected
> until you opt in. If a concurrent run disagrees with a serial run on the same data,
> suspect the scorer before the eval set.

## Example output

Real output from `evallint examples/sample_evalset.jsonl` against the bundled
[deliberately flawed example set](examples/README.md), at an 88-column terminal:

```
evallint  examples/sample_evalset.jsonl

imbalance
  20 cases across 4 classes, imbalance ratio 14.0:1
  WARN  class imbalance ratio is 14.0:1 — 'billing' has 14 cases, 'bug_report' has 1.
        Aggregate accuracy is dominated by 'billing'
  WARN  class 'password_reset' is under-represented: only 4 cases, so its accuracy can
        only land on 5 distinct values
        password_001, password_002, password_003, password_004
  WARN  class 'refund' is under-represented: 5.0% of labelled cases (below 10%); only 1
        case, so its accuracy can only land on 2 distinct values
        refund_001
  WARN  class 'bug_report' is under-represented: 5.0% of labelled cases (below 10%);
        only 1 case, so its accuracy can only land on 2 distinct values
        bug_001

duplicates
  20 cases, 3 near-duplicate clusters covering 7 cases — about 16 distinct scenarios
  (20% redundant)
  WARN  3 near-identical cases (pairwise cosine 0.81-0.91)
        billing_001, billing_002, billing_003
  WARN  2 cases have identical input text
        billing_004, billing_014
  WARN  2 near-identical cases (pairwise cosine 0.98)
        password_001, password_002

7 warnings across 2 checks

Not run
  · discrimination — needs a scoring function that runs your models, so it cannot run
    from the CLI. Use the library: DiscriminationCheck(scorer, ['weak-model',
    'strong-model']).run(load(path))

What this audit cannot tell you
  imbalance
    · This check measures whether the class distribution is UNEVEN, not whether it is
      WRONG. If production traffic really is 70% billing, a 70% billing eval may be
      exactly right. Only you know the target distribution.
    · Classes are read from the 'label' field only. If your categories live in metadata
      or are implied by the text, this check sees nothing.
    · The thresholds are conventions, not statistics. They are a prompt to look, not a
      verdict.
    · Input length is measured in characters, not tokens, so it is a rough proxy for
      cost and complexity.
  duplicates
    · Similarity is computed on the 'input' field alone. Two cases with the same input
      but different 'expected' values may be deliberate (testing consistency, or a known
      ambiguity) — read a cluster before deleting from it.
    · The threshold is a heuristic, not a decision boundary. It was checked against one
      small example set where duplicates and non-duplicates separate cleanly; a domain
      with formulaic phrasing (SQL, legal boilerplate, templated prompts) will push
      unrelated cases above it.
    · Clusters are connected components: if A is similar to B and B to C, all three are
      grouped even when A and C are not themselves similar. The pairwise range reported
      for each cluster shows when this has happened.
    · Inputs longer than the model's token limit (256 word pieces for all-MiniLM-L6-v2)
      are truncated before embedding, so two long cases that differ only in their
      endings can look identical to this check.
    · Every pair is compared, so time and memory grow with the square of the case count.
      Fine for a few thousand cases, not for a corpus.

```

Two things in that output are deliberate. **Discrimination is listed under "Not run"**
rather than omitted — the CLI covers two of three checks and says so, because a report
that quietly covered two thirds of the tool would be the same kind of silent lie this
project exists to catch. And **"What this audit cannot tell you" is not suppressible**;
there is no `--brief` flag that hides it. A clean report is exactly when a reader is most
likely to over-trust it.

## What this does NOT do

- **It is not an eval runner.** It does not execute your eval, call your model, or score
  responses. It audits the dataset those tools consume.
- **It does not replace promptfoo, DeepEval, LangSmith, or OpenAI Evals**, and is not
  trying to. Those run evals; they mostly assume the eval set is sound. evallint checks
  that assumption. Use it *alongside* them, upstream.
- **It audits the set, not the model.** Nothing here tells you whether your model is good.
  A clean evallint report means your eval is capable of measuring something, not that the
  thing it measured is passing.
- **It does not tell you what to delete.** Every finding is a prompt to look, not a
  verdict. Easy cases may be deliberate regression guards; duplicate inputs with different
  `expected` values may be testing consistency on purpose.
- **It cannot tell you whether your distribution is *wrong*, only whether it is *uneven*.**
  If production traffic really is 70% billing, a 70% billing eval may be exactly right.
- **The thresholds are conventions, not statistics.** `0.85` cosine, `10%` share, `5`
  cases — all defaults, all configurable, none derived from theory. The duplicate
  threshold was checked against one 20-case example where duplicates and non-duplicates
  separate at 0.70 vs 0.81.
- **Duplicate detection is O(n²)** in time and memory. Fine for a few thousand cases, not
  for a corpus.
- **Discrimination depends entirely on the model pair you choose.** Two models closer in
  capability than you assumed will make a good eval look non-discriminating, and the check
  cannot tell the difference. It also cannot verify that your "weak" model is actually
  weaker — it can only flag when the numbers contradict the ordering you declared.
- **Only three checks.** Label leakage and ambiguous ground truth are real problems and
  are not implemented.

This section is a feature. Every check also reports its own limitations at runtime, and
`CheckResult` raises if constructed without them — "state what you cannot tell the user"
is enforced by the type, not left to discipline. A tool that states its uncertainty is
more trustworthy than one that doesn't.

## Roadmap (v2+)

- **Label-leakage check** — detect cases where the expected answer is recoverable from the
  input itself, which inflates scores without measuring capability.
- **Ambiguous-ground-truth check** — flag cases where the reference answer is one of
  several defensible responses, so a correct model gets marked wrong.
- **Config file for thresholds.** Five flags is already near the limit of what belongs on
  a command line. As the check count grows, per-check thresholds should move to an
  `evallint.toml` with CLI flags as overrides, rather than adding a flag per check.

Deliberately not planned: a web UI, a database, a hosted service, or becoming an eval
runner.

## Development

```bash
uv run pytest    # 124 tests
```

Every check is tested both ways: it must **fire on known-bad input** and **stay quiet on
known-good input**. A check that warns about everything is worth as little as one that
warns about nothing. Tests avoid re-testing other people's libraries — the duplicate
check's clustering logic runs against a deterministic fake embedder, with two separate
tests exercising the real model end to end.

```
src/evallint/
  schema.py       EvalCase / EvalSet + validation
  io.py           load JSONL / JSON / CSV
  checks/
    base.py       Check interface; CheckResult refuses to exist without limitations
    discrimination.py · duplicates.py · imbalance.py
  report.py       CheckResults -> text (rich) or JSON
  cli.py          `evallint PATH`
```
