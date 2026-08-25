# evallint — Project Spec

## One-line
A tool that audits LLM eval datasets for the flaws that make evaluations silently lie to you.
"Everyone tests their model. Almost nobody tests whether their test set is any good."

## Why this is worth building (the honest case)
The LLM-eval space is crowded (promptfoo, DeepEval, OpenAI Evals, LangSmith). Building
another generic eval *runner* would be a worse version of tools maintained by full teams.
This tool does NOT compete with them. It solves an upstream, mostly-unaddressed problem:
the quality of the eval SET itself. This is differentiated, and it maps directly to real
rigor the author already demonstrated (confound-checking a calibration split in the
ActionLens project). The interview pitch is sharp and true.

## Who it's for
Anyone maintaining an LLM eval set: ML engineers, applied AI teams, anyone who has felt
"my eval says 95% but the model is clearly worse than that in production."

## v1 scope — exactly three checks (deliberately limited)

> **Historical.** This is the v1 design as written on 2026-08-21, kept for the
> record. The tool now has ten analyses — five that run from the file and five
> that need model runs, judges or repeated runs. See `docs/checks.md` for what
> exists, and CHANGELOG.md for how it got there. Nothing below this line
> describes the current product.
Excellence beats breadth. Three checks done flawlessly, then iterate. "I scoped it and
shipped, then expanded" is itself a senior signal.

### Check 1 — Discrimination failure (the flagship)
Detects test cases that fail to *discriminate*: cases where models of clearly different
quality all produce the same outcome. Such cases add noise, not signal, and inflate the
apparent size/reliability of an eval.
- Implementation: run each eval case against >=2 models of known differing capability
  (e.g. a strong and a weak model). Flag cases where all models score identically
  (all pass or all fail). Report the fraction of non-discriminating cases.
- This is the most technically interesting check and the most "me" (it is confound-checking).
- Honest caveat to document: requires actually running models, so it needs API access and
  costs tokens. Design it to work with a user-supplied scoring function so it is model-agnostic.

### Check 2 — Duplicate / near-duplicate detection
Detects semantically identical or near-identical cases that make an eval look bigger than
it is and over-weight one scenario.
- Implementation: embed each case's input (sentence-transformers, local, free), compute
  pairwise cosine similarity, flag pairs above a threshold. Report clusters.
- Demonstrates a real, nameable technique (embeddings + similarity) and runs locally.

### Check 3 — Class imbalance + basic stats
Detects when an eval is dominated by one category, so aggregate accuracy hides failure on
rare classes.
- Implementation: if cases have labels/categories, report distribution, imbalance ratio,
  and a warning if any class is below a threshold share. Also basic stats: n cases, input
  length distribution, missing fields.
- Fast, always-useful, easy win. Good first check to implement to get the skeleton working.

## Architecture (senior-grade, not over-engineered)
Same philosophy as before: modular, tested, reproducible, readable. NO database, no web
server, no heavy framework — it is a CLI tool + library. That restraint is the signal.

```
evallint/
  src/evallint/
    __init__.py
    io.py              # load an eval set from JSON/CSV/JSONL into a common internal format
    schema.py          # the internal EvalCase / EvalSet dataclasses + validation
    checks/
      __init__.py
      base.py          # Check abstract interface: run(eval_set) -> CheckResult
      discrimination.py
      duplicates.py
      imbalance.py
    report.py          # aggregate CheckResults into a readable report (text + optional JSON)
    cli.py             # `evallint path/to/evalset.jsonl` entrypoint
  tests/
    test_io.py
    test_duplicates.py     # e.g. two identical inputs are flagged; distinct ones are not
    test_imbalance.py      # a 90/10 split triggers the warning; balanced does not
    test_discrimination.py # with a mock scorer, a case all-models-pass is flagged
  examples/
    sample_evalset.jsonl   # a small, deliberately-flawed eval set to demo each check firing
  pyproject.toml
  README.md
```

## Key design decisions (be ready to defend each)
- **Model-agnostic via injected scoring function.** The discrimination check takes a
  user-supplied `score(case, model) -> bool/float` so the tool is not tied to any provider.
  This is the single most important design choice — it makes the tool useful to anyone.
- **Checks share a common interface** (`base.Check`) so adding a 4th check later is trivial
  and the report logic never changes. Extensibility without over-abstraction.
- **Local-first where possible.** Duplicates and imbalance run with zero API cost
  (embeddings via local sentence-transformers). Only discrimination needs model calls.
- **Honest reporting.** Every check reports not just "problems found" but its own
  limitations (e.g. duplicate threshold is a heuristic; discrimination depends on the
  chosen model pair). A tool that states its own uncertainty is more trustworthy.

## Tests (the real senior signal)
Each check gets a test proving it fires on a known-bad input and stays quiet on a known-good
one. The `examples/sample_evalset.jsonl` doubles as a live demo and a fixture.

## v1 definition of done
> **Historical, and met at v1.** The `<150 lines` bar below no longer holds:
> 26 of 28 modules exceed it and `audit.py` is 1409 lines. Recorded here as a
> standard that was superseded rather than one that was quietly dropped. What
> replaced it in practice is stated per module in the docstrings, and enforced
> by `tests/test_docs_match_reality.py`.
- `evallint examples/sample_evalset.jsonl` runs the checks that need no model and prints a clean report.
- `pytest` green.
- README with the pitch, install, usage, an example report, and an honest "what this does
  NOT do" section.
- Each module <150 lines, single-responsibility, fully explainable.

## Anti-goals (do NOT build in v1)
- Not an eval *runner* — it audits sets, it does not replace promptfoo/DeepEval.
- No web UI, no database, no cloud, no account system.
- Not all six possible checks — three, done well. *(Historical: superseded.
  Ten analyses shipped between 0.3.0 and 0.17.0, each on explicit request.)*
- No provider lock-in.

## Future (v2+, only after v1 ships and works)
- Add label-leakage and ambiguous-ground-truth checks (both interesting, both harder).
- A `--json` output mode for CI integration ("fail the build if >20% of cases don't discriminate").
- A small writeup / LinkedIn post once it is real and you can demo it.

## Interview narrative

Lead with the measurement, not the feature list. The number is what earns the
next question.

> "I ran Claude Haiku 4.5 and Opus 5 against 100 GSM8K problems, three repeats
> each, exact-match scoring with no LLM judge. On 95 of the 97 cases that scored
> reproducibly, both models gave the identical verdict — so that benchmark
> separates those two models on about 2% of its surface, and the other 98% still
> feed the headline number. It cost $1.68 to find out.
>
> That's why I built evallint: every eval tool tests the model, and almost none
> check whether the eval itself can support the conclusion. It audits the
> dataset and the measurement — ten analyses, five of which need no model at all.
>
> The design choice I'd defend hardest is what it refuses to do. There's no
> overall quality score, because combining redundancy with an unmeasured judge
> needs weights I can't justify, and a reader would act on the number instead of
> the evidence. Two tests exist to make adding one fail."

**If they push, the three follow-ups worth having ready:**

1. *"How do you know your detectors work?"* — A held-out benchmark with
   adversarial negatives. Exact duplicates 1.000 precision, semantic 0.947 at
   τ=0.85. And one detector, `label_in_input`, scores **0.000** — I documented it
   and left it, because fixing it in response to a held-out number contaminates
   the split. The honest caveat: I wrote both the fixtures and the detectors, so
   the bugs that mattered came from real datasets — TruthfulQA gave me 100 false
   positives.

2. *"How do you know your statistics are right?"* — I calibrated them. 200 trials
   of two models constructed genuinely identical; the paired test declared a
   difference 5.5% of the time against a nominal 5%.

3. *"What's the hardest bug you found?"* — My own. Redundancy detection
   allocated 1.5 GiB on 5000 cases: the levels are hash-based, but inside a
   bucket it materialised every pair, and a templated eval is one bucket.
   Replaced it with a star, which gives the identical connected component with
   n-1 edges instead of n(n-1)/2. 53.4 s and 1558 MiB became 0.25 s and 2.2 MiB,
   with a test proving cluster membership is unchanged on both sides of the
   budget.
```
