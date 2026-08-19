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
- `evallint examples/sample_evalset.jsonl` runs all three checks and prints a clean report.
- `pytest` green.
- README with the pitch, install, usage, an example report, and an honest "what this does
  NOT do" section.
- Each module <150 lines, single-responsibility, fully explainable.

## Anti-goals (do NOT build in v1)
- Not an eval *runner* — it audits sets, it does not replace promptfoo/DeepEval.
- No web UI, no database, no cloud, no account system.
- Not all six possible checks — three, done well.
- No provider lock-in.

## Future (v2+, only after v1 ships and works)
- Add label-leakage and ambiguous-ground-truth checks (both interesting, both harder).
- A `--json` output mode for CI integration ("fail the build if >20% of cases don't discriminate").
- A small writeup / LinkedIn post once it is real and you can demo it.

## Interview narrative (why this gets attention)
"I noticed every eval tool tests the model, but none check whether the eval set itself is
sound. I'd already hit this in a computer-vision project where a confound almost made my
results meaningless. So I built a tool that audits eval sets for three failure modes:
cases that don't discriminate between good and bad models, near-duplicate cases, and class
imbalance. The core design choice was making it model-agnostic through an injected scoring
function, so it works with any provider."
```
