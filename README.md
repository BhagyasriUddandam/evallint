# evallint

**evallint audits the reliability of LLM evaluations.**

Everyone tests their model. Almost nobody tests whether the *measurement* can
support the conclusion they draw from it.

## The problem

An unreliable evaluation does not fail loudly — it returns a number, and the
number looks fine.

Suppose a 20-case support-triage eval reports 85%. If 14 of those cases carry one
label, three are the same question reworded, and most produce the same verdict
from every model you compared, then that 85% rests on fewer independent
observations than the case count suggests. Whether that is a problem depends on
what you wanted to learn — but you cannot tell from the 85%.

evallint measures the properties of the evaluation that determine what its number
can and cannot support. It does not tell you the eval is bad. It tells you what
was observed, what was inferred and on what threshold, and what was never looked
at.

## Three different things

Conflating these is the most common way to misread this tool.

| | What it is | Needs | evallint |
|---|---|---|---|
| **Dataset quality** | Properties of the eval *file*: potential redundancy, class distribution, potential leakage, ground-truth ambiguity. | The file alone | **measures it** |
| **Evaluation quality** | Properties of the *measurement procedure*: can it separate the models you compared, is the grader reproducible, does the number come back the same twice, is the comparison powered. | Model runs, judges, or repeated runs | **measures it** |
| **Model quality** | How good the model is. | An evaluation you already trust | **does not measure it** |

The dependency runs one way. Dataset findings bound what evaluation quality can
be; evaluation quality bounds what you can conclude about the model. **Nothing
here tells you a model is good, and nothing here tells you an eval is good.**

## What it measures

Nine analyses. Four run from the file with no model, no network and no API key;
five need something the file cannot supply, and report **NOT ASSESSED** when it
is absent — which is not the same as passing.

### Dataset quality — runs from the file

| Analysis | Reports | Tier |
|---|---|---|
| **Potential redundancy** | near-copy clusters at five levels, each cluster's share of the aggregate | observed / heuristic |
| **Class distribution** | largest-to-smallest ratio, classes too small for a stable per-class accuracy | observed |
| **Potential leakage** | seven deterministic detectors for an answer reachable from its own prompt | heuristic |
| **Ground-truth ambiguity** | seven detectors for references that do not pin down one answer | heuristic |

### Evaluation quality — needs model runs, judges, or repeated runs

| Analysis | Reports | Tier |
|---|---|---|
| **Model separation** | per case: `ceiling`, `floor`, `separating`, `non_monotonic`, `indeterminate` | estimated |
| **Evaluator reliability** | is the *grader* reproducible — kappa, alpha, position bias, self-consistency | estimated |
| **Statistical comparison** | paired tests, effect size, and what difference this eval could have detected | estimated |
| **Redundancy-adjusted coverage** | distinct scenarios as a bracket, and how much to widen your intervals | heuristic |
| **Reproducibility** | three variance sources, reported separately and never summed | estimated |

**Every analysis is documented under eight headings** — what it measures, why it
matters, methodology, assumptions, limitations, example, false positives, false
negatives — in **[docs/checks.md](docs/checks.md)**. Measured false-positive and
false-negative rates are given there with their source.

## Precise language, on purpose

The output says `potential redundancy`, `potential leakage`, `ground-truth
ambiguity`, `provides limited evidence for model separation`, `not identifiable`,
`unresolved`. That is not hedging — each phrase is narrower than the verdict it
replaces, and the narrower claim is the one the method supports.

Two consequences worth stating plainly:

**A non-separating case is not a bad case.** A case where every model gives the
same verdict contributes nothing to *which model is better*. It may still be a
deliberate regression guard, a coverage case, or a floor case whose reference
answer is wrong. The finding says "provides limited evidence for model
separation", and for ceiling cases it is INFO rather than WARNING.

**No check declares a case invalid.** Every finding carries a recommended action,
and none of them is "delete".

## Evidence tiers

Every finding in the unified report carries one, and the rule is enforced in
code rather than documented:

| Tier | What it is | Confidence |
|---|---|---|
| `observed` | a census of the data — counting, not inference | **must be absent**: a count is exact |
| `heuristic` | a chosen threshold with no sampling theory behind it | **required**: else it reads as fact |
| `estimated` | a statistical estimate with an interval and stated assumptions | the interval |

Constructing an `observed` finding with a confidence raises `ValueError`, and so
does a `heuristic` one without. There is **no overall quality score** in any
output format, and two tests exist to make adding one fail.


## Install

Requires Python 3.12+.

```bash
pip install evallint                # core: imbalance + stats, ~33 MB installed
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
| `--map FIELD=COLUMN` | inferred | Point one of evallint's fields at a column in your file. Repeatable. |

```bash
evallint --json evalset.jsonl > report.json      # machine-readable, for CI
evallint --skip-duplicates evalset.jsonl         # no model download
evallint --duplicate-threshold 0.92 set.jsonl    # stricter: only very close pairs
evallint --min-class-share 0.05 set.jsonl        # tolerate smaller classes
evallint --min-class-count 20 set.jsonl          # demand more cases per class
evallint --map input=question set.jsonl          # your column names, not ours
```

### Your column names

evallint's fields are `id`, `input`, `expected`, `label` — and almost no real
eval set uses those names. Five widely-used public datasets were tried cold and
**all five failed on line 1**, because they call the input `question`, `prompt`,
`ctx` or `turns`. Common aliases are now inferred, so most files load unaided:

```bash
$ evallint gsm8k.jsonl --skip-duplicates
  · field map: input <- 'question' (inferred)
  · field map: expected <- 'answer' (inferred)
```

The mapping is always printed. That is the point: a loader that quietly picks
the wrong column produces a confident report about the wrong data, which is the
exact failure this tool exists to catch. So the rules are deliberately timid:

- A column already named `input` / `expected` / `label` is used as-is.
- Otherwise aliases are considered, and **exactly one** candidate is accepted.
- Two candidates is an error naming both, not a coin flip. TruthfulQA has
  `category` *and* `type`; evallint refuses and tells you to pick:

  ```
  Error: cannot tell which column is 'label': category, type are all plausible.
  Choose one explicitly, e.g. --map label=category
  ```

- Alternatives that were passed over are reported too. HellaSwag has a column
  literally named `label`, but there it means *the index of the correct ending*,
  not a class. evallint cannot know that, so it uses the canonical name and
  says what else was available:

  ```
  · field map: label <- 'label'  [WARNING: 'label' used as-is, but
    activity_label also present — check the meaning]
  ```

  That warning is load-bearing. Taking `label` gives "4 classes, ratio 1.2:1",
  which is meaningless — it's counting answer positions. `--map
  label=activity_label` gives **39 classes, ratio 14.0:1**, which is the real
  finding. Same file, opposite conclusion, and the only thing standing between
  them is that evallint said which column it used.

Explicit `--map` always wins, and a mapped field is never overwritten by a
leftover column of the same name (the displaced column is kept in metadata and
reported, not dropped).

Multi-turn sets used to be a genuine limitation rather than a naming one:
MT-Bench's prompt is a *list* of turns, and there was no `--map` for that — it
exited 3 and said so. Schema 2 fixes it; `conversation`, `turns`, `dialogue`,
`chat` and `chat_history` are now aliases for `messages`, and MT-Bench loads
and audits. See [Chat, rubrics and tool calls](#chat-rubrics-and-tool-calls).

### Does evallint actually work? Its own benchmark

```bash
python -m benchmarks.runner --split test              # reportable numbers
python -m benchmarks.runner --split test --semantic    # + the embedding level
```

A tool that reports flaws in other people's data had better be measured itself.
`benchmarks/` holds labelled fixtures for all ten problem classes, with
**adversarial negatives** in every one — a fixture of obvious positives measures
recall and nothing else.

| Detector | P | R | F1 | FPR | ms/item |
|---|---|---|---|---|---|
| exact duplicates | 1.000 | 1.000 | 1.000 | 0.000 | 0.008 |
| semantic duplicates *(tau=0.85)* | 0.947 | 1.000 | 0.973 | 0.004 | 0.512 |
| leakage | 0.667 | 1.000 | 0.800 | 0.250 | 0.019 |
| ambiguous references | 1.000 | 0.667 | 0.800 | 0.000 | 0.008 |
| class imbalance | 1.000 | 0.667 | — | 0.000 | — |

Ceiling / floor / separating / non-monotonic classification: **100% exact
agreement**, 0 disagreements. And the number that reassures me most, because its
target is arithmetic rather than opinion — over 200 trials of two **genuinely
equal** models, the paired test called a real difference **5.5% of the time**
against a nominal alpha of 5%.

**Held out means held out.** DEV is for tuning, TEST carries the reported
numbers, and the splits differ in seed *and* surface content. A test asserts no
fixture shares a positive case between them — added after catching the ambiguity
fixture sharing all six of its positives, where honest TEST recall turned out
*lower* than DEV's.

Three gaps are recorded as **known and deliberately unfixed**, because fixing
them in response to a TEST number would contaminate the split:
`label_in_input` scores precision **0.000** and is the sole reason leakage
precision is 0.667; class imbalance misses a single-class dataset; ambiguity
misses two keyword-list cases.

The honest limitation, stated first in the benchmark's own README: **the fixtures
and the detectors have the same author.** The synthetic suite has never caught a
bug that real datasets did not catch first — TruthfulQA (100 false positives),
MMLU (3), HellaSwag (101) each found one. Committed baselines in
`benchmarks/results/` make regressions fail CI.

Full methodology and limitations: **[benchmarks/README.md](benchmarks/README.md)**.

### The unified audit report

```bash
evallint evalset.jsonl --format terminal          # concise, eleven sections
evallint evalset.jsonl --format json --out r.json # everything, nothing truncated
evallint evalset.jsonl --format html --out r.html # expandable, case-level detail
```

**There is no overall score, in any format.** An "Eval Trust Score: 78/100"
would need a weight for redundancy against a weight for an unmeasured judge, and
there is no defensible source for either number. Two tests exist purely to make
adding one fail: no key anywhere in the JSON may read as a grade, and
`AuditReport` may not gain a `score` attribute.

What replaces it is the thing a score would have compressed:

```json
"evidence_coverage": {
  "analyses_available": 7,
  "analyses_with_evidence": ["dataset_statistics", "ground_truth"],
  "analyses_without_evidence": {
    "discrimination": "needs a scoring function that runs your models ...",
    "evaluator_reliability": "needs repeated judge verdicts on the same cases ...",
    "reproducibility": "needs the SAME eval run more than once ..."
  }
}
```

Four of the seven analyses run from the file alone. The other three need model
runs, judges or repeated runs, so from a CLI they are always **NOT ASSESSED** —
reported in every format and never truncated, because "nothing was found" and
"nobody looked" are opposite states and collapsing them is the exact defect this
tool reports in eval sets. The executive summary says a conclusion is
"unsupported in those respects", not "supported with caveats".

Every finding carries seven fields — severity, evidence tier, confidence,
evidence, affected cases, explanation, limitation, recommended action — and all
seven are required at construction. The `limitation` is the **source analyser's
own**, not one written at the reporting layer, which could contradict it.

The tier is what carries the weight, and it is enforced rather than documented:

| Tier | What it is | Confidence |
|---|---|---|
| `observed` | A census. Counting, not inference. | **must be absent** — a count is exact |
| `heuristic` | A chosen threshold, no sampling theory. | **required** — else it reads as fact |
| `estimated` | An estimate with an interval and assumptions. | the interval |

Constructing an `observed` finding with a confidence raises, and so does a
`heuristic` one without. Recommendations sort **weakest tier first**, so nobody
acts on a cosine threshold thinking it was a count.

The part that does not exist elsewhere: **findings derived from the absence of
evidence.** A judge metric that could not be computed becomes a finding, because
an absent reliability measure is not a passing one. So does an unidentifiable
variance source — unmeasured is not zero.

```
[observed] evaluator_stochasticity is not identifiable: 0 judge(s) recorded
    explanation: This design cannot separate that source of variance, so its
                 contribution is unknown rather than zero.
```

No threshold is invented at this layer. There is no "agreement too low" finding,
because setting that bar would be the arbitrary cutoff the report exists to
avoid; model comparison uses the analyser's own `underpowered` /
`no_information` verdicts, since a detected difference is a result, not a defect.

`--format terminal` truncates case ids and omits raw statistics; `--detail`
adds each finding's explanation, limitation and recommended action. What is
never truncated in any format is the NOT ASSESSED list.

HTML is a single self-contained file — `<details>` for expandable sections, no
CDN, no framework, **no JavaScript**. An audit artefact should still open in five
years, and case ids are HTML-escaped because they come from your file.

Full detail, including the ten-question gate this feature was held to:
**[docs/audit-report.md](docs/audit-report.md)**.

### Chat, rubrics and tool calls

The four-field format is still the whole schema. A file written for evallint 0.1
loads unchanged, and always will. Schema 2 adds seven optional things on top:
**multi-turn `messages`**, **`system` prompts**, **`acceptable` alternative
answers**, **structured `expected` + `expected_schema`**, **`rubric` criteria**,
**`tools`** and **`expected_tool_calls`**.

```json
{"id": "refund-1",
 "system": "You are a support agent. Never promise a refund.",
 "messages": [
   {"role": "user", "content": "My order never arrived."},
   {"role": "assistant", "tool_calls": [{"name": "lookup", "arguments": {"id": "o1"}}]},
   {"role": "tool", "content": "status=lost", "name": "lookup"},
   {"role": "user", "content": "Can I get my money back?"}],
 "expected": "escalate",
 "acceptable": ["offer_replacement"],
 "rubric": ["refuses_politely", "offers_alternative"]}
```

**The design decision that makes this backward compatible:** `input` stays a
required string, and a chat case *derives* it by flattening the conversation.
Every existing check therefore works on chat data with **no edits to any check** —
and a one-turn conversation flattens to byte-identical text, so converting a file
cannot change a single finding. `case.input_is_derived` records that the text was
synthesised, because a report must never quote derived text back as yours.

The `system` prompt is deliberately **excluded** from the flattened input. It is
usually identical across every case, so folding it in would give every case a
long shared prefix — the redundancy check would then report near-duplicates
throughout and the effective-size estimate would collapse, both as artefacts of
the flattening rather than facts about your data.

Declaring a version is optional and does not unlock features — rich fields are
read whenever they are well-formed. It changes *strictness*:

```jsonc
{"evallint_schema": 2}                          // .jsonl: an optional first line
{"evallint_schema": 2, "cases": [ ... ]}        // .json:  an envelope
```

Undeclared, a `messages` value that is not a conversation is kept as metadata
with a note — because real datasets do have a `messages` column meaning
something else, and those files must keep loading. Declared, it is an error. A
future version (`3`) is refused by name rather than misread.

Errors are collected rather than raised one at a time, with the line and the
JSON path inside the record:

```
errs.jsonl: 4 schema problems in 4 cases

  errs.jsonl line 2.system: must be a string, got int
      a system prompt is one block of text; put per-case settings in metadata instead
  errs.jsonl line 5.messages[0].role: 'nobody' is not a valid role; allowed: system, user, assistant, tool
```

Migration is optional, and mostly worth it to make an inferred column mapping
permanent:

```console
$ evallint gsm8k.jsonl --migrate-to gsm8k_v2.jsonl
gsm8k.jsonl -> gsm8k_v2.jsonl
  2 case(s), schema version 2 declared
  columns renamed to canonical fields:
    question -> input
    answer -> expected
    (so --map is no longer needed for this file)
  verified: the written file reloads to identical cases
```

Nothing is written unless the output reloads to identical cases, field by field.
It will not overwrite without `--overwrite`, will not write onto its own source,
and will not write CSV. It also does **not** rewrite simple cases as one-turn
conversations — that is a decision about your eval, not a mechanical one.

Full reference, including the flattening rules and a worked hand-migration:
**[docs/schema-v2.md](docs/schema-v2.md)**.

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
| `3` | **the audit could not be completed** — unreadable file, a check raised, or a check ran only partially |

A check that RAN but could not run fully also exits `3`. On a core install the
redundancy check completes its three deterministic levels and reports that the two
semantic ones were unavailable — a narrower audit than advertised, so it does not go
green. `--skip-duplicates` exits `0`, because that is you choosing the scope.

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

Accepts `.jsonl`, `.ndjson`, `.json`, and `.csv`. Only `input` is required: `id` is
generated as `case_1`, `case_2`... when absent, `expected` and `label` are optional, and
unrecognised fields are preserved rather than dropped.

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

# models ordered WEAKEST FIRST — the declared order is what makes a
# non-monotonic case detectable at all
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
non-reproducible**, and only one of the non-monotonic cases a single run reported survived all
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

**Bound the cost with `sample`.** This is the only expensive check, so on a large set the
real choice is often between auditing a sample and not auditing at all:

```python
DiscriminationCheck(score, models, sample=100, sample_seed=0).run(eval_set)
```

`sample_seed` makes the draw reproducible, so a re-run compares against the same cases
instead of a fresh subset. A sampled run says so everywhere it could be mistaken for a
full one — the summary leads with it, and it is the *first* limitation reported:

```
SAMPLE of 100 of 1319 cases x 2 models: 4 discriminate, 96 do not (96%)

  · Only 100 of 1319 cases were scored (random sample, seed 0). Every figure
    above describes THAT SAMPLE, not your eval set. A clean sample does not mean
    a clean set, and a broken case that was not drawn is not reported here —
    absence of a finding is not evidence.
```

**The remaining two knobs.** `pass_threshold` (default `0.5`) is the score at or above
which a float scorer counts as a pass; it is ignored for bool scorers.
`max_non_discriminating_share` (default `0.5`) is the share of measured cases above which
the check warns.

### What the separation analysis reports

Every case is classified into one of five outcomes, and **none of them is a
defect label**:

| class | pattern | what it means |
|---|---|---|
| `ceiling` | all models pass | provides limited evidence for model separation |
| `floor` | all models fail | limited evidence — and more often a wrong reference than a hard case |
| `separating` | follows the declared order | carries evidence about which model is better |
| `non_monotonic` | a weaker model passed where a stronger failed | evidence about the reference or the ordering |
| `indeterminate` | repeats split evenly | this run cannot assign a verdict |

A ceiling case is not a bad case — it may be a deliberate regression guard, and
an eval with none cannot detect a regression. The wording throughout is
"provides limited evidence for model separation", which is a claim about
evidence rather than about quality.

**Which pairs are actually separated**, not merely how many cases separate
something. An eval can resolve the ends of a capability ladder while saying
nothing about the middle, and that changes what you may conclude from it:

```python
for pair in result.stats["model_pairs"]:
    print(pair["pair"], pair["verdict"], pair["mcnemar_p"])

['small', 'medium']  unresolved   0.25
['small', 'large']   separated    0.00391
['medium', 'large']  unresolved   0.25
```

The comparison is **paired** — both models saw the same cases — so it uses an
exact McNemar test rather than two independent-proportion tests. `unresolved`
and `no_information` are distinct: the second means the two models never once
disagreed, which is not evidence that they are equally capable.

**A minimum detectable effect**, which states what the eval *cannot* do:

```
'haiku' vs 'opus' is UNRESOLVED: observed difference +0.0%, McNemar p=1.
With 2 discordant of 100 cases this eval could only have detected a
difference of about 4.0% or larger, so it cannot support a claim either way
about this pair
```

Verdicts across repeats use a **majority**, not unanimity. Requiring unanimity
and discarding the rest biased the headline number upward as `repeats` rose,
because unanimity is likelier when a model's success probability is extreme —
measured 0.52 → 0.89 going from 1 to 5 repeats on unchanged data. Nothing is
discarded now, so the denominator is fixed.

Full derivations, every threshold with its justification, and the failure modes:
[docs/methodology-discrimination.md](docs/methodology-discrimination.md).

### Does this check actually find anything?

Yes, on a benchmark this project did not write. Run against **100 GSM8K cases** with
Haiku 4.5 and Opus 5, three repeats, and **no LLM judge** — GSM8K answers end in `#### n`,
so grading is a numeric comparison nobody has to trust:

```
100 cases x 2 models x 3 repeats: 2 discriminate, 95 do not (98%), 3 not reproducible
```

98% of a widely-cited reasoning benchmark carries no evidence about which of those two
models is better. Both sit at the ceiling: 98.0% and 98.7%. Repeats also killed one of
the two non-monotonic cases a single run reported, which is exactly why the check refuses to
classify unstable cases.

Full write-up, including what it does **not** show: [docs/findings-gsm8k.md](docs/findings-gsm8k.md).
All 800 API responses are committed, so the numbers recompute for $0.00.

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

leakage
  20 answered cases, none contain their own answer

redundancy
  20 cases, 3 redundant clusters covering 7 cases — about 16 distinct scenarios (1
  duplicate, 2 similar)
  WARN  cluster 1: 3 cases are the same scenario worded differently, with the same
        expected answer [SIMILAR]. Mean pairwise similarity 0.95. Range 0.92-0.97. They
        receive 15% of the aggregate weight where one scenario would receive 5% —
        benchmark weighting risk HIGH. This may be intentional; consistency tests and
        template families are legitimate
        billing_001, billing_002, billing_003
  WARN  cluster 2: 2 cases are identical [DUPLICATE]. Mean pairwise similarity 1.00.
        They receive 10% of the aggregate weight where one scenario would receive 5% —
        benchmark weighting risk HIGH. This may be intentional; consistency tests and
        template families are legitimate
        billing_004, billing_014
  WARN  cluster 3: 2 cases are the same scenario worded differently, with the same
        expected answer [SIMILAR]. Mean pairwise similarity 0.98. They receive 10% of
        the aggregate weight where one scenario would receive 5% — benchmark weighting
        risk HIGH. This may be intentional; consistency tests and template families are
        legitimate
        password_001, password_002
  INFO  similar pairs by threshold — 0.80: 5, 0.85: 5, 0.90: 5, 0.95: 4. That is stable,
        so the threshold is not load-bearing for this data

7 warnings across 3 checks, plus 1 for information

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
  leakage
    · This check CANNOT distinguish a leak from a task where the answer is supposed to
      appear in the input. Extractive QA, span selection and reading comprehension all
      legitimately contain their answers. When most of a set looks like that, this check
      says so once and stops flagging individual cases — read that note before treating
      any finding as a bug.
    · Containment is literal text matching after whitespace and case normalisation. A
      leak that paraphrases the answer, or encodes it as a synonym, a number in a
      different format, or a hint, is invisible here. Absence of a finding is not
      evidence of absence.
    · Token-overlap scoring was deliberately NOT used. On GSM8K the overlap between a
      reference answer and its question reaches 1.00 with no leakage, because the answer
      is a worked solution that restates the question. No threshold separates that from
      a real leak, so no such finding is emitted — this check is quieter than it could
      be, on purpose.
    · Answers shorter than 12 characters are skipped, and sets whose answers come from a
      small closed vocabulary are treated as classification and skipped entirely. A
      genuine leak of a short label will therefore be missed.
    · Only the 'input' and 'expected' fields are read. An answer leaked through
      metadata, a filename, or an id is not seen.
  redundancy
    · SIMILAR IS NOT INVALID. Near-identical cases are often deliberate: consistency
      tests reuse an input on purpose, template families cover an operation across many
      operands, and regression suites repeat a scenario that once broke. This check
      reports how much weight one scenario carries in an averaged score. It does not
      tell you to delete anything.
    · The exact, normalized and template levels are deterministic and cannot be changed
      by tuning. The scenario and semantic levels depend on an embedding model AND a
      threshold, both of which are choices — treat them as evidence to look at, not as
      measurements.
    · Template detection masks numbers and quoted spans only. It does NOT recognise
      named entities, so 'flight from Paris to Rome' and 'flight from Berlin to Madrid'
      are not detected as one template. Nor does it know whether a template family is
      deliberate coverage or accidental bulk.
    · Similarity is computed on whichever fields you selected. With the default
      input+expected, two cases with the same question and DIFFERENT answers are not
      grouped — which is correct, because that is a ground-truth contradiction rather
      than redundancy, but it means this check will not report it. Compare on 'input'
      alone if you want those grouped.
    · Clusters are connected components: if A is redundant with B and B with C, all
      three are grouped even when A and C are not themselves close. The minimum pairwise
      similarity for each cluster is reported so this is visible.
    · Weight-share risk bands (10% high, 5% medium) are conventions with no statistical
      derivation. The raw share is reported so you can apply your own. The principled
      anchor is your eval's own resolution — a cluster that can move the score by more
      than the smallest difference the eval can detect is the one that matters — and
      this check does not compute that.
    · Inputs longer than the embedding model's token limit are truncated before
      encoding, so two long cases differing only in their endings can look identical to
      the semantic levels.
    · Every pair is compared, so semantic analysis costs time and memory proportional to
      the square of the case count.
```

Two things in that output are deliberate. **Discrimination is listed under "Not run"**
rather than omitted — the CLI covers three of four checks and says so, because a report
that quietly covered two thirds of the tool would be the same kind of unstated gap this
project exists to catch. And **"What this audit cannot tell you" is not suppressible**;
there is no `--brief` flag that hides it. A clean report is exactly when a reader is most
likely to over-trust it.

## What evallint cannot tell you

Grouped by *why*, because the reasons differ in kind. The short version:
[docs/checks.md](docs/checks.md#part-3--what-evallint-cannot-tell-you) has the
full list.

### Because it is out of scope by design

- **Whether your model is any good.** evallint audits the measurement, not the
  thing measured. A clean report means the evaluation is capable of measuring
  something, not that what it measured is passing.
- **Whether your eval is "good" or "bad".** No check emits that verdict and no
  report contains an overall score.
- **Whether to delete a case.** Every finding is a prompt to look.
- **It is not an eval runner.** It does not execute your eval, call your model or
  score responses. It does not replace promptfoo, DeepEval, LangSmith or OpenAI
  Evals — those run evals and mostly assume the evaluation is sound. evallint
  checks that assumption, upstream.

### Because it requires knowing your intent

- **Whether your class distribution is wrong**, only whether it is uneven. If
  production traffic really is 70% billing, a 70%-billing eval may be right.
- **Whether redundancy is deliberate.** Consistency tests reuse inputs on
  purpose; regression suites repeat scenarios that once broke.
- **Whether a ceiling case should be removed.** It may be a regression guard.
- **Whether the answer appearing in the prompt is a leak.** For extractive QA,
  span selection and reading comprehension, it is the task.

### Because the information is not in the data

- **Whether your reference answers are correct** — only whether they are
  *unambiguous*, which is a different property. A confidently wrong reference
  passes every check here.
- **Whether your judge is right** — only whether it is *consistent*. Judges from
  one model family share their blind spots, so correlated error is
  indistinguishable from reliability, and only a human-labelled sample separates
  them.
- **Whether your eval covers what matters.** Nothing here knows what you failed
  to test.
- **Whether a non-separating case is easy or your models are too similar.** The
  observation is identical either way.

### Because of a stated methodological limit

- **The thresholds are conventions, not statistics.** `0.85` cosine, `10%` share,
  `5` cases — all defaults, all configurable, none derived from theory. The
  sensitivity curve across thresholds is published rather than hidden.
- **Semantic similarity is a heuristic.** Precision at τ=0.85 measured 0.947
  against author-written labels — precision against *those labels*, not truth.
- **Paraphrased leakage is invisible** to string matching.
- **Redundancy detection is O(n²)** in time and memory. Fine for a few thousand
  cases, not for a corpus.
- **Model separation depends entirely on the model pair you choose.** Two models
  closer in capability than you assumed produce cases that look non-separating,
  and the check cannot tell the difference. It cannot verify your "weak" model is
  weaker — only flag when the numbers contradict the ordering you declared.
- **An exact effective sample size** is not available; the coverage estimate is a
  bracket under a stated assumption.

### Because nobody has looked

Five of the nine analyses need model runs, judges or repeated runs. From a CLI
they report **NOT ASSESSED**, never an empty pass, and the unified report keeps
that structurally distinct and never truncates the list.

**This section is a feature.** Every check also reports its own limitations at
runtime, and `CheckResult` raises if constructed without them — "state what you
cannot tell the user" is enforced by the type rather than left to discipline.


## Roadmap

- **Per-turn analysis.** Schema 2 loads conversations, but the checks read the
  flattened text, so cross-turn leakage is found as overlap within one string
  rather than attributed to a specific turn.
- **A real-dataset benchmark arm.** The synthetic suite has never caught a bug
  that real datasets did not catch first, which is the largest gap in evallint's
  own evidence. See [benchmarks/README.md](benchmarks/README.md).
- **Coverage of a capability space.** Currently unmeasurable from cases alone,
  and named here rather than implemented on a guess.

Deliberately not planned: a web UI, a database, a hosted service, an overall
quality score, or becoming an eval runner.


## Development

```bash
uv run pytest    # 808 tests
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
  mapping.py      your column names -> evallint's fields, always reported
  config.py       evallint.toml / [tool.evallint] discovery
  checks/
    base.py       Check interface; CheckResult refuses to exist without limitations
    discrimination.py · duplicates.py · imbalance.py · leakage.py
  report.py       CheckResults -> text (rich) or JSON
  cli.py          `evallint PATH`
```
