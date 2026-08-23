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

**Reproducibility — does the eval answer the same twice, and why not?** Three
sources of instability, reported **separately and never summed**, because they
need different fixes:

| source | isolated by | the fix |
|---|---|---|
| model stochasticity | ≥2 runs, judge held fixed | lower temperature, or more repeats |
| evaluator stochasticity | ≥2 judges on the *same* run | fix the grader — re-running won't help |
| dataset sampling | bootstrap over cases in one run | add cases — re-running won't help |

```python
from evallint import analyse_reproducibility, RunOutcome
print(analyse_reproducibility(outcomes).render())
```

A single "variance" number would destroy the only actionable information in the
measurement. When the design can't separate two sources, the analyser says
`not identifiable` rather than attributing the noise to one of them — one judge
gives *unidentifiable* evaluator variance, not zero.

**Aggregate stability and per-case stability are different things**, and the
divergence is the finding. This eval has an aggregate standard deviation of
**exactly zero** while **100% of its verdicts flip** between runs:

```
WARNING: the headline score is stable (sd 0.000) while 100% of individual
verdicts flip between runs. Independent flips cancel in an average, so this is
a coincidence rather than reproducibility, and it will not survive a change of
model.
```

Also reports **model ranking stability** — the modal ranking's share of runs,
Kendall's W, and which model pairs swapped order. Three models two points apart
produce a different ranking most runs; the analyser says so instead of reporting
whichever order this run happened to give.

**Redundancy-adjusted coverage — raw cases versus distinct scenarios.** An eval
advertising 500 cases may contain 312 scenarios, and an averaged score gives every
case equal weight, so a scenario appearing eight times counts eight times.

```python
from evallint import estimate_effective_size
print(estimate_effective_size(eval_set).render())
```

```
Raw cases: 500
Semantic clusters: 396
Large clusters (>= 3 cases): 27
Potential redundancy: 21%
Redundancy-adjusted scenario coverage: ~396 distinct scenarios
                                       (estimate; plausible range 396-500)
Design effect: 1.26 — intervals computed on 500 cases are roughly 1.12x too narrow

THIS IS AN ESTIMATE, NOT A STATISTICAL EFFECTIVE SAMPLE SIZE.
```

**That last line is load-bearing.** Kish's effective sample size reduces to
exactly the cluster count when each case is weighted by the inverse of its
cluster size — but that reduction holds *because the weighting assumes
within-cluster correlation of 1*. Two paraphrases are not perfectly redundant: a
model can pass one and fail the other. So the reduction is an assumption, not a
derivation, and the result is reported as a **bracket**:

| bound | assumes |
|---|---|
| cluster count (headline) | cases in a cluster are perfectly redundant, ρ = 1 |
| raw case count | every case is independent, ρ = 0 |

The truth is between. The lower bound leads because it is the one that cannot
flatter the eval.

The **design effect** is the actionable output: it is the factor by which
intervals computed on the raw count are too narrow. A `compare_models` interval
should be widened by roughly its square root when the eval has clusters —
every interval in this library assumes independent cases.

Threshold sensitivity is reported too, so you can see whether the estimate is
threshold-dependent for *your* data rather than assuming.

**Model comparison, paired and honest about it.** Cases are paired across
models, so the tests are paired — only the *discordant* cases carry information
about which model is better, and independent-sample tests would throw the
pairing away:

```python
from evallint import compare_models
print(compare_models({"A": a_outcomes, "B": b_outcomes}).render())
```

```
A: 87.1% (871/1000, 95% CI 84.9%-89.0%)
B: 88.4% (884/1000, 95% CI 86.3%-90.2%)

A 87.1% vs B 88.4% — difference +1.3 percentage points
95% CI (paired bootstrap) [+0.1, +2.4] pp
Statistical significance: McNemar exact p=0.041, on 35 discordant of 1000 cases
Effect size: Cohen's g 0.19 (medium), odds ratio 2.18
Practical significance: NO. The difference is real but smaller than your 2.0%
threshold, so it is detectable without being worth acting on.
```

**Statistical and practical significance are different questions**, and the
verdict is a 2×2 rather than a p-value:

| | effect ≥ threshold | effect < threshold |
|---|---|---|
| **significant** | real and meaningful | real but too small to act on |
| **not significant** | **underpowered** | no meaningful difference |

`underpowered` is the cell people misread as "no difference": the eval saw an
effect worth caring about and *could not resolve it*. The answer is more cases,
not a conclusion. And `no_information` is separate again — two models that never
disagreed tell you nothing, which is not evidence of equivalence.

**You set the practical threshold.** The 2pp default is a placeholder with no
derivation, is labelled as one, and appears in every result.

Three effect sizes, because the obvious one is misleading alone: the **risk
difference** shrinks as models agree more, so the same disagreement looks smaller
on an easy eval — while **Cohen's g** and the **odds ratio** do not. With more
than two models, **Holm-adjusted** p-values are reported: six pairwise tests at
α=0.05 carry a ~26% chance of at least one false positive.

**Evaluator reliability — is the grader itself trustworthy?** Every other check
audits the dataset. This one audits the judge, which is the one place an LLM
judge is legitimately involved: it is the *object* of measurement, not an oracle.

Seven measures: intra-judge consistency, inter-judge agreement, position bias,
order sensitivity, score variance, decision stability, judge correlation. Using
Cohen's kappa, Krippendorff's alpha, Pearson/Spearman and percentile bootstrap
intervals.

```python
from evallint import EvaluatorReliability, collect_verdicts

obs = collect_verdicts(cases, {"a": judge_a, "b": judge_b}, repeats=3)
report = EvaluatorReliability().analyse(obs)
```

evallint never calls your judges — `collect_verdicts` invokes callables *you*
supply, and `analyse` takes plain records, so the analysis itself touches no
provider.

**Correlation is not agreement**, and the module exists partly to show that. Two
judges whose scores differ by a constant offset:

```
Pearson correlation : 0.939
Cohen's kappa       : 0.200
Krippendorff alpha  : 0.025
```

A suite reporting only correlation would call those judges excellent. They agree
barely above chance.

**No metric is computed when its assumptions fail.** Each result carries its
sample size, interval, interpretation and limitations — and `value: None` with a
stated reason is a normal outcome:

| situation | outcome |
|---|---|
| Pearson on a constant series | refused — r is 0/0, *not* 0.0 |
| Spearman on binary verdicts | refused — 100% ties |
| alpha where every rating is identical | refused — undefined, *not* 1.0 |
| bootstrap from 6 observations | refused — resampling its own noise |
| one repeat per item | intra-judge consistency refused, and stated as **not** evidence of consistency |

**Agreement is still not correctness.** Judges from one model family share their
blind spots, so correlated error is indistinguishable from reliability. Every
agreement figure says so.

**Ground-truth quality, seven deterministic detectors plus optional judges.**
Cases whose reference answer may not be well enough defined for a score against
it to mean anything: `multiple_valid_answers`, `subjective_question`,
`missing_criteria`, `ambiguous_wording`, `reference_inconsistency`,
`underspecified_reference`, `multiple_interpretations`.

Each finding reports **case_id, ambiguity_type, evidence, confidence and a
recommended action** — and no recommended action is ever "delete". At dataset
level: the proportion potentially ambiguous, judge agreement, and the set of
cases needing human review.

**A subjective case is not an invalid case.** "Is this response helpful?" has no
single right answer, and an eval of such questions measures preference, which is
a real thing to measure. The finding says the score will be *grader-dependent* —
information for reading the number, not a defect to remove.

**Judges are optional, and the design distrusts them.** No LLM analysis happens
unless you pass judges explicitly, and evallint never calls a provider — you
supply the callable, as with the scorer:

```python
GroundTruthCheck(judges={"a": my_judge_a, "b": my_judge_b}).run(eval_set)
```

- **One judge** → findings capped at LOW confidence and labelled
  `single_judge_uncorroborated`. One model's opinion about ambiguity is an opinion.
- **Two or more** → only cases they *agree* on are reported. Cases they dispute
  are routed to human review, because disagreement is evidence the case is hard
  to adjudicate, not evidence about its answer.
- Agreement is reported with **Fleiss' kappa**, never raw agreement alone — on a
  skewed set two judges agree most of the time by accident. When every judge
  gives every case the same verdict, kappa is **undefined** and reported as such
  rather than as 1.0.
- **Agreement is not correctness.** Judges from one model family share their
  blind spots, so correlated error looks exactly like reliability.

**Leakage, seven deterministic detectors.** A case whose answer can be read off
its own prompt inflates the score while measuring reading. **No LLM judge is used
at any confidence level** — every detector is a deterministic function of the text:

| detector | confidence | catches |
|---|---|---|
| `expected_in_input` | moderate | the reference answer verbatim in the input |
| `label_in_input` | moderate | the class label as a word in the input |
| `explicit_reveal` | **high** | "the correct answer is X" where X *is* the answer |
| `answer_in_instructions` | **high** | the answer in the preamble, before the item marker |
| `metadata_leak` | moderate | the answer in a metadata field that isn't an answer field |
| `fewshot_target` | **high** | the evaluation target inside the case's own demonstrations |
| `high_overlap` | low | answer/input token overlap — **opt-in**, see below |

Confidence is **ordinal with a stated basis, not a probability**. HIGH means the
evidence is hard to explain innocently. MODERATE means the observation is certain
but its interpretation is not — the answer appearing in the input is exactly what
extractive QA looks like. Only HIGH warns; the rest are informational, because a
check that warns about legitimate task design is a check people switch off.

Every finding carries **case ID, type, evidence, confidence and an explanation**,
and the wording is always "potential leakage" — never a verdict that a case is
invalid.

`--leakage-overlap` enables the seventh detector, which is off by default for a
measured reason: on GSM8K, answer/input overlap reaches **100% with no leakage at
all**, because reference answers there are worked solutions that restate the
question. A planted leak scores the same, so no threshold separates them.

Validated at **zero false positives across GSM8K, MMLU, TruthfulQA and
HellaSwag** — a check that fires on real public datasets is worse than no check.

**Semantic redundancy, at five levels.** Only one of the five uses a cosine
threshold, which is deliberate — a single tuned number should not be the ground
truth for whether two cases are the same. `exact`, `normalized` (identical after
folding case, spacing and punctuation) and `template` (identical after masking
numbers and quoted spans) are **deterministic comparisons with nothing to tune**.
`scenario` and `semantic` use embeddings and are reported as heuristic. So
`DUPLICATE` and `SIMILAR` are separate claims, not one score.

Measured on the reference model, that separation earns its place: real
paraphrases span cosine **0.60 to 0.86**, so no single threshold has both good
precision and good recall — while `"What is 5 plus 3"` and `"What is 12 plus 7"`
score only **0.717** and are missed by the semantic level entirely, yet caught
exactly by the template level. Each run also reports how many similar pairs it
would find at 0.80/0.85/0.90/0.95, so you can see whether the threshold is
load-bearing for *your* data.

Comparison uses `input` + `expected` by default: the same question with two
different reference answers is a ground-truth contradiction, not a duplicate, and
collapsing them would hide it. `--compare input` restores input-only matching.
The three deterministic levels need no model, so this runs in a core install.

What it reports is **how much weight one scenario carries in an averaged score** —
a fact about arithmetic, not a judgement. Consistency tests, template families and
regression suites are all legitimate reasons to repeat a scenario, so no finding
here calls a case invalid.

**Class imbalance and basic stats.** Reports class distribution, imbalance ratio, and
per-class counts. Matters because aggregate accuracy hides failure on rare classes: a
model that handles the 70% majority class well and everything else badly still scores
~70% overall. A class becomes unmeasurable two independent ways — its **share** is too
small to move the aggregate, or its **count** is too small for its own accuracy to mean
anything — so both are checked. In a 10,000-case set a 1% class still has 100 cases and is
perfectly measurable; in a 20-case set a 5% class has one.

**Label leakage.** Flags cases whose `expected` answer appears verbatim in their own
`input`. Those cases don't need the capability under test — the answer is sitting in the
prompt — so they inflate the score while measuring reading. Deliberately quiet: token
overlap between an answer and its question was measured and **discarded as a signal**,
because on GSM8K it reaches 1.00 with no leakage at all (the reference answer restates the
question). Short answers and small closed label vocabularies are skipped, and a set where
*most* cases contain their answer is reported once as probable extractive QA rather than
as hundreds of warnings. Zero false positives across GSM8K, MMLU, TruthfulQA and HellaSwag.

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

Multi-turn sets are a genuine limitation rather than a naming one: MT-Bench's
prompt is a *list* of turns, and evallint models one case as one input string.
No `--map` fixes that; it exits 3 and says so.

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
the two inversions a single run reported, which is exactly why the check refuses to
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
- **Only four checks.** Ambiguous ground truth is a real problem and
  are not implemented.

This section is a feature. Every check also reports its own limitations at runtime, and
`CheckResult` raises if constructed without them — "state what you cannot tell the user"
is enforced by the type, not left to discipline. A tool that states its uncertainty is
more trustworthy than one that doesn't.

## Roadmap (v2+)

- **Ambiguous-ground-truth check** — flag cases where the reference answer is one of
  several defensible responses, so a correct model gets marked wrong.
- **Multi-turn eval sets.** One case is currently one input string, so a conversational
  set like MT-Bench (whose prompt is a list of turns) cannot load at all. Supporting it
  is a schema change, not a mapping one.

Deliberately not planned: a web UI, a database, a hosted service, or becoming an eval
runner.

## Development

```bash
uv run pytest    # 613 tests
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
