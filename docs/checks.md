# Check reference

**evallint audits the reliability of LLM evaluations.**

Every check below is documented under the same eight headings: what it measures,
why it matters, methodology, assumptions, limitations, example, false positives,
false negatives. Where a false-positive or false-negative rate has been
*measured*, the number is given with its source; where it has not, that is said
rather than guessed.

---

## Three different things, and evallint measures two of them

Conflating these is the single most common way to misread this tool's output.

| | What it is | Needs | evallint |
|---|---|---|---|
| **Dataset quality** | Properties of the eval *file*: how much of it is redundant, how the classes are distributed, whether answers are visible in prompts, whether references pin down an answer. | The file alone | **Measures it** |
| **Evaluation quality** | Properties of the *measurement procedure*: can it separate the models you are comparing, is the grader reproducible, does the number come back the same twice, is the comparison powered. | Model runs, judges, or repeated runs | **Measures it** |
| **Model quality** | How good the model is. | An eval you already trust | **Does not measure it** |

The dependency runs one way. A finding about dataset quality bounds what
evaluation quality can be, and evaluation quality bounds what you can conclude
about model quality. Nothing here tells you a model is good, and nothing here
tells you an eval is good — the tool reports evidence and gaps, and whether that
is acceptable for your purpose is a judgement it does not have the information to
make.

### On the word "quality"

Used above because it is the word people search for, but every check below
reports something narrower and says so: *potential* redundancy, *potential*
leakage, *potential* ground-truth ambiguity, *limited measurement evidence*. None
of these is a defect until a human looks. A 40%-redundant eval may be a
deliberate consistency suite; a 14:1 class ratio may be your production traffic;
a case every model passes may be a regression guard you want to keep.

---

# Part 1 — Dataset quality

Runs from the file. No model, no network, no API key.

---

## 1. Potential redundancy

### What it measures
How many of the cases are near-copies of each other, at five separately-reported
levels: `exact`, `normalized`, `template`, `scenario`, `semantic`. Reports
clusters, the strongest level in each, and each cluster's **weight share** — the
fraction of the aggregate score that cluster controls.

### Why it matters
An averaged score gives every case equal weight, so a scenario appearing eight
times counts eight times. If one cluster holds 30% of the weight, 30% of your
headline number is one scenario's difficulty.

### Methodology
Five detectors, cheapest first, each independently reported:

| Level | Rule | Tier |
|---|---|---|
| `exact` | byte-identical after mapping | observed |
| `normalized` | identical after case, whitespace and punctuation folding | observed |
| `template` | identical after masking numbers and quoted spans, ≥4 remaining tokens | heuristic |
| `scenario` | identical input with differing expected answers | observed |
| `semantic` | cosine similarity ≥ threshold (default 0.85) on local embeddings | heuristic |

Clusters are **connected components** over the union of all pairs found. The
comparison covers `input` + `expected` by default; `--compare input` reproduces
input-only behaviour.

### Assumptions
- That equal weighting is what you want. If you weight cases yourself, cluster
  weight shares do not apply.
- That the embedding model's notion of similarity matches yours. It is
  `all-MiniLM-L6-v2` by default, and a different model would cluster differently.
- That the threshold is meaningful for your text. It is a chosen number, not a
  derived one — hence `heuristic`.

### Limitations
- **`semantic` needs the optional extra.** Without `pip install
  evallint[embeddings]` the three deterministic levels run and the check reports
  itself as partial rather than clean.
- **Connected components over-cluster on purpose.** A chain A~B~C groups all
  three even when A and C are not similar. That inflates redundancy, making the
  result *more* conservative.
- **Only the compared fields count.** Two cases testing different things through
  near-identical text read as one scenario.
- **Similar is not invalid.** Nothing here recommends deleting a case.

### Example
```python
from evallint import EvalCase, EvalSet
from evallint.checks import RedundancyCheck, CompareFields

cases = (
    EvalCase(id="b1", input="I was charged twice this month.", expected="Refund the duplicate"),
    EvalCase(id="b2", input="I was charged twice this month.", expected="Refund the duplicate"),
    EvalCase(id="b3", input="My card was billed two times in March.", expected="Refund the duplicate"),
    EvalCase(id="s1", input="Where is my order? It has been nine days.", expected="Track it"),
    EvalCase(id="p1", input="How do I reset my password?", expected="Use the reset link"),
)
result = RedundancyCheck(semantic=False, compare=CompareFields.INPUT_EXPECTED).run(EvalSet(cases=cases))
print(result.summary)
for c in result.stats["clusters"]:
    print(c["case_ids"], c["level"], c["certainty"], f"{c['weight_share']:.2f}")
```
```
5 cases, 1 redundant cluster covering 2 cases — about 4 distinct scenarios (1 duplicate, 0 similar)
['b1', 'b2'] exact duplicate 0.40
```
Note what did **not** happen: `b3` is a paraphrase of `b1`/`b2` and was *not*
clustered, because the semantic level was off. That is the honest deterministic
result, and it is why the check reports which levels ran.

### False positives
**Measured: precision 0.947 at τ=0.85** on the benchmark's held-out split
(`benchmarks/README.md`), FPR 0.004 over 435 candidate pairs. `exact` measured
**1.000 precision** — it is byte comparison, so a false positive there would be a
bug.

The realistic false-positive mode is a same-topic pair asking different things
("can I cancel" vs "what happens to my data if I cancel"). One such pair is
clustered at τ=0.85 in the benchmark. The threshold curve is published so you can
see the trade-off rather than trust the default.

### False negatives
**Measured: recall 1.000 at τ=0.85** on paraphrase families in the benchmark, and
**0.000 with the semantic level off** — paraphrases are invisible to the
deterministic levels. Recall collapses to 0.389 at τ=0.90, so the default sits
just below a cliff on that data. Reworded cases sharing no vocabulary are the
expected miss.

---

## 2. Class distribution

### What it measures
How evenly cases are spread across the values of `label`: the ratio of largest to
smallest class, each class's share, and which classes are too small for a stable
per-class accuracy.

### Why it matters
If 14 of 20 cases carry one label, the aggregate is mostly that label's
difficulty. A class with 1 case has an accuracy that can only be 0% or 100%.

### Methodology
A census. Counts per label, largest/smallest ratio, and two thresholds:
`min_class_share` (default 0.10) and `min_class_count` (default 5). For a class of
*n* cases the accuracy can take only *n+1* distinct values, and that is what the
finding reports. Tier: **observed** throughout — arithmetic, not inference.

### Assumptions
- That `label` means a category. HellaSwag has a column literally named `label`
  that holds the index of the correct ending; evallint cannot know that, so it
  reports which other columns were available.
- That an even distribution is what you want. Often it is not.

### Limitations
- **It cannot tell you whether your distribution is wrong, only whether it is
  uneven.** If production traffic is 70% billing, a 70%-billing eval may be
  exactly right.
- Unlabelled cases are counted and reported separately, not silently dropped.
- The two thresholds are conventions. They are printed with the finding.

### Example
```python
from evallint.checks import ImbalanceCheck
result = ImbalanceCheck().run(eval_set)   # 14 billing, 4 shipping, 1 refund, 1 bug_report
print(result.summary)
```
```
20 cases across 4 classes, imbalance ratio 14.0:1
  [warning] class imbalance ratio is 14.0:1 — 'billing' has 14 cases, 'bug_report' has 1.
            Aggregate accuracy is dominated by 'billing'
  [warning] class 'refund' is under-represented: 5.0% of labelled cases (below 10%);
            only 1 case, so its accuracy can only land on 2 distinct values
```

### False positives
**Measured: precision 1.000** on the benchmark. A "false positive" here can only
be a case where the distribution is uneven *and that is fine* — which is a
question about your intent, not about the arithmetic. The check states this rather
than resolving it.

### False negatives
**Measured: recall 0.667** on the benchmark, and the miss is documented as a
**known gap**: a dataset with exactly one label reports "ratio 1.0:1" and no
warning. Technically correct — nothing is imbalanced — and useless, since no
per-class comparison is possible at all.

---

## 3. Potential leakage

### What it measures
Whether the answer is reachable from the prompt without the capability under
test. Seven deterministic detectors: `expected_in_input`, `label_in_input`,
`explicit_reveal`, `answer_in_instructions`, `metadata_leak`, `fewshot_target`,
and an opt-in token-overlap detector.

### Why it matters
A case whose answer sits in its own prompt measures retrieval from context, not
the ability you meant to test — and it inflates the score of any model that can
read.

### Methodology
Deterministic string and word-boundary matching, never an LLM judge. Every
finding carries a type, a confidence, and the evidence that produced it.
Answers under 12 characters are skipped: a short string appearing in a long
prompt is a coincidence waiting to happen. When one type affects more than half
the cases it collapses to a single dataset-level finding, because a pattern
present everywhere is more likely the task design than a defect.

Tier: **heuristic**. Verbatim containment is a *proxy* for leakage, not leakage.

### Assumptions
- That the answer being present is not the point of the task. For extractive QA,
  span selection and reading comprehension it is.
- That `expected` holds the answer rather than a rubric or a rationale.

### Limitations
- **It cannot distinguish a leak from a task where the answer is supposed to
  appear.** The finding says so in its own text.
- **Paraphrased leakage is invisible.** An answer restated in different words is
  not detected by string matching.
- The token-overlap detector is **off by default** because it is measurably
  unreliable: on GSM8K it reaches 100% overlap with no leakage present, since
  reference answers restate the question. Findings from it are labelled LOW
  confidence.

### Example
```python
from evallint.checks import LeakageCheck
print(LeakageCheck().run(eval_set).summary)
```
```
3 answered cases, 1 contain their own answer (33%)
  [warning] POTENTIAL LEAKAGE [expected_in_input, confidence moderate] in case 'L1':
  the reference answer appears verbatim in the input, so this case can be answered by
  copying rather than by the capability under test. If the answer is MEANT to be
  present — extractive QA, span selection, reading comprehension — this is the task
  working, not a leak. Evidence: expected (51 chars) found in input: escalate to the
  billing team within one working day
```

### False positives
**Measured: precision 0.667 overall** on the benchmark's held-out split, and the
per-detector breakdown is the important part:

| Detector | Flagged | Precision |
|---|---|---|
| `expected_in_input` | 4 | **1.000** |
| `label_in_input` | 2 | **0.000** |

`label_in_input` produced every false positive and is the sole reason overall
precision is not 1.000. This is *documented, intended* behaviour — a question
about billing naturally contains the word "billing" — but the cost is now
quantified. Note that **confidence does not separate them**: true and false
positives are all "moderate". The detector *type* does.

**On real data:** 100 false positives on TruthfulQA before a fix (a metadata
exemption matched exact field names and missed `correct_answers`).

### False negatives
**Measured: recall 1.000** on the benchmark — but the benchmark's label is
*definitional*, i.e. the same rule the detector applies, so that number measures
implementation correctness and **not** whether the rule catches real leakage.
Paraphrased leakage, leakage through a shared prefix, and leakage via an
identifier the model has memorised are all outside what string matching can see.
This is the weakest evidence in the whole document, and it is labelled as such.

---

## 4. Ground-truth ambiguity

### What it measures
Whether the reference answer pins down a single correct response. Seven detector
types including `multiple_valid_answers`, `missing_criteria`,
`reference_inconsistency`, and subjective phrasing.

### Why it matters
If a case has several defensible answers and one recorded, a correct model can be
scored wrong. That is a measurement error attributed to the model.

### Methodology
Deterministic detectors first: placeholder references (`TODO`, `TBD`), references
enumerating alternatives ("either", "or"), list-valued references, open-ended
question phrasing, and **reference inconsistency** — two cases sharing an input
with different references, which is structural rather than a judgement.

LLM judges are **optional, off by default, and never authoritative**: one judge
yields LOW confidence tagged `single_judge_uncorroborated`; with two or more only
unanimous findings are reported and disagreements are routed to human review.

Tier: **heuristic**, except reference inconsistency, which is observed.

### Assumptions
- That one reference per case means one correct answer is intended.
- That subjective phrasing correlates with under-specification. It does, weakly.

### Limitations
- **A subjective case is not an invalid case.** "Is this response helpful?" is a
  legitimate thing to evaluate; it just cannot be scored by string match.
- **Keyword-based detection has an English-language, phrasing-specific
  vocabulary.** It will miss constructions it has not been taught.
- **No case is ever declared invalid.** Every finding carries a recommended
  action, and none of them is "delete".

### Example
```python
from evallint.checks import GroundTruthCheck
print(GroundTruthCheck().run(eval_set).summary)
```
```
6 cases, 4 potentially ambiguous (67%), 3 need human review — deterministic detectors
only, no judges configured

  [warning] POTENTIALLY AMBIGUOUS [missing_criteria, confidence high] in case 'A3':
            reference is a placeholder: 'TODO'
  [warning] POTENTIALLY AMBIGUOUS [reference_inconsistency, confidence high] in case 'C1':
            2 cases share this input but carry 2 different references, e.g.
            ['10 working days', '5 working days']
  [info]    POTENTIALLY AMBIGUOUS [multiple_valid_answers, confidence moderate] in case 'A2':
            reference enumerates alternatives ('either')
```

### False positives
**Measured: precision 1.000** on the benchmark's held-out split, with zero false
positives among five adversarial negatives designed to trip a keyword rule —
including a question containing "best-practice" that has one defensible answer,
and an "A or B" *question* whose answer is unambiguous.

**On real data:** 3 false contradictions on MMLU before a fix (punctuation
stripping collapsed three distinct field extensions), and 101 warnings on
HellaSwag, which has no reference answers at all.

### False negatives
**Measured: recall 0.667** on the held-out split — two of six missed, both
keyword-list gaps ("Give a few options for…", "What is the most sensible order
to…"). Recorded as a **known gap** and deliberately unfixed, because fixing it in
response to a held-out number would contaminate the split. The dev split misses
only one, which is what a genuinely held-out set looks like.

---

# Part 2 — Evaluation quality

These need something the file cannot supply: model runs, judges, or repeated
runs. From a command line they will always report **NOT ASSESSED**, and that is
not the same as passing.

---

## 5. Model separation (discrimination)

### What it measures
For each case, whether the models you are comparing produced *different*
verdicts. Cases are classified into `ceiling` (all pass), `floor` (all fail),
`separating` (verdicts differ in the expected order), `non_monotonic` (a weaker
model passed where a stronger failed), and `indeterminate`.

### Why it matters
A case where every model gives the same verdict contributes nothing to the
question "which model is better". If 140 of 200 cases behave that way, the
comparison rests on 60.

### Methodology
An injected `score(case, model) -> bool | float` — evallint never imports a
provider SDK. Per pair of models: exact McNemar test (binomial tail, not χ²),
a paired difference interval, and a **minimum detectable effect**
`(z_α + z_β)·√(p_disc/n)` so an unresolved pair reports what it *could* have
detected. Pair verdicts come from the exact test, not the interval, because on
small discordant counts they disagree.

Tier: **estimated** — intervals and assumptions attached.

### Assumptions
- **That your models really differ in capability**, and in the order you listed
  them. This is the load-bearing assumption: models closer in capability than you
  assumed produce cases that look non-separating.
- That the scorer is deterministic, or that `repeats` is set high enough.
- That a pass/fail verdict captures what you care about.

### Limitations
- **A ceiling case is not a bad case.** It may be a deliberate regression guard.
  The finding says "provides limited evidence for model separation", not "too
  easy", and it is INFO rather than WARNING.
- **A floor case is more often a wrong reference than a hard case**, which is why
  it *is* a warning — the recommendation is to check the grader first.
- Requires at least two models. With one, nothing about separation is knowable.
- Costs money: it runs your models.

### Example
```python
from evallint.checks import DiscriminationCheck
result = DiscriminationCheck(scorer, ["weak", "middle", "strong"]).run(eval_set)
print(result.summary)
```
```
14 cases x 3 models: 5 separate models, 7 provide limited evidence (50%)

  [info]    every model passes 4 cases, including 'weak', so these provide limited
            evidence for model separation. They may still be worth keeping as
            regression guards
  [warning] every model fails 3 cases, including 'strong', so these provide limited
            evidence for model separation. Check the reference answer and the grader
            before assuming difficulty — a case nothing passes is more often wrong
            than hard
  [warning] in 2 cases a weaker model passed where a stronger one failed. That
            ordering is suspicious: it usually means a wrong reference answer or a
            grader artefact rather than a real capability gap
  [info]    'weak' vs 'middle' is UNRESOLVED: observed difference +21.4%,
            McNemar p=0.453. With 7 discordant of 14 cases this eval could only have
            detected a difference of about 52.9% or larger, so it cannot support a
            claim either way about this pair
```
The last line is the shape of an honest negative result: not "the models are the
same", but "this eval cannot tell".

### False positives
**Measured: 100% exact class agreement**, 0 disagreements over 20 cases and 3
models on the benchmark's held-out split. This is the most trustworthy figure in
the document, because the scorer is a fixture — which models pass which case is
*written down* rather than judged, so a disagreement would be a classifier bug
rather than a labelling dispute.

The real false-positive mode is not classification error but **assumption
failure**: if your "weak" model is not actually weaker, correctly-classified
`non_monotonic` cases will point at your reference answers when the problem is
your model ordering.

### False negatives
A case can separate two models on a *dimension you did not test*. With three
models you learn about three pairs; a case that would separate a fourth model is
classified `ceiling` and reported as limited-evidence. That classification is
correct for the comparison you ran and wrong as a statement about the case.

---

## 6. Evaluator reliability

### What it measures
Whether the *grader* is reproducible: inter-judge agreement, chance-corrected
agreement (Cohen's/Fleiss' kappa, Krippendorff's alpha), self-consistency across
repeats, position bias, order sensitivity, and inter-judge correlation.

### Why it matters
If your judge disagrees with itself, every score it produced inherits that noise —
and no amount of re-running the model fixes it.

### Methodology
Judges are **the object of measurement, never a trusted oracle.** Every metric
carries sample size, value, interval where appropriate, interpretation and
limitations. **Metrics are refused when their assumptions are violated**, with
the reason, rather than computed anyway.

The kappa paradox is handled explicitly: when every rater uses one category,
p_expected = 1 and kappa is 0/0. The functions return `None`, and the report says
*undefined* rather than substituting 1.0 or 0.0.

Tier: **estimated**.

### Assumptions
- That the items were graded independently. Judges sharing a prompt template do
  not.
- That the rating scale means the same thing to each judge.

### Limitations
- **Agreement is not correctness.** Judges from one model family share their
  training and their blind spots, so correlated error is indistinguishable from
  reliability. Only a human-labelled sample separates them, and this module
  cannot obtain one. This is stated on every report.
- **Majority-not-unanimity is not neutral.** Filtering to unanimous verdicts
  changes the measured outcome — measured at 0.52 → 0.89 as repeats went 1 → 5.
- One judge measures nothing about inter-judge agreement.

### Example
```python
from evallint.checks import EvaluatorReliability, JudgeObservation
report = EvaluatorReliability().analyse(observations)   # 3 judges, 40 items
for name, m in report.metrics.items():
    print(f"{name}: value={m.value} n={m.n} unmet={m.unmet_reason}")
```
```
inter_judge_agreement: value=0.65 n=40 unmet=None
inter_judge_alpha: value=0.5372222222222223 n=40 unmet=None
position_bias: value=None n=0 unmet=no pairwise observations
order_sensitivity: value=None n=0 unmet=no pairwise observations
judge_correlation:j0|j1: value=None n=0 unmet=insufficient shared numeric scores

  · Every figure here measures the JUDGE, not the dataset and not the model being graded.
  · AGREEMENT IS NOT CORRECTNESS. Judges from one model family share their blind spots,
    so correlated error is indistinguishable from reliability.
  · Not computed because their assumptions were not met: judge_correlation:j0|j1,
    order_sensitivity, position_bias. Each states its own reason rather than reporting
    a number.
```

### False positives
Not a classification problem, so precision does not apply. What *is* checkable is
whether the measurement tracks a known quantity: on the benchmark, chance-corrected
agreement **rises monotonically** with a constructed forced-agreement share from
0.00 to 1.00. At 1.00 kappa becomes undefined — the paradox — and returning
undefined there is the correct output, not a failure.

### False negatives
A judge that is *consistently wrong* is perfectly reliable by every metric here.
Systematic bias shared across judges is invisible. So is a judge that agrees with
itself because it is deterministic rather than because it is reasoning well.

---

## 7. Statistical model comparison

### What it measures
Whether an observed difference between two models is distinguishable from zero,
and whether it is large enough to matter.

### Why it matters
"Model B scored 3 points higher" on 60 cases is compatible with the models being
identical. Reporting it as an improvement is the most common statistical error in
eval reporting.

### Methodology
Cases are **paired** across models, so paired tests are used: exact McNemar
(binomial tail on the discordant pairs), paired-difference variance
`[p_b + p_c − δ²]/n`, paired bootstrap intervals, Wilson intervals for individual
rates (not normal — 98/100 gives [0.943, 1.017] under the normal approximation),
Cohen's g, odds ratio with Haldane correction, and Holm step-down for multiple
pairs.

Every verdict is one of `real_and_meaningful`, `real_but_below_threshold`,
`underpowered`, `no_meaningful_difference`, `no_information` — and never a
p-value alone.

Tier: **estimated**.

### Assumptions
- **Cases are independent.** They are not, if the eval has redundancy clusters —
  which is why the effective-size estimate reports how much to widen intervals.
- The pairing is real: both models saw the same cases.
- The practical threshold (default 2%) is **your** choice, not a statistic. The
  output says so.

### Limitations
- **Underpowered is not "no difference".** It is the absence of evidence about a
  difference, and the answer is more cases, not a smaller p-value.
- Intervals assume independence and are too narrow when clusters exist.
- Cohen's conventional effect-size bands are conventions, printed as such.

### Example
```python
from evallint.checks import compare_models
print(compare_models({"weak": weak_outcomes, "strong": strong_outcomes}).render())
```
```
weak: 66.7% (40/60, 95% CI 54.1%-77.3%)
strong: 85.0% (51/60, 95% CI 73.9%-91.9%)

weak 66.7% vs strong 85.0% — difference +18.3 percentage points
95% CI (paired bootstrap) [+3.3, +33.3] pp
Statistical significance: McNemar exact p=0.0347, on 23 discordant of 60 cases
Effect size: Cohen's g 0.24 (medium by Cohen's conventional bands), odds ratio 2.83
Practical significance: YES. The difference is distinguishable from zero AND at or
above your 2.0% threshold.

Practical threshold: 2.0%. This is YOUR choice, not a statistic — a 'not meaningful'
verdict is only as good as it.
```

### False positives
**Measured: 5.5% over 200 trials of two genuinely equal models**, against a
nominal α of 5% (benchmark held-out split, 60 paired cases). This is the strongest
validation in the document because the target is arithmetic rather than opinion.

### False negatives
**Measured: power 45.5% at a true 20-point difference** over 60 cases, and
**nil at a 5-point difference**. Low power is a property of the sample size, not a
defect — and the analyser reports `underpowered` rather than "no difference",
which is the distinction that matters.

---

## 8. Redundancy-adjusted coverage

### What it measures
How many *distinct scenarios* an eval contains, reported as a bracket rather than
a number.

### Why it matters
An eval advertising 500 cases may contain 312 distinct scenarios, and every
confidence interval computed on 500 is too narrow by roughly the square root of
the design effect.

### Methodology
Deliberately **not** presented as a statistical effective sample size. The
terminology is `effective scenario count`, `redundancy-adjusted coverage`,
`independent-case estimate`.

Kish's `n_eff = (Σw)²/Σw²` reduces exactly to the cluster count under
inverse-cluster weights — but that holds *because* those weights assume
within-cluster correlation of 1. The reduction is an assumption, not a
derivation. So the output is a bracket: lower bound = cluster count (assumes
ρ=1), upper bound = raw case count (assumes ρ=0). The lower bound is the headline
because it cannot flatter the eval.

Tier: **heuristic**.

### Assumptions
- That the clustering threshold is meaningful — hence the sensitivity curve
  across several thresholds, and a `threshold_is_load_bearing` flag.
- That equal weighting applies.

### Limitations
- **It is an estimate, and the output says so in capitals.**
- The lower bound assumes perfect within-cluster redundancy. Two paraphrases are
  not perfectly redundant: a model can pass one and fail the other.
- Redundancy is not a defect.

### Example
```python
from evallint.checks.effective_size import estimate_from_clusters
print(estimate_from_clusters([3, 2], raw_cases=20, threshold=0.85).render())
```
```
Raw cases: 20
Semantic clusters: 17
Potential redundancy: 15%
Redundancy-adjusted scenario coverage: ~17 distinct scenarios (estimate; plausible range 17-20)
Design effect: 1.18 — intervals computed on 20 cases are roughly 1.08x too narrow

THIS IS AN ESTIMATE, NOT A STATISTICAL EFFECTIVE SAMPLE SIZE.
```

### False positives / false negatives
Not a detector, so neither applies. The estimate inherits the redundancy check's
error modes exactly: over-clustering inflates redundancy and makes the estimate
more conservative; under-clustering flatters the eval.

---

## 9. Reproducibility

### What it measures
Whether the same eval gives the same answer twice — and **which of three separate
sources** the instability comes from.

### Why it matters
An unstable score has at least three causes needing different fixes. Collapsing
them into one number destroys the only actionable information in the measurement.

### Methodology
Three sources, reported separately and **never summed**:

| Source | Isolated by | The fix |
|---|---|---|
| model stochasticity | ≥2 runs, judge held fixed | lower temperature, or average more repeats |
| evaluator stochasticity | ≥2 judges scoring the *same* run | fix the grader — re-running cannot help |
| dataset sampling | bootstrap over cases within one run | add cases — re-running cannot help |

They are additive only under a variance-components model with independent
effects, which these designs do not guarantee. `as_dict()` contains no total, and
a test asserts its absence.

Also reports aggregate stability, per-case stability, and model ranking stability
(modal ranking share, Kendall's W without tie correction, so a reported value is a
*lower bound* when ties are present).

Tier: **estimated**.

### Assumptions
- That runs are exchangeable — same prompts, same scorer, same conditions.
- That a cached response is a genuine repeat. It is not, and the report says so
  when variance is exactly zero.

### Limitations
- **The design determines what is identifiable.** With one judge, evaluator
  variance is `not identifiable`, **not zero** — reporting 0.0 would claim the
  grader is noiseless.
- With one run, nothing about run-to-run stability is claimed.
- Kendall's W has no tie correction (conservative).

### Example
```python
from evallint.checks import analyse_reproducibility
print(analyse_reproducibility(outcomes).render())
```
```
Design: 4 run(s), 2 model(s), 20 case(s), judge not recorded

VARIANCE SOURCES (reported separately, never summed)
  model_stochasticity        sd 0.0000   n=4
  evaluator_stochasticity    not identifiable   n=0
    not identifiable from this design: 0 judge(s) recorded; needs at least 2 scoring
    the same run to separate grader noise from model noise
  dataset_sampling           sd 0.2000   n=20

PER-CASE STABILITY: 0% of 40 cases kept the same verdict in every run; 40 are flip-prone

  · WARNING: the headline score is stable (sd 0.000) while 100% of individual verdicts
    flip between runs. Independent flips cancel in an average, so this is a coincidence
    rather than reproducibility, and it will not survive a change of model.
```
That divergence is the finding: a stable aggregate over unstable cases is
arithmetic luck, not reproducibility.

### False positives
Not a classification problem. What is checkable is attribution, and it is: a
deterministic model with disagreeing judges yields model variance of **exactly
0.0000** and evaluator variance of **0.1768** — all instability correctly
assigned to the grader.

### False negatives
Variance shared across all runs is invisible. A model that is consistently wrong
in the same way looks perfectly reproducible, because it is — reproducibility and
correctness are different properties.

---

# Part 3 — What evallint cannot tell you

Grouped by why, because the reasons differ in kind.

### Because it is out of scope by design

- **Whether your model is any good.** evallint audits the measurement, not the
  thing measured.
- **Whether your eval is "good" or "bad".** No check emits such a verdict and no
  report contains an overall score. Weighing redundancy against an unmeasured
  judge would need invented weights.
- **Whether a case should be deleted.** No finding recommends it.
- **How to fix your model.** Every recommendation is about the evaluation.

### Because it requires knowing your intent

- **Whether your class distribution is wrong**, only whether it is uneven. 70%
  billing may match production.
- **Whether redundancy is deliberate.** Consistency tests reuse inputs on
  purpose; regression suites repeat scenarios that once broke.
- **Whether a ceiling case should be removed.** It may be a regression guard.
- **Whether a subjective case belongs in your eval.** It may be exactly the thing
  you need to evaluate.
- **Whether the answer appearing in the prompt is a leak.** For extractive QA it
  is the task.

### Because the information is not in the data

- **Whether your reference answers are correct.** evallint checks whether they
  are *unambiguous*, which is a different property. A confidently wrong reference
  passes every check here.
- **Whether your judge is right.** Only whether it is *consistent*. Agreement
  among judges from one model family may be correlated error.
- **Whether your eval covers what matters.** Coverage of a capability space
  cannot be measured from cases alone; nothing here knows what you failed to test.
- **Whether a non-separating case is easy or your models are too similar.** The
  observation is the same either way.

### Because of a stated methodological limit

- **Whether two cases are semantically identical.** A cosine threshold is a
  heuristic. Precision at τ=0.85 measured 0.947 against author-written labels,
  which is precision against those labels rather than truth.
- **An exact effective sample size.** The coverage estimate is a bracket under a
  stated assumption.
- **Whether paraphrased leakage exists.** String matching cannot see it.
- **Absolute memory or runtime.** Benchmark memory comes from `tracemalloc` and
  reports 114 KiB while loading a 90 MB model.

### Because nobody has looked

Three of nine analyses need model runs, judges or repeated runs. From a CLI they
report **NOT ASSESSED** — which is not a pass. The unified report keeps this
structurally distinct and never truncates the list.

---

# Part 4 — Reproducible methodology

Every number in this document came from a command. These reproduce them.

### The examples above

```bash
git clone https://github.com/BhagyasriUddandam/evallint && cd evallint
uv pip install -e ".[embeddings]"
python -m benchmarks.runner --split test --semantic
```

All check outputs quoted here were generated by running the snippet shown beside
them against the eval set described in it. Nothing is illustrative.

### The measured false-positive and false-negative rates

```bash
python -m benchmarks.runner --split test --json out.json    # deterministic
python -m benchmarks.runner --split test --semantic          # + embedding level
```

Methodology, label bases and the held-out split rule:
[`benchmarks/README.md`](../benchmarks/README.md). The important caveat, stated
first there: **the fixtures and the detectors have the same author.**

### The α-calibration result

```bash
python -m benchmarks.runner --split test --only statistical_noise --trials 200
```

Constructs 200 trials of two genuinely equal models over 60 paired cases and
counts how often the test declares a difference. Target 5%, measured 5.5%.

### The real-dataset false-positive history

Not vendored, for licence and size reasons. To reproduce, point evallint at a
local copy:

```bash
evallint truthfulqa.csv --skip-duplicates --format json --out tqa.json
```

| Dataset | Measured | Cause |
|---|---|---|
| TruthfulQA | 100 false positives (leakage) | metadata exemption matched exact field names |
| MMLU | 3 false contradictions (ground truth) | punctuation stripping collapsed field extensions |
| HellaSwag | 101 warnings (ground truth) | dataset has no reference answers at all |
| GSM8K | 98% of cases non-separating | genuine finding — [`findings-gsm8k.md`](findings-gsm8k.md) |

### The GSM8K finding

```bash
# 250 cases, judge-free exact-match scoring, two models
python scripts/gsm8k_harness.py --n 250 --models weak,strong
```

Full methodology, cost accounting and caveats:
[`findings-gsm8k.md`](findings-gsm8k.md) and
[`methodology-discrimination.md`](methodology-discrimination.md).

---

## Further reading

| Document | What is in it |
|---|---|
| [`audit-report.md`](audit-report.md) | The unified report: eleven sections, evidence tiers, three formats |
| [`schema-v2.md`](schema-v2.md) | Chat, rubrics, tool calls, and the flattening rules |
| [`methodology-discrimination.md`](methodology-discrimination.md) | The statistics behind model separation, derived |
| [`findings-gsm8k.md`](findings-gsm8k.md) | A real measurement on a public benchmark |
| [`architecture-v2.md`](architecture-v2.md) | Design across eight dimensions, including what was declined |
| [`../benchmarks/README.md`](../benchmarks/README.md) | How evallint is measured, and by whom |
