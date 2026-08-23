# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] — 2026-08-22

### Added

**The unified audit report.** Eleven sections, three formats
(`--format terminal|json|html`), and deliberately **no overall score**.

**What replaces the score.** An "Eval Trust Score: 78/100" would need a weight
for redundancy against a weight for an unmeasured judge, and no defensible
source for either number exists. So the report publishes `evidence_coverage`
instead: which analyses produced evidence, and which did not and what it would
take. Two tests exist purely to make adding a score fail — no key anywhere in
the JSON may read as a grade, and `AuditReport` may not gain a `score`
attribute.

**The evidence tier is enforced, not documented.** Designed in
`docs/architecture-v2.md` and implemented here for the first time. Every finding
is OBSERVED (a census, exact), HEURISTIC (a chosen threshold, no sampling
theory) or ESTIMATED (an interval and stated assumptions). Two rules raise in
`__post_init__`: an OBSERVED finding may not carry a confidence, because a count
is exact and attaching "HIGH" implies a doubt that does not exist; a HEURISTIC
finding must carry one, because a threshold-derived signal without a confidence
reads as a fact. Recommendations sort weakest-tier-first, so nobody acts on a
cosine threshold thinking it was a count.

**"Not assessed" is structurally distinct from "nothing found."** A section that
could not run is NOT_RUN with a reason naming the exact call to make; it may not
carry findings, and a reason is required. Four of seven analyses run from a file
alone, so from a CLI the other three are always NOT ASSESSED — reported in every
format, never truncated, and the executive summary says a conclusion is
"unsupported in those respects" rather than "supported with caveats". Collapsing
those two states is the exact defect this package reports in eval sets, and it
was present in evallint's own reporting until now.

**Findings derived from the absence of evidence**, which is the part that does
not exist in other reporting tools. A judge metric the reliability module
REFUSED to compute becomes a finding, because an absent reliability measure is
not a passing one. So does an unidentifiable variance source: unmeasured is not
zero, and reporting 0.0 would claim a stability never measured. Both are
OBSERVED — whether a metric was computed is a fact about the design, not an
estimate.

**No threshold is invented at the reporting layer.** There is no "agreement too
low" finding, because setting that bar would be the arbitrary cutoff this report
exists to avoid. Computed metrics go into `stats` with the value, interval and
interpretation their own analyser wrote. Model comparison uses the analyser's own
`underpowered` / `no_information` verdicts, since a detected difference is a
result rather than a defect.

**Sections 1, 2, 3 and 11 are derived views**, computed from sections 4-10 rather
than written, so a finding cannot appear in "critical findings" phrased
differently from how it appears in its own section. A test asserts the unified
report and the legacy report contain exactly the same set of findings.

**Leakage shares section 7** with ground truth. The eleven requested sections
have no leakage entry and dropping the analyser would have lost information the
CLI already reports; both answer "can this case be scored as intended", since an
under-specified reference and an answer sitting in the prompt are two ways for
the answer to fail to test what you meant. The grouping is stated in the section
summary.

**Formats.** `terminal` truncates case ids and omits statistics, bounded by a
test at under 120 lines; `--detail` adds the prose fields. `json` carries
everything. `html` is one self-contained file using `<details>` — no CDN, no
framework, no JavaScript, because a report that fails to render when a CDN moves
is worse than a plain one. Case ids are HTML-escaped: they come from a user's
file, and a test asserts `<script>alert('x')</script>` as a case id cannot
execute.

`render_text` and `to_dict` gained a `schema_notes` argument in 0.12.0; the
legacy `--json` output is otherwise unchanged. 63 exports, up from 54.

### Fixed
- The three report adapters read `report.warnings` and `report.summary` via
  `getattr(..., default)`, but `ReliabilityReport`, `ComparisonReport` and
  `ReproducibilityReport` expose neither — they expose `notes`. Every one of
  those sections would have rendered as ASSESSED with zero findings while the
  underlying analysis had warnings, which is precisely the silent pass the
  module docstring rails against. The adapters now read each report's actual
  structure: refused metrics, unidentifiable variance sources, unstable
  rankings, and per-pair verdicts.

### Tests
38 new in `test_audit.py`, all deterministic — no sampling and no model call in
the report layer, so the same input gives byte-identical output. Two fixture
errors were corrected after running them: a leakage fixture whose answers were
under the detector's 12-character floor, and a truncation test whose largest
finding covered 2 cases so `max_cases=2` could never truncate — it would have
passed for the wrong reason.

## [0.12.0] — 2026-08-22

### Added

**Schema version 2.** Seven optional additions to the dataset representation:
multi-turn `messages`, `system` prompts, `acceptable` alternative answers,
structured `expected` plus `expected_schema`, `rubric` criteria, `tools`, and
`expected_tool_calls`. The four-field format is unchanged and remains the whole
schema for anyone who does not need the rest.

**`input` stayed a required string, and that is the design.** Every check reads
`case.input` as text — `len()` in imbalance, embedding in redundancy,
`normalise()` in five leakage detectors. So a chat case DERIVES `input` by
flattening its conversation, which means all eleven existing checks work on chat
data with no edits to any check. A one-turn conversation flattens to
byte-identical text, so converting a file cannot change a single finding, and
`input_is_derived` records that the text was synthesised rather than written.

**The `system` prompt is excluded from the flattened input.** It is usually
identical across every case, so folding it in would give every case a long
shared prefix: the redundancy check would report near-duplicates throughout and
the effective-size estimate would collapse toward one cluster, both as artefacts
of the flattening. A test builds two cases sharing a 20-fold-repeated system
prompt and asserts their inputs share no token at all.

**Versioning changes strictness, not features.** Rich fields are read whenever
they are well-formed, declared or not, because dropping a user's conversation
into metadata to be pedantic about a header line would be the silent data loss
this package exists to report. Undeclared, a `messages` value that is not a
conversation is kept as metadata with a note — real datasets do have a
`messages` column meaning something else. Declared `"evallint_schema": 2`, it is
an error, and a key one edit from a real field (`sytem`) is an error with a
suggestion. A future version is refused by name rather than misread. The version
is never inferred from the fields present.

**Errors are collected, not raised one at a time.** A 5000-case file with 40
mistakes takes one run to diagnose. Each issue carries the file, the line, the
JSON path inside the record (`messages[1].role`), the bad value, and what was
allowed. A consequence is not stacked on its cause: a case whose `messages`
failed to parse also has no `input`, and only the first is reported.

**MT-Bench loads.** Its prompt is a list of plain strings, which previously
exited 3 with "no column could be used as 'input'". A list of strings is now
read as successive USER turns — an interpretation, so it is reported every time
it is applied, and migration writes explicit role objects so the note stops.

**`evallint FILE --migrate-to OUT`** and `evallint.migrate_file`. Mostly worth
it to make an inferred column mapping permanent: GSM8K needs `--map
input=question` on every run, and after migrating the mapping is the identity.
Nothing is written unless the output reloads to cases identical field by field;
it refuses to overwrite without `--overwrite`, to write onto its own source, and
to write CSV. It deliberately does NOT rewrite simple cases as one-turn
conversations.

**Schema notes get their own report heading**, "How your file was read", rather
than being folded into "Not run". A field demoted to metadata changes what every
check above actually looked at, so filing it under a heading that says something
did not run would misdescribe it.

New: `Message`, `Role`, `ToolCall`, `ToolSpec`, `Rubric`, `Criterion`,
`derive_input`, `SchemaValidationError`, `ValidationIssue`, `MigrationReport`,
`migrate_file`, `EvalCase.as_dict`, `EvalCase.acceptable_answers`,
`EvalSet.load_notes`, `EvalSet.chat_cases`. 54 exports, up from 40.

### Fixed
- `check_version` raised outside the `LoadError` wrapper, so a single mistyped
  version line gave the CLI a traceback instead of `Error: ...`.
- The lenient demotion path said "kept as metadata" after the parser had already
  popped the key, so the value was actually gone — a message wrong in the one
  direction that matters, since the user would stop looking for their data.
- The affected-case count in a validation error was derived by splitting the
  issue path on `.`, so a source file named `x.jsonl` collapsed every issue into
  one bucket and the count was silently always 1.
- `mapping` had three loops over a hardcoded field list that had to agree; they
  now share `MAPPABLE`. When they disagreed, a mapped value could be overwritten
  by the column sharing its name.

### Tests
685, up from 613. The first section of `test_schema_v2.py` fixes version-1
behaviour in place — the four-field record, the four-argument constructor, the
existing aliases, and above all the exact TEXT each check sees — because the
feature is worthless if it breaks a file someone already has. `test_migrate.py`
proves every refusal leaves the destination absent, including by breaking
`as_dict` on purpose to exercise the verification path.

## [0.11.0] — 2026-08-22

### Added
- **`analyse_reproducibility`** — decomposes the instability of a repeated
  evaluation into three sources, reported SEPARATELY and never summed:

      model stochasticity      >= 2 runs, judge held fixed
                               -> lower temperature, or average more repeats
      evaluator stochasticity  >= 2 judges scoring the SAME run
                               -> fix the grader; re-running the model cannot help
      dataset sampling         bootstrap over cases within one run
                               -> add cases; re-running cannot help

  They are additive only under a variance-components model with independent
  effects, which these designs do not guarantee, and a single total would not
  tell you which fix to apply. `as_dict()` contains no total, and a test asserts
  that.

- **The design determines what is identifiable.** With one judge, evaluator
  variance is reported as NOT IDENTIFIABLE rather than zero -- reporting 0.0
  would claim the grader is noiseless. With one run, nothing about run-to-run
  stability is claimed, and the report states that this is not evidence of
  reproducibility.

- **Aggregate and per-case stability are reported separately**, and their
  divergence is called out. A test constructs an eval whose headline score has
  standard deviation EXACTLY zero while 100% of its individual verdicts flip
  between runs: independent flips cancel in an average, so that is a coincidence
  rather than reproducibility.

- **Model ranking stability**: modal ranking and its share of runs, distinct
  ranking count, top-model stability, Kendall's W, and which model pairs swapped
  order. Three models two percentage points apart produce a different ranking in
  most runs, and the analyser says so instead of reporting whichever order this
  run gave.

- **`kendalls_w`** added to `agreement`. No tie correction, so a reported value
  is a LOWER bound on agreement when ties are present -- conservative, and stated
  wherever it is used.

- 40 exports, up from 38.

### Fixed
- A real bug in the sampling-variance path: it read `resamples` out of the
  bootstrap's detail dict and called `len()` on an int. The sample size is now
  the number of cases bootstrapped, which is the figure that actually governs
  the interval.

### Tests
27 new, in the two required kinds. DETERMINISTIC tests construct outcomes by
hand so expected values are exact -- including designs where one variance source
is provably zero, which is how the separation is verified: a deterministic model
with disagreeing judges must show model variance of exactly 0.0 and all the
instability in the evaluator term. SEEDED STOCHASTIC tests fix the seed so a
genuinely random model or judge cannot make the suite flake.

Kendall's W is checked against a hand-computed case (W = 0.75 for one adjacent
swap over three items, with the arithmetic in the docstring).

One test bound was corrected after running it. It asserted more than 10
flip-prone cases and the observed count was exactly 10. For a coin-flip model
over 5 runs, flip-prone needs min(passes, fails) >= 2, which for Bin(5, 0.5) has
probability (C(5,2) + C(5,3))/32 = 0.625, so the expectation over 20 cases is
12.5 and 10 is well within noise. The bound is now derived from the binomial
rather than eyeballed.

## [0.10.0] — 2026-08-22

### Added
- **`estimate_effective_size`** — distinguishes the raw number of cases from the
  approximately independent information they contain, using the redundancy
  check's clustering.

      Raw cases: 500
      Semantic clusters: 396
      Large clusters (>= 3 cases): 27
      Potential redundancy: 21%
      Redundancy-adjusted scenario coverage: ~396 distinct scenarios
                                             (estimate; plausible range 396-500)
      Design effect: 1.26 -- intervals computed on 500 cases are roughly
                            1.12x too narrow

- **It is NOT presented as a statistical effective sample size**, because the
  method does not support that claim. The terminology throughout is
  "redundancy-adjusted coverage", "effective scenario count" and
  "independent-case estimate", and the output states in capitals that it is an
  estimate.

  The reasoning is disclosed rather than buried. Kish's effective sample size

      n_eff = (sum w_i)^2 / sum w_i^2

  reduces to exactly the cluster count when each case is weighted by the inverse
  of its cluster size -- but only BECAUSE that weighting assumes within-cluster
  correlation of 1. Two paraphrases are not perfectly redundant: a model can
  pass one and fail the other. The reduction is an assumption, not a derivation,
  and a test asserts that the caveat saying so is present.

- **The result is a bracket, not a number.** Lower bound = cluster count
  (assumes rho = 1), upper bound = raw count (assumes rho = 0), truth between.
  The lower bound is the headline because it cannot flatter the eval.

- **Design effect** as the actionable output: the factor by which intervals
  computed on the raw count are too narrow. A `compare_models` interval should
  be widened by roughly its square root when the eval has clusters, since every
  interval in this library assumes independent cases.

- **Threshold sensitivity**, with the embedder memoised so four thresholds cost
  one embedding pass rather than four. A test asserts the embedder is called
  exactly once.

- `estimate_from_clusters` is pure and takes cluster sizes, so the whole
  analysis is testable with no model at all. 38 exports, up from 37.

### Measured on the four scenarios
      fully unique      raw=12  scenarios=12  redundancy= 0%  DEFF= 1.00
      exact duplicates  raw=12  scenarios= 9  redundancy=25%  DEFF= 1.33
      highly redundant  raw=20  scenarios= 2  redundancy=90%  DEFF=10.00
      templated         raw=12  scenarios= 2  redundancy=83%  DEFF= 6.00

The templated case needs no embedder: the template detector is exact.

### Tests
22 new, covering all four required scenarios plus the terminology constraints --
that the output refuses the effective-sample-size claim, that the Kish reduction
is disclosed as an assumption, that the connected-components bias is declared as
running in the CONSERVATIVE direction, and that nothing describes redundancy as
a defect.

One test arithmetic error was corrected after running it: clusters of 3 and 2
over 20 cases leave 15 singletons, so 17 scenarios rather than 16.

## [0.9.0] — 2026-08-22

### Added
- **`compare_models`** — paired statistical comparison of two or more models on
  a shared eval set. Cases are paired, so the tests are paired: only the
  DISCORDANT cases carry information about which model is better, and
  independent-sample tests would throw the pairing away, inflate the variance
  and hide real differences.

  Reports per-model Wilson intervals, the paired difference with a paired
  bootstrap interval, an exact McNemar test, three effect sizes, a minimum
  detectable effect, and a verdict.

- **Statistical and practical significance are separate verdicts**, in a 2x2:

                          | effect >= threshold | effect < threshold
      significant         | real and meaningful | real but too small
      not significant     | UNDERPOWERED        | no meaningful difference

  `underpowered` is the cell people misread as "no difference": the eval saw an
  effect worth caring about and could not resolve it. The answer is more cases,
  not a conclusion. `no_information` is separate again -- two models that never
  disagreed tell you nothing, which is not evidence of equivalence.

- **Three effect sizes**, because the obvious one misleads alone. The risk
  difference shrinks as models agree more, so the same disagreement looks
  smaller on an easy eval; Cohen's g and the odds ratio do not. Verified by a
  test: the same 2-vs-18 split gives a 4x larger risk difference on a 100-case
  set than a 400-case one, with identical g and odds ratio.

- **The practical threshold is the user's.** The 2pp default is a placeholder
  with no derivation, is labelled as one, and is reported in every result. A
  test asserts that changing it moves the verdict while leaving the p-value and
  effect size untouched.

- **Holm-adjusted p-values** whenever more than one pair is compared. Four
  models is six tests, and six unadjusted tests at alpha=0.05 carry roughly a
  26% chance of at least one false positive.

- New statistics in `separation`: `mcnemar_odds_ratio` (log-scale interval,
  Haldane correction applied and disclosed when a cell is zero), `cohens_g`,
  `paired_bootstrap_difference` (resamples CASES, keeping each case's two
  outcomes together), `holm_adjust`.

- 37 exports, up from 36.

### The brief's example, reproduced
      A: 87.1% (871/1000, 95% CI 84.9%-89.0%)
      B: 88.4% (884/1000, 95% CI 86.3%-90.2%)
      difference +1.3 pp, 95% CI (paired bootstrap) [+0.1, +2.4] pp
      McNemar exact p=0.041 on 35 discordant of 1000
      Cohen's g 0.19 (medium), odds ratio 2.18
      -> real, but below a 2pp threshold: detectable without being worth acting on

A p-value-only report would have said "significant" and stopped.

### Tests
33 new, covering the five required scenarios -- equal models, small samples,
large differences, borderline differences, highly imbalanced datasets -- plus
multiple comparisons and the reporting contract. Contingency tables are built
exactly so every expected value is derivable.

One test was corrected after running it: it claimed to check the case where the
Wald interval and the exact test disagree, but used n=40, where they agree. The
phenomenon needs the discordant cases to be a large share of n (3 of 12, not 3
of 40). The corrected test records why.

## [0.8.0] — 2026-08-22

### Added
- **`EvaluatorReliability`** — measures whether the GRADER is trustworthy.
  Every other check audits the dataset; this audits the judge, which is the one
  place an LLM judge is legitimately involved, because it is the object of
  measurement rather than an oracle.

  Seven measures: intra-judge consistency, inter-judge agreement, position bias,
  order sensitivity, score variance, binary decision stability, judge
  correlation.

- **`evallint.checks.reliability_stats`** — Krippendorff's alpha (nominal and
  interval), Pearson, Spearman, percentile bootstrap intervals. Standard library
  only. Alpha is used rather than Fleiss' kappa where the data is ragged: a
  judge that errors on some cases leaves a different number of ratings per item,
  which Fleiss cannot take.

- **Every result is a `MetricResult`**, carrying metric, value, sample size,
  interval, interpretation and limitations. A reliability number without its
  sample size and assumptions is not interpretable, and this module exists to
  judge the judges.

- **evallint never calls your judges.** `collect_verdicts` and `collect_choices`
  invoke callables you supply; `analyse` takes plain `JudgeObservation` records
  and touches no provider. Same contract as the injected scorer.

- 36 exports, up from 33.

### Assumptions are gated, not assumed
`value: None` with a stated reason is a normal outcome:

      Pearson on a constant series        refused -- r is 0/0, not 0.0
      Spearman on binary verdicts         refused -- 100% ties
      alpha with one category in use      refused -- undefined, not 1.0
      bootstrap from < 10 observations    refused -- resampling its own noise
      one repeat per item                 intra-judge consistency refused, and
                                          stated as NOT evidence of consistency
      ordinal alpha                       refused rather than approximated

Returning 0.0 in those cases would claim the judges were no better than chance;
returning 1.0 would claim perfection. Both are inventions.

### The demonstration
Two scoring judges whose values differ by a constant offset:

      Pearson correlation : 0.939
      Cohen's kappa       : 0.200
      Krippendorff alpha  : 0.025

A suite reporting only correlation would call them excellent. They agree barely
above chance. Correlation is not agreement, and both are reported.

### Tests
32 new, built on judges with KNOWN behaviour so expected values are derivable
rather than recorded: a deterministic judge must score 1.0 on consistency; a
coin-flip judge at 5 repeats has P(unanimous) = 2 x 0.5^5 = 0.0625 so must score
near zero; a judge that always picks the first-presented option must show
position bias 1.0 and order sensitivity 1.0. Krippendorff's alpha is checked
against a hand-computed coincidence matrix (alpha = 0.125, with D_o and D_e
asserted separately).

Not wired into the CLI: its input is judge observations, not a dataset.

## [0.7.0] — 2026-08-22

### Added
- **`GroundTruthCheck`** — seven deterministic detectors for references that
  may not be well enough defined for a score against them to mean anything:
  `multiple_valid_answers`, `subjective_question`, `missing_criteria`,
  `ambiguous_wording`, `reference_inconsistency`, `underspecified_reference`,
  `multiple_interpretations`.

  Every finding reports case_id, ambiguity_type, evidence, confidence and a
  recommended action. No recommended action is ever "delete". At dataset level:
  proportion potentially ambiguous, judge agreement, and the cases needing
  human review.

- **A subjective case is not an invalid case.** "Is this response helpful?"
  measures preference, which is a legitimate thing to measure. The finding says
  the score is grader-dependent. A test asserts no finding contains an
  affirmative accusation ("is invalid", "is broken", "delete this") -- and
  requires the denials to be present, since that is the actual requirement.

- **Optional multi-judge analysis, built to distrust judges.** Off unless
  judges are passed explicitly, and evallint never calls a provider -- you
  supply the callable, as with the scorer and embedder.
    * one judge  -> LOW confidence, labelled single_judge_uncorroborated
    * two or more -> only cases they AGREE on are reported; disputed cases are
      routed to human review, because disagreement is evidence the case is hard
      to adjudicate rather than evidence about its answer
    * agreement reported with Fleiss' kappa, never raw agreement alone
    * a judge that raises is recorded and the deterministic results survive

- **`evallint.checks.agreement`** — Cohen's kappa, Fleiss' kappa, raw and
  pairwise agreement. Standard library only, exported at package level. Handles
  the kappa paradox honestly: when every rater uses one category throughout,
  kappa is 0/0 and the functions return None. Reporting 1.0 there would claim
  perfect reliability from raters who never made a distinction.

- 33 exports, up from 32. `GroundTruthCheck` runs in the CLI (deterministic
  detectors only -- judges cannot be configured from a command line without
  locking the tool to one provider).

### Fixed
Both found by running against the four real public datasets rather than by
reasoning about the code:

- **Three false contradictions on MMLU.** Reference inconsistency used the
  aggressive normaliser, which strips punctuation, so

      Q(sqrt(2) + sqrt(3)) -> 1
      Q(sqrt(2), sqrt(3))  -> 1
      Q(sqrt(2)*sqrt(3))   -> 2

  all collapsed to one string and the check reported a HIGH-confidence WARNING
  that the dataset contradicts itself. Those are three mathematically distinct
  field extensions. Comparison now keeps punctuation. The cost is stated: "What
  is 2+2?" and "What is 2 + 2" are no longer compared, because a missed
  inconsistency is a gap while a fabricated one is a confident wrong answer.

- **101 warnings on HellaSwag.** It has no reference answers at all, so
  `missing_criteria` fired on 100 of 100 cases. Correct detection, useless
  presentation: that is one fact about the file. Any single ambiguity type
  affecting most of the set is now reported once.

## [0.6.0] — 2026-08-22

### Added
- **Leakage analyser: seven deterministic detectors**, up from one. `NO LLM
  JUDGE is used at any confidence level` — every detector is a deterministic
  function of the text, because a judge would make the result depend on a
  model's opinion and two findings in this project were already retracted for
  exactly that.

      expected_in_input       moderate   answer verbatim in the input
      label_in_input          moderate   class label as a word in the input
      explicit_reveal         HIGH       "the correct answer is X", X = answer
      answer_in_instructions  HIGH       answer in the preamble, pre-marker
      metadata_leak           moderate   answer in a non-answer metadata field
      fewshot_target          HIGH       target inside the case's own demos
      high_overlap            low        token overlap -- OPT-IN

- **Per-finding reporting**: case ID, leakage type, evidence, confidence,
  severity and an explanation, in `stats["potential_leaks"]` and in the
  rendered findings.

- **Confidence is ordinal with a stated basis, not a probability.** Attaching
  0.87 to "this looks like leakage" would invent precision that does not exist.
  HIGH means hard to explain innocently; MODERATE means the observation is
  certain but its interpretation is not — the answer appearing in the input is
  exactly what extractive QA looks like. Only HIGH warns.

- **`--leakage-overlap` / `overlap=True`** for the seventh detector, off by
  default for a measured reason: on GSM8K, answer/input overlap reaches 100%
  with NO leakage, because reference answers restate the question, and a
  planted leak scores the same. A test deliberately asserts that false positive
  so the reason stays documented rather than becoming folklore.

- Terminology is "potential leakage" throughout, and a test asserts no finding
  contains "invalid", "bad case", "must be removed" or similar.

### Fixed
- **100 false positives on TruthfulQA.** The metadata detector exempted
  answer-bookkeeping fields by exact key name, so `correct_answers` — which
  naturally contains `best_answer` — was not exempt, and every one of 100 cases
  was flagged. The exemption now matches substrings. Found by running the check
  against the four real public datasets rather than by reasoning about it.
  Validated at **zero false positives** on GSM8K, MMLU, TruthfulQA and
  HellaSwag.

### Compatibility
- `LeakageCheck`'s existing stats keys are unchanged and detector 1 behaves
  exactly as before, including its short-answer and closed-vocabulary guards.
  One existing test was updated: its fixture text "Hint: the correct label is
  ..." is now caught by two detectors rather than one, which is correct.

## [0.5.0] — 2026-08-22

### Added
- **`RedundancyCheck` — semantic redundancy at five levels**, superseding the
  narrower `DuplicateCheck` in the CLI. Only ONE of the five uses a cosine
  threshold, which is the point: a single tuned number should not be the ground
  truth for whether two cases are the same.

      exact        byte-identical comparison text            DUPLICATE
      normalized   identical after case/space/punct fold     DUPLICATE
      template     identical after masking numbers and       DUPLICATE
                   quoted spans
      scenario     cosine >= threshold AND same expected     SIMILAR
      semantic     cosine >= threshold                       SIMILAR

  `DUPLICATE` and `SIMILAR` are separate claims. The first three are
  comparisons with nothing to tune; the last two depend on a model and a
  threshold and are labelled heuristic in the output, the stats and the
  severity.

  **Measured justification for multiple mechanisms.** On the reference model,
  genuine paraphrases span cosine 0.599 / 0.659 / 0.860 — no single threshold
  has both good precision and good recall over that range. And "What is 5 plus
  3" vs "What is 12 plus 7" scores 0.717, *below* the 0.85 default, so the
  semantic level misses it entirely while the template level catches it exactly
  and without a threshold.

- **Threshold sensitivity curve** on every run: how many similar pairs would be
  found at 0.80 / 0.85 / 0.90 / 0.95, and whether that count is stable. If it
  is, the threshold is not load-bearing for your data; if it swings, you have
  been told rather than left to assume. On the bundled example it is stable
  (5, 5, 5, 4).

- **Scenario weighting.** Per cluster: size, mean and range of pairwise
  similarity, the share of aggregate weight it carries, excess weight from
  redundancy, and a HIGH/MEDIUM/LOW band. The bands are conventions and the raw
  share is always reported so you can apply your own; the limitations say the
  principled anchor is the eval's own resolution.

- **Field selection.** `compare` defaults to `input+expected`, because the same
  question with two different reference answers is a ground-truth contradiction
  rather than a duplicate and collapsing them would hide it.
  `--compare input` restores the older input-only behaviour, and
  `input+expected+metadata` is also available.

- **Runs without the embeddings extra.** The three deterministic levels need no
  model, so a core install still gets useful redundancy analysis. `levels_run`
  reports which detectors were available, so a missing embedder narrows the
  audit visibly rather than silently.

- `CompareFields`, `RedundancyCheck`, `RedundancyLevel` exported. 32 exports,
  up from 29.

### Changed
- The CLI now runs `RedundancyCheck` in place of `DuplicateCheck`, and the
  report section is named `redundancy`. `--duplicate-threshold` keeps its name
  and meaning; `--compare` is new.
- No finding says a case is invalid, and a test asserts the absence of
  "invalid", "bad case", "useless", "delete these" and similar. Every cluster
  message ends by noting the redundancy may be intentional.

- **`CheckResult.partial`** — reasons a check ran but could not run FULLY.
  Additive, defaults empty, so existing checks are unaffected. The CLI treats a
  non-empty `partial` exactly like a check that raised: exit 3, not 0.

  This was needed because the upgrade would otherwise have LOOSENED the exit
  contract. Previously a core install exited 3 because `DuplicateCheck` raised.
  `RedundancyCheck` succeeds in degraded mode, which would have exited 0 — a
  narrowed audit reporting a clean gate, the exact failure this project reports
  in eval sets. `--skip-duplicates` still exits 0, because that is the caller
  choosing scope rather than a capability being absent.

### Compatibility
- `DuplicateCheck` is unchanged and still exported, for anyone relying on
  input-only semantic detection. Its 21 tests are untouched.
- Two CLI tests monkeypatched `evallint.cli.DuplicateCheck` and were repointed.

### Fixed
- The embedder call previously caught bare `Exception` and degraded silently.
  That swallowed a genuine bug during development and reported a narrower audit
  as a complete one. Only `ImportError` degrades now; anything else propagates
  and the CLI records the run as incomplete.

## [0.4.0] — 2026-08-22

### Changed
- **The discrimination check is now a multi-model separation analysis.** It no
  longer answers "do these two models agree?" but "how much evidence about
  model separation does this eval carry, and between which capability levels?".

  Five per-case outcomes, and **none of them is a defect label**: `ceiling`,
  `floor`, `separating`, `non_monotonic`, `indeterminate`. A ceiling case may be
  a deliberate regression guard, so the wording throughout is now "provides
  limited evidence for model separation" — a claim about evidence, not quality.
  The words "too easy" are gone.

- **Majority verdicts instead of unanimity, which fixes a measurement bias.**
  The previous version classified a case only when every model's verdict was
  unanimous across repeats and discarded the rest. That filter is not neutral:
  unanimity is likelier when a model's success probability is extreme, so
  survivors were enriched for the ceiling category. Measured on unchanged data,
  the reported share moved 0.52 -> 0.65 -> 0.74 -> 0.89 as repeats went
  1 -> 2 -> 3 -> 5, while the surviving population collapsed from 100 cases to
  9. The check's own advice was to raise repeats before trusting the figure, so
  following it made the number worse.

  Every scored case is now classified and the denominator is always `n_cases`.
  Raising `repeats` moves cases out of `indeterminate` rather than deleting
  them.

### Added
- **Per-pair separation with uncertainty.** Every model pair is reported, not
  only adjacent ones, with the paired difference, an interval, an exact McNemar
  p-value, and a minimum detectable effect. `separated`,
  `ordering_contradicted`, `unresolved`, `no_information`. The last is distinct
  on purpose: two models that never disagreed provide no evidence of a
  difference, which is not evidence there is none.
- **`evallint.checks.separation`** — Wilson intervals, exact McNemar, paired
  difference, minimum detectable effect. Standard library only. Exported at
  package level (`wilson_interval`, `mcnemar_exact_p`,
  `minimum_detectable_effect`) so any number the check reports can be
  re-derived rather than taken on trust.
- `CaseClass` and `CaseSeparation` exported. 29 exports, up from 25.
- New stats keys: `case_classes`, `n_by_class`, `case_ids_by_class`,
  `limited_evidence_share` and its interval, `separating_share` and its
  interval, `n_statistically_supported`, `model_pairs`, `separated_pairs`,
  `contradicted_pairs`, `unresolved_pairs`, `confidence_level`.
- `docs/methodology-discrimination.md` — the full derivation, every threshold
  with its justification, and what the analysis cannot conclude.

### Fixed
- Wilson interval bounds are now exact at p = 0 and p = 1, where the algebra
  gives 0 and 1 identically and floating point gave 0.9999999999999999.

### Compatibility
- `n_measured`, `n_unstable`, `non_discriminating_share`, `effective_n_cases`
  and every `*_case_ids` key are computed exactly as before and retained, so
  existing JSON consumers see no change. They carry the selection bias above:
  **do not compare them across different `repeats` values.** Use
  `limited_evidence_share`, whose denominator is fixed.
- The scorer contract is unchanged: `scorer(case, model) -> bool | float`.
- Finding wording and the summary line changed. Three existing tests asserted
  on the old text and were updated.

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

[0.13.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.13.0
[0.12.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.12.0
[0.11.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.11.0
[0.10.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.10.0
[0.9.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.9.0
[0.8.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.8.0
[0.7.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.7.0
[0.6.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.6.0
[0.5.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.5.0
[0.4.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.4.0
[0.3.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.3.0
[0.2.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.2.0
[0.1.0]: https://github.com/BhagyasriUddandam/evallint/releases/tag/v0.1.0
