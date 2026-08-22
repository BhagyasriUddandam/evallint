# evallint v2 architecture — an evaluation-validity auditor

**Status: design proposal. Nothing here is implemented.**

## The principle

> evallint assesses whether an LLM evaluation can support trustworthy
> conclusions, rather than declaring individual dataset examples good or bad.

The unit of judgement moves from the **case** to the **conclusion**. v1 answers
"is case 31 a duplicate?". v2 answers "can this eval distinguish model A from
model B, and at what resolution?".

Worked example, computed from the GSM8K run already in this repo
(`docs/findings-gsm8k.md`, 100 cases, 3 repeats, no LLM judge):

```
observed        Haiku 4.5 98.0%   Opus 5 98.7%      gap +0.7 pp
paired analysis 98 both pass, 1 weak-only, 1 strong-only
                discordance 2.0%
                gap 95% CI  [-2.8 pp, +2.8 pp]
                minimum detectable effect, 80% power  4.0 pp

conclusion this eval LICENSES:    "the two models differ by less than 4 pp,
                                   or not at all"
conclusion it CANNOT LICENSE:     "Opus 5 is better than Haiku 4.5 here"
```

The second block is the product. No score appears anywhere.

## What this architecture explicitly rejects

**A composite trust score (e.g. `78/100`).** It would require weighting
incommensurable quantities — a duplicate rate, a κ, a confidence interval width
— with weights that have no defensible derivation. The resulting number is
precisely the artefact this tool exists to warn about: a single confident
figure that looks fine and hides everything underneath. `CheckResult` already
refuses to exist without limitations; a composite score has no honest
limitations text, because the honest text is "the weights are invented".

Per-dimension status is reported. The collapse to one number is refused, and the
refusal is stated in the output so it reads as a position rather than an
omission.

---

# The evidence ladder

Every statement evallint emits carries exactly one tier. This is the core
architectural device: it prevents a heuristic from being read as a measurement.

| tier | definition | uncertainty | may license a conclusion? |
|---|---|---|---|
| **`OBSERVED`** | A census of the data. Counting, not inference. "14 of 20 cases have label `billing`." | none — exact | yes, about *this dataset* only |
| **`HEURISTIC`** | A signal produced by a chosen threshold with no sampling theory behind it. "cosine ≥ 0.85 ⇒ near-duplicate." | threshold provenance + sensitivity range | **no**, alone |
| **`ESTIMATED`** | A quantity with a stated estimator, an interval, and named assumptions. "non-discriminating share 0.52, 95% CI [0.42, 0.62], assuming per-case independence." | interval + assumption list | yes, within stated assumptions |
| **`RECOMMENDED`** | An action. Must cite the specific claims it rests on and their tiers. | inherits the weakest cited tier | n/a — it *is* the advice |

Two hard rules:

1. **No promotion.** A `HEURISTIC` claim may never be reported as `ESTIMATED`,
   even when it is numeric. Cosine 0.85 does not become statistical by having
   decimals.
2. **Tier inheritance.** A `RECOMMENDED` claim resting on any `HEURISTIC`
   evidence is labelled as heuristic-backed. This is what stops "delete these
   8 cases" from acquiring authority it hasn't earned.

## Type architecture, and backward compatibility

Existing public API is preserved exactly. `CheckResult(check, summary,
findings, stats, limitations)` keeps its signature, its `warnings` property, and
its "raises without limitations" invariant. All four v1 checks continue to work
unmodified.

Additive change:

```
Claim
  tier          : Tier                    OBSERVED|HEURISTIC|ESTIMATED|RECOMMENDED
  dimension     : Dimension               one of the eight below
  statement     : str                     what is claimed, in words
  quantity      : Quantity | None         value + interval + estimator name
  assumptions   : tuple[str, ...]         empty only for OBSERVED
  evidence      : tuple[str, ...]         ids of the claims this rests on
  case_ids      : tuple[str, ...]         supporting cases, never truncated
  licenses      : tuple[str, ...]         conclusions this claim supports
  forbids       : tuple[str, ...]         conclusions it does NOT support

Quantity
  value, interval (lo, hi), level (e.g. 0.95), estimator, n, n_effective

CheckResult                               UNCHANGED, plus:
  claims        : tuple[Claim, ...] = ()   defaults empty -> v1 checks unaffected
```

`findings` remains the rendering surface. When a check emits `claims` and no
`findings`, the report derives findings from claims (`ESTIMATED`/`OBSERVED`
whose interval excludes the acceptable region ⇒ `WARNING`; `HEURISTIC` ⇒
`INFO`). A v1 check emitting only `findings` renders as it does today.

`Severity` gains `CRITICAL` for defects that make the audit itself
uninterpretable (malformed data, contradictory ground truth). `--fail-on`
extends to `{never, critical, warning, any}` with `warning` keeping its current
meaning, so existing CI invocations do not change behaviour.

---

# The eight dimensions

Cross-dimension dependencies are real and are declared, because several
dimensions are unsound when computed in isolation:

```
1 integrity ──────► everything            (nothing is valid on malformed data)
2 redundancy ─────► 6 statistical         (n_effective, not n, sets interval width)
3 discrimination ─► 4 ground truth        (inversions are reference-quality evidence)
                 └► 8 difficulty          (per-case success rates are the input)
5 judge ──────────► 3, 4, 6               (an unreliable judge invalidates all three)
7 reproducibility ► 6                     (run variance widens every interval)
```

---

## 1. Dataset integrity

**Measures** conformance of the file to a schema that can be audited at all:
required fields present, types consistent, encoding sound, ids unique,
structural shape supported (single-turn vs `messages`).

**Definition.** A census, so rates not estimates. For field *f*:
completeness *C_f* = |{i : f present and non-empty}| / n. Type consistency
*T_f* = share of the modal type for *f*. No interval — this is the whole
population.

**Inputs** the raw file only. No model, no embedder, no cost.

**Output** per-field completeness and type-consistency tables; the resolved
column mapping (already implemented in `mapping.py`); a list of unauditable
records with reasons.

**Uncertainty** none. This tier is `OBSERVED` throughout.

**Can conclude** that specific fields are absent, ragged, mistyped, or that ids
collide; which downstream dimensions are therefore unavailable.

**Cannot conclude** anything about whether present values are *correct*. A file
can be perfectly conformant and entirely wrong.

**Failure modes** column mapping picks a plausible wrong column (mitigated:
`mapping.py` refuses ambiguity and always reports its choice); a valid
alternative shape is rejected as malformed (current state for multi-turn —
MT-Bench exits 3); silent coercion, e.g. label `1` and `"1"` merging (currently
deliberate and documented in `schema.py`).

**Warnings** `CRITICAL` when a required field is missing for any record, since
every other dimension is then computed on a subset the user didn't choose.
`WARNING` when completeness is partial, naming the affected ids.

**Tests** golden files per format; ragged JSONL where field sets differ across
rows; BOM'd CSV (regression — this silently renamed every case once); duplicate
ids; `messages` shape accepted rather than rejected; mapping ambiguity raising
rather than guessing.

---

## 2. Redundancy

**Measures** how many *independent* scenarios the eval contains, and how much
weight any single scenario carries in the aggregate.

**Definition.** Weight each case by the inverse of its cluster size,
*w_i = 1/|C(i)|*, then use Kish's effective sample size

  *n_eff = (Σ w_i)² / Σ w_i²*

which for hard clusters reduces to the cluster count. The design effect
*DEFF = n / n_eff* is exported for dimension 6. Maximum scenario weight
*W_max = max_k |C_k| / n* bounds how much one rephrased question can move the
headline number.

**The threshold problem, and the fix.** v1 picks τ = 0.85, measured on one
20-case set — the weakest threshold provenance in the codebase. v2 reports
*n_eff(τ)* as a **sensitivity curve** over τ ∈ {0.80, 0.85, 0.90, 0.95}. If
n_eff is stable across that range the threshold is not load-bearing and the
conclusion is robust; if it swings, the finding is reported as
threshold-dependent and the range is shown. This converts an arbitrary constant
into a stated sensitivity.

**Inputs** case inputs; an embedder (injected — contract unchanged).

**Output** clusters with pairwise similarity ranges; n_eff(τ) curve; DEFF;
W_max; contradiction set (see dimension 4).

**Uncertainty** `HEURISTIC` by construction. Never `ESTIMATED`: there is no
sampling model for "these two texts mean the same thing".

**Can conclude** that the eval has at most n_eff independent scenarios at a
stated τ, and that intervals computed on n are too narrow by √DEFF.

**Cannot conclude** that any cluster member is redundant *for the user's
purpose* — duplicate inputs with different expected values may be deliberate
consistency tests, and near-duplicates may be intentional regression guards.

**Failure modes** transitive chaining groups cases below τ (verified: members
at cosine 0.62 clustered because each is 0.90 to a shared neighbour) — v2 must
report the cluster's minimum pairwise similarity beside n_eff, and offer
single-linkage vs complete-linkage as an explicit choice rather than
hard-coding connected components; input truncation at the embedder's token
limit makes long documents that differ only in their endings look identical;
formulaic domains (SQL, legal boilerplate) push unrelated cases above any τ;
O(n²) memory — 400 MB at n=10,000, 10 GB at n=50,000, currently unguarded.

**Warnings** `WARNING` when W_max exceeds a stated share; `INFO` with the
n_eff(τ) curve always; `CRITICAL` never — redundancy degrades a conclusion, it
does not invalidate the data.

**Tests** transitive-chain characterisation (pins the 0.62 grouping before any
linkage change); n_eff(τ) monotonicity in τ; DEFF = 1 exactly when all cases
are singletons; complete- vs single-linkage divergence on a constructed set;
pre-flight guard fires before allocating at large n; deterministic fake embedder
for all logic, real model for two end-to-end paths (existing pattern).

---

## 3. Model discrimination

**Measures** whether the eval can separate models of differing capability, and
at what resolution.

### The v1 defect this dimension exists to fix

v1 classifies a case only if every model's verdict is **unanimous** across
repeats, and discards the rest. That filter is not neutral with respect to the
outcome it measures. Measured on a synthetic 100-case set with a weak model at
p = 0.6 and a strong model at p = 1.0:

| repeats | n_measured | non_disc_share | n_ceiling |
|---|---|---|---|
| 1 | 100 | 0.52 | 52 |
| 2 | 55 | 0.65 | 36 |
| 3 | 27 | 0.74 | 20 |
| 5 | 9 | **0.89** | 8 |

For p = 0.6, P(5 unanimous passes) = 0.078 while P(5 unanimous fails) = 0.010,
so survivors are enriched for "weak model consistently passes" — the ceiling
category. **Raising `repeats` inflates the headline share.** v1's own
limitations instruct users to run 3+ repeats before trusting any figure, so
following the advice worsens the bias.

### v2 definition

No case is ever discarded. For case *i*, model *m*, *r* repeats yielding
*k_{i,m}* successes, estimate *p̂_{i,m} = k/r* with a Wilson interval. Classify
using intervals, with **`INDETERMINATE` as a first-class outcome that keeps its
count in the denominator**:

| class | condition (ordered weak → strong) |
|---|---|
| `CEILING` | lower bound of min_m p̂ > pass threshold |
| `FLOOR` | upper bound of max_m p̂ < pass threshold |
| `DISCRIMINATING` | CI for p̂_strong − p̂_weak excludes 0, expected sign |
| `INVERTED` | CI excludes 0, wrong sign |
| `INDETERMINATE` | intervals overlap — *r* is too small to resolve this case |

The denominator is always *n*. Increasing *r* moves cases out of
`INDETERMINATE` into resolved classes; it does not change the population. The
bias above disappears because nothing is filtered.

An honest consequence to state loudly: at *r* = 3 most per-case classifications
are `INDETERMINATE`, because three Bernoulli trials cannot resolve a per-case
difference. That is a true statement about what three repeats support, and
reporting it is the point.

Set level: discriminating share with a binomial interval over cases, **widened
by √DEFF from dimension 2**, since clustered cases are not independent
observations.

**Inputs** injected `scorer(case, model) -> bool | float` (contract unchanged),
two or more models ordered weakest first, *r* ≥ 1, optional `sample`.

**Output** per-case class with interval; set-level shares with intervals;
per-model pass rates with Wilson intervals; the `INDETERMINATE` count as a
statement about resolution.

**Uncertainty** `ESTIMATED`, assumptions: repeats are i.i.d. per (case, model);
the declared capability ordering is correct; scorer is deterministic given the
model output.

**Can conclude** that a stated share of cases carries no evidence about which
model is better, at a stated resolution; that the eval behaves as a set of size
n_eff × (1 − non-discriminating share).

**Cannot conclude** that a non-discriminating case is a *bad* case — ceiling
cases may be deliberate regression guards. Cannot verify the declared ordering
is right; it can only report that the numbers contradict it. Cannot generalise
beyond the chosen model pair: two models closer in capability than assumed make
a good eval look non-discriminating, and the check cannot tell the difference.

**Failure modes** wrong declared ordering inverts the inversion finding;
non-thread-safe user scorers corrupt results silently at `max_workers > 1`
(already documented); a response cache defeats `repeats` entirely unless the
attempt index is in the cache key (this bug shipped once — see
`scripts/gsm8k_discrimination.py`); `pass_threshold` collapses 0.95 vs 0.55 to
"both pass".

**Warnings** `CRITICAL` if `repeats = 1` and any per-case conclusion is being
reported, since nothing can be known about stability. `WARNING` when
`INDETERMINATE` exceeds a stated share — the message being "this run cannot
resolve most cases; raise r or accept set-level claims only".

**Tests** the repeats-bias table pinned as a characterisation test, so any fix
must consciously change it; `INDETERMINATE` counted in the denominator at every
r; denominator invariant to r; majority vs unanimity divergence on the real
GSM8K cache (100 cases retained vs 27); exact call-count bounds under
`sample`/`repeats`/`max_workers`; inversion semantics with 4 models; float
scores thresholded not ranked.

---

## 4. Ground-truth quality

**Measures** whether reference answers can bear the weight of being called
correct.

**Definition — three signals, ordered by strength.**

*Contradiction rate* (`OBSERVED`, free, currently unmeasured). Within
near-duplicate clusters from dimension 2, the share whose `expected` values
disagree:
  *R_contra = |{clusters with ≥2 distinct expected}| / |clusters|*
Near-identical inputs with different reference answers are either a deliberate
consistency test or a labelling error, and the eval cannot be scored coherently
until the user says which. This is the strongest ground-truth signal available
without a model.

*Leakage* (`HEURISTIC`, implemented in `leakage.py`). Verbatim containment of
`expected` in `input`, guarded by minimum answer length and closed-vocabulary
detection. Measured zero false positives across GSM8K, MMLU, TruthfulQA,
HellaSwag. Note the discarded signal: token overlap reaches 1.00 on GSM8K with
no leakage, because reference answers restate the question — no threshold
separates that from a real leak, so no such finding is emitted.

*Inversion evidence* (`ESTIMATED`, consumes dimension 3). A case where a weaker
model reliably passes and a stronger one reliably fails is evidence about the
reference, not the models. Real instance from this repo: GSM8K `case_13` asks
"how many years before he starts earning money"; the reference says 13
(break-even at 12, profit in 13), Opus 5 answered 12 in all three runs. Both
readings are defensible. That is ambiguous ground truth, detected without any
judge, and it survived three repeats.

*Ambiguity estimation* requires judges and is therefore specified in dimension
5, not here.

**Inputs** inputs + expected; dimension 2 clusters; dimension 3 inversions.

**Output** contradiction clusters with their conflicting answers; leakage
findings with evidence; inversion cases promoted to reference-quality
candidates.

**Uncertainty** mixed tier by signal, labelled per claim.

**Can conclude** that specific references are mutually inconsistent, or
recoverable from the prompt, or contradicted by a capability-ordered model pair.

**Cannot conclude** that a reference is *wrong* — only that it cannot be
simultaneously true with another reference, or that it is obtainable without the
capability under test. Correctness needs a human.

**Failure modes** inherits every dimension-2 clustering failure, so a bad
cluster produces a spurious contradiction; deliberate consistency tests read as
contradictions; paraphrased leakage invisible; inversion evidence contaminated
by an asymmetric harness (real instance: capping one model at `effort: low`
while the other ran uncapped).

**Warnings** `CRITICAL` for contradictions — the eval is not coherently
scoreable until resolved. `WARNING` for leakage. `INFO` for inversion
candidates, since they are pointers for review.

**Tests** contradiction detection on constructed identical-input/different-answer
pairs; deliberate consistency tests reported distinguishably; leakage
false-positive suite over the four real dataset shapes (exists); `case_13`
pinned as a golden ambiguity specimen.

---

## 5. Evaluator / judge reliability

**Measures** whether the grader is reliable enough for its verdicts to mean
anything. This is the **one legitimate use of an LLM judge in this project**,
because the judge is the object of measurement rather than a trusted oracle.

**Definition.** Raw agreement is inadequate and must never be reported alone:
with a 90% base rate, 90% agreement is chance. Use chance-corrected measures.

Two raters: Cohen's *κ = (p_o − p_e) / (1 − p_e)*.
More than two: Fleiss' *κ*. Missing/ordinal data: Krippendorff's *α = 1 − D_o/D_e*.
Report κ with a bootstrap interval over cases.

*Self-preference* must be defined as a **difference of differences**, not an
absolute rate. For judge *j* and model *m*, let *A_{j,m}* be the pass rate
awarded. Self-preference is
  *S_j = (A_{j,j} − Ā_{·,j}) − mean_{m≠j}(A_{j,m} − Ā_{·,m})*
i.e. how much more generously *j* treats its own output than others do,
*relative to* how it treats everyone else's. This project previously retracted a
"~15pp self-preference" finding for exactly this reason: the asymmetry was +15pp
on own output but +25pp on others', which is general **leniency**, the opposite
conclusion.

*Human anchor.* κ between each judge and a human-labelled subsample.

**Inputs** two or more judges over the same (case, response) pairs; optionally a
human-labelled subsample.

**Output** κ per judge pair with interval; per-judge leniency; self-preference
*S_j*; judge–human κ where available; a propagated verdict-uncertainty term for
dimensions 3, 4 and 6.

**Uncertainty** `ESTIMATED`. Assumptions: judges score independently; the
human subsample is representative; the response set is fixed across judges.

**Can conclude** that judges do or do not agree beyond chance, and that a
stated share of verdict variance originates in the grader rather than the model.

**Cannot conclude** that a high-agreement judge is **correct**. Three judges
from one model family share their training and their blind spots; correlated
error looks exactly like reliability. Without a human anchor, κ measures
consensus, not accuracy — and this must be stated on every judge claim lacking
one.

**Failure modes** judges from the same family (correlated error read as
agreement); a judge grading its own output; prompt-order and position effects;
judge nondeterminism confounded with model nondeterminism unless repeats are
attributed to the right source; cost, since m judges multiply spend by m.

**Warnings** `CRITICAL` when κ falls below a stated floor — downstream
conclusions are then unsupported and dimensions 3/4/6 must be reported as
unavailable rather than computed. `WARNING` when no human anchor exists.
`WARNING` when any judge graded its own model's output.

**Tests** κ against hand-computed values on small fixed tables, including the
high-agreement-low-κ case that raw agreement would misreport; Fleiss' κ for 3+
raters; self-preference formula reproducing the retracted-finding numbers and
yielding "leniency" rather than "self-preference"; mocked judge clients (no API
key in tests, existing pattern); judge-family collinearity surfaced as a warning.

---

## 6. Statistical reliability

**Measures** how precise the eval's headline numbers are, and what size of
effect it could detect at all.

**Definition.**

*Accuracy interval.* Wilson score interval, not the normal approximation, which
breaks at the extremes this domain lives in — a normal CI on 98% at n=100
crosses 100%:

  centre = (p̂ + z²/2n) / (1 + z²/n)
  half-width = z/(1 + z²/n) · √( p̂(1−p̂)/n + z²/4n² )

*Model comparison is paired*, because both models see the same cases. Two
independent-proportion tests are the wrong tool. Use McNemar on the discordant
counts *b* (A passes, B fails) and *c* (B passes, A fails):

  δ̂ = (c − b)/n,   Var(δ̂) = [p_b + p_c − δ²]/n

*Minimum detectable effect* — the most useful output in this dimension:

  MDE ≈ (z_{1−α/2} + z_{1−β}) · √( p_disc / n )

Worked from this repo's real GSM8K data (n=100, 3 repeats, majority verdict):

```
98 both pass · 1 weak-only · 1 strong-only · discordance 2.0%
δ̂ = +0.0 pp    95% CI [−2.8, +2.8] pp
MDE at 80% power, α = .05  =  4.0 pp
```

So that eval cannot resolve anything below 4 pp, and the reported 0.7 pp gap is
inside the noise. This is what stops teams celebrating a 1–2 pp improvement.

*Redundancy correction.* All intervals use *n_eff* from dimension 2, or state
that they assume independence and are therefore optimistic by √DEFF.

*Per-class power.* For each class in dimension 8, the same MDE computation.
This is the rigorous replacement for `min_class_count = 5`: a 3-case class has a
Wilson interval roughly ±50 pp and can support no conclusion at all — that is a
derivation, not a convention.

**Inputs** per-case per-model verdicts; n_eff; α and power (defaults stated).

**Output** every reported rate with a Wilson interval; paired δ̂ with interval
and exact McNemar p; MDE overall and per class; explicit "this comparison is
underpowered" statements.

**Uncertainty** `ESTIMATED`. Assumptions: cases independent after n_eff
correction; the eval set is the population of interest, or a random sample of
it — *stated explicitly*, because a hand-curated eval is not a random sample of
production traffic, and the interval then describes only re-running on this set.

**Can conclude** that a difference is or is not resolvable at this n; the
resolution floor of the eval.

**Cannot conclude** that a resolvable difference **generalises** to production.
No interval fixes a non-representative sample. Nor does non-significance mean
equality.

**Failure modes** normal approximation at extreme p (avoided by Wilson);
unpaired tests on paired data (inflates variance, hides real differences);
ignoring DEFF (intervals too narrow); multiple comparisons across many
model pairs without correction; treating a curated set as a random sample.

**Warnings** `WARNING` whenever an observed difference is below MDE, worded as
"this eval cannot support that comparison". `WARNING` when DEFF > a stated
bound. `INFO` reporting MDE on every run, whether or not a comparison was made.

**Tests** Wilson interval against published tables, including p = 0 and p = 1
where the normal approximation fails; McNemar against `statsmodels`-verified
fixtures without adding the dependency (hard-coded expected values); MDE
monotone decreasing in n and increasing in discordance; the GSM8K numbers above
pinned as an end-to-end golden case; DEFF > 1 widening intervals.

---

## 7. Reproducibility

**Measures** whether the eval returns the same answer twice, and **where the
instability lives**.

**Definition.** With *r* runs of the whole eval producing scores *S_1..S_r*:

*Aggregate stability* — SD and range of *S*.
*Per-case stability* — flip rate, and per-case verdict entropy
*H_i = −Σ p log p* over observed verdicts.

**These must be reported separately** and the distinction stated, because they
diverge: an eval can have a 40% per-case flip rate and a rock-steady aggregate,
since independent flips cancel. This project's own bundled 20-case set shows
8 of 20 verdicts (40%) non-reproducible across three runs while the aggregate
moves little. A stable headline number over unstable cases is not a
reproducible eval — it is a coincidence that will not survive a new model.

*Variance attribution* — decompose total verdict variance into case, run and
residual components (ANOVA-style), and report the intraclass correlation
  *ICC = σ²_between-case / (σ²_between-case + σ²_within-case)*.
Where dimension 5 is available, separate model-sampling variance from
judge-sampling variance by holding one fixed. This is what makes the finding
actionable: "your instability is in the grader" and "your instability is in the
model" require different fixes.

**Inputs** *r* ≥ 2 full runs, or *r* repeats per case; ideally with the judge
held fixed for one arm.

**Output** aggregate SD/range; per-case flip rates and entropies; variance
components; ICC; the count of cases whose verdict never stabilised.

**Uncertainty** `ESTIMATED`, and unusually honest about small *r*: with *r* = 3
the variance estimate itself has enormous uncertainty, which must be shown.

**Can conclude** that a reported score is or is not stable at this sampling
configuration, and which component dominates.

**Cannot conclude** that a stable score is **correct** — a deterministic wrong
grader is perfectly reproducible. Reproducibility is necessary, not sufficient.

**Failure modes** a response cache making repeats free and therefore
meaningless (real, shipped once — the cache key must include an attempt index);
`temperature=0` unavailable on current frontier models, so determinism may not
be purchasable at all; conflating judge and model variance; too few runs to
estimate variance while reporting it as though estimated.

**Warnings** `CRITICAL` if repeats were served from cache without an attempt
key, since the reproducibility number is then a tautology. `WARNING` when
per-case flip rate is high but aggregate is stable — the most misleading
combination, and the one users will otherwise read as success.

**Tests** cache-key attempt-index regression (exists in
`tests/test_gsm8k_harness.py`); flip rate and aggregate SD computed on
constructed run matrices with known variance; the "stable aggregate, unstable
cases" case asserted to warn; ICC against hand-computed values; refusal to
report variance components below a minimum *r*.

---

## 8. Coverage and difficulty

**Measures** what the eval covers, and where each case sits on a difficulty
scale — the two together determining which capability band the eval can measure.

**Definition.**

*Coverage* — class distribution from `imbalance.py`, reframed as power rather
than count. Report per class: *n_c*, share, and its own MDE from dimension 6. A
class that cannot support a conclusion is reported as **unmeasurable**, with the
interval as proof, replacing the convention `min_class_count = 5`.

*Imbalance summary.* v1 uses max/min, which one singleton class dominates:
a 10,000-case set with 9,999 evenly spread and one class of 1 reports the same
ratio as a genuine 9999:1 split. v2 reports max/min **and** normalised Shannon
entropy *H/ log k* plus the Gini coefficient, so a single rare class cannot
alone characterise the distribution.

*Difficulty* — empirical, per case, over a model ladder: *p̂_i = mean_m p̂_{i,m}*
with an interval, and the distribution of *p̂* across the set. A good eval is
not one where everything is hard; it is one whose difficulty mass sits where the
models under test actually differ.

*IRT, conditionally.* A 2-parameter logistic model
  *P(correct | θ_m, a_i, b_i) = [1 + exp(−a_i(θ_m − b_i))]⁻¹*
separates difficulty *b_i* from item discrimination *a_i* — and *a_i* is
formally the quantity dimension 3 estimates. **Identifiability must gate this**:
with 2 models there are 2 observations per item and 2 free parameters per item
plus abilities, so the model is not identifiable. Fit only when the ladder is
large enough (M ≥ 5 as a floor, more in practice); otherwise report the
empirical *p̂* curve and say why IRT was not fitted. Offering an unidentifiable
fit would be the worst failure this tool could commit.

**Inputs** labels for coverage; per-case per-model verdicts across a ladder for
difficulty.

**Output** per-class power table with unmeasurable classes named; entropy and
Gini alongside max/min; difficulty histogram with intervals; IRT parameters
*only* when identifiable, with the condition stated either way.

**Uncertainty** coverage counts are `OBSERVED`; per-class power and difficulty
are `ESTIMATED`; IRT is `ESTIMATED` with strong parametric assumptions
(unidimensionality, local independence) that must be listed and are frequently
violated by real eval sets.

**Can conclude** which classes the eval can and cannot measure; which
capability band the difficulty mass addresses.

**Cannot conclude** that the distribution is **wrong**. If production traffic
is 70% billing, a 70% billing eval may be exactly right — only the user knows
the target distribution. Nor can it conclude coverage of anything not encoded
in `label`: categories living in metadata or implied by text are invisible.

**Failure modes** labels that are not classes (real: HellaSwag's `label` is the
correct-ending index, giving "4 classes, 1.2:1" instead of the true "39 classes,
14.0:1" — mitigated by `mapping.py` reporting alternatives); IRT fitted when
unidentifiable; difficulty conflated with brokenness, since a case every model
fails may have a wrong reference rather than being hard (dimension 4 owns that
distinction).

**Warnings** `WARNING` per unmeasurable class, with its interval as evidence.
`INFO` when max/min and entropy disagree about whether the set is imbalanced —
that disagreement is itself informative. `WARNING` when IRT was requested but
the ladder is too small.

**Tests** per-class MDE against hand-computed values; entropy/Gini on uniform
and maximally-skewed distributions; the singleton-class characterisation pinning
current max/min behaviour before changing it; IRT identifiability refusal at
M < 5; HellaSwag label/activity_label golden case.

---

# Output contract

The report is organised by dimension, and each dimension states one of four
statuses, never a score:

```
DIMENSION            STATUS            BASIS
integrity            supported         census, 100% complete
redundancy           supported         n_eff 16 of 20 at tau 0.85; stable 0.80-0.95
discrimination       underpowered      74 of 100 INDETERMINATE at r=3
ground truth         attention         1 contradiction cluster (CRITICAL)
judge reliability    not assessed      no judge configured
statistical          supported         MDE 4.0 pp; observed gap 0.7 pp
reproducibility      attention         per-case flips 40%, aggregate SD 0.8 pp
coverage/difficulty  partial           2 of 4 classes unmeasurable
```

Statuses are `supported` / `attention` / `underpowered` / `not assessed`.
`not assessed` is distinct from `supported` — the v1 exit-code-3 principle
generalised to every dimension. Then the two lists that constitute the product:

```
THIS EVAL CAN SUPPORT
  · "models differ by less than 4 pp, or not at all"        [ESTIMATED]
  · "at most 16 independent scenarios are present"          [HEURISTIC, tau=0.85]

THIS EVAL CANNOT SUPPORT
  · "Opus 5 outperforms Haiku 4.5"          gap 0.7 pp is below MDE 4.0 pp
  · "class 'refund' accuracy is 100%"       n=1, interval [2.5%, 100%]
  · any per-case conclusion                 74% INDETERMINATE at r=3

NO COMPOSITE SCORE IS REPORTED
  These dimensions have no common unit. A weighted average of a duplicate rate,
  a kappa and a confidence interval would be a number with invented weights --
  the exact artefact this tool exists to report.
```

## Compatibility and staging

| stage | content | breaks anything? |
|---|---|---|
| 1 | `Claim`/`Quantity`/`Tier` types; `CheckResult.claims=()`; report renders claims when present | no — v1 checks untouched |
| 2 | Dimension 6 (Wilson, McNemar, MDE) as a new check consuming existing stats | no — additive |
| 3 | Discrimination `INDETERMINATE` path behind an opt-in flag, unanimity remaining default | no — opt-in |
| 4 | n_eff/DEFF from duplicates feeding dimension 6 | no — additive stats keys |
| 5 | Dimensions 4, 7, 8 extensions | no |
| 6 | Dimension 5 (judges); `Severity.CRITICAL`; `--fail-on critical` | `--fail-on` gains a value; existing values unchanged |
| 7 | Flip discrimination default to `INDETERMINATE`; deprecate unanimity-only | **yes — major version** |

Stage 7 is the only breaking change, and it is a bug fix: the current default
produces a measurably biased number.

## Test strategy

Two categories, both required before any behaviour changes:

**Characterisation tests** pin today's behaviour, including the defects, so a
fix has to change them deliberately rather than by accident: the repeats-bias
table, the transitive-cluster grouping at 0.62, max/min imbalance under a
singleton class, float scores thresholded not ranked, and the `--json` payload
key set, which consumers depend on and which is currently unpinned.

**Property tests** for the new statistics: interval width monotone decreasing in
n; MDE monotone in n and discordance; κ ≤ raw agreement always; n_eff ≤ n
always with equality iff no clusters; DEFF ≥ 1; every `ESTIMATED` claim carries
a non-empty assumption list; no `HEURISTIC` claim carries an interval.

The last two are enforceable in the type layer, the same way
`CheckResult` already refuses to exist without limitations — which is the
mechanism that has kept this project honest so far, and the one worth extending
rather than replacing.
