# evallint's own benchmark

**Purpose: let a contributor tell whether a change actually improves evallint.**

Registered detector names, for `--only`:
`exact_duplicates`, `semantic_duplicates`, `class_imbalance`, `leakage`,
`ambiguous_references`, `ceiling_floor_discrimination`, `judge_instability`,
`statistical_noise`.

```bash
python -m benchmarks.runner --split test                    # reportable numbers
python -m benchmarks.runner --split dev                      # for tuning
python -m benchmarks.runner --split test --semantic           # + embedding level
python -m benchmarks.runner --split test --json out.json
```

`benchmarks/results/*.json` are committed baselines. `tests/test_benchmark.py`
compares the current code against them and fails on regression, so a change that
quietly makes a detector worse cannot pass CI.

---

## Read this before reading any number below

**The fixtures and the detectors have the same author.** A detector can score
perfectly here and still fail on data neither the fixture nor the detector
anticipated. The synthetic numbers measure *whether each detector does what it
was designed to do* — not whether the design finds real problems.

The only numbers here not written by the person who wrote the detectors are the
[real-dataset results](#real-dataset-results), and they are the ones that have
actually caught bugs.

**What "labelled" means differs by category**, and it matters more than the
metrics:

| Basis | Categories | What a high score means |
|---|---|---|
| **Objective by construction** | exact duplicates, ceiling/floor/separating, statistical noise | The detector is correct. A label error would be a bug in `fixtures.py`. |
| **Definitional** | leakage | The implementation matches its own rule. Says *nothing* about whether the rule is a good proxy. |
| **Author judgement** | semantic duplicates, ambiguous references | Precision *against these labels*. A second annotator would move some. |

---

## Held-out split, and how it can be broken

`DEV` is for tuning and debugging. `TEST` carries the reported numbers and
**nothing may be tuned against it**. The splits differ in seed *and* in surface
content — DEV uses a customer-support vocabulary, TEST a healthcare/transit one,
with different paraphrase templates and different ambiguity cases.

Every threshold in `evallint` was chosen before these fixtures existed, on real
datasets and hand-built unit cases. That is the honest version of held-out: the
fixtures came after the thresholds, so they cannot have influenced them.

**How a future contributor destroys this guarantee:** change a threshold because
a TEST number looked bad. If you need to tune, tune on DEV, and only then look at
TEST. Running `--split dev` prints a banner and tags the output
`"reportable": false` for exactly this reason.

The separation is enforced, not trusted: a test asserts no fixture has
byte-identical positive cases across splits. It was added after catching the
ambiguity fixture sharing all six of its positives between splits — TEST was not
held out at all for that category, and the honest TEST recall turned out to be
**lower** than DEV's once fixed.

---

## Results — TEST split, evallint 0.13.0

Deterministic detectors, `--split test`:

| Detector | Unit | P | R | F1 | FPR | FNR | ms/item |
|---|---|---|---|---|---|---|---|
| exact duplicates | case pair | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.008 |
| semantic duplicates *(no embeddings)* | case pair | n/a | 0.000 | n/a | 0.000 | 1.000 | 0.006 |
| class imbalance | dataset | 1.000 | 0.667 | — | 0.000 | 0.333 | — |
| leakage | case | 0.667 | 1.000 | 0.800 | 0.250 | 0.000 | 0.019 |
| ambiguous references | case | 1.000 | 0.667 | 0.800 | 0.000 | 0.333 | 0.008 |

With `--semantic` (adds the embedding level):

| Detector | P | R | F1 | FPR | ms/item |
|---|---|---|---|---|---|
| semantic duplicates | 0.947 | 1.000 | 0.973 | 0.004 | 0.512 |

**Discrimination** — `ceiling_floor_discrimination`, covering ceiling, floor,
separating and non-monotonic cases in one pass:
**100% exact class agreement**, 0 disagreements over 20 cases and 3 models. The
strongest result in the benchmark, and the most trustworthy — the scorer is a
fixture, so which models pass which case is written down rather than judged.

**Statistical noise** — `statistical_noise`, 200 trials per arm, 60 paired cases, α = 0.05:

| True difference | Significant share | Reading |
|---|---|---|
| 0.00 | **5.5%** | false-positive rate under a true null — the target is 5% |
| 0.05 | 5.0% | power at a 5-point difference: none, at 60 cases |
| 0.20 | 45.5% | power at a 20-point difference |

The 5.5% is the single most reassuring number here, because the target is
arithmetic rather than opinion: an exact McNemar test at α = 0.05 should call a
difference about 5% of the time when there is none. The 5.0% at a true 5-point
difference is *not* a bug — it is what no power looks like, and the analyser
reports `underpowered` rather than "no difference".

**Judge instability** — `judge_instability`: chance-corrected agreement rises monotonically with the
constructed forced-agreement share (0.00 → 1.00). At share 1.00 kappa is
undefined — the kappa paradox — and returning undefined there is the correct
output, not a failure.

### Threshold sensitivity, semantic duplicates (TEST)

Reported as a curve because a single number would hide the trade-off. **This is
not how the default was chosen** — 0.85 predates these fixtures.

| τ | P | R | F1 | FPR |
|---|---|---|---|---|
| 0.75 | 0.857 | 1.000 | 0.923 | 0.013 |
| 0.80 | 0.857 | 1.000 | 0.923 | 0.013 |
| **0.85** *(default)* | 0.947 | 1.000 | 0.973 | 0.004 |
| 0.90 | 1.000 | 0.389 | 0.560 | 0.000 |
| 0.95 | n/a | 0.000 | n/a | 0.000 |

Recall collapses between 0.85 and 0.90, which is the useful finding: the default
sits just below a cliff on this data. Precision is `n/a` at 0.95 because nothing
was flagged, which is a different result from precision 0.

---

## Known gaps, with baselines

These are **measured deficiencies left unfixed on purpose**: fixing them in
response to a TEST number would contaminate the split. A contributor who fixes
one should see the number move.

**1. `label_in_input` precision is 0.000 on this fixture** (leakage, TEST).
Per-detector-type breakdown:

| Detector type | Flagged | P | R |
|---|---|---|---|
| `expected_in_input` | 4 | 1.000 | 1.000 |
| `label_in_input` | 2 | **0.000** | 0.000 |

Both of `label_in_input`'s flags were false positives, and it is the sole reason
overall leakage precision is 0.667 rather than 1.000. This is *documented,
intended* behaviour — the detector's own message says a question about billing
naturally containing the word "billing" is often legitimate — but the cost is now
quantified rather than described. Note that **confidence stratification does not
separate these**: the true and false positives are all "moderate". The detector
*type* does.

**2. Class imbalance misses a single-class dataset** (recall 0.667). A dataset
with one label reports "ratio 1.0:1" and no warning. Technically correct —
nothing is imbalanced — and completely useless, since no per-class comparison is
possible at all. The fixture keeps it labelled a positive rather than moving the
goalpost.

**3. Ambiguity recall is 0.667**, missing two of six on TEST:
`amb_list` ("Give a few options for renewing a lapsed membership") and
`amb_subjective` ("What is the most sensible order to book appointments in?").
Both are keyword-list gaps in the open-ended and subjective detectors. DEV misses
only one, which is what a genuinely held-out split looks like.

---

## Real-dataset results

These were measured against public datasets during development, and are recorded
here because they are the only numbers in this document that were not produced by
fixtures the author also wrote. Each one **found a real bug**:

| Dataset | Measured | Cause | Fix |
|---|---|---|---|
| TruthfulQA | 100 false positives (leakage) | metadata exemption matched exact field names, missing `correct_answers` | substring matching |
| MMLU | 3 false contradictions (ground truth) | punctuation stripping collapsed three distinct field extensions | conservative normaliser that keeps punctuation |
| HellaSwag | 101 warnings (ground truth) | the dataset has no reference answers at all | widespread-type collapse |
| GSM8K | 98% of cases non-discriminating | genuine finding, not a bug | documented in `docs/findings-gsm8k.md` |

**These are historical measurements, not re-run by this benchmark.** They need
the datasets on disk, which are not vendored here for licence and size reasons.
To reproduce, point `evallint` at a local copy:

```bash
evallint truthfulqa.csv --skip-duplicates --format json --out tqa.json
```

The gap this leaves is the most important limitation of the whole benchmark: the
synthetic suite has never caught a bug that the real datasets did not catch
first. Adding a downloadable real-dataset arm — opt-in, not in CI — is the single
highest-value contribution someone could make here.

---

## Limitations

1. **Same author for fixtures and detectors.** A shared blind spot passes both
   splits silently. The real-dataset table above is the only guard.
2. **TEST is a held-out sample from the same generator design**, not an
   independent benchmark. It protects against overfitting to specific strings,
   not against a shared conception of what problems look like.
3. **Author-judgement labels** on semantic duplicates and ambiguous references.
   Reported precision is precision against those labels.
4. **The leakage label is the detector's own rule.** A perfect score there
   measures implementation, not validity.
5. **Fixtures are tens of cases**, so one misclassification moves precision
   several points. Treat differences under ~5 points as noise.
6. **Memory figures come from `tracemalloc`** and exclude numpy and torch
   buffers. The embedding path reports **114 KiB peak** while loading a ~90 MB
   model — a concrete demonstration that the number is a relative indicator and
   not a footprint.
7. **Runtime is wall clock on one machine.** The recorded platform string exists
   so you can tell whether two runs are comparable; across machines they are not.
8. **No F1 for three categories.** Class imbalance has a five-dataset universe;
   judge instability and statistical noise are not classification problems. An
   F1 for them would be a number with no interpretation.
9. **Power figures are specific to 60 cases**, and low power there is a property
   of the sample size rather than a defect in the test.
10. **The benchmark measures detectors, not the audit report** that composes
    them. That has its own 38 tests.

## Adding a detector benchmark

1. Add a fixture generator to `fixtures.py` taking a `Split`, with content that
   differs between splits. Include **adversarial negatives** — a fixture of
   obvious positives measures recall and nothing else.
2. State the `label_basis` honestly. If it is your judgement, say so.
3. Add a `bench_*` function to `runner.py` and register it in `ALL_DETECTORS`.
4. Use `score_sets(predicted, actual, universe)`. The universe is required, so a
   detector that flags everything cannot look good.
5. Regenerate the baselines and commit them in the same change as the code, so
   the diff shows the effect.
