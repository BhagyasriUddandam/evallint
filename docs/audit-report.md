# The unified audit report

```bash
evallint evalset.jsonl --format terminal          # concise
evallint evalset.jsonl --format json  --out r.json
evallint evalset.jsonl --format html  --out r.html
```

---

## The ten questions, applied to this feature

The gate was applied before any code was written. The honest answers, including
the ones that constrained the design:

**1. What exact problem does this solve?**
Eleven analyses exist in this package and there is no way to see them together.
Worse, the ones that need a scorer, judges or repeated runs simply do not appear
in CLI output, so a reader cannot tell "nothing was found" from "nobody looked".
That specific confusion is the problem: it is the same defect this package
reports in eval sets, present in its own reporting.

**2. Is this already solved well by an existing tool?**
No, because no other tool audits the dataset. promptfoo, DeepEval and lm-eval
report eval *results*; this reports evidence about whether those results can
support a conclusion. There is no overlap to duplicate.

**3. What evidence supports the methodology?**
The report itself contains no new methodology — that is deliberate. It composes
analyses whose methods are documented and tested elsewhere
(`docs/methodology-discrimination.md`, `docs/architecture-v2.md`). The one new
mechanism is the **evidence tier**, and its justification is that the alternative
was tested and found wanting: a single score requires weights across
incommensurable evidence, and no defensible source for those weights exists.

**4. Can the result be validated?**
Yes, and this is where the design is strongest. Every claim in the report is
traceable to the analyser that produced it, and a test asserts the unified report
and the legacy report contain *exactly* the same set of findings — so the
composition cannot invent or lose one.

**5. What are the false-positive and false-negative modes?**
The report inherits them from its sources and adds two of its own:

- *False positive:* an `OBSERVED` finding — a class with 1 case, say — is
  reported even when the small class is deliberate. The report cannot know your
  intent, so it says what it counted and states that it does not know whether
  that is a problem.
- *False negative, and the more serious one:* an analysis that did not run
  cannot produce findings. This is why `NOT_RUN` is structurally distinct from
  `RAN` with no findings, why the not-run list is never truncated in any format,
  and why the executive summary says a conclusion is "unsupported in those
  respects" rather than "supported with caveats".

**6. Can the feature operate without an LLM?** Yes, entirely. Four of the seven
analyses run from the file with no model, no network and no optional extra. The
report is a pure composer: it calls no model and runs no analyser, so it cannot
decide on your behalf to spend money.

**7. If an LLM is required, how do we measure the reliability of that LLM?**
No LLM is required. Where one is *used* — a judge behind a rubric — its
reliability is measured by `EvaluatorReliability`, and the report's treatment of
that is the point: a judge metric that could not be computed becomes a
**finding**, because an absent reliability measure is not a passing one.

**8. Does the feature help determine whether an evaluation conclusion is
trustworthy?** That is its only purpose. `evidence_coverage` answers exactly
"which respects is a conclusion from this eval supported in, and which not".

**9. Does it provide evidence rather than an arbitrary score?** Yes, enforced
rather than intended: a test asserts no key anywhere in the JSON could be read as
a grade, and another asserts `AuditReport` has gained no `score` attribute.

**10. Can we write a reproducible benchmark for the feature?** Yes — 38 tests,
all deterministic. There is no sampling and no model call anywhere in the report
layer, so the same input gives byte-identical output.

---

## What replaces the single score

Nothing is compressed. What a score would have summarised is reported as
**evidence coverage**:

```json
{
  "analyses_available": 7,
  "analyses_with_evidence": ["dataset_statistics", "ground_truth"],
  "analyses_without_evidence": {
    "discrimination": "needs a scoring function that runs your models ...",
    "evaluator_reliability": "needs repeated judge verdicts on the same cases ...",
    "reproducibility": "needs the SAME eval run more than once ..."
  }
}
```

An "Eval Trust Score: 78/100" would need a weight for redundancy against a
weight for an unmeasured judge. There is no defensible source for either number,
so the score would be arbitrary — and a reader would act on it instead of reading
the evidence underneath. Two tests exist purely to make adding one fail.

## The evidence tier

Every finding carries one. Designed in `docs/architecture-v2.md`, enforced here.

| Tier | What it is | Confidence |
|---|---|---|
| `OBSERVED` | A census of the data. Counting, not inference. | **must be absent** |
| `HEURISTIC` | A signal from a chosen threshold, no sampling theory. | **required** |
| `ESTIMATED` | A statistical estimate with an interval and assumptions. | the interval |
| `RECOMMENDED` | An action. Inherits the weakest tier it cites. | inherited |

Two rules, enforced in `__post_init__` rather than left to discipline:

**No promotion.** An `OBSERVED` finding may not carry a confidence — a census is
exact, and attaching "HIGH" to it implies a doubt that does not exist. A
`HEURISTIC` finding *must* carry one, because a threshold-derived signal without
a confidence reads as a fact. Both raise `ValueError`.

**Recommendations sort weakest-tier-first**, so "this rests on a cosine
threshold" appears above "this is a count". Nobody should act on a heuristic
believing it was a census.

## Every finding carries seven fields

All seven are required; a blank one raises.

```json
{
  "section": "ground_truth",
  "severity": "warning",
  "tier": "heuristic",
  "confidence": "not stated by the source analyser",
  "evidence": "POTENTIAL LEAKAGE [expected_in_input, confidence moderate] in case 'c0': the reference answer appears verbatim in the input ...",
  "affected_cases": ["c0"],
  "n_affected_cases": 1,
  "explanation": "8 answered cases, 2 contain their own answer (25%)",
  "limitation": "This check CANNOT distinguish a leak from a task where the answer is supposed to appear in the input ...",
  "recommended_action": "Inspect the listed cases by hand. A confirmed leak means ..."
}
```

`limitation` is the **source analyser's own** stated limitation, not one written
at the reporting layer. A limitation invented here could contradict the analyser
that produced the finding.

`confidence` is `null` for `OBSERVED` findings and, where an analyser genuinely
never computed one, the literal string `"not stated by the source analyser"` —
which is honest rather than a fabricated `"medium"`.

## The eleven sections

Four are **derived views**, computed from the other seven rather than written, so
a finding cannot appear in "critical findings" phrased differently from how it
appears in its own section:

| # | Section | JSON key | Kind |
|---|---|---|---|
| 1 | Executive summary | `executive_summary` | derived |
| 2 | Critical findings | `critical_findings` | derived — `WARNING` severity, most-cases-first |
| 3 | Warnings | `warnings` | derived — `INFO` severity |
| 4 | Dataset statistics | `dataset_statistics` | analysis (+ imbalance) |
| 5 | Redundancy analysis | `redundancy` | analysis |
| 6 | Discrimination analysis | `discrimination` | analysis |
| 7 | Ground-truth analysis | `ground_truth` | analysis (+ **leakage**) |
| 8 | Evaluator reliability | `evaluator_reliability` | analysis |
| 9 | Statistical reliability | `statistical_reliability` | analysis (effective size + model comparison) |
| 10 | Reproducibility | `reproducibility` | analysis |
| 11 | Recommendations | `recommendations` | derived — deduplicated, weakest tier first |

**Why leakage shares section 7.** The eleven requested sections have no leakage
entry, and dropping the analyser would have lost information the CLI already
reports. Both answer the same question — *can this case be scored as intended?* —
because an under-specified reference and an answer sitting in the prompt are two
different ways for the answer to fail to test what you meant. The grouping is
stated in the section summary so nobody has to infer it.

## Findings derived from the absence of evidence

This is the part that does not exist in other reporting tools, and the reason the
adapters are not one-liners.

A judge metric that **could not be computed** becomes a finding:

```
[observed] judge_correlation:j0|j1 was not computed: insufficient shared numeric scores
    explanation: This metric was refused because its assumptions were not met,
                 so it provides no evidence either way. An absent reliability
                 measure is not a passing one.
```

So does an **unidentifiable variance source**:

```
[observed] evaluator_stochasticity is not identifiable: 0 judge(s) recorded;
           needs at least 2 scoring the same run
    explanation: This design cannot separate that source of variance, so its
                 contribution is unknown rather than zero. Reporting it as 0.0
                 would claim a stability that was never measured.
```

Both are `OBSERVED`: whether a metric was computed is a *fact about the design*,
not an estimate, so it carries no confidence.

**No threshold is invented at this layer.** There is no "agreement too low"
finding, because setting that bar would be exactly the arbitrary cutoff this
report exists to avoid. Every computed metric goes into `stats` with the value,
interval and interpretation its own analyser wrote. For model comparison the
report uses the analyser's **own verdict vocabulary** — only `underpowered` and
`no_information` become findings, because a detected difference is a *result*,
not a defect.

## The three formats

**`--format terminal`** is concise: findings one line each with their tier,
truncated case ids, no raw statistics. What is *never* truncated is the not-run
list, because that is the part a reader is most likely to mistake for a pass. A
test bounds the whole output at under 120 lines. `--detail` adds each finding's
explanation, limitation and recommended action.

**`--format json`** carries everything, nothing truncated, every case id.

**`--format html`** is a single self-contained file: `<details>` for expandable
sections, case-level lists, and per-section raw statistics. No CDN, no
framework, **no JavaScript** — a report that fails to render because a CDN moved
is worse than a plain one, and an audit artefact should still open in five years.
Case ids and source paths are HTML-escaped, since they come from a user's file
and are untrusted input; a test asserts a case id of `<script>alert('x')</script>`
cannot execute.

## Limitations

- **The report inherits every limitation of its sources**, and states them per
  finding rather than summarising them away.
- **It cannot know your intent.** A 14:1 class imbalance may be your production
  traffic distribution. A 40% redundancy rate may be a deliberate consistency
  suite. The report says what it counted and that it does not know whether that
  is a problem.
- **Four of seven analyses run from a file.** The other three need model runs,
  judges or repeated runs, and from a CLI they will always be `NOT ASSESSED`.
  That is a property of the information available, not a gap to be closed by
  better code.
- **`critical` means "would distort a number you are about to quote"**, which is
  what `Severity.WARNING` has always meant here. It does not mean the eval is
  broken, and the report never says that.
- **No overall score, in any format.** Whether the evidence is acceptable for
  your purpose is a judgement the tool does not have the information to make.

## Library use

```python
from evallint import load, run_audit, render_terminal
from evallint.checks import ImbalanceCheck, LeakageCheck, GroundTruthCheck

eval_set = load("evalset.jsonl")
report = run_audit(
    eval_set,
    imbalance=ImbalanceCheck().run(eval_set),
    leakage=LeakageCheck().run(eval_set),
    ground_truth=GroundTruthCheck().run(eval_set),
    # and, when you have them:
    #   discrimination=DiscriminationCheck(scorer, models).run(eval_set),
    #   evaluator_reliability=EvaluatorReliability().analyse(observations),
    #   reproducibility=analyse_reproducibility(outcomes),
    #   model_comparison=compare_models(outcomes),
    #   effective_size=estimate_effective_size(eval_set),
)
render_terminal(report)

for gap, reason in report.evidence_coverage["analyses_without_evidence"].items():
    print(f"unsupported: {gap} — {reason}")
```

`run_audit` runs nothing itself. It is a pure composer, so it cannot download a
model or call an API behind your back, and every argument being `None` is a valid
call that produces a report saying nothing was assessed — the honest output for
that input.
