# Methodology: model separation analysis

How evallint decides what an eval can and cannot tell you about a set of
models. Implemented in `src/evallint/checks/discrimination.py`, with the
statistics isolated in `src/evallint/checks/separation.py`.

## The question

Not *"is this case good?"* but *"how much evidence about model separation does
this eval carry, and between which capability levels?"*

The distinction is not cosmetic. A case every model passes is not defective —
it may be a deliberate regression guard, and an eval containing none cannot
detect a regression. What such a case does is **provide limited evidence for
model separation**. That is a statement about evidence, and the wording is used
consistently in place of any quality judgement.

## Per-case taxonomy

Models are supplied ordered weakest first. Each case is assigned exactly one
class, and every scored case gets one — nothing is discarded.

| class | pattern | what it means |
|---|---|---|
| `ceiling` | all pass | limited evidence for separation; may be a regression guard |
| `floor` | all fail | limited evidence for separation; **more often a wrong reference than a hard case** |
| `separating` | follows declared order, ≥1 change | carries evidence about which model is better |
| `non_monotonic` | a weaker model passed where a stronger failed | evidence about the reference or the ordering, not capability |
| `indeterminate` | repeats split exactly evenly | the run cannot assign a verdict |

`floor` and `non_monotonic` warrant more attention than `ceiling`, for a reason
unrelated to difficulty: a case nothing passes, or one the weaker model wins, is
more often a broken reference answer or a broken grader than a hard case. They
are pointers at the ground truth.

## Majority, not unanimity

This is the central methodological change from the earlier implementation.

The previous version classified a case only when every model's verdict was
unanimous across repeats, and discarded the rest. **That filter is not neutral
with respect to the outcome it measures.** For a model that passes a case with
probability *p*, unanimity is far more likely when *p* is extreme:

  at *p* = 0.6 with 5 repeats, P(all pass) = 0.078, P(all fail) = 0.010

so survivors are enriched for *"this model consistently passes"* — the ceiling
category. Measured on a synthetic 100-case set, weak model at *p* = 0.6, strong
model deterministic:

| repeats | surviving cases | non-discriminating share |
|---|---|---|
| 1 | 100 | 0.52 |
| 2 | 55 | 0.65 |
| 3 | 27 | 0.74 |
| 5 | 9 | **0.89** |

The share nearly doubles while the data does not change. And the check's own
advice was to raise `repeats` before trusting the figure, so following the
advice made the number worse.

The analysis now takes the **majority verdict** per model and classifies every
case. The denominator is always `n_cases`. Raising `repeats` moves cases out of
`indeterminate` into resolved classes rather than deleting them.

A tie is only reachable with an even number of repeats and yields
`indeterminate` rather than an arbitrary tie-break. **Prefer odd repeat counts.**

### Backward compatibility

`n_measured`, `n_unstable`, `non_discriminating_share`, `effective_n_cases` and
the `*_case_ids` keys are computed exactly as before and retained, so existing
JSON consumers see no change. They carry the selection bias described above, so
**do not compare them across different `repeats` values.** Use
`limited_evidence_share`, whose denominator is fixed.

## Uncertainty

### Per case

A model's success count out of `repeats` gets a **Wilson score interval**:

    centre = (p̂ + z²/2n) / (1 + z²/n)
    half   = z/(1 + z²/n) · √( p̂(1−p̂)/n + z²/4n² )

Wilson rather than the normal approximation because eval accuracies live at the
extremes. At 98/100 the normal interval is [0.943, **1.017**] — it contains an
impossible value — and at 10/10 it is [1.0, 1.0], claiming certainty from ten
observations.

A classification is marked `supported` only when every model's interval
excludes 0.5. **At `repeats=1` nothing is ever supported**, because one
observation cannot bound a probability. Even 3/3 does not qualify: its Wilson
interval is [0.44, 1.0], whose lower bound is below one half. That is a true
statement about what three trials establish, not a limitation of the code.

### Per model pair

Both models are scored on the **same cases**, so the comparison is **paired**.
Treating the two as independent proportions discards the pairing, inflates the
variance, and hides real differences. Only discordant cases carry information:

    δ̂ = (c − b) / n         b = weaker passed & stronger failed
    Var(δ̂) = [p_b + p_c − δ̂²] / n

The **decision** comes from an exact McNemar test — *b* ~ Binomial(*b*+*c*, ½)
under the null — not from whether the interval excludes zero.

**Those two disagree, and where they disagree the interval is wrong.** Three
discordant cases all favouring one model give a Wald interval of
[+0.005, +0.495], excluding zero, while the exact p-value is **0.25**. And when
every comparable case is discordant in one direction the Wald interval collapses
to zero width and claims certainty — the same defect that rules out the normal
approximation for a proportion. So the interval describes magnitude and carries
an `interval_reliable` flag (false below 10 discordant cases); the verdict comes
from the test.

Each pair reports one of:

| verdict | meaning |
|---|---|
| `separated` | exact p < α and the stronger model led |
| `ordering_contradicted` | exact p < α and the **weaker** model led |
| `unresolved` | the eval could not resolve this pair at this n |
| `no_information` | the two never once disagreed |
| `not_comparable` | no case had determinate verdicts for both |

`no_information` is deliberately distinct from `unresolved`. Two models that
never disagree provide no evidence of a difference, which is **not** evidence
there is none.

### Minimum detectable effect

    MDE ≈ (z_{1−α/2} + z_{1−β}) · √( p_discordant / n )

The most useful output here, because it states what the eval *cannot* do. From
the real GSM8K run in this repository (100 cases, 3 repeats, no LLM judge —
see [findings-gsm8k.md](findings-gsm8k.md)):

```
98 both pass · 1 weak-only · 1 strong-only · discordance 2.0%
δ̂ = +0.0 pp     95% CI [−2.8, +2.8] pp
MDE at 80% power, α = .05  =  4.0 pp
```

That eval cannot resolve anything below 4 percentage points. The observed gap
between Haiku 4.5 and Opus 5 was 0.7 pp, which is inside the noise. Reported as
`unresolved`, not as a ranking.

MDE is `None`, never `0.0`, when there is no discordance — a zero would read as
"can detect any difference", the exact opposite of the truth.

## Every threshold, and why

| constant | value | justification |
|---|---|---|
| `pass_threshold` | 0.5 | Only converts a float score to a verdict. A convention; move it if your scorer's scale differs. |
| support test | 0.5 | A verdict is resolved when the interval for that model's success probability excludes one half. Follows from majority voting rather than being chosen separately. |
| `confidence_level` | 0.95 | A convention. Exported in `stats` so a reader can substitute their own. |
| `ALPHA` | 0.05 | Derived from the confidence level, not chosen independently, so the two cannot drift apart. |
| `NORMAL_APPROX_MIN_DISCORDANT` | 10 | Conventional rule of thumb for a binomial normal approximation. Earns its place: at 3 discordant cases the Wald interval and the exact test give opposite answers. |
| `max_non_discriminating_share` | 0.5 | Warning level only. A convention with no statistical content, stated as such in the limitations. |

No threshold appears in the pair decision beyond α.

## What this analysis cannot conclude

- **That a non-separating case is a bad case.** Ceiling cases can be deliberate
  regression guards. The check reports how much of the eval does separating
  work; it does not tell you what to delete.
- **That the declared capability ordering is correct.** It can only report that
  the numbers contradict it.
- **Anything beyond the chosen models.** Two models closer in capability than
  assumed make a good eval look non-separating, and the analysis cannot tell the
  difference.
- **That an unresolved pair is equally capable.** Absence of evidence is not
  evidence of absence, which is why `no_information` exists as its own verdict.
- **That results generalise to production.** A curated eval set is not a random
  sample of real traffic, and no interval fixes that.

## Failure modes

- A wrong declared ordering inverts the `non_monotonic` finding.
- A non-thread-safe user scorer corrupts results silently at `max_workers > 1`;
  evallint cannot detect this.
- A response cache defeats `repeats` entirely unless the attempt index is part
  of the cache key. This bug shipped once in this repository — the repeats all
  returned one stored response, agreed by construction, and would have reported
  perfect reproducibility having tested nothing.
- `pass_threshold` collapses 0.95 vs 0.55 into "both pass", so a genuine margin
  can be invisible to a binary taxonomy.
- Cases are treated as independent. If the eval contains near-duplicates, the
  intervals here are narrower than they should be — the duplicate check's
  cluster count is the input needed to correct that, and doing so is not yet
  implemented.

## Reproducing the numbers

Every statistic is exported, so nothing needs to be taken on trust:

```python
from evallint import wilson_interval, mcnemar_exact_p, minimum_detectable_effect

wilson_interval(98, 100)              # (0.9300, 0.9945)
mcnemar_exact_p(1, 9)                 # 0.021484375  == 22/1024 exactly
minimum_detectable_effect(0.02, 100)  # 0.0400
```

Tests in `tests/test_separation.py` check each of these against hand-computed
values rather than against recorded output, because a test that captures what
the code currently says cannot detect a wrong estimator.
