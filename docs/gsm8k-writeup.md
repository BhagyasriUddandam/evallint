# 98 of 100 GSM8K problems can't tell Haiku 4.5 from Opus 5

**TL;DR** — I ran two Claude models of clearly different capability against 100
GSM8K problems, three times each, with no LLM judge. On 95 of the 97 cases that
scored reproducibly, both models gave the identical verdict. That benchmark
separates those two models on about 2% of its surface, and the other 98% still
feed the headline number.

Cost to establish this: **$1.68**.

---

## Why this matters more than it sounds

A case where every model you're comparing produces the same verdict contributes
nothing to the question *"which model is better."* It isn't a bad case — it may
be a perfectly good regression guard — but it carries no separation evidence.

So when you read "89% vs 91% on GSM8K", the 2-point gap rests on a much smaller
number of cases than 100. On 60 cases with 20 discordant pairs, an exact McNemar
test puts a gap like that at p ≈ 0.5. The number looks authoritative and cannot
support the claim.

## Method

| | |
|---|---|
| dataset | GSM8K test split, first 100 rows, verbatim from the HuggingFace datasets-server |
| models | `claude-haiku-4-5` (declared weak) → `claude-opus-5` (declared strong) |
| repeats | 3 |
| calls | 600 |
| grader | **none — exact numeric comparison** |
| cost | $1.68 across this and one earlier run (800 calls total) |

**There is no LLM judge**, and that matters more than the cost saving. Every
GSM8K answer ends in a literal `#### n` marker, so grading is `float(a) ==
float(b)`. Two earlier attempts at this used a judge and produced results I
couldn't separate from judge noise. Removing the judge removed the ambiguity.

Three repeats, because a single run cannot distinguish a genuinely
non-separating case from a flaky one. Cases whose repeats split evenly are
reported as `indeterminate` rather than silently classified.

## Result

```
100 cases x 2 models x 3 repeats: 2 discriminate, 95 do not (98%), 3 not reproducible
```

| model | pass rate |
|---|---|
| `claude-haiku-4-5` | 294/300 = 98.0% |
| `claude-opus-5` | 296/300 = 98.7% |

Both models sit near the ceiling, so almost every case lands in the `ceiling`
class: everyone passes, nobody learns anything about ordering.

**3 of 100 cases were not reproducible** across the three repeats, which is
roughly 3% noise. Any single-run figure on this data carries that uncertainty,
and it is why the check refuses to classify unstable cases rather than reporting
whichever verdict the last run happened to give.

## The one case worth reading

One case survived all three repeats as a genuine ordering anomaly — the *weaker*
model right where the stronger one was wrong:

> Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it
> will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to
> water and feed the tree. How many years will it take before he starts earning
> money?

Net income is $10.50 − $3 = $7.50/year, so the tree pays for itself after 12
years and profit *begins* in year 13. The reference answer is 13. Opus 5 answered
**12** in all three runs — it computed break-even and stopped.

That is not a capability gap. It's a question whose wording admits two readings,
with one of them recorded as ground truth. Which is a different defect entirely,
and the reason a "stronger model failed here" finding points at the reference
before it points at the model.

## What this does and doesn't say

**It does not say GSM8K is a bad benchmark.** It says GSM8K cannot separate
*these two models*, because both are near its ceiling. That's what saturation
looks like, and it's a property of the pairing as much as the dataset — the same
100 cases would likely separate a 7B model from Opus.

**It does not say those 98 cases are worthless.** They still detect regression.
They just don't inform a ranking.

**What it does say** is that the headline number gave no indication of any of
this, and that finding out took one afternoon and $1.68.

## Reproduce it

```bash
pip install evallint
```

```python
from evallint import load
from evallint.checks import DiscriminationCheck

def score(case, model):                     # your client, your keys
    out = my_client.complete(model=model, prompt=case.input)
    return extract_final_number(out) == extract_final_number(case.expected)

result = DiscriminationCheck(
    score, ["claude-haiku-4-5", "claude-opus-5"],   # WEAKEST FIRST
    repeats=3,
).run(load("gsm8k.jsonl"))

print(result.summary)
```

The classification, the per-pair McNemar tests and the minimum detectable effect
all come out of that one call. Full methodology, including how the pair verdicts
are derived and why the exact test overrides the interval on small discordant
counts: [`methodology-discrimination.md`](methodology-discrimination.md).

## Caveats I'd want a reader to have

- **100 cases, one dataset, one model pair.** This is a demonstration of a
  measurement, not a survey of benchmarks.
- **First 100 rows, not a random sample.** Convenient, and it means the result
  is about that slice.
- **3% of cases were non-reproducible**, so treat the 98% as approximately 98%.
- **The declared ordering is an assumption.** The whole method rests on Opus 5
  being stronger than Haiku 4.5. If two models are closer than you assumed, a
  perfectly good eval looks non-separating and the tool cannot tell the
  difference.
- **The detectors were benchmarked by their own author.** Measured precision and
  recall are in [`../benchmarks/README.md`](../benchmarks/README.md), including
  one detector that scores 0.000 precision and is documented rather than
  quietly fixed.
