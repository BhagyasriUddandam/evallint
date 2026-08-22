# GSM8K cannot tell Haiku 4.5 from Opus 5

A run of evallint's discrimination check against a public benchmark, on data
the author did not write.

**Result: 98% of 100 GSM8K cases give both models the same verdict.** A
100-case benchmark that discriminates like a 2-case one.

## Why this run exists

evallint's other two checks had been run against five real public eval sets.
The discrimination check — the flagship, and the only one that is not a
commodity — had only ever run on the 20-case example bundled with this repo,
written by the author. "It works on my own data" was the project's weakest
claim.

## Method

| | |
|---|---|
| dataset | GSM8K test split, first 100 rows, verbatim from the HuggingFace datasets-server |
| models | `claude-haiku-4-5` (declared weak) → `claude-opus-5` (declared strong) |
| repeats | 3 |
| calls | 600 |
| grader | **none — exact numeric comparison** |
| cost | $1.68 across this and one earlier run (800 calls total) |

**There is no LLM judge.** Every GSM8K answer ends in a literal marker:

```
...18 every day at the farmer's market.
#### 18
```

So grading is: parse the model's `ANSWER: <n>` line, parse the reference `####
n`, compare the numbers. This matters more than the cost saving. Two earlier
findings in this project had to be retracted because a model was grading its
own answers; removing the judge removes that entire class of artifact. What is
left is arithmetic anyone can re-derive.

The eval set was loaded through evallint's own mapping layer, which inferred
`input <- 'question'` and `expected <- 'answer'` — GSM8K does not use
evallint's field names, and no real eval set does.

## Result

```
100 cases x 2 models x 3 repeats: 2 discriminate, 95 do not (98%), 3 not reproducible

WARN  3 of 100 cases gave a DIFFERENT verdict across 3 identical runs, so they
      cannot be classified and are excluded from the counts below
      case_74, case_88, case_94
WARN  98% of measured cases (95/97) give every model the same verdict, so they
      carry no evidence about which model is better. This eval discriminates
      like a 2-case set
INFO  every model passes 95 cases, including 'claude-haiku-4-5' — too easy to
      separate anything
WARN  in 1 case a weaker model passed where a stronger one failed
      case_13
```

Pass rates, over all 300 calls per model:

| model | pass rate |
|---|---|
| `claude-haiku-4-5` | 294/300 = 98.0% |
| `claude-opus-5` | 296/300 = 98.7% |

Both models sit at the ceiling. A 0.7 percentage point gap across 100 cases is
not a measurement of anything; it is what two saturated models look like. GSM8K
is still widely cited as a reasoning benchmark, and against this pair it has
almost no remaining resolving power.

## What repeats changed

The first single run reported **two** inversions, case_13 and case_88. Running
the identical configuration three times:

- **case_88 did not reproduce.** Its verdict changed between identical runs, so
  it is sampling noise, not a property of the data. Reporting it would have
  produced a fourth retraction in this project.
- **case_13 survived all three repeats**, and is the more interesting case
  anyway. See below.

3 of 100 cases were non-reproducible, so single-run figures on this data carry
roughly 3% noise. That is why the check refuses to classify unstable cases and
excludes them from its denominator rather than folding them in.

## case_13: ambiguous ground truth, found in the wild

> Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year
> it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to
> water and feed the tree. How many years will it take before he starts earning
> money on the lemon tree?

GSM8K's reference answer is **13**, reasoning that break-even arrives at 12
years ($90 / $7.50) and profit therefore begins in year 13. Opus 5 answered
**12** in all three runs, treating break-even as the answer.

Both readings are defensible. This is not a model error and not a grader bug —
it is a question with two correct answers depending on how "starts earning
money" is read, being scored as though it has one. That is exactly the
**ambiguous ground truth** failure mode, which is on evallint's roadmap as an
unimplemented check. Here is a real instance of it inside a benchmark in
constant use.

## Where the author was wrong

The first run capped Opus at `output_config: {effort: "low"}` while Haiku ran
uncapped, and Opus produced less than half the output tokens (73 vs 162 mean).
The stated hypothesis was that this handicap explained the inversions, and that
equalising it would remove them.

**That hypothesis was mostly wrong.** With `effort` omitted so both models run
at their own defaults, Opus's mean output rose only from 73 to 97 tokens, still
far below Haiku's 164. Opus is simply terse on grade-school arithmetic —
adaptive thinking correctly declines to think hard about it. The effort
asymmetry was real, but it was not what invalidated case_88. **Repeats were.**

The conclusion was right and the mechanism was wrong, which is worth stating
plainly: it would have been easy to publish "configuration artifact" and be
believed.

## What this does not show

- **One dataset, one domain.** GSM8K is grade-school arithmetic. Nothing here
  transfers to legal reasoning, code review, or any domain with disputable
  answers.
- **One model pair.** The check's own limitations say the verdict depends
  entirely on the pair chosen. A weaker pair — say two small models — would
  find GSM8K discriminating perfectly well. The finding is "GSM8K cannot
  separate *these two frontier models*", not "GSM8K is a bad benchmark".
- **100 cases, not the full split.** GSM8K's test set is 1319 rows. This is the
  first 100.
- **Saturation is not the same as uselessness.** A benchmark every model passes
  is still a regression guard. It just cannot rank.

## Reproducing this

All 800 API responses are committed in `scripts/.gsm8k_cache.json`, so the
numbers above can be recomputed for **$0.00**:

```bash
python scripts/gsm8k_discrimination.py --sample 100 --repeats 3
```

Every call is served from disk. Delete the cache to pay for it again. A project
about not taking numbers on trust should ship the numbers.
