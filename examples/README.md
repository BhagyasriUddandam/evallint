# examples/

## sample_evalset.jsonl — a deliberately flawed eval set

20 cases from a plausible customer-support triage eval. Every flaw below was
planted on purpose so each check has something real to catch. This file is both
the live demo and a test fixture.

JSONL has no comment syntax, and adding a `#`-comment rule to the loader would
make it a non-standard parser, so the flaws are documented here instead.

### Planted flaw 1 — near-duplicates (for the duplicate check)

**Exact duplicate pair.** `billing_004` and `billing_014` have byte-identical
inputs ("How do I update the credit card on my account?"). Cosine similarity
should be 1.0. If the check misses this, it is broken.

**Near-duplicate cluster of 3.** `billing_001`, `billing_002`, `billing_003`
are three phrasings of "I was charged twice this month". Different words, same
scenario, same expected answer. This is the realistic case: nobody notices it
in a spreadsheet, and it silently gives one scenario triple weight.

**Near-duplicate pair, word-order only.** `password_001` ("I forgot my password
and can't log in.") and `password_002` ("I can't log in because I forgot my
password.") are the same sentence reordered.

So 6 of 20 cases sit in a duplicate cluster. Stated the other way: this eval
looks like 20 cases but carries roughly 16 distinct scenarios.

**Control group.** The remaining cases are genuinely distinct (sales tax, VAT
receipts, failed payments, account lockout, a UI bug). A check that flags those
as duplicates is too aggressive — they exist to prove the check stays quiet on
known-good input.

### Planted flaw 2 — class imbalance (for the imbalance check)

| label | count | share |
|---|---|---|
| `billing` | 14 | 70% |
| `password_reset` | 4 | 20% |
| `refund` | 1 | 5% |
| `bug_report` | 1 | 5% |

Imbalance ratio 14:1. Two classes have a single case each, which means their
per-class accuracy can only ever be 0% or 100% — no useful signal. A model that
handles billing well and everything else badly still scores ~70% overall, which
is exactly the way an aggregate number hides a failure.

### Planted flaw 3 — non-discriminating cases (for the discrimination check)

Not detectable from this file alone: the discrimination check needs model
outputs, not just the dataset. Several cases here are trivial enough that any
competent model should get them right (`billing_008` sales tax, `billing_013`
invoice went up), so they are the cases most likely to be flagged as
non-discriminating once real models are scored against them. That prediction
is untested until the check runs — it is not a claimed result.

## sample_evalset_v2.jsonl — the same shape in schema 2

Three cases exercising every schema-2 field, so the format has a working
reference that is also loaded by the test suite:

- **`chat_001`** — a three-turn conversation, a `system` prompt, one
  `acceptable` alternative to `expected`, and a weighted `rubric` with a scale.
- **`chat_002`** — a tool call and its result inside `messages`, a declared
  `tools` inventory, `expected_tool_calls` as ground truth, a structured
  (object-valued) `expected`, and an `expected_schema`.
- **`chat_003`** — a plain four-field case, unchanged from version 1, sitting in
  the same file. That mixture is the point: schema 2 does not require rewriting
  cases that were already fine.

Two things to notice when you run it:

```console
$ evallint examples/sample_evalset_v2.jsonl --skip-duplicates
```

`chat_001` and `chat_002` have `input_is_derived = True` — their `input` was
flattened from `messages`, and every check read that flattened text. `chat_003`
has `input_is_derived = False`, because those words are the ones in the file.

And the `system` prompt is byte-identical across the two chat cases, but it does
not appear in either `input`. If it did, the two cases would share a long prefix
and the redundancy check would call them near-duplicates — a finding about the
flattening rather than about the eval.
