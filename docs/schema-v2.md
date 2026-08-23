# Schema version 2 — chat, rubrics, tools

**Nothing here is required.** The four-field format is still the whole schema,
and a file written for evallint 0.1 loads unchanged on 0.12 and will keep
loading. Read the first section, stop there if it covers you, and come back when
you need a conversation or a tool call.

```json
{"id": "1", "input": "What is 2+2?", "expected": "4", "label": "arithmetic"}
```

Only `id` and `input` are required. Everything below is optional and defaults to
empty.

---

## The one thing worth understanding before the field list

`input` is a **required, non-empty string** and always will be. Every check in
this package reads it as text: `len(case.input)` in imbalance, embedding in
redundancy, `normalise(case.input)` in five leakage detectors.

So a multi-turn case does not replace `input` — it **derives** it. `messages` is
the authoritative field, and `input` is that conversation flattened to text.

```python
>>> from evallint import load
>>> case = load("chat.jsonl").cases[0]
>>> case.messages
(Message(role=Role.USER, content='Hi'), Message(role=Role.ASSISTANT, ...))
>>> case.input
'user: Hi\nassistant: Hello\nuser: Refund?'
>>> case.input_is_derived
True
```

Two consequences:

1. **Every existing check works on chat data on day one.** Leakage detection
   across turns, redundancy between conversations, class balance over a chat
   set — none of those checks was modified, and none of them needs to be.
2. **A derived value is marked as derived.** `input_is_derived` exists so a
   report never quotes synthesised text back as something you wrote.

### The flattening rules, in full

| Input | Flattens to |
|---|---|
| One `user` turn, no tool calls, no name | its `content`, **verbatim, with no prefix** |
| Anything else | `role: content` per turn, joined by newlines, in order |
| A turn with a `name` | `role(name): content` |
| A tool call | `assistant: [tool_call] name(arg=value)`, keys sorted |
| The `system` **field** | **not included** — see below |

Rule 1 is what makes migration safe: the chat form of a simple case produces
text **byte-identical** to the four-field form, so converting a file cannot
change a single finding.

### Why `system` is excluded from `input`

This is the least obvious rule, and it has teeth.

A system prompt is usually identical across every case in a set. Prepending it
to each `input` would give every case a long shared prefix. That inflates
pairwise cosine similarity and token overlap, so the redundancy check would
report near-duplicates throughout and the effective-size estimate would collapse
toward a single cluster — both as artefacts of the flattening rather than
properties of your data.

`system` therefore stays its own field. A `system` **turn** you put inside
`messages` yourself is your choice and is flattened like any other turn.

---

## Declaring a version

```jsonc
// .json  — an envelope
{"evallint_schema": 2, "cases": [ ... ]}
```

```jsonc
// .jsonl — an optional first line, then one case per line
{"evallint_schema": 2}
{"id": "1", "input": "..."}
```

CSV cannot declare a version and is always read leniently.

**Declaring a version does not unlock features.** `messages`, `rubric` and the
rest are read whenever they are present and well-formed, version or no version.
Dropping your conversation into `metadata` in order to be pedantic about a header
line would be exactly the silent data loss this tool exists to complain about.

What the version changes is **strictness**:

| | Undeclared (version 1) | Declared version 2 |
|---|---|---|
| `messages` that is not a conversation | kept as metadata, with a note | **error** |
| A key one edit from a real field (`sytem`) | kept as metadata, silently | **error, with a suggestion** |
| A future version (`3`) | — | **refused by name** |

The lenient column exists because real datasets have columns named `labels`,
`inputs`, and yes, `messages` meaning something else. Those files must keep
loading. The strict column exists because a file you *maintain* is better off
failing loudly on a typo.

A version this build does not know is refused by name, not parsed on a guess —
a version-3 file misread under version-2 rules would produce a confident report
about data that was parsed wrong:

```
future.jsonl: 1 schema problem

  evallint_schema: this file declares schema version 3, which evallint 0.11.0 cannot read
      supported versions: 1, 2. Upgrade evallint, or lower the declared version if the file does not actually use newer features.
```

The version is **never inferred** from the fields present. A version-1 file read
under version-2 rules would fail on data that was always legal.

---

## The seven additions

### 1. Chat / multi-turn messages

```json
{"id": "refund-1",
 "messages": [
   {"role": "user", "content": "My order never arrived."},
   {"role": "assistant", "content": "I'm sorry — let me look."},
   {"role": "user", "content": "Can I get my money back?"}],
 "expected": "escalate"}
```

Roles are a closed set: `system`, `user`, `assistant`, `tool`. An open string
field would turn `"assistent"` into a turn nothing recognises and nothing
rejects.

A list of plain strings is also accepted, because MT-Bench and several
HuggingFace sets store conversations that way:

```json
{"question_id": 81, "turns": ["Write a travel post.", "Now rewrite it."]}
```

Those become successive **user** turns. That is an interpretation, not a fact
about the file, so it is reported every time it is applied:

```
How your file was read
  · a list of plain strings was read as successive USER turns (the MT-Bench
    shape). If they are alternating user/assistant turns, rewrite them as
    {"role": ..., "content": ...} objects. — 2 cases, first at
    mtbench.jsonl line 1.messages
```

Migrating the file writes explicit role objects, and the note stops.

### 2. System prompts

```json
{"id": "a", "system": "You are a support agent. Never promise a refund.",
 "messages": [{"role": "user", "content": "Refund?"}]}
```

### 3. Metadata

Unchanged from version 1: any key evallint does not recognise is preserved in
`case.metadata`. You can also nest it explicitly under `metadata`, which
additionally silences the strict-mode near-miss warning for keys that look like
fields.

### 4. Multiple acceptable answers

```json
{"id": "nl", "input": "What is the capital of the Netherlands?",
 "expected": "Amsterdam",
 "acceptable": ["The Hague", "Den Haag"]}
```

`expected` stays the canonical answer — it is what a report quotes.
`case.acceptable_answers()` returns all of them, canonical first:

```python
>>> case.acceptable_answers()
('Amsterdam', 'The Hague', 'Den Haag')
```

A bare string is one answer, not its characters. `"acceptable": "yes"` gives
`("yes",)`, never `("y", "e", "s")`.

### 5. Structured expected outputs

```json
{"id": "invoice", "input": "Extract the total from this invoice: ...",
 "expected": {"total": 42.50, "currency": "EUR"},
 "expected_schema": {"type": "object", "required": ["total", "currency"]}}
```

`expected_schema` is **stored, not enforced**. Validating arguments against it
would mean shipping a JSON Schema implementation, and being wrong about a spec
would produce confident findings about correct data. Use it to tell your own
scorer what shape to expect.

### 6. Evaluation criteria / rubrics

```json
{"id": "essay", "input": "Write a paragraph on tides.",
 "rubric": {
   "criteria": [
     {"name": "accuracy", "description": "No factual errors", "weight": 3},
     {"name": "concision", "weight": 1}],
   "scale": [1, 5]}}
```

Shorthands: a bare list is a list of criteria, and a bare string is a criterion
name. `"rubric": ["accuracy", "tone"]` is valid.

Weights are **relative, and stored as you wrote them.** 3 and 1 mean the same as
0.75 and 0.25; the shares are derived, so the file and the report cannot
disagree:

```python
>>> case.rubric.weight_shares()
{'accuracy': 0.75, 'concision': 0.25}
```

Two refusals worth knowing: duplicate criterion names (their weights would
silently add), and all-zero weights (no aggregate would be definable, and
falling back to an unweighted mean would invent a scoring rule you did not ask
for).

**A rubric is an input to a judge, not ground truth.** Storing one does not make
the resulting scores reliable. If a judge applies it, use
`EvaluatorReliability` to find out whether the grading is reproducible.

### 7. Tool calls

```json
{"id": "weather",
 "messages": [
   {"role": "user", "content": "What's the weather in Oslo?"},
   {"role": "assistant",
    "tool_calls": [{"id": "t1", "name": "get_weather",
                    "arguments": {"city": "Oslo"}}]},
   {"role": "tool", "content": "4C, rain", "tool_call_id": "t1",
    "name": "get_weather"}],
 "tools": [{"name": "get_weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}}}],
 "expected_tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}]}
```

`Message.tool_calls` is what the assistant **did**. `expected_tool_calls` is
what it **should** do — ground truth.

Provider shapes load as-is, so a transcript pasted out of an API response works
without hand-editing: a call nested under `"function"`, arguments serialised as
a JSON string, and `input_schema` in place of `parameters` are all accepted.

Two refusals:

- Only an `assistant` turn may carry tool calls.
- An `expected_tool_calls` entry naming a tool absent from `tools` is an
  **error**, not a finding. The case could only ever fail, so no scoring run
  could produce a useful number from it.

---

## Validation errors

Errors are **collected, not raised one at a time.** A 5000-case file with
mistakes in 40 cases takes one run to diagnose:

```
errs.jsonl: 4 schema problems in 4 cases

  errs.jsonl line 2.system: must be a string, got int
      a system prompt is one block of text; put per-case settings in metadata instead
  errs.jsonl line 3.acceptable: must be a list of answers, got an object
      an object is one structured answer, so put it in a list: "acceptable": [{...}]
  errs.jsonl line 4.expected_schema: must be a JSON Schema object, got list
      e.g. {"type": "object", "required": ["total"]}
  errs.jsonl line 5.messages[0].role: 'nobody' is not a valid role; allowed: system, user, assistant, tool
```

A suggestion appears only when there is a near miss. `'nobody'` gets none, since
it is not close to any role; `'assistent'` gets `did you mean 'assistant'?`.

Each issue carries the file, the line, the JSON path within the record, what was
wrong, and what was allowed. Line numbers are **real file lines** — a version
header occupies line 1, so the first case is on line 2.

Beyond 20 issues the message prints a count (`... and 143 more`); all of them
stay on `SchemaValidationError.issues` for a script that wants them.

A **consequence is not stacked on its cause**. A case whose `messages` failed to
parse also has no `input`, but only the first is reported — otherwise one
mistake would read as two.

---

## Migration

Migration is **optional**. It does two things, both about the file rather than
the tool:

```console
$ evallint gsm8k.jsonl --migrate-to gsm8k_v2.jsonl
gsm8k.jsonl -> gsm8k_v2.jsonl
  2 case(s), schema version 2 declared
  columns renamed to canonical fields:
    question -> input
    answer -> expected
    (so --map is no longer needed for this file)
  verified: the written file reloads to identical cases
```

(Two cases because that is the size of the GSM8K-shaped fixture this was run
against, not a claim about GSM8K itself.)

1. **It makes the column mapping permanent.** GSM8K needs `--map input=question`
   on every run, or relies on inference being re-done and re-reported each time.
   After migrating, the mapping is the identity and the flag is gone.
2. **It declares the version**, turning the lenient demotions above into errors.

Library equivalent:

```python
from evallint import migrate_file
report = migrate_file("gsm8k.jsonl", "gsm8k_v2.jsonl")
print(report.render())
```

### What migration will not do

It does **not** turn `input` into a one-turn `messages` list. The conversion
would be lossless, but it would make a simple file harder to read in exchange
for nothing. Converting a single-prompt case into a conversation is a decision
about your eval, so it is left to whoever adds the second turn — see below.

It does not touch your data: field names change, the version line is added, and
nothing else. No reordering, no normalising, no dropping of unrecognised columns.

### Nothing is written unless it verifies

Migration reloads what it produced and compares every case field by field
against the input. If anything differs, it raises and writes **nothing**:

```
Error: out.jsonl NOT written: case 'a' changed during migration
(label: 'cls' became None). This is a bug in evallint, not in your file —
please report it with the input that caused it.
```

It also refuses to overwrite an existing file (without `--overwrite`), to write
onto its own source, and to write CSV — which can hold neither a nested
conversation nor a version declaration.

---

## Worked migration: adding a second turn by hand

**Before** — a single-prompt case testing whether the model refuses:

```json
{"id": "refund-1", "input": "Can I get a refund?", "expected": "no_promise",
 "label": "policy"}
```

**Step 1 — run the mechanical part.**

```console
$ evallint evalset.jsonl --migrate-to evalset_v2.jsonl
```

**Step 2 — turn the case you care about into a conversation.** The single user
turn is exactly equivalent to the `input` you had, so start there and add:

```json
{"id": "refund-1",
 "system": "You are a support agent. Never promise a refund.",
 "messages": [
   {"role": "user", "content": "My order never arrived."},
   {"role": "assistant", "content": "I'm sorry to hear that. Let me check."},
   {"role": "user", "content": "Can I get a refund?"}],
 "expected": "no_promise",
 "acceptable": ["escalate_to_human"],
 "rubric": ["refuses_politely", "offers_alternative"],
 "label": "policy"}
```

**Step 3 — confirm what evallint now sees.**

```python
>>> case = load("evalset_v2.jsonl").cases[0]
>>> case.input                      # what every check reads
"user: My order never arrived.\nassistant: I'm sorry to hear that. Let me check.\nuser: Can I get a refund?"
>>> case.system                     # held apart, so it cannot inflate similarity
'You are a support agent. Never promise a refund.'
>>> case.acceptable_answers()
('no_promise', 'escalate_to_human')
```

The case now exercises the multi-turn behaviour, and every check that ran on the
one-line version still runs.

---

## Field reference

| Field | Type | Version | Notes |
|---|---|---|---|
| `id` | string | 1 | Auto-generated as `case_N` if absent |
| `input` | string | 1 | **Required** unless `messages` is present |
| `expected` | anything | 1 | Scalar, list, or (v2) an object |
| `label` | string / number | 1 | Numbers normalise to strings |
| *(anything else)* | anything | 1 | Preserved in `metadata` |
| `messages` | list of turns | 2 | Derives `input` |
| `system` | string | 2 | Excluded from `input` |
| `acceptable` | list | 2 | Additional correct answers |
| `expected_schema` | object | 2 | JSON Schema; stored, not enforced |
| `rubric` | object or list | 2 | Criteria with relative weights |
| `tools` | list | 2 | Tools the model was allowed to call |
| `expected_tool_calls` | list | 2 | Calls it should have made |

Column aliases are inferred for `messages` (`conversation`, `turns`, `dialogue`,
`dialog`, `chat`, `chat_history`, `conversations`) and `system`
(`system_prompt`, `system_message`, `system_instruction`), on the same terms as
the existing aliases: exactly one candidate is accepted, two or more is an error
naming them, and the resolved mapping is always reported.

---

## What this does not do

- **It does not interpret `expected_schema` or `rubric`.** Both are stored for
  your scorer and your judge. No check reads them.
- **No check yet reasons per-turn.** Checks see the flattened `input`, so
  cross-turn leakage is detected as text overlap within one string rather than
  attributed to a specific turn. Per-turn analysis is a separate change and is
  not pretended to exist here.
- **`acceptable` is not read by `GroundTruthCheck`.** Declaring alternatives
  does not currently suppress or trigger any ambiguity finding; the check still
  looks only at `expected`.
