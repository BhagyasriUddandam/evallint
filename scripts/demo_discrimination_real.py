#!/usr/bin/env python
"""One-off validation: run Check 1 against REAL models.

NOT part of the shipped library. Nothing in src/evalcheck imports this, and
this file imports the discrimination check without modifying it — which is the
point. The check never learns that Anthropic exists; it only ever sees a
`score(case, model) -> bool` callable. Swapping the two functions below for
OpenAI, Bedrock, or a local model would require no change to src/.

WHAT IT COSTS
    n_cases x 2 models generation calls, plus one judge call per generated
    answer. On the bundled 20-case example that is 80 calls on the first run
    and 0 on every run after, because every response is cached to disk.

BUDGET PROTECTION
    Each response is written to the cache file the moment it arrives, not at
    the end. A crash on case 15 keeps the 14 answers you already paid for.
    Delete the cache file to force a fresh run.

USAGE
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/demo_discrimination_real.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import anthropic

from evalcheck.checks import DiscriminationCheck, ScorerError
from evalcheck.io import load

# Ordered WEAKEST FIRST — the check uses that ordering to detect inversions,
# so getting it backwards would invert the finding rather than raise.
WEAK_MODEL = "claude-haiku-4-5"
STRONG_MODEL = "claude-opus-5"

# Deliberately a third model, neither of the two being scored. If the strong
# model graded its own answers, self-preference would inflate its pass rate and
# manufacture discrimination that isn't there — the exact confound this tool
# exists to catch. Sharing a model family is a residual limitation, stated in
# the output.
JUDGE_MODEL = "claude-sonnet-5"

EVAL_SET = Path(__file__).resolve().parents[1] / "examples" / "sample_evalset.jsonl"
CACHE_PATH = Path(__file__).resolve().parent / ".response_cache.json"

# Without an explicit timeout the SDK default is 10 minutes, so one stalled
# connection parks the whole run with no output — it looks like a hang, not a
# failure. These two bound it instead.
#
# The SDK retries timeouts, so worst case per call is
# REQUEST_TIMEOUT_S x (MAX_RETRIES + 1) = 90s, not 30s. Bounded either way.
# If legitimate calls start timing out (a long thinking trace on a hard case),
# raise the timeout rather than the retry count.
REQUEST_TIMEOUT_S = 30.0
MAX_RETRIES = 2

# List prices per million tokens (input, output), for the run-cost estimate
# only. Sonnet 5 has lower introductory pricing at the time of writing; using
# the standard rate overestimates, which is the safe direction for a budget.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Two scorer prompts, run as an A/B.
#
# The original prompt granted a persona and nothing else, so a model that
# correctly refused to invent account access or policy timing was graded WRONG
# for it — the harness rewarded confident fabrication over calibrated
# uncertainty. Both variants below fix that by granting the authority the
# reference answers assume.
#
# They are split because the POLICY variant contains text that is close to
# verbatim reference answers (billing_005 expects "applies at the next renewal
# and is prorated"; billing_007 expects the 5-7 day pending-authorization
# line). A case passing under POLICY is therefore weak evidence — the model may
# only be repeating the prompt. AUTHORITY leaks no reference text, so it is the
# clean test of "did the agent simply lack authority?". Run both and compare.
_PERSONA = (
    "You are a customer support agent for a SaaS product. Answer the "
    "customer's message directly and concisely. Do not ask clarifying "
    "questions."
)

_AUTHORITY = (
    "You have full access to this customer's account: their billing history, "
    "individual charges, subscription status, and payment methods. You can "
    "see the specific transactions they are asking about. You are authorized "
    "to issue refunds yourself, without escalating."
)

_POLICY = (
    "Billing policy:\n"
    "- Plan and billing-cycle changes take effect at the next renewal and are "
    "prorated against the unused portion of the current period.\n"
    "- A confirmed duplicate charge is refunded in full; refunds reach the "
    "original payment method in 5-10 business days.\n"
    "- A pending authorization is not a completed charge. It often appears "
    "alongside the real charge and drops off the statement within 5-7 days."
)

_ACT = (
    "Act on the account rather than telling the customer to go check it "
    "themselves, and say what you have done."
)

ANSWER_SYSTEM_AUTHORITY = "\n\n".join([_PERSONA, _AUTHORITY, _ACT])
ANSWER_SYSTEM_POLICY = "\n\n".join([_PERSONA, _AUTHORITY, _POLICY, _ACT])

VARIANTS = {
    "authority": ANSWER_SYSTEM_AUTHORITY,
    "policy": ANSWER_SYSTEM_POLICY,
}

JUDGE_SYSTEM = "You are a strict, consistent grader."

JUDGE_TEMPLATE = """You are grading one case of a model evaluation.

CUSTOMER MESSAGE:
{question}

REFERENCE ANSWER (what a correct response must convey):
{expected}

CANDIDATE ANSWER:
{answer}

Does the candidate convey the same substantive resolution as the reference?
Ignore differences in tone, length, politeness, and phrasing — grade only
whether the substance matches. Reply with exactly one word: PASS or FAIL."""


def cache_key(key_parts: dict[str, Any]) -> str:
    """Hash a request identity into a cache key.

    One function, used by both the write path and --show. If the two built
    keys separately they could drift, and --show would silently report
    "not in cache" for entries that are sitting right there.

    sort_keys so an identical request always hashes identically; without it,
    dict ordering could produce a cache miss and a charge.
    """
    return hashlib.sha256(
        json.dumps(key_parts, sort_keys=True).encode("utf-8")
    ).hexdigest()


def answer_key(model: str, system: str, case_input: str) -> dict[str, Any]:
    """Identity of a generated answer.

    EVERY input that can change the response belongs here. The system prompt
    is one of them: it was omitted at first, which meant editing ANSWER_SYSTEM
    to run a different experiment would silently serve answers generated under
    the OLD prompt — the experiment would look like it ran and would report
    nothing but the previous run's results.

    Keyed on the input text rather than the case id, so two cases with
    byte-identical inputs share one cached answer. On the bundled example that
    saves two calls, because billing_004 and billing_014 are the planted exact
    duplicate.
    """
    return {"kind": "answer", "model": model, "system": system, "input": case_input}


def judge_key(case_id: str, system: str, prompt: str) -> dict[str, Any]:
    """Identity of a grading verdict.

    Keyed on the fully rendered judge prompt, not just the answer text. The
    rendered prompt already contains the question, the reference answer, and
    the candidate answer, so editing JUDGE_TEMPLATE *or* correcting an
    'expected' value in the eval set correctly invalidates the old verdict.
    Keying on the answer alone would have kept grading against a reference
    that no longer exists.

    ``case_id`` is redundant for correctness (the prompt determines the
    result) and is kept only to keep distinct cases in distinct slots.
    """
    return {
        "kind": "judge",
        "model": JUDGE_MODEL,
        "case": case_id,
        "system": system,
        "prompt": prompt,
    }


def judge_prompt_for(case, answer_text: str) -> str:
    """Render the judge prompt. One function so the write path and --show
    cannot format it differently and hash to different keys."""
    return JUDGE_TEMPLATE.format(
        question=case.input, expected=case.expected, answer=answer_text
    )


class ResponseCache:
    """A write-through JSON cache so a re-run never re-pays for a response.

    Written to disk on every miss rather than once at the end: an exception on
    case 15 of 20 must not throw away the 14 answers already paid for.

    Spend is accumulated here, at the point of the miss, rather than totalled
    afterwards from a list of records — cached responses cost nothing, and a
    tally that could not tell a hit from a miss would report a re-run as
    costing full price.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        self.spend_usd = 0.0
        self._data: dict[str, Any] = {}
        if path.is_file():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get_or_call(self, key_parts: dict[str, Any], call) -> dict[str, Any]:
        key = cache_key(key_parts)

        if key in self._data:
            self.hits += 1
            return self._data[key]

        self.misses += 1
        record = call()
        in_rate, out_rate = PRICING.get(record["model"], (0.0, 0.0))
        self.spend_usd += record["usage"]["input"] / 1e6 * in_rate
        self.spend_usd += record["usage"]["output"] / 1e6 * out_rate

        self._data[key] = record
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        return record


def request_kwargs(model: str) -> dict[str, Any]:
    """Per-model request parameters. The knobs are not the same on every model.

    Two real differences, each of which would be a runtime error or silent cost
    if ignored:
      - `output_config.effort` is rejected by Haiku 4.5.
      - On the Claude 5 models thinking is ON when the field is omitted, and
        max_tokens caps thinking PLUS the answer. A max_tokens sized for the
        answer alone would truncate the response mid-sentence.
    """
    if model.startswith("claude-haiku"):
        return {"max_tokens": 1000}
    return {"max_tokens": 4000, "output_config": {"effort": "low"}}


def call_model(
    client: anthropic.Anthropic, model: str, system: str, prompt: str
) -> dict[str, Any]:
    try:
        response = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **request_kwargs(model),
        )
    except anthropic.APITimeoutError as exc:
        # Already retried MAX_RETRIES times by the SDK before landing here.
        raise RuntimeError(
            f"{model} timed out after {MAX_RETRIES + 1} attempts of "
            f"{REQUEST_TIMEOUT_S:.0f}s"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(f"{model} could not be reached: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError(f"{model} rate limited after retries: {exc}") from exc

    # Check stop_reason before touching content: a refusal returns HTTP 200
    # with an empty content list, so indexing straight into content[0] would
    # raise IndexError and disguise a policy decline as a code bug.
    if response.stop_reason == "refusal":
        raise RuntimeError(f"{model} declined the request (stop_reason=refusal)")

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(
            f"{model} returned no text (stop_reason={response.stop_reason})"
        )

    return {
        "model": model,
        "text": text,
        "usage": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    }


def build_scorer(
    client: anthropic.Anthropic, cache: ResponseCache, answer_system: str
):
    """Return the `score(case, model) -> bool` the check consumes.

    This closure is the entire integration surface. evalcheck calls it once per
    case per model and interprets only the bool it returns.

    ``answer_system`` is passed in rather than read from a module global so the
    value that goes into the cache key is provably the value that goes to the
    API — the two cannot drift apart.
    """

    def score(case, model: str) -> bool:
        if case.expected is None:
            raise RuntimeError(
                f"case '{case.id}' has no 'expected' value, so it cannot be graded"
            )

        answer = cache.get_or_call(
            answer_key(model, answer_system, case.input),
            lambda: call_model(client, model, answer_system, case.input),
        )

        prompt = judge_prompt_for(case, answer["text"])
        verdict = cache.get_or_call(
            judge_key(case.id, JUDGE_SYSTEM, prompt),
            lambda: call_model(client, JUDGE_MODEL, JUDGE_SYSTEM, prompt),
        )

        return verdict["text"].strip().upper().startswith("PASS")

    return score


def _block(label: str, text: str) -> None:
    print(f"  {label}")
    for line in textwrap.wrap(text, width=76) or [""]:
        print(f"    {line}")
    print()


def show_cases(case_ids: list[str], answer_system: str) -> int:
    """Print the full paper trail for named cases. Reads only; never calls out.

    This is the audit surface for the check's own verdicts: a case recorded as
    'inverted' is either a real capability inversion or a grader artefact, and
    the only way to tell them apart is to read what the models actually said.
    """
    if not CACHE_PATH.is_file():
        print(f"no cache at {CACHE_PATH} — run the script without --show first.",
              file=sys.stderr)
        return 1

    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    cases = {case.id: case for case in load(EVAL_SET)}

    missing = [cid for cid in case_ids if cid not in cases]
    if missing:
        print(f"not in the eval set: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"reading {len(data)} cached responses from {CACHE_PATH.name}")
    print("no API calls will be made\n")

    for case_id in case_ids:
        case = cases[case_id]
        print("=" * 80)
        print(f"CASE {case_id}")
        print("=" * 80)
        _block("INPUT", case.input)
        _block("EXPECTED (reference answer)", str(case.expected))

        for label, model in (("weak", WEAK_MODEL), ("strong", STRONG_MODEL)):
            answer = data.get(cache_key(answer_key(model, answer_system, case.input)))
            if answer is None:
                print(
                    f"  --- {model} ({label}) --- not in cache "
                    "(no run yet, or the prompts changed since the last one)\n"
                )
                continue

            prompt = judge_prompt_for(case, answer["text"])
            verdict = data.get(cache_key(judge_key(case_id, JUDGE_SYSTEM, prompt)))
            if verdict is None:
                ruling = "NO VERDICT IN CACHE"
            else:
                raw = verdict["text"].strip()
                passed = raw.upper().startswith("PASS")
                ruling = f"{'PASS' if passed else 'FAIL'}  (judge said: {raw!r})"

            print(f"  --- {model} ({label}) --- judged {ruling}")
            _block("ANSWER", answer["text"])

    return 0


def report_cost(cache: ResponseCache) -> None:
    total = cache.hits + cache.misses
    print("\nCOST")
    print(f"  API calls this run : {cache.misses}")
    print(f"  served from cache  : {cache.hits} of {total}")
    print(f"  estimated spend    : ${cache.spend_usd:.4f} (list prices)")
    print(f"  re-run cost        : $0.0000 — delete {CACHE_PATH.name} to force a fresh run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--show",
        nargs="+",
        metavar="CASE_ID",
        help="Print the cached answers and judge verdicts for these cases and "
        "exit. Reads the cache only — makes no API calls and needs no API key.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="policy",
        help="Which scorer system prompt to use. 'authority' grants account "
        "access and refund authority only; 'policy' adds billing policy facts "
        "(some of which are close to reference answers). Default: policy.",
    )
    args = parser.parse_args(argv)
    answer_system = VARIANTS[args.variant]

    if args.show:
        # Return before the client is constructed, so --show cannot spend money
        # even by accident.
        return show_cases(args.show, answer_system)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export it and re-run.", file=sys.stderr)
        return 1

    eval_set = load(EVAL_SET)
    client = anthropic.Anthropic(
        timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES
    )
    cache = ResponseCache(CACHE_PATH)

    planned = len(eval_set) * 2 * 2  # 2 models, each answer graded once
    print(f"eval set   : {eval_set.source} ({len(eval_set)} cases)")
    print(f"variant    : {args.variant} ({len(answer_system)} chars of system prompt)")
    print(f"weak model : {WEAK_MODEL}")
    print(f"strong     : {STRONG_MODEL}")
    print(f"judge      : {JUDGE_MODEL}")
    print(f"cache      : {CACHE_PATH}")
    print(f"up to {planned} API calls, minus whatever the cache already holds\n")

    check = DiscriminationCheck(
        build_scorer(client, cache, answer_system), [WEAK_MODEL, STRONG_MODEL]
    )

    try:
        result = check.run(eval_set)
    except ScorerError as exc:
        # A failed call aborts the run rather than being scored as a FAIL, and
        # that is deliberate: `score` must return a bool, so "continuing" past
        # a timeout would mean inventing a verdict. A fabricated FAIL would
        # land in n_floor or n_inverted and be indistinguishable from a real
        # one — a silent lie in the exact tool built to catch silent lies.
        #
        # Resumability comes from the cache instead: every response already
        # paid for is on disk, so re-running the same command picks up where
        # this left off and only pays for what is missing.
        print(f"\nscoring failed: {exc}", file=sys.stderr)
        print(
            "re-run the same command to resume — cached responses are kept.",
            file=sys.stderr,
        )
        report_cost(cache)
        return 1

    print(f"summary : {result.summary}\n")
    print(f"FINDINGS ({len(result.findings)})")
    for finding in result.findings:
        print(f"  [{finding.severity.value.upper():7}] {finding.message}")
        if finding.case_ids:
            shown = ", ".join(finding.case_ids[:6])
            extra = len(finding.case_ids) - 6
            more = f" (+{extra} more)" if extra > 0 else ""
            print(f"            cases: {shown}{more}")

    print("\nSTATS")
    print(json.dumps(result.stats, indent=2))

    print(f"\nLIMITATIONS ({len(result.limitations)})")
    for limitation in result.limitations:
        print(f"  - {limitation}")
    print(
        "  - DEMO-SPECIFIC: the grader is an LLM judge, so a case can be "
        "recorded as failed because the judge was wrong rather than because "
        "the model was. The judge is a third model to avoid self-preference "
        "bias, but it shares a model family with both scored models."
    )

    report_cost(cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
