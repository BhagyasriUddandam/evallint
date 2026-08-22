#!/usr/bin/env python
"""Run the discrimination check against GSM8K — a public set I did not write.

NOT part of the shipped library. Nothing here is imported by src/.

WHY THIS EXISTS
The two cheap checks have been run against five real public eval sets. The
discrimination check — the flagship, and the only one that is not a commodity —
had only ever run on the 20-case example bundled with this repo, which I wrote
myself. "It works on my own data" is the weakest claim in the project.

WHY GSM8K
Because it can be scored WITHOUT AN LLM JUDGE. Every GSM8K answer ends in a
literal marker:

    ...18 every day at the farmer's market.\\n#### 18

So grading is: extract the model's final number, extract the reference number,
compare. That matters more than the cost saving. Two earlier findings in this
project had to be retracted because a model was grading its own answers, and
removing the judge means this result cannot be a grading artifact of that kind.
What remains is a plain arithmetic comparison anyone can re-derive.

WHAT IT DOES NOT PROVE
GSM8K is grade-school arithmetic. A capability gap measured here says nothing
about a capability gap on legal reasoning or code review. It is one dataset,
one domain, one model pair.

COST CONTROL
  - --smoke makes exactly ONE call per model, prints real token counts and the
    computed cost, and exits. Always run this first. It exists because a Kimi
    run earlier in this project would have produced 40 confident, wrong results
    if one call had not been checked by hand beforehand.
  - Every response is cached on disk, keyed on everything that changes the
    output, so a re-run costs nothing.
  - --sample bounds the case count through the check's own sampling.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _anthropic_backend import call_model, make_client  # noqa: E402
from _cache import ResponseCache  # noqa: E402
from _env import load_env_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evallint.checks import DiscriminationCheck  # noqa: E402
from evallint.io import load_with_mapping  # noqa: E402
from evallint.report import render_text  # noqa: E402

DATA = Path("/tmp/extern_evals/gsm8k.jsonl")
CACHE = Path(__file__).resolve().parent / ".gsm8k_cache.json"

WEAK = "claude-haiku-4-5"
STRONG = "claude-opus-5"

# Published list prices, USD per million tokens. Only used to report what a run
# cost; nothing depends on them being current, and they are printed so a wrong
# figure here is visible rather than buried.
PRICING = {
    WEAK: (1.00, 5.00),
    STRONG: (5.00, 25.00),
}

# The scorer needs ONE parseable number back. Asking for a marker is more
# reliable than grabbing the last number in free prose, where "I spent $12 of
# the $20" ends on the wrong figure.
SYSTEM = (
    "Solve the grade-school math problem. Reason briefly, then end your reply "
    "with a final line of exactly this form and nothing after it:\n"
    "ANSWER: <number>\n"
    "The number must be plain digits with no units, no commas and no currency "
    "symbol."
)

_MARKER = re.compile(r"ANSWER:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_REFERENCE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


class UnparseableAnswerError(RuntimeError):
    """The model's reply had no ANSWER: line.

    Deliberately an error rather than a False verdict. Scoring an unparseable
    reply as "wrong" is how you get a clean-looking run that measured nothing —
    the exact mistake a judge in this project made earlier, where every
    unrecognised reply became FAIL and the output looked like a real result.
    """


def _number(text: str, pattern: re.Pattern[str], what: str) -> float:
    match = None
    for match in pattern.finditer(text):  # last match wins
        pass
    if match is None:
        raise UnparseableAnswerError(
            f"no {what} found in: ...{text[-160:]!r}"
        )
    return float(match.group(1).replace(",", ""))


def make_scorer(cache: ResponseCache, client, verbose: bool):
    """Build the injected scorer. This is the whole integration surface:
    evallint calls score(case, model) and never learns who the provider is."""

    def score(case, model: str) -> bool:
        record = cache.get_or_call(
            # Every input that changes the answer is in the key. Omitting the
            # system prompt here was a real bug earlier in this project: it
            # served answers from a previous prompt and looked fine.
            {"model": model, "system": SYSTEM, "prompt": case.input},
            lambda: call_model(client, model, SYSTEM, case.input),
        )
        got = _number(record["text"], _MARKER, "ANSWER: line")
        want = _number(case.expected, _REFERENCE, "#### reference")
        if verbose:
            flag = "ok " if got == want else "MISS"
            print(f"    {flag} {case.id:<10} {model:<20} got {got:g} want {want:g}")
        return got == want

    return score


def cost_of(cache: ResponseCache) -> dict[str, dict]:
    """Sum real token usage out of the cache, per model."""
    totals: dict[str, dict] = {}
    for record in cache.values():
        model = record.get("model")
        usage = record.get("usage") or {}
        t = totals.setdefault(model, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
        t["calls"] += 1
        t["in"] += usage.get("input", 0)
        t["out"] += usage.get("output", 0)
    for model, t in totals.items():
        pin, pout = PRICING.get(model, (0.0, 0.0))
        t["usd"] = t["in"] / 1e6 * pin + t["out"] / 1e6 * pout
    return totals


def report_cost(cache: ResponseCache) -> float:
    totals = cost_of(cache)
    print("\n  cost (from real token usage in the cache):")
    grand = 0.0
    for model, t in sorted(totals.items()):
        print(
            f"    {model:<22} {t['calls']:>4} calls  "
            f"in {t['in']:>7}  out {t['out']:>7}  ${t['usd']:.4f}"
        )
        grand += t["usd"]
    print(f"    {'TOTAL':<22} {'':>4}        {'':>10}  {'':>11}  ${grand:.4f}")
    return grand


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=100,
                    help="how many GSM8K cases to score (default 100)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="ONE call per model, print real cost, then stop")
    ap.add_argument("--verbose", action="store_true",
                    help="print every case's got/want")
    args = ap.parse_args()

    load_env_file()
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (checked env and .env)", file=sys.stderr)
        return 2
    if not DATA.exists():
        print(f"{DATA} not found — re-download the GSM8K sample", file=sys.stderr)
        return 2

    # Dogfooding: GSM8K uses question/answer, not input/expected, so this also
    # exercises the mapping layer rather than hand-writing the field names.
    eval_set, mapping = load_with_mapping(DATA)
    print(f"  loaded {len(eval_set)} cases from {DATA.name}")
    for line in mapping.explain():
        print(f"    {line}")

    cache = ResponseCache(CACHE)
    client = make_client()
    print(f"  cache: {len(cache)} responses already stored")

    if args.smoke:
        case = eval_set.cases[0]
        print(f"\n  SMOKE TEST — one call per model on {case.id}")
        print(f"    question: {case.input[:100]}...")
        scorer = make_scorer(cache, client, verbose=True)
        for model in (WEAK, STRONG):
            before = len(cache)
            correct = scorer(case, model)
            cached = " (from cache)" if len(cache) == before else ""
            print(f"    {model}: correct={correct}{cached}")
        total = report_cost(cache)
        n = args.sample * 2
        print(f"\n  extrapolated to {n} calls: "
              f"~${total / 2 * args.sample:.2f} (2 calls measured)")
        print("  re-run without --smoke to do the full run")
        return 0

    result = DiscriminationCheck(
        make_scorer(cache, client, args.verbose),
        [WEAK, STRONG],              # WEAKEST FIRST — inversion depends on it
        sample=args.sample,
        sample_seed=args.sample_seed,
        max_workers=args.max_workers,
    ).run(eval_set)

    print()
    render_text([result], source=str(DATA), notes=[
        "scored by exact numeric comparison against GSM8K's '#### N' marker — "
        "NO LLM judge, so no self-grading artifact is possible",
    ])
    report_cost(cache)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        # The backend raises RuntimeError with an already-actionable message
        # (bad key, timeout, refusal). Printing that alone beats a traceback
        # whose only useful line is the last one.
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
