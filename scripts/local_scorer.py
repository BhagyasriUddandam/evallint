#!/usr/bin/env python
"""Local (Ollama) scorer backend for Check 1 — the discrimination check.

NOT part of the shipped library. Like the Anthropic backend, this file imports
the check and does not modify it. That is the claim being tested: the check
never learns which provider is behind the scorer. Swapping a paid frontier API
for three local GGUF models required ZERO changes to
src/evalcheck/checks/discrimination.py.

WHY A SECOND BACKEND
    1. It is the actual proof that the injected-scorer interface is
       provider-agnostic. One backend is a design claim; two is evidence.
    2. Local inference is free, so the "results depend entirely on the model
       pair you chose" limitation can finally be tested rather than merely
       stated — run a wide pair and a narrow pair and compare.

MODEL LADDER (verified present via /api/tags)
    llama3.2:3b   3.2B   no thinking capability
    qwen3:8b      8.2B   thinking-capable
    qwen3:14b    14.8B   thinking-capable

USAGE
    ollama serve                       # if not already running
    uv run python scripts/local_scorer.py --pair wide
    uv run python scripts/local_scorer.py --pair narrow
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Protocol

# scripts/ is not a package, so make sibling modules importable whether this
# file is run directly or loaded by path from a test.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cache import ResponseCache  # noqa: E402
from evalcheck.checks import DiscriminationCheck, ScorerError  # noqa: E402
from evalcheck.io import load  # noqa: E402

OLLAMA_URL = "http://localhost:11434"
REQUEST_TIMEOUT_S = 180.0  # local generation is slower than an API call

EVAL_SET = Path(__file__).resolve().parents[1] / "examples" / "sample_evalset.jsonl"
CACHE_PATH = Path(__file__).resolve().parent / ".local_cache.json"

# Weakest -> strongest. This ordering is load-bearing: the check uses model
# order to detect inversions, so a mis-ordered pair inverts the finding.
MODEL_LADDER = ("llama3.2:3b", "qwen3:8b", "qwen3:14b")

PAIRS = {
    "wide": ("llama3.2:3b", "qwen3:14b"),
    "narrow": ("qwen3:8b", "qwen3:14b"),
}

# Verified against /api/show: qwen3 reports the "thinking" capability,
# llama3.2 does not. Sending `think` to a model without the capability is an
# error — the same shape of trap as sending `effort` to Haiku on the Anthropic
# side, so it gets the same treatment: decide per model, never blanket.
THINKING_MODELS = frozenset({"qwen3:8b", "qwen3:14b"})

DEFAULT_JUDGE = "qwen3:14b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 0

# Thinking models can still emit reasoning inline even with think disabled.
# The judge parse keys on the response STARTING with PASS, so a leaked
# <think> block would silently turn every verdict into FAIL — a systematic
# grading failure that looks like a capability finding.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

ANSWER_SYSTEM = (
    "You are a customer support agent for a SaaS product. Answer the "
    "customer's message directly and concisely. Do not ask clarifying "
    "questions.\n"
    "\n"
    "You have full access to this customer's account: their billing history, "
    "individual charges, subscription status, and payment methods. You can "
    "see the specific transactions they are asking about. You are authorized "
    "to issue refunds yourself, without escalating.\n"
    "\n"
    "Act on the account rather than telling the customer to go check it "
    "themselves, and say what you have done."
)

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


class ChatClient(Protocol):
    """The whole surface a scorer backend needs.

    Declared as a Protocol so the tests can pass a plain fake object and never
    import ollama, open a socket, or require a running service.
    """

    def chat(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float,
        seed: int,
    ) -> dict[str, Any]: ...


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks a reasoning model may emit inline."""
    return _THINK_BLOCK.sub("", text).strip()


def chat_options(model: str, temperature: float, seed: int) -> dict[str, Any]:
    """Per-model request body for /api/chat.

    `think` is sent ONLY for models that declare the capability. Sending it to
    llama3.2:3b is an error, and sending nothing to qwen3 leaves reasoning on,
    which is slow and pollutes the graded text.
    """
    body: dict[str, Any] = {
        "stream": False,
        # seed alongside temperature: temperature 0 is greedy decoding, but
        # pinning the seed too makes the run reproducible rather than merely
        # low-variance.
        "options": {"temperature": temperature, "seed": seed},
    }
    if model in THINKING_MODELS:
        body["think"] = False
    return body


class OllamaClient:
    """Talks to a local Ollama server over its HTTP API.

    httpx rather than the `ollama` package: it is already present, and one
    less dependency for a demo script. The import is lazy so this module can
    be imported (and tested) with no HTTP library and no server.
    """

    def __init__(
        self, base_url: str = OLLAMA_URL, timeout: float = REQUEST_TIMEOUT_S
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http: Any = None

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=self.timeout)
        return self._http

    def chat(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float,
        seed: int,
    ) -> dict[str, Any]:
        import httpx

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **chat_options(model, temperature, seed),
        }
        try:
            response = self._client().post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"{model} timed out after {self.timeout:.0f}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"cannot reach Ollama at {self.base_url} — is `ollama serve` "
                f"running? ({exc})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{model} returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        return parse_chat_response(model, response.json())


def parse_chat_response(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Turn an /api/chat body into the record shape the cache stores.

    Deliberately strict: an empty answer is raised rather than cached, because
    a cached empty string would be graded FAIL forever and look like a real
    model failure.
    """
    text = strip_thinking((payload.get("message") or {}).get("content") or "")
    if not text:
        raise RuntimeError(f"{model} returned no usable text (after thinking strip)")

    return {
        "model": model,
        "text": text,
        "usage": {
            "input": payload.get("prompt_eval_count") or 0,
            "output": payload.get("eval_count") or 0,
        },
    }


def answer_key(
    model: str,
    system: str,
    case_input: str,
    temperature: float,
    seed: int,
    repeat: int = 1,
) -> dict[str, Any]:
    """Identity of a generated answer.

    Every input that can change the response is here: model, system prompt,
    the case text, temperature, and seed. Leaving any of them out means a
    later experiment silently reads answers produced under different settings.
    """
    return {
        "backend": "ollama",
        "kind": "answer",
        "model": model,
        "system": system,
        "input": case_input,
        "temperature": temperature,
        "seed": seed,
        "repeat": repeat,
    }


def judge_key(
    judge_model: str,
    case_id: str,
    system: str,
    prompt: str,
    temperature: float,
    seed: int,
    repeat: int = 1,
) -> dict[str, Any]:
    """Identity of a grading verdict.

    ``judge_model`` is in the key because the judge is selectable here, unlike
    the Anthropic backend where it was a module constant. A configurable value
    that is not in the key is precisely the staleness bug this project already
    recorded as a lesson.
    """
    return {
        "backend": "ollama",
        "kind": "judge",
        "judge_model": judge_model,
        "case": case_id,
        "system": system,
        "prompt": prompt,
        "temperature": temperature,
        "seed": seed,
        "repeat": repeat,
    }


def judge_prompt_for(case, answer_text: str) -> str:
    """Render the judge prompt. One function so every path formats it the same
    way and therefore hashes to the same key."""
    return JUDGE_TEMPLATE.format(
        question=case.input, expected=case.expected, answer=answer_text
    )


def build_scorer(
    client: ChatClient,
    cache: ResponseCache,
    judge_model: str,
    answer_system: str = ANSWER_SYSTEM,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    repeat: int = 1,
):
    """Return the `score(case, model) -> bool` the check consumes.

    Same closure shape as the Anthropic backend, and the check cannot tell
    them apart. That is the entire point of this file.
    """

    def score(case, model: str) -> bool:
        if case.expected is None:
            raise RuntimeError(
                f"case '{case.id}' has no 'expected' value, so it cannot be graded"
            )

        answer = cache.get_or_call(
            answer_key(model, answer_system, case.input, temperature, seed, repeat),
            lambda: client.chat(
                model, answer_system, case.input, temperature, seed
            ),
        )

        prompt = judge_prompt_for(case, answer["text"])
        verdict = cache.get_or_call(
            judge_key(
                judge_model, case.id, JUDGE_SYSTEM, prompt, temperature, seed, repeat
            ),
            lambda: client.chat(
                judge_model, JUDGE_SYSTEM, prompt, temperature, seed
            ),
        )

        return verdict["text"].strip().upper().startswith("PASS")

    return score


def judge_limitations(judge_model: str, pair: tuple[str, str]) -> list[str]:
    """State what this judge choice cannot tell you.

    With a three-model ladder and the strongest model in both requested pairs,
    an independent judge and a capable judge are mutually exclusive. Rather
    than pick silently, say which trade was made.
    """
    notes = []
    if judge_model in pair:
        notes.append(
            f"JUDGE IS ALSO A SCORED MODEL: '{judge_model}' grades its own "
            "answers in this run. Self-preference bias inflates its pass rate, "
            "which pushes results toward 'the strong model discriminates' "
            "regardless of whether it does. Treat the pass rate for "
            f"'{judge_model}' as an upper bound, not a measurement."
        )
    if judge_model != MODEL_LADDER[-1]:
        notes.append(
            f"Judge '{judge_model}' is not the strongest available model "
            f"('{MODEL_LADDER[-1]}'), so grading quality is itself a "
            "limitation — a wrong verdict is indistinguishable from a model "
            "failure in the output."
        )
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pair",
        choices=sorted(PAIRS),
        default="wide",
        help="Which capability gap to score. wide=%s, narrow=%s"
        % (PAIRS["wide"], PAIRS["narrow"]),
    )
    parser.add_argument(
        "--judge",
        default=DEFAULT_JUDGE,
        help=f"Grading model (default {DEFAULT_JUDGE}). Pick one outside the "
        "scored pair for an independent verdict, at the cost of a weaker judge.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Repeat index, part of the cache key so repeats do not collide.",
    )
    parser.add_argument("--base-url", default=OLLAMA_URL)
    args = parser.parse_args(argv)

    weak, strong = PAIRS[args.pair]
    eval_set = load(EVAL_SET)
    cache = ResponseCache(CACHE_PATH)  # no pricing: local inference is free

    print(f"eval set   : {eval_set.source} ({len(eval_set)} cases)")
    print(f"pair       : {args.pair}  weak={weak}  strong={strong}")
    print(f"judge      : {args.judge}")
    print(f"sampling   : temperature={args.temperature} seed={args.seed}")
    print(f"repeat     : {args.repeat}")
    print(f"cache      : {CACHE_PATH} ({len(cache)} entries)\n")

    client = OllamaClient(base_url=args.base_url)
    check = DiscriminationCheck(
        build_scorer(
            client,
            cache,
            args.judge,
            ANSWER_SYSTEM,
            args.temperature,
            args.seed,
            args.repeat,
        ),
        [weak, strong],
    )

    try:
        result = check.run(eval_set)
    except ScorerError as exc:
        print(f"\nscoring failed: {exc}", file=sys.stderr)
        print(
            "re-run the same command to resume — cached responses are kept.",
            file=sys.stderr,
        )
        return 1

    print(f"summary : {result.summary}\n")
    print(f"FINDINGS ({len(result.findings)})")
    for finding in result.findings:
        print(f"  [{finding.severity.value.upper():7}] {finding.message}")
        if finding.case_ids:
            shown = ", ".join(finding.case_ids[:6])
            extra = len(finding.case_ids) - 6
            print(f"            cases: {shown}{' (+%d more)' % extra if extra > 0 else ''}")

    print("\nSTATS")
    print(json.dumps(result.stats, indent=2))

    print(f"\nLIMITATIONS ({len(result.limitations)})")
    for limitation in result.limitations:
        print(f"  - {limitation}")
    for note in judge_limitations(args.judge, (weak, strong)):
        print(f"  - DEMO-SPECIFIC: {note}")
    print(
        "  - DEMO-SPECIFIC: local quantized models (Q4_K_M) are weaker than "
        "their full-precision counterparts, so absolute pass rates here do not "
        "transfer to the hosted versions of the same models."
    )

    total = cache.hits + cache.misses
    print("\nCOST")
    print(f"  generations this run : {cache.misses}")
    print(f"  served from cache    : {cache.hits} of {total}")
    print("  spend                : $0.0000 (local inference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
