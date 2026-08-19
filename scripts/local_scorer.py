#!/usr/bin/env python
"""Local (Ollama) scorer backend for Check 1 — the discrimination check.

NOT part of the shipped library. Like the Anthropic backend, this file imports
the check and does not modify it. That is the claim being tested: the check
never learns which provider is behind the scorer. Swapping a paid frontier API
for three local GGUF models required ZERO changes to
src/evallint/checks/discrimination.py.

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
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# scripts/ is not a package, so make sibling modules importable whether this
# file is run directly or loaded by path from a test.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cache import ResponseCache  # noqa: E402
from _env import DEFAULT_ENV_PATH, load_env_file  # noqa: E402
from evallint.checks import DiscriminationCheck, ScorerError  # noqa: E402
from evallint.io import load  # noqa: E402

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

# Grading can come from a different provider than the answers. With a
# three-model local ladder the strongest model sits inside both configured
# pairs, so a local judge always grades its own answers somewhere — an
# Anthropic judge is the only way to get a grader that is independent of both
# scored models.
JUDGE_BACKENDS = ("ollama", "anthropic")
# Opus 5 rather than Sonnet 5: the judge should be the most capable model
# available, since a wrong verdict is indistinguishable from a model failure
# in the output. It is also still independent of both scored local models.
ANTHROPIC_JUDGE_MODEL = "claude-opus-5"

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
        self.calls = 0  # actual generations; cache hits never reach here
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

        self.calls += 1
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
    judge_backend: str,
    judge_model: str,
    case_id: str,
    system: str,
    prompt: str,
    temperature: float | None,
    seed: int | None,
    repeat: int = 1,
) -> dict[str, Any]:
    """Identity of a grading verdict.

    ``judge_backend`` is in the key as well as ``judge_model``, because model
    names alone do not identify a grader once two providers are in play — and
    because sampling means something different on each side (Ollama honours
    temperature and seed; the Claude 5 models reject temperature outright, so
    the Anthropic judge stores None for both). Keying on the model name only
    would let a backend switch read the previous backend's verdicts, which is
    the same staleness bug already recorded in tasks/lessons.md.
    """
    return {
        "backend": judge_backend,
        "kind": "judge",
        "judge_model": judge_model,
        "case": case_id,
        "system": system,
        "prompt": prompt,
        "temperature": temperature,
        "seed": seed,
        "repeat": repeat,
    }


@dataclass(frozen=True)
class Judge:
    """Everything about the grader, in one object.

    Bundled because the five fields must stay consistent with each other:
    the client that gets called, the model name, and the sampling values that
    were actually sent all feed the same cache key. Passing them as five loose
    arguments is how they drift.
    """

    backend: str
    model: str
    client: Any
    temperature: float | None
    seed: int | None


class AnthropicJudgeClient:
    """Adapts the Anthropic transport to the local backend's `chat` shape.

    Deliberately DROPS temperature and seed. `claude-sonnet-5` returns a 400
    for any non-default temperature and has no seed parameter at all, so the
    Anthropic judge cannot be made deterministic the way the Ollama judge can.
    That is a real trade — self-preference bias is removed, sampling noise in
    the grader is introduced — and it is reported as a limitation rather than
    hidden here.
    """

    def __init__(self, client: Any = None, model: str = ANTHROPIC_JUDGE_MODEL) -> None:
        self._client = client
        self.model = model
        self.calls = 0

    def _ensure(self) -> Any:
        if self._client is None:
            import _anthropic_backend

            self._client = _anthropic_backend.make_client()
        return self._client

    def chat(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        import _anthropic_backend

        self.calls += 1
        # temperature/seed intentionally not forwarded — see class docstring.
        return _anthropic_backend.call_model(self._ensure(), model, system, prompt)


def judge_prompt_for(case, answer_text: str) -> str:
    """Render the judge prompt. One function so every path formats it the same
    way and therefore hashes to the same key."""
    return JUDGE_TEMPLATE.format(
        question=case.input, expected=case.expected, answer=answer_text
    )


def build_scorer(
    client: ChatClient,
    cache: ResponseCache,
    judge: Judge,
    answer_system: str = ANSWER_SYSTEM,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    repeat: int = 1,
):
    """Return the `score(case, model) -> bool` the check consumes.

    Same closure shape as the Anthropic backend, and the check cannot tell
    them apart. That is the entire point of this file — and with an Anthropic
    judge the single run spans two providers without the check noticing.
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
                judge.backend,
                judge.model,
                case.id,
                JUDGE_SYSTEM,
                prompt,
                judge.temperature,
                judge.seed,
                repeat,
            ),
            lambda: judge.client.chat(
                judge.model, JUDGE_SYSTEM, prompt, judge.temperature, judge.seed
            ),
        )

        return verdict["text"].strip().upper().startswith("PASS")

    return score


def make_judge(
    backend: str,
    model: str | None,
    pair: tuple[str, str],
    temperature: float,
    seed: int,
    base_url: str = OLLAMA_URL,
    client: Any = None,
) -> Judge:
    """Assemble the grader for this run.

    Sampling values are resolved HERE, once, so that the values placed in the
    cache key are the same values the client will actually send.
    """
    if backend not in JUDGE_BACKENDS:
        raise ValueError(f"unknown judge backend {backend!r}")

    if backend == "anthropic":
        return Judge(
            backend="anthropic",
            model=model or ANTHROPIC_JUDGE_MODEL,
            client=client or AnthropicJudgeClient(),
            # None, not 0.0: the Claude 5 models reject temperature and have no
            # seed. Recording the values that were NOT sent would be a lie in
            # the cache key.
            temperature=None,
            seed=None,
        )

    return Judge(
        backend="ollama",
        model=model or DEFAULT_JUDGE,
        client=client or OllamaClient(base_url=base_url),
        temperature=temperature,
        seed=seed,
    )


def judge_limitations(judge: Judge, pair: tuple[str, str]) -> list[str]:
    """State what this judge choice cannot tell you.

    Every option here trades one confound for another, so the job is to name
    which trade was made rather than to imply the grader is neutral.
    """
    notes = []

    if judge.backend == "anthropic":
        # Independent of both scored models, so no self-preference — but
        # independence is not neutrality.
        notes.append(
            f"CROSS-FAMILY JUDGE: '{judge.model}' is from a different model "
            "family than the scored models, so it is independent of both — no "
            "self-preference bias. But it may systematically favour or "
            "disfavour the answer STYLE of a different family (length, "
            "hedging, formatting, how directly an answer is asserted). That is "
            "a different confound, not the absence of one: a shift in results "
            "versus a local judge is evidence the grader changed, not proof "
            "either grader is right."
        )
        notes.append(
            "NON-DETERMINISTIC GRADING: the Claude 5 models reject a "
            "temperature parameter, so unlike the Ollama judge at temperature "
            "0 this grader cannot be pinned. Repeated runs can return different "
            "verdicts for identical answers — use --repeat to measure that "
            "rather than assuming a single run is stable."
        )
        return notes

    if judge.model in pair:
        notes.append(
            f"JUDGE IS ALSO A SCORED MODEL: '{judge.model}' grades its own "
            "answers in this run. Self-preference bias inflates its pass rate, "
            "which pushes results toward 'the strong model discriminates' "
            "regardless of whether it does. Treat the pass rate for "
            f"'{judge.model}' as an upper bound, not a measurement. "
            "--judge-backend anthropic removes this confound."
        )
    if judge.model != MODEL_LADDER[-1]:
        notes.append(
            f"Judge '{judge.model}' is not the strongest available model "
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
        "--judge-backend",
        choices=JUDGE_BACKENDS,
        default="ollama",
        help="Where grading runs. 'ollama' (default) keeps everything local "
        "and free but the judge is always one of the scored models. "
        f"'anthropic' grades with {ANTHROPIC_JUDGE_MODEL} — independent of "
        "both scored models, costs money, needs ANTHROPIC_API_KEY. Answers "
        "always come from Ollama either way.",
    )
    parser.add_argument(
        "--judge",
        default=None,
        help=f"Grading model. Defaults to {DEFAULT_JUDGE} for the ollama "
        f"backend and {ANTHROPIC_JUDGE_MODEL} for the anthropic backend.",
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

    if args.judge_backend == "anthropic":
        load_env_file()
    if args.judge_backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "--judge-backend anthropic needs ANTHROPIC_API_KEY. Put it in "
            f"{DEFAULT_ENV_PATH.name} at the repo root, export it, or use "
            "the default --judge-backend ollama.",
            file=sys.stderr,
        )
        return 1

    judge = make_judge(
        args.judge_backend,
        args.judge,
        (weak, strong),
        args.temperature,
        args.seed,
        base_url=args.base_url,
    )

    # Pricing only matters for the Anthropic judge. Ollama models are absent
    # from the table, so answer generations contribute exactly $0 and the
    # cache's spend counter is, by construction, judge spend alone.
    pricing = None
    if judge.backend == "anthropic":
        import _anthropic_backend

        pricing = _anthropic_backend.PRICING
    cache = ResponseCache(CACHE_PATH, pricing=pricing)

    answer_client = OllamaClient(base_url=args.base_url)

    print(f"eval set   : {eval_set.source} ({len(eval_set)} cases)")
    print(f"pair       : {args.pair}  weak={weak}  strong={strong}")
    print(f"answers    : ollama (local, free)")
    print(f"judge      : {judge.backend}:{judge.model}", end="")
    print(" [independent of both scored models]"
          if judge.backend == "anthropic" or judge.model not in (weak, strong)
          else " [ALSO A SCORED MODEL — see limitations]")
    print(f"sampling   : answers temperature={args.temperature} seed={args.seed}", end="")
    print(f" | judge temperature={judge.temperature} seed={judge.seed}")
    print(f"repeat     : {args.repeat}")
    print(f"cache      : {CACHE_PATH} ({len(cache)} entries)\n")

    check = DiscriminationCheck(
        build_scorer(
            answer_client,
            cache,
            judge,
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
    for note in judge_limitations(judge, (weak, strong)):
        print(f"  - DEMO-SPECIFIC: {note}")
    print(
        "  - DEMO-SPECIFIC: local quantized models (Q4_K_M) are weaker than "
        "their full-precision counterparts, so absolute pass rates here do not "
        "transfer to the hosted versions of the same models."
    )

    total = cache.hits + cache.misses
    judge_calls = getattr(judge.client, "calls", 0)
    print("\nCOST")
    print(f"  cache misses this run : {cache.misses} (of {total} lookups)")
    print(f"  served from cache     : {cache.hits}")
    print(
        f"  answers  : {max(cache.misses - judge_calls, 0)} local generations"
        "  -> $0.0000"
    )
    if judge.backend == "anthropic":
        print(f"  grading  : {judge_calls} {judge.model} calls  -> ${cache.spend_usd:.4f}")
        print(f"  TOTAL SPEND           : ${cache.spend_usd:.4f}")
    else:
        print(f"  grading  : {judge_calls} local calls  -> $0.0000")
        print("  TOTAL SPEND           : $0.0000 (fully local)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
