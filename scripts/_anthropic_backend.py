"""Anthropic transport, shared by the demo backends.

NOT part of the shipped library — `src/evalcheck` never imports a provider SDK.

Extracted for the same reason as `_cache.py`: two copies of a request builder
drift, and the per-model quirks encoded here are exactly the kind that fail at
runtime rather than at import. One place to be wrong is better than two.

Used by:
    demo_discrimination_real.py   answers AND grading via Anthropic
    local_scorer.py               grading only (answers come from Ollama)
"""

from __future__ import annotations

from typing import Any

import anthropic

__all__ = [
    "MAX_RETRIES",
    "PRICING",
    "REQUEST_TIMEOUT_S",
    "UnsupportedSamplingError",
    "call_model",
    "make_client",
    "request_kwargs",
    "supports_temperature",
]

# Without an explicit timeout the SDK default is 10 minutes, so one stalled
# connection parks the whole run with no output — it looks like a hang, not a
# failure. These two bound it instead.
#
# The SDK retries timeouts, so worst case per call is
# REQUEST_TIMEOUT_S x (MAX_RETRIES + 1) = 90s, not 30s. Bounded either way.
REQUEST_TIMEOUT_S = 30.0
MAX_RETRIES = 2

# List prices per million tokens (input, output), for run-cost reporting only.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Sampling parameters (temperature / top_p / top_k) were REMOVED from the
# Claude 5 family and from Opus 4.7/4.8 — sending temperature at all returns a
# 400 on Opus 5, and any non-default value returns a 400 on Sonnet 5. They
# remain available on Opus 4.6 / Sonnet 4.6 and the 4.5-generation models.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
)


def supports_temperature(model: str) -> bool:
    """Whether this model accepts a `temperature` request parameter."""
    return not model.startswith(_NO_SAMPLING_PREFIXES)


class UnsupportedSamplingError(RuntimeError):
    """A temperature was requested for a model that rejects the parameter."""


def make_client(
    timeout: float = REQUEST_TIMEOUT_S, max_retries: int = MAX_RETRIES
) -> anthropic.Anthropic:
    """Construct a bounded client. Reads ANTHROPIC_API_KEY from the env."""
    return anthropic.Anthropic(timeout=timeout, max_retries=max_retries)


def request_kwargs(model: str, temperature: float | None = None) -> dict[str, Any]:
    """Per-model request parameters. The knobs are not the same on every model.

    Three real differences, each of which would be a runtime error or silent
    cost if ignored:
      - `output_config.effort` is rejected by Haiku 4.5.
      - On the Claude 5 models thinking is ON when the field is omitted, and
        max_tokens caps thinking PLUS the answer. A max_tokens sized for the
        answer alone would truncate the response mid-sentence.
      - `temperature` is rejected outright by the Claude 5 family and Opus
        4.7/4.8. Rather than let that surface as a 400 partway through a paid
        run, refuse before the first call.
    """
    if model.startswith("claude-haiku"):
        kwargs: dict[str, Any] = {"max_tokens": 1000}
    else:
        kwargs = {"max_tokens": 4000, "output_config": {"effort": "low"}}

    if temperature is not None:
        if not supports_temperature(model):
            raise UnsupportedSamplingError(
                f"{model} does not accept a temperature parameter (the Claude 5 "
                f"family and Opus 4.7/4.8 removed sampling parameters; sending "
                f"one returns HTTP 400). Either drop --temperature, or switch "
                f"to models that still accept it (e.g. claude-opus-4-6, "
                f"claude-sonnet-4-6, claude-haiku-4-5)."
            )
        kwargs["temperature"] = temperature

    return kwargs


def call_model(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    prompt: str,
    temperature: float | None = None,
) -> dict[str, Any]:
    """One request. Returns the record shape the shared cache stores."""
    try:
        response = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            **request_kwargs(model, temperature),
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
