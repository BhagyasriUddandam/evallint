"""Tests for scripts/gsm8k_discrimination.py.

This harness is not shipped, but it produces NUMBERS THAT GET REPORTED, and
every bug it has had so far was invisible in its output rather than a crash:

  - the cost report summed usage["input_tokens"] where the backend returns
    usage["input"], so it printed $0.0000 for every run
  - the smoke-test extrapolation divided a per-case cost by 2 and then
    multiplied by the case count, quoting half the real spend
  - `repeats=N` was a tautology: one cache key per (model, prompt) meant all N
    repeats returned the same stored response, agreed by construction, and
    reported "0 unstable"

None of those would fail a run. They would produce a confident, wrong,
plausible-looking result — which is the exact failure mode this project exists
to report, so the harness gets tests too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    """Load a scripts/ module by path — scripts/ is not an installed package.

    Registered in sys.modules before exec_module for the same reason as
    test_local_scorer: an unregistered module breaks name-based lookups done
    during module execution.
    """
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gsm8k = _load("gsm8k_discrimination")

from evallint.schema import EvalCase  # noqa: E402


class SpyCache:
    """Records the keys asked for, and never calls anything real."""

    def __init__(self, text: str = "ANSWER: 42") -> None:
        self.keys: list[dict] = []
        self.text = text

    def get_or_call(self, key_parts, call):
        self.keys.append(dict(key_parts))
        return {
            "model": key_parts["model"],
            "text": self.text,
            "usage": {"input": 1, "output": 1},
        }

    def values(self):
        return []


CASE = EvalCase(id="c1", input="How many?", expected="reasoning\n#### 42")


# --- the repeats-versus-cache bug ----------------------------------------


def test_each_repeat_gets_a_distinct_cache_key() -> None:
    """REGRESSION. Without this, repeats=3 returns the same cached response
    three times, every verdict agrees, and the check reports 0 unstable — a
    tautology that reads as "fully reproducible"."""
    cache = SpyCache()
    score = gsm8k.make_scorer(cache, client=None, verbose=False)

    for _ in range(3):
        score(CASE, "model-a")

    frozen = {tuple(sorted(k.items())) for k in cache.keys}
    assert len(frozen) == 3, "three repeats must produce three distinct keys"
    assert [k.get("attempt") for k in cache.keys] == [None, 1, 2]


def test_attempt_zero_omits_the_attempt_field() -> None:
    """The first attempt must not carry `attempt: 0`, so responses already on
    disk from a same-config run are reused rather than re-bought."""
    cache = SpyCache()
    gsm8k.make_scorer(cache, client=None, verbose=False)(CASE, "model-a")
    assert cache.keys == [
        {
            "model": "model-a",
            "system": gsm8k.SYSTEM,
            "prompt": CASE.input,
            # effort IS in the key: it changes the answer, so a run at a
            # different effort must not be served the previous run's responses.
            "effort": gsm8k.EFFORT,
        }
    ]
    assert "attempt" not in cache.keys[0]


def test_attempt_counter_is_per_case_and_per_model() -> None:
    """Counting globally would give case 2 an attempt number of 1 on its FIRST
    call, missing the cache and paying for a response it already had."""
    cache = SpyCache()
    score = gsm8k.make_scorer(cache, client=None, verbose=False)
    other = EvalCase(id="c2", input="Other?", expected="#### 42")

    score(CASE, "model-a")
    score(other, "model-a")
    score(CASE, "model-b")

    assert [k.get("attempt") for k in cache.keys] == [None, None, None]


def test_the_prompt_and_system_are_in_the_key() -> None:
    """Omitting the system prompt was a real bug here: it served answers
    generated under a previous prompt and the output looked fine."""
    cache = SpyCache()
    gsm8k.make_scorer(cache, client=None, verbose=False)(CASE, "m")
    key = cache.keys[0]
    assert key["system"] == gsm8k.SYSTEM
    assert key["prompt"] == CASE.input


# --- answer parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Reasoning. ANSWER: 18", 18.0),
        ("She spent $12 of the $20.\nANSWER: 8", 8.0),  # last number is a trap
        ("ANSWER: 1,234", 1234.0),
        ("answer: -5", -5.0),
        ("ANSWER: $42", 42.0),
        ("ANSWER: 7\nANSWER: 9", 9.0),  # last marker wins
    ],
)
def test_answer_parsing(reply: str, expected: float) -> None:
    assert gsm8k._number(reply, gsm8k._MARKER, "ANSWER: line") == expected


def test_an_unparseable_reply_raises_instead_of_scoring_wrong() -> None:
    """Scoring an unparseable reply as False is how a judge in this project
    produced a clean-looking run that had measured nothing."""
    with pytest.raises(gsm8k.UnparseableAnswerError):
        gsm8k._number("I think it is probably 18.", gsm8k._MARKER, "ANSWER: line")


def test_scorer_compares_against_the_gsm8k_marker() -> None:
    right = gsm8k.make_scorer(SpyCache("ANSWER: 42"), None, False)
    wrong = gsm8k.make_scorer(SpyCache("ANSWER: 41"), None, False)
    assert right(CASE, "m") is True
    assert wrong(CASE, "m") is False


# --- the cost report ------------------------------------------------------


def test_cost_uses_the_keys_the_backend_actually_returns() -> None:
    """REGRESSION. This summed usage["input_tokens"], but call_model returns
    usage["input"], so every run reported $0.0000."""

    class Cache:
        def values(self):
            return [
                {"model": gsm8k.WEAK, "usage": {"input": 100, "output": 300}},
                {"model": gsm8k.STRONG, "usage": {"input": 100, "output": 900}},
            ]

    totals = gsm8k.cost_of(Cache())
    weak_in, weak_out = gsm8k.PRICING[gsm8k.WEAK]
    strong_in, strong_out = gsm8k.PRICING[gsm8k.STRONG]

    assert totals[gsm8k.WEAK]["usd"] == pytest.approx(
        100 / 1e6 * weak_in + 300 / 1e6 * weak_out
    )
    assert totals[gsm8k.STRONG]["usd"] == pytest.approx(
        100 / 1e6 * strong_in + 900 / 1e6 * strong_out
    )
    assert all(t["usd"] > 0 for t in totals.values()), "cost must never be zero"


def test_unknown_model_costs_zero_rather_than_crashing() -> None:
    """A model missing from PRICING should not take down a completed run — the
    results are already bought by then."""

    class Cache:
        def values(self):
            return [{"model": "some-new-model", "usage": {"input": 5, "output": 5}}]

    assert gsm8k.cost_of(Cache())["some-new-model"]["usd"] == 0.0


# --- the CI constraint, encoded ------------------------------------------


def test_the_sdk_is_not_imported_at_module_scope() -> None:
    """REGRESSION. A module-level `from _anthropic_backend import ...` pulls in
    the anthropic SDK, which is a dev-only dependency. That broke all four CI
    jobs that install the core package alone -- they could not even COLLECT
    this file. Deferring the import into the functions that call it keeps the
    harness importable, and every test here runnable, with no provider SDK
    present. This asserts it statically so the regression fails locally.
    """
    import ast

    tree = ast.parse((SCRIPTS / "gsm8k_discrimination.py").read_text())
    top_level: list[str] = []
    for node in tree.body:  # module scope only, not function bodies
        if isinstance(node, ast.Import):
            top_level += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module)

    offenders = [
        n for n in top_level
        if n == "anthropic" or n.startswith("_anthropic_backend")
    ]
    assert not offenders, (
        f"{offenders} imported at module scope; import it inside the function "
        "that needs it, or the core-only CI jobs cannot collect these tests"
    )
