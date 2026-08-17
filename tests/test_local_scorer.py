"""Tests for the local (Ollama) scorer backend.

Every test here uses a fake chat client. No socket is opened, no model is
loaded, and the suite passes with Ollama not installed and not running — which
is the point of taking the client as a parameter rather than constructing one
inside the scorer.

What is under test is the BACKEND's contract: that it returns the bool the
check expects, keys its cache on everything that affects a response, and does
not send a request parameter to a model that rejects it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    """Load a scripts/ module by path — scripts/ is not an installed package."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local_scorer = _load("local_scorer")
_cache = _load("_cache")

from evalcheck.checks import DiscriminationCheck, ScorerError  # noqa: E402
from evalcheck.schema import EvalCase, EvalSet  # noqa: E402

WEAK, STRONG = local_scorer.PAIRS["wide"]
JUDGE = local_scorer.DEFAULT_JUDGE


class FakeClient:
    """Stands in for Ollama. Records calls; answers from a lookup table.

    Verdicts are keyed on (case_input, scored_model) rather than case id,
    because the rendered judge prompt contains the case's input and expected
    answer but never its id — there is nothing else in the prompt to match on.
    """

    def __init__(self, verdicts: dict[tuple[str, str], bool] | None = None) -> None:
        self.calls: list[dict] = []
        self.verdicts = verdicts or {}

    def chat(self, model, system, prompt, temperature, seed):
        self.calls.append(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "temperature": temperature,
                "seed": seed,
            }
        )
        if "REFERENCE ANSWER" in prompt:  # this is a grading call
            graded = self._graded_model(prompt)
            passed = any(
                verdict
                for (case_input, scored), verdict in self.verdicts.items()
                if scored == graded and case_input in prompt
            )
            text = "PASS" if passed else "FAIL"
        else:
            text = f"answer from {model}"
        return {"model": model, "text": text, "usage": {"input": 10, "output": 5}}

    @staticmethod
    def _graded_model(prompt: str) -> str:
        """Which scored model produced the candidate answer in this prompt."""
        for m in local_scorer.MODEL_LADDER:
            if f"answer from {m}" in prompt:
                return m
        return ""


def make_set(n: int = 3) -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(id=f"c{i}", input=f"question {i}", expected=f"answer {i}")
            for i in range(1, n + 1)
        )
    )


@pytest.fixture
def cache(tmp_path):
    return _cache.ResponseCache(tmp_path / "cache.json")


# --------------------------------------------------------------------------
# The scorer satisfies the check's contract
# --------------------------------------------------------------------------


def test_scorer_returns_bool_and_check_consumes_it(cache) -> None:
    """The whole claim: the check runs against a local backend unmodified.

    Only the strong model passes, so all three cases discriminate.
    """
    client = FakeClient(
        verdicts={(f"question {i}", STRONG): True for i in (1, 2, 3)}
    )
    score = local_scorer.build_scorer(client, cache, JUDGE)

    result = DiscriminationCheck(score, [WEAK, STRONG]).run(make_set())

    assert result.check == "discrimination"
    assert result.stats["models"] == [WEAK, STRONG]
    assert result.stats["n_scorer_calls"] == 6  # 3 cases x 2 models
    assert result.stats["n_discriminating"] == 3
    assert result.stats["per_model_pass_rate"] == {WEAK: 0.0, STRONG: 1.0}


def test_pass_verdict_is_true_and_fail_is_false(cache) -> None:
    eval_set = make_set(1)
    case = eval_set.cases[0]
    client = FakeClient(verdicts={("question 1", STRONG): True})
    score = local_scorer.build_scorer(client, cache, JUDGE)

    assert score(case, STRONG) is True
    assert score(case, WEAK) is False


def test_case_without_expected_is_rejected(cache) -> None:
    eval_set = EvalSet(cases=(EvalCase(id="x", input="q", expected=None),))
    score = local_scorer.build_scorer(FakeClient(), cache, JUDGE)

    with pytest.raises(ScorerError, match="no 'expected' value"):
        DiscriminationCheck(score, [WEAK, STRONG]).run(eval_set)


# --------------------------------------------------------------------------
# Cache correctness — the lesson from tasks/lessons.md, enforced
# --------------------------------------------------------------------------


def test_second_run_makes_no_client_calls(tmp_path) -> None:
    eval_set = make_set()
    path = tmp_path / "c.json"

    first = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(first, _cache.ResponseCache(path), JUDGE),
        [WEAK, STRONG],
    ).run(eval_set)
    assert first.calls, "first run should have called the client"

    second = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(second, _cache.ResponseCache(path), JUDGE),
        [WEAK, STRONG],
    ).run(eval_set)
    assert second.calls == [], "cached run must not call the client"


@pytest.mark.parametrize(
    ("field", "a", "b"),
    [
        ("system", {"system": "SYS A"}, {"system": "SYS B"}),
        ("temperature", {"temperature": 0.0}, {"temperature": 0.7}),
        ("seed", {"seed": 0}, {"seed": 1}),
        ("model", {"model": WEAK}, {"model": STRONG}),
        ("repeat", {"repeat": 1}, {"repeat": 2}),
    ],
)
def test_answer_key_covers_every_response_determinant(field, a, b) -> None:
    """If changing an input would change the answer, it must change the key."""
    base = dict(
        model=WEAK, system="s", case_input="q", temperature=0.0, seed=0, repeat=1
    )
    key_a = _cache.cache_key(local_scorer.answer_key(**{**base, **a}))
    key_b = _cache.cache_key(local_scorer.answer_key(**{**base, **b}))

    assert key_a != key_b, f"{field} is missing from the answer cache key"


def test_judge_model_is_part_of_the_judge_key() -> None:
    """The judge is selectable here, so it MUST be keyed — otherwise switching
    judges would silently reuse the previous judge's verdicts."""
    base = dict(
        case_id="c1", system="s", prompt="p", temperature=0.0, seed=0, repeat=1
    )
    a = _cache.cache_key(local_scorer.judge_key(judge_model="qwen3:8b", **base))
    b = _cache.cache_key(local_scorer.judge_key(judge_model="qwen3:14b", **base))

    assert a != b


def test_changing_the_judge_forces_fresh_grading(tmp_path) -> None:
    eval_set = make_set(1)
    path = tmp_path / "c.json"

    first = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(first, _cache.ResponseCache(path), "qwen3:14b"),
        [WEAK, STRONG],
    ).run(eval_set)

    second = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(second, _cache.ResponseCache(path), "qwen3:8b"),
        [WEAK, STRONG],
    ).run(eval_set)

    judge_calls = [c for c in second.calls if "REFERENCE ANSWER" in c["prompt"]]
    assert judge_calls, "switching judges must re-grade, not reuse old verdicts"
    assert all(c["model"] == "qwen3:8b" for c in judge_calls)


def test_cache_is_written_through_on_every_miss(tmp_path) -> None:
    """A crash mid-run must not discard completed work."""
    path = tmp_path / "c.json"
    cache = _cache.ResponseCache(path)
    case = make_set(1).cases[0]

    local_scorer.build_scorer(FakeClient(), cache, JUDGE)(case, WEAK)

    assert path.is_file()
    assert len(_cache.ResponseCache(path)) == 2  # one answer + one verdict


# --------------------------------------------------------------------------
# Per-model request shaping — `think` is not universally accepted
# --------------------------------------------------------------------------


def test_think_is_sent_only_to_thinking_capable_models() -> None:
    """llama3.2:3b has no thinking capability; sending `think` is an error.

    Same trap as `effort` on Haiku in the Anthropic backend.
    """
    assert "think" not in local_scorer.chat_options("llama3.2:3b", 0.0, 0)
    for model in local_scorer.THINKING_MODELS:
        assert local_scorer.chat_options(model, 0.0, 0)["think"] is False


def test_sampling_options_are_passed_through() -> None:
    opts = local_scorer.chat_options("qwen3:14b", 0.0, 7)["options"]
    assert opts == {"temperature": 0.0, "seed": 7}


# --------------------------------------------------------------------------
# Response parsing — a reasoning model can poison the verdict
# --------------------------------------------------------------------------


def test_thinking_blocks_are_stripped() -> None:
    """A leaked <think> block would stop every verdict starting with PASS,
    turning a grading artefact into what looks like a capability finding."""
    assert local_scorer.strip_thinking("<think>hmm, maybe</think>PASS") == "PASS"
    assert local_scorer.strip_thinking("<think>\nmulti\nline\n</think>\nFAIL") == "FAIL"
    assert local_scorer.strip_thinking("PASS") == "PASS"


def test_thinking_only_response_is_an_error_not_an_empty_answer() -> None:
    """Caching an empty string would grade FAIL forever and look like a real
    model failure."""
    with pytest.raises(RuntimeError, match="no usable text"):
        local_scorer.parse_chat_response(
            "qwen3:14b", {"message": {"content": "<think>only thinking</think>"}}
        )


def test_parse_chat_response_extracts_text_and_usage() -> None:
    record = local_scorer.parse_chat_response(
        "qwen3:8b",
        {
            "message": {"content": "  PASS  "},
            "prompt_eval_count": 123,
            "eval_count": 45,
        },
    )

    assert record == {
        "model": "qwen3:8b",
        "text": "PASS",
        "usage": {"input": 123, "output": 45},
    }


def test_missing_usage_counts_default_to_zero() -> None:
    record = local_scorer.parse_chat_response("qwen3:8b", {"message": {"content": "x"}})
    assert record["usage"] == {"input": 0, "output": 0}


# --------------------------------------------------------------------------
# Judge independence is reported, not assumed
# --------------------------------------------------------------------------


def test_judge_inside_the_scored_pair_is_flagged() -> None:
    notes = local_scorer.judge_limitations("qwen3:14b", PAIR := ("qwen3:8b", "qwen3:14b"))
    assert any("JUDGE IS ALSO A SCORED MODEL" in n for n in notes)
    assert PAIR[1] in notes[0]


def test_independent_judge_is_not_flagged_for_self_grading() -> None:
    notes = local_scorer.judge_limitations("qwen3:8b", ("llama3.2:3b", "qwen3:14b"))
    assert not any("ALSO A SCORED MODEL" in n for n in notes)


def test_weaker_judge_is_flagged_as_a_grading_limitation() -> None:
    notes = local_scorer.judge_limitations("qwen3:8b", ("llama3.2:3b", "qwen3:14b"))
    assert any("not the strongest available model" in n for n in notes)


def test_both_configured_pairs_contain_the_default_judge() -> None:
    """Documents the real constraint: with this three-model ladder there is no
    pair where the strongest model both scores and stays out of the judging."""
    for name, pair in local_scorer.PAIRS.items():
        assert local_scorer.DEFAULT_JUDGE in pair, name


# --------------------------------------------------------------------------
# The suite must never need a running service
# --------------------------------------------------------------------------


def test_module_imports_without_ollama_running() -> None:
    """Constructing the real client must not connect; the HTTP client is built
    lazily on first use."""
    client = local_scorer.OllamaClient(base_url="http://127.0.0.1:1")
    assert client._http is None
    assert client.base_url == "http://127.0.0.1:1"


def test_model_ladder_is_ordered_weakest_first() -> None:
    """Order is load-bearing: the check reads inversion from model order."""
    for _, (weak, strong) in local_scorer.PAIRS.items():
        assert local_scorer.MODEL_LADDER.index(weak) < local_scorer.MODEL_LADDER.index(
            strong
        )
