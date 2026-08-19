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
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    """Load a scripts/ module by path — scripts/ is not an installed package.

    The module MUST be registered in sys.modules before exec_module: dataclass
    field resolution looks its own module up by name, and an unregistered
    module makes that lookup return None (AttributeError at import time).
    """
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


local_scorer = _load("local_scorer")
_cache = _load("_cache")
_env = _load("_env")

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


def ollama_judge(client, model: str = JUDGE):
    """A local-backend Judge wired to the given fake client."""
    return local_scorer.Judge(
        backend="ollama", model=model, client=client, temperature=0.0, seed=0
    )


def anthropic_judge(client, model: str = local_scorer.ANTHROPIC_JUDGE_MODEL):
    """An Anthropic-backend Judge: no temperature, no seed (both rejected)."""
    return local_scorer.Judge(
        backend="anthropic", model=model, client=client, temperature=None, seed=None
    )


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
    score = local_scorer.build_scorer(client, cache, ollama_judge(client))

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
    score = local_scorer.build_scorer(client, cache, ollama_judge(client))

    assert score(case, STRONG) is True
    assert score(case, WEAK) is False


def test_case_without_expected_is_rejected(cache) -> None:
    eval_set = EvalSet(cases=(EvalCase(id="x", input="q", expected=None),))
    score = local_scorer.build_scorer(FakeClient(), cache, ollama_judge(FakeClient()))

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
        local_scorer.build_scorer(first, _cache.ResponseCache(path), ollama_judge(first)),
        [WEAK, STRONG],
    ).run(eval_set)
    assert first.calls, "first run should have called the client"

    second = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(second, _cache.ResponseCache(path), ollama_judge(second)),
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
        judge_backend="ollama", case_id="c1", system="s", prompt="p",
        temperature=0.0, seed=0, repeat=1,
    )
    a = _cache.cache_key(local_scorer.judge_key(judge_model="qwen3:8b", **base))
    b = _cache.cache_key(local_scorer.judge_key(judge_model="qwen3:14b", **base))

    assert a != b


def test_changing_the_judge_forces_fresh_grading(tmp_path) -> None:
    eval_set = make_set(1)
    path = tmp_path / "c.json"

    first = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(first, _cache.ResponseCache(path), ollama_judge(first, "qwen3:14b")),
        [WEAK, STRONG],
    ).run(eval_set)

    second = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(second, _cache.ResponseCache(path), ollama_judge(second, "qwen3:8b")),
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

    local_scorer.build_scorer(FakeClient(), cache, ollama_judge(FakeClient()))(case, WEAK)

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
    notes = local_scorer.judge_limitations(
        ollama_judge(None, "qwen3:14b"), PAIR := ("qwen3:8b", "qwen3:14b")
    )
    assert any("JUDGE IS ALSO A SCORED MODEL" in n for n in notes)
    assert PAIR[1] in notes[0]


def test_independent_judge_is_not_flagged_for_self_grading() -> None:
    notes = local_scorer.judge_limitations(
        ollama_judge(None, "qwen3:8b"), ("llama3.2:3b", "qwen3:14b")
    )
    assert not any("ALSO A SCORED MODEL" in n for n in notes)


def test_weaker_judge_is_flagged_as_a_grading_limitation() -> None:
    notes = local_scorer.judge_limitations(
        ollama_judge(None, "qwen3:8b"), ("llama3.2:3b", "qwen3:14b")
    )
    assert any("not the strongest available model" in n for n in notes)


def test_both_configured_pairs_contain_the_default_judge() -> None:
    """Documents the real constraint: with this three-model ladder there is no
    pair where the strongest model both scores and stays out of the judging."""
    for name, pair in local_scorer.PAIRS.items():
        assert local_scorer.DEFAULT_JUDGE in pair, name


# --------------------------------------------------------------------------
# Mixed backend: answers from Ollama, grading from Anthropic
# --------------------------------------------------------------------------


class FakeAnthropicJudge:
    """Stands in for the Anthropic judge client.

    Asserts the contract the real adapter must honour: temperature and seed
    are NOT forwarded, because claude-sonnet-5 rejects temperature outright
    and has no seed parameter.
    """

    def __init__(self, verdict: str = "PASS") -> None:
        self.calls: list[dict] = []
        self.verdict = verdict

    def chat(self, model, system, prompt, temperature=None, seed=None):
        assert temperature is None, "temperature must not reach the Anthropic judge"
        assert seed is None, "seed must not reach the Anthropic judge"
        self.calls.append({"model": model, "prompt": prompt})
        return {
            "model": model,
            "text": self.verdict,
            "usage": {"input": 400, "output": 5},
        }


def test_answers_stay_local_while_grading_goes_to_anthropic(cache) -> None:
    """One run, two providers, and the check cannot tell."""
    answers = FakeClient()
    grader = FakeAnthropicJudge(verdict="PASS")
    score = local_scorer.build_scorer(answers, cache, anthropic_judge(grader))

    result = DiscriminationCheck(score, [WEAK, STRONG]).run(make_set(2))

    assert all(c["model"] in (WEAK, STRONG) for c in answers.calls), (
        "answer generation must stay on the local models"
    )
    assert all("REFERENCE ANSWER" not in c["prompt"] for c in answers.calls), (
        "no grading call should have gone to the local client"
    )
    assert len(grader.calls) == 4  # 2 cases x 2 models, all graded remotely
    assert all(c["model"] == local_scorer.ANTHROPIC_JUDGE_MODEL for c in grader.calls)
    assert result.stats["n_scorer_calls"] == 4


def test_switching_judge_backend_does_not_reuse_verdicts(tmp_path) -> None:
    """THE point of putting the backend in the key: an ollama verdict must not
    be served to an anthropic-judged run, or the comparison is meaningless."""
    path = tmp_path / "c.json"
    eval_set = make_set(1)

    local = FakeClient()
    DiscriminationCheck(
        local_scorer.build_scorer(local, _cache.ResponseCache(path), ollama_judge(local)),
        [WEAK, STRONG],
    ).run(eval_set)

    answers = FakeClient()
    grader = FakeAnthropicJudge()
    DiscriminationCheck(
        local_scorer.build_scorer(
            answers, _cache.ResponseCache(path), anthropic_judge(grader)
        ),
        [WEAK, STRONG],
    ).run(eval_set)

    assert grader.calls, "backend switch must re-grade, not reuse ollama verdicts"
    assert answers.calls == [], "answers were unchanged and should still be cached"


def test_judge_backend_is_in_the_judge_key() -> None:
    base = dict(
        judge_model="m", case_id="c1", system="s", prompt="p", repeat=1
    )
    ollama = _cache.cache_key(
        local_scorer.judge_key(judge_backend="ollama", temperature=0.0, seed=0, **base)
    )
    anthropic = _cache.cache_key(
        local_scorer.judge_key(
            judge_backend="anthropic", temperature=None, seed=None, **base
        )
    )
    assert ollama != anthropic


def test_make_judge_records_only_sampling_that_is_actually_sent() -> None:
    """The key must not claim a temperature the request never carried."""
    local = local_scorer.make_judge("ollama", None, (WEAK, STRONG), 0.0, 7, client=object())
    assert (local.backend, local.temperature, local.seed) == ("ollama", 0.0, 7)

    remote = local_scorer.make_judge(
        "anthropic", None, (WEAK, STRONG), 0.0, 7, client=object()
    )
    assert remote.model == local_scorer.ANTHROPIC_JUDGE_MODEL
    assert remote.temperature is None and remote.seed is None


def test_unknown_judge_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown judge backend"):
        local_scorer.make_judge("openai", None, (WEAK, STRONG), 0.0, 0)


def test_anthropic_judge_model_is_the_strongest_available() -> None:
    """A wrong verdict is indistinguishable from a model failure in the
    output, so the grader should be the most capable model on offer."""
    assert local_scorer.ANTHROPIC_JUDGE_MODEL == "claude-opus-5"


def test_anthropic_judge_client_does_not_connect_on_construction() -> None:
    client = local_scorer.AnthropicJudgeClient()
    assert client._client is None
    assert client.model == local_scorer.ANTHROPIC_JUDGE_MODEL
    assert client.calls == 0


# --------------------------------------------------------------------------
# Judge limitations differ by backend
# --------------------------------------------------------------------------


def test_anthropic_judge_does_not_trigger_self_preference_warning() -> None:
    notes = local_scorer.judge_limitations(
        anthropic_judge(None), ("qwen3:8b", "qwen3:14b")
    )
    assert not any("ALSO A SCORED MODEL" in n for n in notes)


def test_anthropic_judge_is_flagged_as_cross_family() -> None:
    notes = local_scorer.judge_limitations(
        anthropic_judge(None), ("qwen3:8b", "qwen3:14b")
    )
    assert any("CROSS-FAMILY JUDGE" in n for n in notes)
    assert any("answer STYLE" in n for n in notes)


def test_anthropic_judge_is_flagged_as_non_deterministic() -> None:
    """Trading self-preference bias for sampling noise is a real trade and the
    output has to say so."""
    notes = local_scorer.judge_limitations(
        anthropic_judge(None), ("qwen3:8b", "qwen3:14b")
    )
    assert any("NON-DETERMINISTIC" in n for n in notes)


# --------------------------------------------------------------------------
# .env loading — convenience that must not become a footgun
# --------------------------------------------------------------------------


def test_env_file_populates_missing_keys(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-fromfile\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = _env.load_env_file(env)

    assert loaded == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-fromfile"


def test_exported_value_always_beats_the_file(tmp_path, monkeypatch) -> None:
    """A key you exported deliberately must not be replaced by a stale file."""
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-stale\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit")

    loaded = _env.load_env_file(env)

    assert loaded == []
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-explicit"


def test_loader_returns_names_never_values(tmp_path, monkeypatch) -> None:
    """Nothing here may leak a secret into a log or a terminal recording."""
    env = tmp_path / ".env"
    env.write_text("SECRET_TOKEN=super-secret-value\n")
    monkeypatch.delenv("SECRET_TOKEN", raising=False)

    loaded = _env.load_env_file(env)

    assert loaded == ["SECRET_TOKEN"]
    assert "super-secret-value" not in repr(loaded)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("KEY=plain", "plain"),
        ('KEY="double quoted"', "double quoted"),
        ("KEY='single quoted'", "single quoted"),
        ("export KEY=exported", "exported"),
        ("  KEY=surrounded  ", "surrounded"),
        ("KEY=with=equals", "with=equals"),
    ],
)
def test_line_shapes(tmp_path, monkeypatch, line, expected) -> None:
    env = tmp_path / ".env"
    env.write_text(line + "\n")
    monkeypatch.delenv("KEY", raising=False)

    _env.load_env_file(env)

    assert os.environ["KEY"] == expected


def test_comments_and_blanks_are_ignored(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nREAL_KEY=1\nnot-a-pair\n")
    monkeypatch.delenv("REAL_KEY", raising=False)

    assert _env.load_env_file(env) == ["REAL_KEY"]


def test_missing_env_file_is_not_an_error(tmp_path) -> None:
    """CI configures the environment directly; absence is the normal case."""
    assert _env.load_env_file(tmp_path / "nope.env") == []


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
