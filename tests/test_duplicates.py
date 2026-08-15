"""Tests for the duplicate check.

Almost every test here injects a fake embedder instead of loading a real
model. That is the point of making the embedder injectable: these tests are
fast, offline, deterministic, and they test the CLUSTERING logic rather than
re-testing sentence-transformers, which is not our code.

One test at the bottom does use the real model end to end, and skips cleanly
if it cannot be loaded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from evalcheck.checks import DuplicateCheck
from evalcheck.io import load
from evalcheck.schema import EvalCase, EvalSet

EXAMPLE_SET = Path(__file__).resolve().parents[1] / "examples" / "sample_evalset.jsonl"


def make_set(texts: list[str]) -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(id=f"d{i}", input=text, expected="ok", label="x")
            for i, text in enumerate(texts, start=1)
        )
    )


def bag_of_words(texts: Sequence[str]) -> np.ndarray:
    """A tiny deterministic stand-in for a sentence embedder.

    Identical text scores 1.0, disjoint vocabulary scores 0.0, and word order
    is ignored — close enough to a real embedder's behaviour to exercise the
    clustering, with none of the cost or nondeterminism.
    """
    vocab = sorted({word for text in texts for word in text.lower().split()})
    index = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(texts), len(vocab)))
    for row, text in enumerate(texts):
        for word in text.lower().split():
            matrix[row, index[word]] += 1
    return matrix


def at_angles(degrees: list[float]):
    """An embedder placing each case at a chosen angle on the unit circle.

    Cosine similarity between two cases is then cos(angle difference), which
    lets a test state the exact similarity matrix it wants.
    """

    def embedder(texts: Sequence[str]) -> np.ndarray:
        assert len(texts) == len(degrees)
        radians = np.radians(degrees)
        return np.stack([np.cos(radians), np.sin(radians)], axis=1)

    return embedder


def messages(result) -> str:
    return " | ".join(f.message for f in result.findings)


# --------------------------------------------------------------------------
# Stays quiet on known-good input
# --------------------------------------------------------------------------


def test_distinct_cases_are_not_flagged() -> None:
    """Three unrelated cases share no vocabulary. Nothing should be reported."""
    eval_set = make_set(
        [
            "the cat sat on the mat",
            "quantum mechanics describes subatomic particles",
            "banana bread needs ripe fruit",
        ]
    )
    result = DuplicateCheck(embedder=bag_of_words).run(eval_set)

    assert result.findings == ()
    assert result.stats["n_clusters"] == 0
    assert result.stats["effective_n_cases"] == 3
    assert "no near-duplicates" in result.summary


def test_similar_but_below_threshold_is_not_flagged() -> None:
    """Two cases 40 degrees apart (cosine 0.77) are related, not duplicates."""
    result = DuplicateCheck(embedder=at_angles([0, 40])).run(make_set(["a", "b"]))

    assert result.findings == ()
    assert result.stats["max_similarity"] == pytest.approx(math.cos(math.radians(40)), abs=1e-3)


# --------------------------------------------------------------------------
# Fires on known-bad input
# --------------------------------------------------------------------------


def test_identical_cases_are_flagged_as_identical() -> None:
    eval_set = make_set(
        ["how do I reset my password", "how do I reset my password", "unrelated topic here"]
    )
    result = DuplicateCheck(embedder=bag_of_words).run(eval_set)

    assert len(result.findings) == 1
    assert "identical input text" in result.findings[0].message
    assert result.findings[0].case_ids == ("d1", "d2")
    assert result.stats["n_redundant_cases"] == 1
    assert result.stats["effective_n_cases"] == 2


def test_reordered_words_are_caught_as_near_duplicates() -> None:
    """The realistic case: same question, different word order."""
    eval_set = make_set(
        [
            "I forgot my password and cannot log in",
            "I cannot log in and I forgot my password",
        ]
    )
    result = DuplicateCheck(embedder=bag_of_words).run(eval_set)

    assert len(result.findings) == 1
    assert result.findings[0].case_ids == ("d1", "d2")


def test_cluster_of_three_is_reported_as_one_finding() -> None:
    """Three phrasings of one scenario is one problem, not three pair reports."""
    eval_set = make_set(
        [
            "charged twice for my subscription",
            "charged twice for my subscription",
            "charged twice for my subscription",
            "where do I download an invoice",
        ]
    )
    result = DuplicateCheck(embedder=bag_of_words).run(eval_set)

    assert len(result.findings) == 1
    assert result.findings[0].case_ids == ("d1", "d2", "d3")
    assert result.stats["n_cases_in_clusters"] == 3
    assert result.stats["effective_n_cases"] == 2


# --------------------------------------------------------------------------
# Transitivity: the documented weakness of connected components
# --------------------------------------------------------------------------


def test_transitive_cluster_reports_its_pairwise_range() -> None:
    """A~B and B~C but A!~C still forms one cluster of three.

    That is a real property of connected components, not a bug, so the finding
    reports the pairwise range (here 0.64-0.91) to make it visible that the
    weakest link in the cluster is below the threshold.
    """
    result = DuplicateCheck(embedder=at_angles([0, 25, 50])).run(make_set(["a", "b", "c"]))

    assert len(result.findings) == 1
    assert result.findings[0].case_ids == ("d1", "d2", "d3")
    assert "0.64-0.91" in result.findings[0].message

    cluster = result.stats["clusters"][0]
    assert cluster["min_similarity"] == pytest.approx(0.6428, abs=1e-3)
    assert cluster["max_similarity"] == pytest.approx(0.9063, abs=1e-3)
    assert cluster["min_similarity"] < DuplicateCheck().threshold


def test_transitivity_is_named_in_the_limitations() -> None:
    result = DuplicateCheck(embedder=bag_of_words).run(make_set(["a b", "c d"]))
    assert any("connected components" in lim for lim in result.limitations)


# --------------------------------------------------------------------------
# Configuration and robustness
# --------------------------------------------------------------------------


def test_threshold_is_configurable() -> None:
    eval_set = make_set(["a", "b"])

    assert DuplicateCheck(embedder=at_angles([0, 25])).run(eval_set).findings
    strict = DuplicateCheck(threshold=0.95, embedder=at_angles([0, 25]))
    assert strict.run(eval_set).findings == ()


@pytest.mark.parametrize("threshold", [0, -0.5, 1.5])
def test_nonsense_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        DuplicateCheck(threshold=threshold)


def test_unnormalised_embeddings_still_work() -> None:
    """Cosine similarity ignores vector length, and so must this check: a
    custom embedder returning long vectors must not change the answer."""

    def scaled(texts: Sequence[str]) -> np.ndarray:
        return bag_of_words(texts) * np.array([[1.0], [500.0], [0.01]])

    eval_set = make_set(["same text here", "same text here", "totally different words"])
    result = DuplicateCheck(embedder=scaled).run(eval_set)

    assert result.findings[0].case_ids == ("d1", "d2")


def test_zero_vector_does_not_divide_by_zero() -> None:
    def with_zero(texts: Sequence[str]) -> np.ndarray:
        return np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    result = DuplicateCheck(embedder=with_zero).run(make_set(["a", "b", "c"]))

    assert result.findings[0].case_ids == ("d1", "d2")


def test_bad_embedder_shape_is_rejected() -> None:
    result_shape = lambda texts: np.zeros((len(texts) + 1, 4))  # noqa: E731
    with pytest.raises(ValueError, match="one vector per text"):
        DuplicateCheck(embedder=result_shape).run(make_set(["a", "b"]))


def test_single_case_set_is_handled() -> None:
    """EvalSet allows one case, so this must not crash on an empty pair list."""
    result = DuplicateCheck(embedder=bag_of_words).run(make_set(["only one"]))

    assert result.findings == ()
    assert "nothing to compare" in result.summary
    assert result.stats["max_similarity"] is None


def test_injected_embedder_is_recorded_in_stats() -> None:
    """The report must say what produced the numbers, not imply a model ran."""
    result = DuplicateCheck(embedder=bag_of_words).run(make_set(["a b", "c d"]))
    assert result.stats["embedder"] == "custom"
    assert DuplicateCheck().model_name == "all-MiniLM-L6-v2"


def test_a_case_is_never_its_own_duplicate() -> None:
    """The similarity matrix diagonal is all 1.0 and must be excluded."""
    result = DuplicateCheck(embedder=bag_of_words).run(
        make_set(["alpha beta", "gamma delta"])
    )
    assert result.stats["n_clusters"] == 0


# --------------------------------------------------------------------------
# End to end with the real model (skips if it cannot be loaded)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_check() -> DuplicateCheck:
    check = DuplicateCheck()
    try:
        check._get_embedder()(["warm up the model"])
    except Exception as exc:  # pragma: no cover - depends on local cache/network
        pytest.skip(f"sentence-transformers model unavailable: {exc}")
    return check


def test_example_evalset_duplicates_are_caught_by_the_real_model(
    real_check: DuplicateCheck,
) -> None:
    """The three clusters planted in examples/sample_evalset.jsonl, found with
    the actual embedding model rather than a stand-in."""
    result = real_check.run(load(EXAMPLE_SET))
    clusters = {tuple(c["case_ids"]) for c in result.stats["clusters"]}

    assert clusters == {
        ("billing_004", "billing_014"),
        ("billing_001", "billing_002", "billing_003"),
        ("password_001", "password_002"),
    }
    assert result.stats["n_redundant_cases"] == 4
    assert result.stats["effective_n_cases"] == 16
    assert "identical input text" in messages(result)


def test_real_model_does_not_flag_the_distinct_cases(
    real_check: DuplicateCheck,
) -> None:
    """Specificity: the 14 genuinely distinct cases must stay unflagged."""
    result = real_check.run(load(EXAMPLE_SET))
    flagged = {case_id for f in result.findings for case_id in f.case_ids}

    assert "billing_008" not in flagged  # sales tax
    assert "bug_001" not in flagged  # UI bug
    assert "refund_001" not in flagged  # refund request
    assert len(flagged) == 7


def test_the_documented_separation_gap_is_real(real_check: DuplicateCheck) -> None:
    """The DEFAULT_THRESHOLD comment justifies 0.85 with measured numbers.

    Pin those numbers so the justification cannot quietly rot: if the model or
    the example set changes, this fails loudly instead of leaving a comment
    asserting something no longer true.
    """
    eval_set = load(EXAMPLE_SET)
    cases = list(eval_set)
    similarity = real_check._similarity_matrix([c.input for c in cases])
    np.fill_diagonal(similarity, 0.0)

    result = real_check.run(eval_set)
    clustered = [set(c["case_ids"]) for c in result.stats["clusters"]]

    within, across = [], []
    for i in range(len(cases)):
        for j in range(i + 1, len(cases)):
            same = any({cases[i].id, cases[j].id} <= c for c in clustered)
            (within if same else across).append(float(similarity[i, j]))

    # Duplicates and non-duplicates separate cleanly, with a real gap between.
    assert max(across) == pytest.approx(0.70, abs=0.02)
    assert min(within) == pytest.approx(0.81, abs=0.02)
    assert max(across) < min(within)

    # And the honest caveat: 0.85 sits ABOVE that gap, so the weakest pair in
    # the billing_001-003 cluster is joined transitively, not by a direct edge.
    assert min(within) < real_check.threshold
