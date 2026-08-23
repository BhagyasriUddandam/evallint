"""Tests for the semantic redundancy analyser.

The centrepiece is FIXTURE below: a small labelled set of duplicate and
non-duplicate pairs, used to measure precision and recall rather than to assert
that the code does what it currently does.

Why that matters here more than elsewhere. Every measured cosine value in this
file came from running all-MiniLM-L6-v2, not from intuition:

    genuine paraphrases      0.599, 0.659, 0.860
    same template, numbers   0.717
    same domain, different    0.317, -0.001
    unrelated                 0.109

Two consequences drive the design under test. Real paraphrases span 0.60 to
0.86, so ONE threshold cannot have both high precision and high recall on them.
And the template pair scores 0.717 — below the default 0.85 — so the semantic
level MISSES a redundancy that the template level catches exactly. That is the
argument for five mechanisms instead of one tuned number.
"""

from __future__ import annotations

import importlib.util

import pytest

from evallint.checks.redundancy import (
    DEFAULT_THRESHOLD,
    MIN_SKELETON_TOKENS,
    CompareFields,
    RedundancyCheck,
    RedundancyLevel,
    normalise,
    skeleton,
)
from evallint.schema import EvalCase, EvalSet

# numpy moved to the [embeddings] extra when a review found it was 25 MiB of a
# 32 MiB core install used for one median. The tests below build fake embedders
# that return arrays, so they need it.
#
# `pytestmark` rather than `importorskip`: importorskip raises during COLLECTION,
# which removes these tests from the collected count and makes the README's
# stated test total depend on which extras happen to be installed. A skipif mark
# collects them and skips them, so the count is stable everywhere.
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_NUMPY, reason="needs numpy from the [embeddings] extra"
)
if _HAS_NUMPY:
    import numpy as np

# =========================================================================
# Labelled fixture
#
# (id, input, expected). Duplicate groups are declared separately so precision
# and recall are computed against a stated ground truth rather than inferred.
# =========================================================================

FIXTURE = [
    # --- exact duplicates -------------------------------------------------
    # NOTE: deliberately a different topic from the pq_* billing pair below.
    # A first version used another double-charge sentence here, and the semantic
    # level correctly linked the two groups -- the fixture's ground truth was
    # wrong, not the detector.
    ("ex_a", "The tracking page shows no updates for my delivery.", "Open a lost-parcel claim."),
    ("ex_b", "The tracking page shows no updates for my delivery.", "Open a lost-parcel claim."),
    # --- normalized-only duplicates (case, spacing, punctuation) ----------
    ("nm_a", "How do I cancel my subscription?", "Cancel it from billing."),
    ("nm_b", "how do i   CANCEL my subscription!!!", "Cancel it from billing."),
    # --- template family: identical structure, different numbers ----------
    ("tp_a", "What is 5 plus 3 in total?", "8"),
    ("tp_b", "What is 12 plus 7 in total?", "19"),
    ("tp_c", "What is 100 plus 250 in total?", "350"),
    # --- paraphrase pair, same answer (measured cosine 0.860) -------------
    ("pp_a", "How do I reset my password?", "Use the reset link."),
    ("pp_b", "How can I change my password?", "Use the reset link."),
    # --- paraphrase pair, same answer (measured cosine 0.659) -------------
    ("pq_a", "I was billed twice for the same order.", "Refund the extra charge."),
    ("pq_b", "My card got charged two times for one purchase.", "Refund the extra charge."),
    # --- hard negatives: same domain, genuinely different scenarios -------
    ("hn_a", "My flight to Rome was cancelled.", "Offer a rebooking."),
    ("hn_b", "I want to upgrade to business class.", "Quote the fare difference."),
    ("hn_c", "The hotel charged a resort fee I never agreed to.", "Dispute the fee."),
    # --- unrelated --------------------------------------------------------
    ("un_a", "Explain the return policy for electronics.", "14 days, unopened."),
]

#: Ground truth: every pair inside a group is redundant, every other pair is not.
DUPLICATE_GROUPS = [
    {"ex_a", "ex_b"},
    {"nm_a", "nm_b"},
    {"tp_a", "tp_b", "tp_c"},
    {"pp_a", "pp_b"},
    {"pq_a", "pq_b"},
]

#: The subset detectable WITHOUT any embedding model.
DETERMINISTIC_GROUPS = [
    {"ex_a", "ex_b"},
    {"nm_a", "nm_b"},
    {"tp_a", "tp_b", "tp_c"},
]


def fixture_set() -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(id=cid, input=text, expected=exp) for cid, text, exp in FIXTURE
        ),
        source="fixture",
    )


def truth_pairs(groups) -> set[frozenset[str]]:
    pairs = set()
    for group in groups:
        members = sorted(group)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(frozenset((members[i], members[j])))
    return pairs


def found_pairs(result) -> set[frozenset[str]]:
    """Pairs implied by the reported clusters."""
    pairs = set()
    for cluster in result.stats["clusters"]:
        ids = sorted(cluster["case_ids"])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add(frozenset((ids[i], ids[j])))
    return pairs


def precision_recall(found, truth) -> tuple[float, float]:
    if not found:
        return (1.0, 0.0) if truth else (1.0, 1.0)
    true_positive = len(found & truth)
    precision = true_positive / len(found)
    recall = true_positive / len(truth) if truth else 1.0
    return precision, recall


# =========================================================================
# Deterministic levels: exact precision and recall, no model needed
# =========================================================================


def test_deterministic_levels_have_perfect_precision_and_recall() -> None:
    """No embedder at all. Exact, normalized and template redundancy is found
    completely and with no false positives — these levels are comparisons, not
    thresholds, so there is nothing to tune wrong."""
    result = RedundancyCheck(semantic=False).run(fixture_set())

    precision, recall = precision_recall(
        found_pairs(result), truth_pairs(DETERMINISTIC_GROUPS)
    )
    assert precision == 1.0, f"false positives: {found_pairs(result) - truth_pairs(DETERMINISTIC_GROUPS)}"
    assert recall == 1.0, f"missed: {truth_pairs(DETERMINISTIC_GROUPS) - found_pairs(result)}"


def test_deterministic_levels_find_no_false_positives_among_negatives() -> None:
    """The hard negatives — same domain, different scenario — must not group."""
    result = RedundancyCheck(semantic=False).run(fixture_set())
    clustered = {cid for c in result.stats["clusters"] for cid in c["case_ids"]}
    for negative in ("hn_a", "hn_b", "hn_c", "un_a"):
        assert negative not in clustered


def test_each_level_is_attributed_correctly() -> None:
    result = RedundancyCheck(semantic=False).run(fixture_set())
    by_ids = {frozenset(c["case_ids"]): c for c in result.stats["clusters"]}

    assert by_ids[frozenset({"ex_a", "ex_b"})]["level"] == "exact"
    assert by_ids[frozenset({"nm_a", "nm_b"})]["level"] == "normalized"
    assert by_ids[frozenset({"tp_a", "tp_b", "tp_c"})]["level"] == "template"


def test_the_template_level_catches_what_cosine_misses() -> None:
    """The measured argument for multiple mechanisms.

    'What is 5 plus 3' and 'What is 12 plus 7' have a real cosine of 0.717,
    BELOW the 0.85 default, so the semantic level does not see them. The
    template level does, exactly and without a threshold.
    """
    result = RedundancyCheck(semantic=False).run(fixture_set())
    template = next(
        c for c in result.stats["clusters"] if c["level"] == "template"
    )
    assert set(template["case_ids"]) == {"tp_a", "tp_b", "tp_c"}
    assert template["certainty"] == "duplicate"


def test_semantic_levels_are_reported_as_not_run_when_absent() -> None:
    result = RedundancyCheck(semantic=False).run(fixture_set())
    assert result.stats["levels_run"] == ["exact", "normalized", "template"]
    assert result.stats["semantic_available"] is False


# =========================================================================
# Precision / recall of the SEMANTIC level, against the real model
# =========================================================================


def _real_model_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


real_model = pytest.mark.skipif(
    not _real_model_available(),
    reason="needs the evallint[embeddings] extra",
)


@real_model
def test_default_threshold_is_high_precision_low_recall() -> None:
    """Measured, not assumed. Real paraphrases in this fixture sit at 0.599,
    0.659 and 0.860, so a single threshold at 0.85 finds one of them. That is a
    deliberate trade: no false positives, at the cost of recall the
    deterministic levels have to make up.
    """
    result = RedundancyCheck(threshold=DEFAULT_THRESHOLD).run(fixture_set())
    precision, recall = precision_recall(found_pairs(result), truth_pairs(DUPLICATE_GROUPS))

    assert precision == 1.0, (
        "the default threshold must not produce false positives: "
        f"{found_pairs(result) - truth_pairs(DUPLICATE_GROUPS)}"
    )
    # Deterministic levels give 4 of 6 true pairs; the 0.860 paraphrase adds one.
    assert recall >= 4 / 6
    assert recall < 1.0, "if this now finds everything, re-measure the fixture"


@real_model
def test_a_lower_threshold_buys_recall_without_losing_precision_here() -> None:
    """On this fixture the positives start at 0.599 and the negatives stop at
    0.317, so 0.55 separates them completely. That is 15 cases — weak evidence,
    and the reason the check reports a sensitivity curve instead of claiming
    0.55 is correct in general."""
    result = RedundancyCheck(threshold=0.55).run(fixture_set())
    precision, recall = precision_recall(found_pairs(result), truth_pairs(DUPLICATE_GROUPS))

    assert precision == 1.0
    assert recall == 1.0


@real_model
def test_an_absurdly_low_threshold_destroys_precision() -> None:
    """The counter-test. Without it, "lower is better" would look free."""
    result = RedundancyCheck(threshold=0.05).run(fixture_set())
    precision, _ = precision_recall(found_pairs(result), truth_pairs(DUPLICATE_GROUPS))
    assert precision < 0.5


@real_model
def test_scenario_and_semantic_are_distinguished_by_the_expected_answer() -> None:
    """Same wording distance, different answer -> `semantic`, not `scenario`."""
    same_answer = EvalSet(
        cases=(
            EvalCase(id="a", input="How do I reset my password?", expected="Use the link."),
            EvalCase(id="b", input="How can I change my password?", expected="Use the link."),
        ),
        source="t",
    )
    diff_answer = EvalSet(
        cases=(
            EvalCase(id="a", input="How do I reset my password?", expected="Use the link."),
            EvalCase(id="b", input="How can I change my password?", expected="Call support."),
        ),
        source="t",
    )
    assert (
        RedundancyCheck(compare=CompareFields.INPUT).run(same_answer)
        .stats["clusters"][0]["level"] == "scenario"
    )
    assert (
        RedundancyCheck(compare=CompareFields.INPUT).run(diff_answer)
        .stats["clusters"][0]["level"] == "semantic"
    )


@real_model
def test_threshold_sensitivity_curve_is_reported() -> None:
    """The honest answer to "is 0.85 right?" is "here is what happens if it
    isn't"."""
    stats = RedundancyCheck().run(fixture_set()).stats
    curve = stats["threshold_sensitivity"]
    assert sorted(curve) == ["0.80", "0.85", "0.90", "0.95"]
    counts = [curve[k] for k in sorted(curve)]
    assert counts == sorted(counts, reverse=True), "must be monotone in threshold"


# =========================================================================
# Clustering, weighting and reporting — deterministic fake embedder
# =========================================================================


def fake_embedder(mapping: dict[str, list[float]]):
    """Vectors keyed by the text, so similarity is exactly controlled."""

    def embed(texts):
        return np.array([mapping[t] for t in texts], dtype=np.float32)

    return embed


def simple_set(texts: dict[str, str], expected: str = "same") -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(id=cid, input=text, expected=expected)
            for cid, text in texts.items()
        ),
        source="t",
    )


#: Filler wording that shares no template. Written out as words rather than
#: numbered, because "sentence number 3" and "sentence number 7" ARE the same
#: template and the template detector correctly groups them -- which broke the
#: first version of these tests.
FILLER = [
    "the harbour was quiet before dawn",
    "she repaired the bicycle chain herself",
    "an unexpected parcel arrived on Tuesday",
    "the recipe calls for smoked paprika",
    "his passport expired last winter",
    "they repainted the garden fence green",
    "the lecture hall smelled of chalk",
    "a heron stood in the shallow river",
    "the printer jammed again this morning",
    "we cancelled the picnic because of wind",
    "the violin needed fresh strings",
    "a stray cat adopted the bookshop",
    "the bridge closes for repairs soon",
    "her thesis defence went smoothly",
    "the market sells excellent olives",
    "an owl nested in the old chimney",
    "the kettle whistles too loudly",
    "he memorised the entire timetable",
    "the pond froze over completely",
    "a mural covers the northern wall",
    "the clock tower chimes off key",
    "she prefers strong coffee unsweetened",
    "the ferry runs only in summer",
    "our neighbour grows enormous pumpkins",
    "the library extended its hours",
    "a comet was visible last night",
    "the bakery closes on Mondays",
    "they found the missing key eventually",
    "the trail loops around the lake",
    "his handwriting is nearly illegible",
    "the orchestra tuned for a while",
    "a fox crossed the empty road",
]


def comparison_key(input_text: str, expected: str = "same") -> str:
    """The text a default-configured check will actually embed."""
    return f"{input_text}\n{expected}"


def test_cluster_reports_size_similarity_and_weighting_risk() -> None:
    """The requested output shape: size, average similarity, weighting risk."""
    texts = {f"c{i}": FILLER[i] for i in range(10)}
    # first 8 point the same direction -> one cluster of 8 out of 10
    vectors = {
        comparison_key(text): ([1.0, 0.0] if i < 8 else [0.0, 1.0])
        for i, text in enumerate(texts.values())
    }
    result = RedundancyCheck(embedder=fake_embedder(vectors)).run(simple_set(texts))

    cluster = max(result.stats["clusters"], key=lambda c: c["size"])
    assert cluster["size"] == 8
    assert cluster["mean_similarity"] == pytest.approx(1.0)
    assert cluster["weight_share"] == pytest.approx(0.8)
    assert cluster["weighting_risk"] == "HIGH"

    message = next(f.message for f in result.findings if "cluster" in f.message)
    assert "8 cases" in message
    assert "weighting risk HIGH" in message


def test_weighting_risk_bands_are_configurable_and_reported_raw() -> None:
    texts = {f"c{i}": FILLER[i] for i in range(20)}
    vectors = {
        comparison_key(text): ([1.0, 0.0] if i < 2 else [0.0, 1.0])
        for i, text in enumerate(texts.values())
    }
    embedder = fake_embedder(vectors)

    # 2 of 20 = 10% -> HIGH by default
    default = RedundancyCheck(embedder=embedder).run(simple_set(texts))
    pair = next(c for c in default.stats["clusters"] if c["size"] == 2)
    assert pair["weight_share"] == pytest.approx(0.10)
    assert pair["weighting_risk"] == "HIGH"

    # raise the band and the same cluster is no longer high
    relaxed = RedundancyCheck(
        embedder=embedder, high_weight_share=0.5, medium_weight_share=0.25
    ).run(simple_set(texts))
    pair = next(c for c in relaxed.stats["clusters"] if c["size"] == 2)
    assert pair["weight_share"] == pytest.approx(0.10)  # the fact is unchanged
    assert pair["weighting_risk"] == "LOW"  # only the label moved


def test_effective_case_count_and_excess_weight() -> None:
    texts = {f"c{i}": FILLER[i] for i in range(6)}
    vectors = {
        comparison_key(text): ([1.0, 0.0] if i < 3 else [0.0, 1.0])
        for i, text in enumerate(texts.values())
    }
    stats = RedundancyCheck(embedder=fake_embedder(vectors)).run(simple_set(texts)).stats

    # two clusters of 3, so 4 redundant, 2 distinct scenarios
    assert stats["n_redundant_cases"] == 4
    assert stats["effective_n_cases"] == 2
    cluster = stats["clusters"][0]
    assert cluster["excess_weight_share"] == pytest.approx(2 / 6)


def test_duplicate_and_similar_are_counted_separately() -> None:
    """Requirement: separate "similar" from "duplicate"."""
    cases = (
        EvalCase(id="d1", input="identical text here for both", expected="x"),
        EvalCase(id="d2", input="identical text here for both", expected="x"),
        EvalCase(id="s1", input="a wholly different sentence one", expected="x"),
        EvalCase(id="s2", input="a wholly different sentence two", expected="x"),
    )
    vectors = {
        "identical text here for both\nx": [1.0, 0.0],
        "a wholly different sentence one\nx": [0.0, 1.0],
        "a wholly different sentence two\nx": [0.0, 1.0],
    }
    stats = RedundancyCheck(embedder=fake_embedder(vectors)).run(
        EvalSet(cases=cases, source="t")
    ).stats

    assert stats["n_duplicate_clusters"] == 1
    assert stats["n_similar_clusters"] == 1
    levels = {c["level"]: c["certainty"] for c in stats["clusters"]}
    assert levels["exact"] == "duplicate"
    assert levels["scenario"] == "similar"


def test_deterministic_duplicates_warn_and_heuristic_similarity_informs() -> None:
    """A certain redundancy is a WARNING because it cannot be tuned away.

    A threshold-dependent similarity is INFO -- unless its aggregate weight is
    high enough to move the headline on its own, in which case the weighting is
    worth attention however the grouping was found. Both clusters here are
    deliberately small relative to n so the weight rule is not what decides the
    severity.
    """
    from evallint.checks import Severity

    n_filler = 28
    cases = (
        EvalCase(id="d1", input="identical text here for both", expected="x"),
        EvalCase(id="d2", input="identical text here for both", expected="x"),
        EvalCase(id="s1", input=FILLER[0], expected="y"),
        EvalCase(id="s2", input=FILLER[1], expected="y"),
        *[
            EvalCase(id=f"f{i}", input=FILLER[i + 2], expected="z")
            for i in range(n_filler)
        ],
    )
    total = 4 + n_filler

    # One-hot directions so the filler cases are mutually orthogonal and cannot
    # cluster. Two dimensions is not enough for that, which is what made an
    # earlier version of this test group 30 cases into one 94%-weight cluster.
    def onehot(index: int) -> list[float]:
        vec = [0.0] * total
        vec[index] = 1.0
        return vec

    vectors = {comparison_key("identical text here for both", "x"): onehot(0)}
    # s1 and s2 share a direction -> similar, but only 2 of 32 cases
    vectors[comparison_key(FILLER[0], "y")] = onehot(1)
    vectors[comparison_key(FILLER[1], "y")] = onehot(1)
    for i in range(n_filler):
        vectors[comparison_key(FILLER[i + 2], "z")] = onehot(i + 2)

    result = RedundancyCheck(embedder=fake_embedder(vectors)).run(
        EvalSet(cases=cases, source="t")
    )

    duplicate = next(f for f in result.findings if "[DUPLICATE]" in f.message)
    similar = next(f for f in result.findings if "[SIMILAR]" in f.message)

    assert duplicate.severity is Severity.WARNING
    assert similar.severity is Severity.INFO
    # Neither cluster is large enough for the weight rule to be what decided it.
    for cluster in result.stats["clusters"]:
        assert cluster["weighting_risk"] != "HIGH"


def test_a_high_weight_similar_cluster_is_warned_about_anyway() -> None:
    """The exception to the rule above: heuristic grouping still warrants a
    warning when one scenario controls enough of the score to decide a
    comparison by itself."""
    from evallint.checks import Severity

    texts = {f"c{i}": FILLER[i] for i in range(10)}
    vectors = {
        comparison_key(text): ([1.0, 0.0] if i < 5 else [0.0, 1.0])
        for i, text in enumerate(texts.values())
    }
    result = RedundancyCheck(embedder=fake_embedder(vectors)).run(simple_set(texts))

    similar = [f for f in result.findings if "[SIMILAR]" in f.message]
    assert similar, "expected heuristic clusters"
    assert all(f.severity is Severity.WARNING for f in similar)
    assert all(
        c["weighting_risk"] == "HIGH"
        for c in result.stats["clusters"]
        if c["certainty"] == "similar"
    )


# =========================================================================
# Field selection
# =========================================================================


def test_input_only_comparison_is_preserved_on_request() -> None:
    """Requirement: keep input-only detection available explicitly.

    Same question, DIFFERENT reference answers. Under the default that is not
    redundancy — it is a ground-truth contradiction, and collapsing the two
    would hide it. Under compare="input" they group, which is what the older
    behaviour did.
    """
    cases = (
        EvalCase(id="a", input="What is the capital of France?", expected="Paris"),
        EvalCase(id="b", input="What is the capital of France?", expected="Lyon"),
    )
    eval_set = EvalSet(cases=cases, source="t")

    default = RedundancyCheck(semantic=False).run(eval_set)
    assert default.stats["n_clusters"] == 0

    input_only = RedundancyCheck(
        semantic=False, compare=CompareFields.INPUT
    ).run(eval_set)
    assert input_only.stats["n_clusters"] == 1
    assert input_only.stats["clusters"][0]["level"] == "exact"


def test_metadata_can_be_included() -> None:
    cases = (
        EvalCase(id="a", input="Same question asked here", expected="x",
                 metadata={"locale": "en"}),
        EvalCase(id="b", input="Same question asked here", expected="x",
                 metadata={"locale": "fr"}),
    )
    eval_set = EvalSet(cases=cases, source="t")

    assert RedundancyCheck(semantic=False).run(eval_set).stats["n_clusters"] == 1
    assert (
        RedundancyCheck(semantic=False, compare=CompareFields.ALL)
        .run(eval_set).stats["n_clusters"] == 0
    )


def test_metadata_key_order_does_not_change_the_result() -> None:
    a = EvalCase(id="a", input="q", expected="x", metadata={"p": 1, "q": 2})
    b = EvalCase(id="b", input="q", expected="x", metadata={"q": 2, "p": 1})
    check = RedundancyCheck(semantic=False, compare=CompareFields.ALL)
    assert check.comparison_text(a) == check.comparison_text(b)


# =========================================================================
# Language: nothing here calls a case invalid
# =========================================================================


def test_no_finding_claims_a_case_is_invalid() -> None:
    """Requirement, asserted rather than trusted."""
    texts = {f"c{i}": "the very same sentence repeated verbatim" for i in range(4)}
    vectors = {comparison_key("the very same sentence repeated verbatim"): [1.0, 0.0]}
    result = RedundancyCheck(embedder=fake_embedder(vectors)).run(simple_set(texts))

    text = " ".join(f.message for f in result.findings).lower()
    for banned in ("invalid", "bad case", "worthless", "useless", "delete these", "remove these"):
        assert banned not in text, f"found {banned!r} in a finding"
    assert "may be intentional" in text
    joined = " ".join(result.limitations).lower()
    assert "does not tell you to delete anything" in joined


# =========================================================================
# Degenerate input and configuration
# =========================================================================


def test_single_case_has_nothing_to_compare() -> None:
    eval_set = EvalSet(cases=(EvalCase(id="a", input="only one"),), source="t")
    result = RedundancyCheck(semantic=False).run(eval_set)
    assert result.stats["n_clusters"] == 0
    assert "nothing to compare" in result.summary


def test_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="threshold"):
        RedundancyCheck(threshold=0)
    with pytest.raises(ValueError, match="threshold"):
        RedundancyCheck(threshold=1.5)
    with pytest.raises(ValueError, match="weight shares"):
        RedundancyCheck(high_weight_share=0.1, medium_weight_share=0.5)


def test_missing_embeddings_degrades_rather_than_failing() -> None:
    """A broken embedder must not throw away the deterministic results."""

    def broken(texts):
        raise ImportError("no model here")

    result = RedundancyCheck(embedder=broken).run(fixture_set())
    assert result.stats["semantic_available"] is False
    assert result.stats["n_clusters"] == 3  # the deterministic ones survive
    assert "no model here" in result.stats["semantic_skipped_reason"]
    assert any("semantic levels did not run" in f.message for f in result.findings)


# =========================================================================
# The helpers
# =========================================================================


def test_normalise_folds_case_space_and_punctuation() -> None:
    assert normalise("  Hello,   WORLD!! ") == "hello world"
    assert normalise("a-b_c") == "a b c"


def test_skeleton_masks_numbers_and_quoted_spans() -> None:
    assert skeleton("What is 5 plus 3 in total?") == skeleton(
        "What is 12 plus 7 in total?"
    )
    assert skeleton('Translate "hello there" into French now') == skeleton(
        'Translate "goodbye friend" into French now'
    )


def test_skeleton_refuses_to_be_too_generic() -> None:
    """Without the floor, every short arithmetic question collapses into one
    family. MIN_SKELETON_TOKENS is the smallest value that keeps the bundled
    example and the four real public datasets free of template false positives.
    """
    assert skeleton("What is 5?") is None
    assert len(normalise(skeleton("What is 5 plus 3 in total?")).split()) >= MIN_SKELETON_TOKENS


def test_level_duplicate_classification() -> None:
    assert RedundancyLevel.EXACT.is_duplicate
    assert RedundancyLevel.NORMALIZED.is_duplicate
    assert RedundancyLevel.TEMPLATE.is_duplicate
    assert not RedundancyLevel.SCENARIO.is_duplicate
    assert not RedundancyLevel.SEMANTIC.is_duplicate


# =========================================================================
# Partial-run signalling
# =========================================================================


def test_missing_semantic_levels_are_declared_partial() -> None:
    """A check that ran but did less than it claims must not let the CLI report
    a clean gate. Declared through CheckResult.partial rather than by the CLI
    special-casing this check."""

    def unavailable(texts):
        raise ImportError("sentence-transformers is not installed")

    result = RedundancyCheck(embedder=unavailable).run(fixture_set())
    assert result.partial
    assert "semantic and scenario levels did not run" in result.partial[0]


def test_explicitly_disabling_semantic_is_not_partial() -> None:
    """semantic=False is the caller choosing scope, exactly like
    --skip-duplicates. Not an incomplete audit."""
    result = RedundancyCheck(semantic=False).run(fixture_set())
    assert result.partial == ()


def test_a_full_run_is_not_partial() -> None:
    texts = {f"c{i}": FILLER[i] for i in range(4)}
    vectors = {comparison_key(t): [1.0, 0.0] for t in texts.values()}
    result = RedundancyCheck(embedder=fake_embedder(vectors)).run(simple_set(texts))
    assert result.partial == ()


# --------------------------------------------------------------------------
# Scale: one large family must not cost quadratic memory
# --------------------------------------------------------------------------


def _templated(n: int) -> EvalSet:
    """n cases that all collapse to ONE skeleton once numbers are masked.

    The realistic shape this guards against: a templated eval ("what is {a} +
    {b}") is a single template family, so the bucket is the whole dataset.
    """
    return EvalSet(
        cases=tuple(
            EvalCase(
                id=f"c{i}",
                input=f"Question number {i} about topic {i} with padding text?",
                expected=f"answer {i} with enough characters to count",
            )
            for i in range(n)
        )
    )


def test_a_star_produces_the_same_cluster_as_every_pair() -> None:
    """The claim the optimisation rests on, tested rather than asserted.

    Clusters are connected components, so linking every bucket member to the
    first gives the identical component with size-1 edges instead of
    size*(size-1)/2. Verified by running the same data on both sides of the
    budget.
    """
    from evallint.checks import redundancy as module

    eval_set = _templated(60)

    exhaustive = module.RedundancyCheck(semantic=False).run(eval_set)

    original = module.MAX_BUCKET_PAIRS
    try:
        module.MAX_BUCKET_PAIRS = 10  # forces the star path for a 60-member bucket
        starred = module.RedundancyCheck(semantic=False).run(eval_set)
    finally:
        module.MAX_BUCKET_PAIRS = original

    assert starred.stats["truncated_buckets"], "the star path did not trigger"
    assert not exhaustive.stats["truncated_buckets"]

    # Cluster MEMBERSHIP must be identical -- that is the guarantee.
    def members(result):
        return sorted(tuple(sorted(c["case_ids"])) for c in result.stats["clusters"])

    assert members(starred) == members(exhaustive)
    assert starred.stats["n_clusters"] == exhaustive.stats["n_clusters"]
    assert starred.stats["n_redundant_cases"] == exhaustive.stats["n_redundant_cases"]
    assert starred.stats["effective_n_cases"] == exhaustive.stats["effective_n_cases"]


def test_truncation_is_reported_not_silent() -> None:
    from evallint.checks import redundancy as module

    original = module.MAX_BUCKET_PAIRS
    try:
        module.MAX_BUCKET_PAIRS = 10
        stats = module.RedundancyCheck(semantic=False).run(_templated(40)).stats
    finally:
        module.MAX_BUCKET_PAIRS = original

    assert len(stats["truncated_buckets"]) == 1
    record = stats["truncated_buckets"][0]
    assert record["bucket_size"] == 40
    assert record["pairs_not_materialised"] == 40 * 39 // 2 - 39


def test_ten_thousand_cases_in_one_family_stay_within_budget() -> None:
    """Before the pair budget this allocated about 6 GiB and took minutes.

    The bounds are deliberately loose -- this is a guard against a return to
    quadratic behaviour, not a performance benchmark, and CI machines vary.
    """
    import time
    import tracemalloc

    eval_set = _templated(10_000)
    tracemalloc.start()
    started = time.perf_counter()
    result = RedundancyCheck(semantic=False).run(eval_set)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.stats["n_clusters"] == 1
    assert peak / 1024**2 < 200, f"peak {peak / 1024**2:.0f} MiB looks quadratic"
    assert elapsed < 30, f"{elapsed:.1f}s looks quadratic"


def test_the_semantic_level_refuses_an_unaffordable_matrix() -> None:
    """Refusing with the arithmetic beats an OOM kill, and the deterministic
    levels still run."""
    from evallint.checks import redundancy as module

    original = module.MAX_SEMANTIC_CASES
    try:
        module.MAX_SEMANTIC_CASES = 5
        check = module.RedundancyCheck(
            semantic=True, embedder=lambda texts: [[1.0, 0.0]] * len(texts)
        )
        result = check.run(_templated(20))
    finally:
        module.MAX_SEMANTIC_CASES = original

    assert result.partial, "an unaffordable matrix must mark the run partial"
    assert "GiB" in result.partial[0]
    # A size refusal must not be reported as a missing install.
    assert "pip install" not in result.partial[0]
    assert "semantic" not in result.stats["levels_run"]
