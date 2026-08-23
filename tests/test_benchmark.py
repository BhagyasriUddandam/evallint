"""Tests for the benchmark itself.

A benchmark nobody checks is a benchmark that quietly stops measuring. Three
things are proven here:

**1. The split really is held out.** A test asserts no fixture has
byte-identical positive cases across DEV and TEST. This was added after catching
the ambiguity fixture sharing all six of its positives — TEST was not held out at
all for that category, and the honest TEST recall turned out to be *lower* than
DEV's once fixed. The invariant is enforced rather than remembered.

**2. The metrics are arithmetically right**, including the cases where a rate is
undefined. Precision 0.0 ("everything it flagged was wrong") and precision None
("it flagged nothing") are different results, and collapsing them would make a
silent detector look like a broken one.

**3. The committed baselines still hold.** This is the point of the whole
directory: a change that makes a detector worse fails here, with the numbers in
the message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.fixtures import (
    Split,
    ambiguity_fixture,
    discrimination_fixture,
    exact_duplicate_fixture,
    imbalance_fixtures,
    judge_fixture,
    leakage_fixture,
    noise_trials,
    semantic_duplicate_fixture,
)
from benchmarks.metrics import Confusion, measure, score_sets
from benchmarks.runner import ALL_DETECTORS, run_benchmark

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"

CASE_FIXTURES = (
    exact_duplicate_fixture,
    semantic_duplicate_fixture,
    leakage_fixture,
    ambiguity_fixture,
)

#: How far a detector may drift below its committed baseline before this is a
#: regression. Fixtures are tens of cases, so one reclassification moves
#: precision by several points -- 0.05 is roughly "one case" and anything
#: larger is a real change in behaviour.
TOLERANCE = 0.05


# ==========================================================================
# 1. The split is genuinely held out
# ==========================================================================


@pytest.mark.parametrize("builder", CASE_FIXTURES, ids=lambda f: f.__name__)
def test_no_positive_case_is_shared_between_splits(builder) -> None:
    """The invariant the whole benchmark rests on.

    If TEST reuses DEV's positives, TEST is not held out and every number
    reported from it is a training-set number.
    """
    dev, test = builder(Split.DEV), builder(Split.TEST)
    dev_text = {
        (c.input, str(c.expected)) for c in dev.eval_set if c.id in dev.positives
    }
    test_text = {
        (c.input, str(c.expected)) for c in test.eval_set if c.id in test.positives
    }
    shared = dev_text & test_text
    assert not shared, (
        f"{builder.__name__}: {len(shared)} positive case(s) are byte-identical "
        "across splits, so TEST is not held out for this category"
    )


def test_the_two_splits_use_different_seeds() -> None:
    from benchmarks.fixtures import _SEED

    assert _SEED[Split.DEV] != _SEED[Split.TEST]


def test_noise_trials_differ_between_splits() -> None:
    """Seeded, so a trial is never shared."""
    dev = next(iter(noise_trials(Split.DEV, n_trials=1)))
    test = next(iter(noise_trials(Split.TEST, n_trials=1)))
    assert dev != test


def test_discrimination_fixture_content_differs_between_splits() -> None:
    dev = {c.input for c in discrimination_fixture(Split.DEV).eval_set}
    test = {c.input for c in discrimination_fixture(Split.TEST).eval_set}
    assert not dev & test


# ==========================================================================
# 2. The fixtures are internally consistent
# ==========================================================================


@pytest.mark.parametrize("builder", CASE_FIXTURES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("split", list(Split))
def test_labels_refer_to_cases_that_exist(builder, split) -> None:
    fixture = builder(split)
    ids = {c.id for c in fixture.eval_set}
    assert fixture.positives <= ids
    for pair in fixture.positive_pairs:
        assert pair <= ids, f"pair {sorted(pair)} names a case not in the set"
        assert len(pair) == 2


@pytest.mark.parametrize("builder", CASE_FIXTURES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("split", list(Split))
def test_every_fixture_has_adversarial_negatives(builder, split) -> None:
    """A fixture of obvious positives measures recall and nothing else, so
    precision would be unmeasurable and any reported value meaningless."""
    fixture = builder(split)
    assert fixture.negatives, "no negatives: precision cannot be measured"
    assert len(fixture.negatives) >= 4


@pytest.mark.parametrize("builder", CASE_FIXTURES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("split", list(Split))
def test_every_fixture_states_its_label_basis(builder, split) -> None:
    """Author judgement must be declared as such, or a reader will take a
    precision figure as precision against truth."""
    basis = builder(split).label_basis
    assert len(basis) > 20
    assert any(
        word in basis.lower()
        for word in ("objective", "judgement", "definitional", "arithmetic",
                     "constructed")
    )


def test_pair_labels_are_consistent_with_case_labels() -> None:
    for builder in (exact_duplicate_fixture, semantic_duplicate_fixture):
        for split in Split:
            fixture = builder(split)
            from_pairs = {cid for pair in fixture.positive_pairs for cid in pair}
            assert from_pairs == fixture.positives


def test_imbalance_fixtures_contain_both_classes_of_label() -> None:
    for split in Split:
        labels = [is_imbalanced for _, is_imbalanced, _ in imbalance_fixtures(split)]
        assert any(labels) and not all(labels), (
            "needs both imbalanced and balanced datasets, or precision and "
            "recall cannot both be measured"
        )


def test_judge_fixture_agreement_share_is_respected() -> None:
    """At a forced share of 1.0 every item must be unanimous by construction."""
    fixture = judge_fixture(Split.TEST, agreement_rate=1.0, n_items=30)
    by_item: dict[str, set[bool]] = {}
    for observation in fixture.observations:
        by_item.setdefault(observation.item, set()).add(observation.verdict)
    assert all(len(verdicts) == 1 for verdicts in by_item.values())


def test_discrimination_fixture_scorer_matches_its_own_labels() -> None:
    """The labels are only objective if the scorer really behaves as claimed.

    A bug here would make the strongest category in the benchmark silently
    measure the wrong thing, so it is checked against the four definitions
    directly rather than trusted.
    """
    fixture = discrimination_fixture(Split.TEST)
    for case in fixture.eval_set:
        outcomes = {m: fixture.scorer(case, m) for m in fixture.models}
        kind = fixture.expected_class[case.id]
        if kind == "ceiling":
            assert all(outcomes.values())
        elif kind == "floor":
            assert not any(outcomes.values())
        elif kind == "separating":
            assert outcomes["weak"] is False and outcomes["strong"] is True
        else:  # non_monotonic: the middle model must break the ordering
            assert outcomes["weak"] is True
            assert outcomes["middle"] is False
            assert outcomes["strong"] is True


# ==========================================================================
# 3. The metrics are arithmetically correct
# ==========================================================================


def test_confusion_rates() -> None:
    c = Confusion(tp=8, fp=2, fn=4, tn=86)
    assert c.n == 100
    assert c.precision == pytest.approx(0.8)
    assert c.recall == pytest.approx(8 / 12)
    assert c.f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
    assert c.false_positive_rate == pytest.approx(2 / 88)
    assert c.false_negative_rate == pytest.approx(4 / 12)


def test_a_detector_that_flags_nothing_has_undefined_precision() -> None:
    """Not 0.0. "Everything it flagged was wrong" and "it flagged nothing" are
    different results, and reporting 0.0 for the second would make a silent
    detector indistinguishable from a broken one."""
    c = Confusion(tp=0, fp=0, fn=5, tn=95)
    assert c.precision is None
    assert c.recall == 0.0
    assert c.f1 is None
    assert c.false_negative_rate == 1.0


def test_a_fixture_with_no_negatives_has_undefined_false_positive_rate() -> None:
    c = Confusion(tp=5, fp=0, fn=0, tn=0)
    assert c.false_positive_rate is None
    assert c.precision == 1.0


def test_score_sets_counts_true_negatives_from_the_universe() -> None:
    confusion = score_sets(
        predicted={"a", "b"}, actual={"b", "c"}, universe={"a", "b", "c", "d", "e"}
    )
    assert (confusion.tp, confusion.fp, confusion.fn, confusion.tn) == (1, 1, 1, 2)


def test_score_sets_rejects_a_granularity_mismatch() -> None:
    """Comparing cases against pairs would produce numbers that look fine and
    mean nothing, so it raises instead."""
    with pytest.raises(ValueError, match="outside the universe"):
        score_sets(predicted={"pair_x"}, actual={"a"}, universe={"a", "b"})


def test_f1_is_omitted_with_a_stated_reason() -> None:
    payload = Confusion(tp=1, fp=1, fn=1, tn=1).as_dict(with_f1=False)
    assert payload["f1"] is None
    assert "no useful interpretation" in payload["f1_omitted_because"]


def test_measure_reports_timing_and_a_memory_caveat() -> None:
    m = measure(lambda: sum(range(1000)), n_items=10, repeats=2)
    assert m.seconds_best > 0
    assert m.seconds_best <= m.seconds_median + 1e-9
    assert m.as_dict()["ms_per_item"] is not None
    # The caveat travels with the number, so it cannot be quoted without it.
    assert "tracemalloc only" in m.as_dict()["python_peak_caveat"]


def test_measure_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        measure(lambda: None, n_items=1, repeats=0)


# ==========================================================================
# 4. The runner, and the committed baselines
# ==========================================================================


def test_the_dev_split_is_marked_unreportable() -> None:
    """So a stale dev run cannot be mistaken for a result by anything reading
    the file."""
    assert run_benchmark(Split.DEV, only=["exact_duplicates"])["reportable"] is False
    assert run_benchmark(Split.TEST, only=["exact_duplicates"])["reportable"] is True


def test_every_run_carries_its_limitations() -> None:
    payload = run_benchmark(Split.TEST, only=["exact_duplicates"])
    assert len(payload["limitations"]) >= 8
    assert any("same author" in line for line in payload["limitations"])


def test_an_unknown_detector_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        run_benchmark(Split.TEST, only=["not_a_detector"])


def test_exact_duplicates_are_found_perfectly() -> None:
    """The objective category. Anything less than perfect here is a real bug,
    not a labelling dispute -- byte-identical is byte-identical."""
    result = run_benchmark(Split.TEST, only=["exact_duplicates"])["detectors"][
        "exact_duplicates"
    ]
    assert result["confusion"]["precision"] == 1.0
    assert result["confusion"]["recall"] == 1.0
    assert result["near_miss_check"]["passed"], (
        "normalised-but-not-exact duplicates were clustered at the exact level, "
        "which loses the distinction the levels exist to make"
    )


def test_discrimination_classes_are_recovered_exactly() -> None:
    """The scorer is a fixture, so this is arithmetic. A disagreement means the
    classifier is wrong, and the test says which case."""
    result = run_benchmark(Split.TEST, only=["ceiling_floor_discrimination"])[
        "detectors"
    ]["ceiling_floor_discrimination"]
    assert result["exact_class_agreement"] == 1.0, result["confusions"]


def test_the_null_false_positive_rate_is_near_alpha() -> None:
    """The strongest check available: with two genuinely equal models, an exact
    test at alpha = 0.05 should call a difference about 5% of the time.

    The upper bound is 0.12 rather than 0.05 because 80 trials leave real
    binomial slack: at a true rate of 0.05, the standard error is
    sqrt(0.05 * 0.95 / 80) = 0.024, so 0.12 is about three standard errors up.
    Being materially ABOVE that would mean the test is anti-conservative, which
    is the failure that matters -- reporting differences that are not there.
    """
    arms = run_benchmark(Split.TEST, only=["statistical_noise"], trials=80)[
        "detectors"
    ]["statistical_noise"]["arms"]
    null = next(arm for arm in arms if arm["true_difference"] == 0.0)
    assert null["significant_share"] <= 0.12, (
        f"false-positive rate {null['significant_share']:.1%} under a true null "
        "is materially above alpha; the test is anti-conservative"
    )


def test_judge_agreement_is_monotonic_in_the_constructed_agreement() -> None:
    result = run_benchmark(Split.TEST, only=["judge_instability"])["detectors"][
        "judge_instability"
    ]
    assert result["monotonicity_check"]["passed"], result["monotonicity_check"]


@pytest.mark.parametrize(
    "detector,metric,floor",
    [
        ("exact_duplicates", "precision", 1.0),
        ("exact_duplicates", "recall", 1.0),
        ("leakage", "recall", 1.0),
        ("ambiguous_references", "precision", 1.0),
    ],
)
def test_committed_baseline_is_not_regressed(detector, metric, floor) -> None:
    """The point of the whole directory.

    Only metrics whose baseline is a round 1.0 are pinned as floors here, so
    this cannot drift into pinning noise. The full committed baselines live in
    benchmarks/results/ and the comparison below covers everything.
    """
    result = run_benchmark(Split.TEST, only=[detector])["detectors"][detector]
    value = result["confusion"][metric]
    assert value is not None and value >= floor - 1e-9, (
        f"{detector} {metric} is {value}, baseline {floor}. If this change is "
        "intended, regenerate benchmarks/results/ in the same commit."
    )


def test_current_results_match_the_committed_baseline() -> None:
    """Compare every metric against the committed file.

    Tolerance is one-case-worth, because the fixtures are small. A drift beyond
    it is a behaviour change and should come with regenerated baselines in the
    same commit.
    """
    baseline_path = RESULTS / "test.json"
    assert baseline_path.is_file(), (
        "no committed baseline; run "
        "`python -m benchmarks.runner --split test --json benchmarks/results/test.json`"
    )
    baseline = json.loads(baseline_path.read_text())
    current = run_benchmark(Split.TEST, only=["exact_duplicates", "leakage",
                                             "ambiguous_references",
                                             "class_imbalance"])

    drifted = []
    for name, result in current["detectors"].items():
        was = baseline["detectors"][name]["confusion"]
        now = result["confusion"]
        for metric in ("precision", "recall", "f1", "false_positive_rate",
                       "false_negative_rate"):
            before, after = was.get(metric), now.get(metric)
            if before is None or after is None:
                # None is a distinct result, so a change INTO or OUT OF it is a
                # drift regardless of tolerance.
                if before is not after:
                    drifted.append(f"{name}.{metric}: {before} -> {after}")
                continue
            if abs(before - after) > TOLERANCE:
                drifted.append(f"{name}.{metric}: {before} -> {after}")

    assert not drifted, (
        "benchmark results drifted from the committed baseline:\n  "
        + "\n  ".join(drifted)
        + "\n\nIf this is intended, regenerate benchmarks/results/*.json in the "
        "same commit so the diff shows the effect."
    )


def test_the_committed_baseline_is_from_the_test_split() -> None:
    baseline = json.loads((RESULTS / "test.json").read_text())
    assert baseline["split"] == "test"
    assert baseline["reportable"] is True


def test_every_registered_detector_appears_in_the_baseline() -> None:
    """So adding a detector without regenerating the baseline is caught."""
    baseline = json.loads((RESULTS / "test.json").read_text())
    missing = sorted(set(ALL_DETECTORS) - set(baseline["detectors"]))
    assert not missing, (
        f"{missing} have no committed baseline; regenerate "
        "benchmarks/results/test.json"
    )


def test_the_readme_documents_every_detector() -> None:
    text = (RESULTS.parent / "README.md").read_text()
    missing = [
        name
        for name in ALL_DETECTORS
        if name.replace("_", " ") not in text and name not in text
    ]
    assert not missing, f"undocumented in benchmarks/README.md: {missing}"
