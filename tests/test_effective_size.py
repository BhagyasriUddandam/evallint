"""Tests for redundancy-adjusted coverage.

Four required scenarios, each with a deterministic dataset: fully unique data,
exact duplicates, highly redundant data, and templated data. The templated case
needs no embedder at all — the template detector is exact — which makes it the
cheapest end-to-end test in the file.

The other half of these tests is about TERMINOLOGY. The estimate must not be
presented as a statistical effective sample size, because the method does not
support that claim, so several tests assert on the wording and on the presence
of the bracket.
"""

from __future__ import annotations

import numpy as np
import pytest

from evallint.checks.effective_size import (
    LARGE_CLUSTER_MIN,
    estimate_effective_size,
    estimate_from_clusters,
)
from evallint.schema import EvalCase, EvalSet

# Filler sentences that share no template and no vocabulary, so the
# deterministic detectors leave them alone. Numbered wording would form one
# template family, which is a mistake made earlier in this project.
UNIQUE_TEXTS = [
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
]


def make_set(texts: list[str], expected: str = "same") -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(id=f"c{i}", input=text, expected=expected)
            for i, text in enumerate(texts)
        ),
        source="t",
    )


def orthogonal_embedder(texts_to_group: dict[str, int], dimensions: int):
    """One-hot vectors, so cases in the same group are identical and cases in
    different groups are orthogonal. Two dimensions cannot make many groups
    mutually orthogonal, which broke an earlier test in this project."""

    def embed(texts):
        rows = []
        for text in texts:
            vector = [0.0] * dimensions
            vector[texts_to_group[text]] = 1.0
            rows.append(vector)
        return np.array(rows, dtype=np.float32)

    return embed


# =========================================================================
# 1. FULLY UNIQUE DATA
# =========================================================================


def test_fully_unique_data_has_no_redundancy() -> None:
    eval_set = make_set(UNIQUE_TEXTS)
    groups = {f"{t}\nsame": i for i, t in enumerate(UNIQUE_TEXTS)}
    estimate = estimate_effective_size(
        eval_set,
        embedder=orthogonal_embedder(groups, len(UNIQUE_TEXTS)),
        sensitivity=False,
    )

    assert estimate.raw_cases == 12
    assert estimate.scenario_clusters == 12
    assert estimate.redundant_cases == 0
    assert estimate.potential_redundancy == 0.0
    assert estimate.large_clusters == 0
    assert estimate.largest_cluster == 1
    # The bracket collapses when there is nothing to adjust for.
    assert estimate.independent_case_estimate == (12, 12)
    assert estimate.design_effect == pytest.approx(1.0)
    assert estimate.interval_inflation == pytest.approx(1.0)


def test_unique_data_needs_no_interval_widening() -> None:
    estimate = estimate_from_clusters([], raw_cases=500)
    assert estimate.design_effect == pytest.approx(1.0)
    assert "roughly 1.00x too narrow" in estimate.render()


# =========================================================================
# 2. EXACT DUPLICATES
# =========================================================================


def test_exact_duplicates_are_found_without_any_embedder() -> None:
    """Deterministic, so this needs no model: 4 copies of one case plus 8
    unique ones is 9 scenarios, not 12."""
    eval_set = make_set(["the very same sentence"] * 4 + UNIQUE_TEXTS[:8])
    estimate = estimate_effective_size(eval_set, semantic=False)

    assert estimate.raw_cases == 12
    assert estimate.scenario_clusters == 9
    assert estimate.redundant_cases == 3
    assert estimate.potential_redundancy == pytest.approx(3 / 12)
    assert estimate.largest_cluster == 4
    assert estimate.large_clusters == 1
    assert estimate.independent_case_estimate == (9, 12)


def test_a_pair_is_not_counted_as_a_large_cluster() -> None:
    """A pair is the smallest possible redundancy and barely moves an average.
    LARGE_CLUSTER_MIN is a convention, and it is 3 for that reason."""
    estimate = estimate_from_clusters([2, 2, 2], raw_cases=100)
    assert estimate.large_clusters == 0
    assert estimate.redundant_cases == 3

    estimate = estimate_from_clusters([2, 3], raw_cases=100)
    assert estimate.large_clusters == 1
    assert LARGE_CLUSTER_MIN == 3


def test_the_size_distribution_includes_singletons() -> None:
    estimate = estimate_from_clusters([4, 2], raw_cases=10)
    assert estimate.cluster_size_distribution == {1: 4, 2: 1, 4: 1}
    assert sum(size * count for size, count in
               estimate.cluster_size_distribution.items()) == 10


# =========================================================================
# 3. HIGHLY REDUNDANT DATA
# =========================================================================


def test_a_highly_redundant_set_collapses_to_few_scenarios() -> None:
    """20 cases in two groups of 10. Two scenarios, 90% redundancy, and a
    design effect of 10 — intervals computed on 20 cases are more than three
    times too narrow."""
    texts = [f"group one variant {chr(97 + i)}" for i in range(10)]
    texts += [f"group two variant {chr(97 + i)}" for i in range(10)]
    groups = {f"{t}\nsame": (0 if i < 10 else 1) for i, t in enumerate(texts)}
    estimate = estimate_effective_size(
        make_set(texts), embedder=orthogonal_embedder(groups, 2), sensitivity=False
    )

    assert estimate.raw_cases == 20
    assert estimate.scenario_clusters == 2
    assert estimate.potential_redundancy == pytest.approx(0.9)
    assert estimate.design_effect == pytest.approx(10.0)
    assert estimate.interval_inflation == pytest.approx(10.0**0.5)
    assert estimate.largest_cluster == 10


def test_redundancy_is_never_described_as_a_defect() -> None:
    """Consistency tests, template families and regression suites all repeat a
    scenario on purpose."""
    text = estimate_from_clusters([10, 10], raw_cases=20).render()
    assert "Redundancy is not a defect" in text
    assert "does not recommend deleting any case" in text
    for banned in ("invalid", "bad case", "delete these", "remove these"):
        assert banned not in text.lower()


# =========================================================================
# 4. TEMPLATED DATA
# =========================================================================


def test_a_templated_set_collapses_to_its_template_families() -> None:
    """Identical structure, different numbers. Caught exactly by the template
    detector with no embedder and no threshold, which is why this is the
    cheapest scenario to check."""
    texts = [f"What is {i} plus {i + 1} in total?" for i in range(1, 9)]
    texts += [f"How many days are in month number {i} of the year?" for i in range(1, 5)]
    estimate = estimate_effective_size(make_set(texts), semantic=False)

    assert estimate.raw_cases == 12
    # Two template families, so two scenarios.
    assert estimate.scenario_clusters == 2
    assert estimate.potential_redundancy == pytest.approx(10 / 12)
    assert estimate.large_clusters == 2
    assert "template" in estimate.method


def test_a_templated_set_reports_a_large_design_effect() -> None:
    texts = [f"What is {i} plus {i + 1} in total?" for i in range(1, 13)]
    estimate = estimate_effective_size(make_set(texts), semantic=False)
    assert estimate.scenario_clusters == 1
    assert estimate.design_effect == pytest.approx(12.0)
    assert "12 cases are roughly 3.46x too narrow" in estimate.render()


# =========================================================================
# TERMINOLOGY: this is not a statistical effective sample size
# =========================================================================


def test_the_output_refuses_the_effective_sample_size_claim() -> None:
    text = estimate_from_clusters([3, 3], raw_cases=20).render()
    assert "THIS IS AN ESTIMATE, NOT A STATISTICAL EFFECTIVE SAMPLE SIZE." in text
    # The approved vocabulary.
    assert "Redundancy-adjusted scenario coverage" in text
    assert "estimate; plausible range" in text


def test_the_kish_reduction_is_disclosed_as_an_assumption() -> None:
    """The cluster count IS Kish's n_eff under inverse-cluster weights, and that
    equality holds only because the weights assume rho = 1. Presenting the
    reduction as a derivation would be the overclaim."""
    caveats = " ".join(estimate_from_clusters([3], raw_cases=10).caveats)
    assert "Kish" in caveats
    assert "assumption, not a derivation" in caveats
    assert "correlation of 1" in caveats


def test_the_headline_is_the_conservative_end_of_a_bracket() -> None:
    estimate = estimate_from_clusters([5, 5], raw_cases=100)
    low, high = estimate.independent_case_estimate
    assert low == estimate.scenario_clusters == 92
    assert high == estimate.raw_cases == 100
    assert low <= high
    caveats = " ".join(estimate.caveats)
    assert "PERFECTLY redundant" in caveats
    assert "true independent-information count is higher" in caveats


def test_connected_components_bias_is_declared_as_conservative() -> None:
    """Chaining inflates cluster sizes, which lowers the estimate. That makes it
    more conservative, not less, and saying which direction an error runs is the
    difference between a caveat and an excuse."""
    caveats = " ".join(estimate_from_clusters([3], raw_cases=10).caveats)
    assert "connected components" in caveats
    assert "MORE conservative, not less" in caveats


# =========================================================================
# Threshold sensitivity
# =========================================================================


def test_a_stable_cluster_count_is_reported_as_not_load_bearing() -> None:
    estimate = estimate_from_clusters(
        [3], raw_cases=100,
        threshold_sensitivity={"0.80": 98, "0.85": 98, "0.90": 98, "0.95": 99},
    )
    assert estimate.threshold_is_load_bearing is False
    assert "not load-bearing here" in estimate.render()


def test_a_swinging_cluster_count_is_reported_as_threshold_dependent() -> None:
    estimate = estimate_from_clusters(
        [3], raw_cases=100,
        threshold_sensitivity={"0.80": 40, "0.85": 70, "0.90": 90, "0.95": 99},
    )
    assert estimate.threshold_is_load_bearing is True
    assert "threshold-dependent" in estimate.render()


def test_sensitivity_is_absent_without_semantic_clustering() -> None:
    """The deterministic levels have no threshold, so there is nothing to vary
    and no curve to report."""
    estimate = estimate_effective_size(make_set(UNIQUE_TEXTS), semantic=False)
    assert estimate.threshold_sensitivity == {}
    assert estimate.threshold_is_load_bearing is None


def test_the_embedder_is_called_once_across_thresholds() -> None:
    """Four thresholds must not cost four embedding passes."""
    calls = []
    groups = {f"{t}\nsame": i for i, t in enumerate(UNIQUE_TEXTS)}
    base = orthogonal_embedder(groups, len(UNIQUE_TEXTS))

    def counting(texts):
        calls.append(1)
        return base(texts)

    estimate_effective_size(
        make_set(UNIQUE_TEXTS), embedder=counting, sensitivity=True
    )
    assert len(calls) == 1, f"embedder called {len(calls)} times"


# =========================================================================
# Input validation and the JSON shape
# =========================================================================


def test_singletons_must_not_be_passed_as_clusters() -> None:
    with pytest.raises(ValueError, match="groups of 2 or more"):
        estimate_from_clusters([1, 3], raw_cases=10)


def test_clusters_cannot_exceed_the_case_count() -> None:
    with pytest.raises(ValueError, match="but the eval has"):
        estimate_from_clusters([8, 8], raw_cases=10)


def test_negative_raw_cases_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_from_clusters([], raw_cases=-1)


def test_an_empty_eval_does_not_divide_by_zero() -> None:
    estimate = estimate_from_clusters([], raw_cases=0)
    assert estimate.potential_redundancy == 0.0
    assert estimate.design_effect == 1.0


def test_as_dict_is_json_shaped() -> None:
    import json

    payload = estimate_from_clusters(
        [3, 2], raw_cases=20, threshold=0.85,
        threshold_sensitivity={"0.85": 16},
    ).as_dict()
    json.dumps(payload)
    # 20 cases, clusters of 3 and 2 -> 5 clustered, 15 singletons,
    # so 15 + 2 groups = 17 scenarios.
    assert payload["independent_case_estimate"] == [17, 20]
    assert payload["cluster_size_distribution"] == {"1": 15, "2": 1, "3": 1}
    assert payload["caveats"]
