"""Tests for the reproducibility analyser.

Two kinds, as required. DETERMINISTIC tests construct outcomes by hand so every
expected value is exact — including designs where one variance source is
provably zero. SEEDED STOCHASTIC tests use a fixed seed so a genuinely random
judge or model produces the same numbers on every run.

The sharpest test is `test_a_stable_aggregate_can_hide_total_case_instability`:
an eval whose headline score has standard deviation EXACTLY zero while 100% of
its individual verdicts flip between runs. Averaging hides it completely.
"""

from __future__ import annotations

import random

import pytest

from evallint.checks.agreement import kendalls_w
from evallint.checks.reproducibility import (
    FLIP_PRONE_SHARE,
    RunOutcome,
    analyse_reproducibility,
)

CASES = [f"c{i}" for i in range(20)]


def outcome(run, model, case, passed, judge="j1"):
    return RunOutcome(run=run, model=model, case=case, passed=passed, judge=judge)


# =========================================================================
# DETERMINISTIC: model stochasticity isolated
# =========================================================================


def test_a_deterministic_model_has_exactly_zero_model_variance() -> None:
    """Same verdicts in every run, so there is nothing for re-running to
    reveal. Zero is the correct answer AND is stated ambiguously on purpose:
    a cached harness looks identical from here."""
    outcomes = [
        outcome(r, "m", c, int(c[1:]) % 2 == 0) for r in range(4) for c in CASES
    ]
    report = analyse_reproducibility(outcomes)
    model = report.variance_sources["model_stochasticity"]

    assert model.value == 0.0
    assert model.n == 4
    assert "cached" in model.interpretation
    assert "measures the cache" in " ".join(model.limitations)


def test_a_known_score_sequence_gives_the_hand_computed_sd() -> None:
    """Runs scoring 10/20, 12/20, 14/20, 16/20 -> 0.5, 0.6, 0.7, 0.8.
    Sample standard deviation of those four values is sqrt(0.0166..) = 0.12910.
    """
    passes = {0: 10, 1: 12, 2: 14, 3: 16}
    outcomes = [
        outcome(r, "m", c, index < passes[r])
        for r in range(4)
        for index, c in enumerate(CASES)
    ]
    report = analyse_reproducibility(outcomes)

    aggregate = report.aggregate_stability["score_sd"]
    assert aggregate.value == pytest.approx(0.129099, abs=1e-5)
    assert aggregate.interval == (0.5, 0.8)
    assert "50.0%" in aggregate.interpretation and "80.0%" in aggregate.interpretation


# =========================================================================
# DETERMINISTIC: evaluator stochasticity isolated
# =========================================================================


def test_a_deterministic_model_with_disagreeing_judges_isolates_the_grader() -> None:
    """The key separation. The model is deterministic, so model variance must be
    exactly zero, and ALL instability must be attributed to the judges."""
    outcomes = []
    for run in range(3):
        for case in CASES:
            base = int(case[1:]) % 2 == 0
            outcomes.append(outcome(run, "m", case, base, judge="lenient"))
            outcomes.append(
                outcome(run, "m", case, base and int(case[1:]) % 4 != 0, judge="strict")
            )
    report = analyse_reproducibility(outcomes)

    assert report.variance_sources["model_stochasticity"].value == 0.0
    evaluator = report.variance_sources["evaluator_stochasticity"]
    assert evaluator.value is not None and evaluator.value > 0.1
    assert "no amount of re-running the model will reduce it" in evaluator.interpretation
    assert "Averaging more model runs does not reduce this" in " ".join(
        evaluator.limitations
    )


def test_agreeing_judges_give_zero_evaluator_variance() -> None:
    outcomes = []
    for run in range(2):
        for case in CASES:
            passed = int(case[1:]) % 2 == 0
            for judge in ("a", "b", "c"):
                outcomes.append(outcome(run, "m", case, passed, judge=judge))
    evaluator = analyse_reproducibility(outcomes).variance_sources[
        "evaluator_stochasticity"
    ]
    assert evaluator.value == 0.0
    assert "agreed exactly" in evaluator.interpretation


def test_one_judge_makes_evaluator_variance_unidentifiable() -> None:
    """Not zero — unidentifiable. With one grader there is nothing to compare it
    against, and reporting 0.0 would claim the grader is noiseless."""
    outcomes = [outcome(r, "m", c, True) for r in range(3) for c in CASES]
    evaluator = analyse_reproducibility(outcomes).variance_sources[
        "evaluator_stochasticity"
    ]
    assert evaluator.value is None
    assert "needs at least 2" in evaluator.unmet_reason


def test_an_untagged_judge_is_reported_as_a_design_gap() -> None:
    outcomes = [
        RunOutcome(run=r, model="m", case=c, passed=True)
        for r in range(3)
        for c in CASES
    ]
    report = analyse_reproducibility(outcomes)
    assert report.n_judges == 0
    assert any("Tag outcomes with the judge" in note for note in report.notes)


# =========================================================================
# DETERMINISTIC: dataset sampling variability
# =========================================================================


def test_dataset_sampling_is_computable_from_a_single_run() -> None:
    """A different question from run-to-run variance, and answerable without
    repeating anything."""
    outcomes = [outcome(0, "m", c, int(c[1:]) < 14) for c in CASES]
    report = analyse_reproducibility(outcomes)

    sampling = report.variance_sources["dataset_sampling"]
    assert sampling.computed
    assert sampling.value > 0
    assert "ADDING cases, not by re-running" in sampling.interpretation
    # And the run-based sources are correctly unavailable.
    assert report.variance_sources["model_stochasticity"].value is None


def test_dataset_sampling_needs_ten_cases() -> None:
    outcomes = [outcome(0, "m", f"c{i}", True) for i in range(5)]
    sampling = analyse_reproducibility(outcomes).variance_sources["dataset_sampling"]
    assert sampling.value is None
    assert "too few to bootstrap" in sampling.unmet_reason


def test_sampling_variability_shrinks_as_cases_are_added() -> None:
    """The mechanism the metric points at: more cases, narrower interval."""
    small = [outcome(0, "m", f"c{i}", i % 2 == 0) for i in range(20)]
    large = [outcome(0, "m", f"c{i}", i % 2 == 0) for i in range(400)]
    narrow = analyse_reproducibility(large).variance_sources["dataset_sampling"]
    wide = analyse_reproducibility(small).variance_sources["dataset_sampling"]
    assert narrow.value < wide.value


# =========================================================================
# The three sources are never collapsed
# =========================================================================


def test_the_three_sources_are_reported_separately_and_never_summed() -> None:
    outcomes = []
    for run in range(3):
        for case in CASES:
            base = int(case[1:]) % 2 == 0
            outcomes.append(outcome(run, "m", case, base, judge="a"))
            outcomes.append(outcome(run, "m", case, base and run != 1, judge="b"))
    report = analyse_reproducibility(outcomes)

    assert set(report.variance_sources) == {
        "model_stochasticity",
        "evaluator_stochasticity",
        "dataset_sampling",
    }
    payload = report.as_dict()
    # No total anywhere.
    assert "total_variance" not in payload
    assert "total" not in str(payload["variance_sources"].keys())
    assert any("NOT summed" in note for note in report.notes)
    assert "reported separately, never summed" in report.render()


def test_each_source_names_the_fix_it_implies() -> None:
    """The reason for keeping them apart: they need different remedies."""
    outcomes = []
    for run in range(3):
        for case in CASES:
            outcomes.append(outcome(run, "m", case, run == 0, judge="a"))
            outcomes.append(outcome(run, "m", case, True, judge="b"))
    sources = analyse_reproducibility(outcomes).variance_sources

    assert "temperature" in sources["model_stochasticity"].interpretation
    assert "re-running" in sources["evaluator_stochasticity"].interpretation
    assert "ADDING cases" in sources["dataset_sampling"].interpretation


# =========================================================================
# Per-case stability
# =========================================================================


def test_a_stable_aggregate_can_hide_total_case_instability() -> None:
    """The finding this analyser exists to surface.

    Each run flips a different half of the cases, so the aggregate score is
    EXACTLY 50% every time — standard deviation zero — while not one single case
    keeps its verdict. Averaging conceals it completely.
    """
    outcomes = [
        outcome(r, "m", c, (i + r) % 2 == 0)
        for r in range(4)
        for i, c in enumerate(CASES)
    ]
    report = analyse_reproducibility(outcomes)

    assert report.aggregate_stability["score_sd"].value == 0.0
    assert report.case_stability["stable_share"] == 0.0
    warnings = [n for n in report.notes if n.startswith("WARNING")]
    assert warnings, "the divergence must be called out"
    assert "coincidence rather than reproducibility" in warnings[0]


def test_flip_prone_cases_are_named_with_their_threshold() -> None:
    stable = [outcome(r, "m", "steady", True) for r in range(4)]
    flipping = [outcome(r, "m", "wobbly", r % 2 == 0) for r in range(4)]
    report = analyse_reproducibility(stable + flipping)

    stability = report.case_stability
    assert stability["n_cases_assessed"] == 2
    assert stability["n_stable"] == 1
    assert stability["flip_prone_case_ids"] == ["m:wobbly"]
    assert stability["flip_prone_threshold"] == FLIP_PRONE_SHARE
    assert "A convention, not a statistic" in stability["threshold_note"]


def test_case_stability_needs_two_runs() -> None:
    report = analyse_reproducibility([outcome(0, "m", c, True) for c in CASES])
    assert report.case_stability["assessed"] is False
    assert report.case_stability["stable_share"] is None
    assert any("NOT evidence the eval is reproducible" in n for n in report.notes)


# =========================================================================
# Model ranking stability
# =========================================================================


def test_well_separated_models_rank_identically_in_every_run() -> None:
    outcomes = []
    for run in range(4):
        for model, threshold in (("small", 4), ("medium", 10), ("large", 18)):
            for index, case in enumerate(CASES):
                outcomes.append(outcome(run, model, case, index < threshold))
    ranking = analyse_reproducibility(outcomes).ranking_stability

    assert ranking["modal_ranking"] == ["large", "medium", "small"]
    assert ranking["modal_ranking_share"] == 1.0
    assert ranking["distinct_rankings"] == 1
    assert ranking["top_model_stable"] is True
    assert ranking["kendalls_w"] == pytest.approx(1.0)
    assert ranking["unstable_pairs"] == []


def test_a_deliberately_swapped_ranking_is_detected() -> None:
    """Two runs where the order of the top two models reverses."""
    outcomes = []
    for run in range(2):
        a_threshold, b_threshold = (12, 10) if run == 0 else (10, 12)
        for model, threshold in (("A", a_threshold), ("B", b_threshold)):
            for index, case in enumerate(CASES):
                outcomes.append(outcome(run, model, case, index < threshold))
    ranking = analyse_reproducibility(outcomes).ranking_stability

    assert ranking["distinct_rankings"] == 2
    assert ranking["modal_ranking_share"] == 0.5
    assert ranking["top_model_stable"] is False
    assert ranking["unstable_pairs"] == [("A", "B")]
    assert ranking["kendalls_w"] == pytest.approx(0.0)


def test_ranking_stability_needs_two_models_and_two_runs() -> None:
    single_model = [outcome(r, "m", c, True) for r in range(3) for c in CASES]
    assert analyse_reproducibility(single_model).ranking_stability["assessed"] is False

    single_run = [outcome(0, m, c, True) for m in ("a", "b") for c in CASES]
    assert analyse_reproducibility(single_run).ranking_stability["assessed"] is False


def test_kendalls_w_is_declared_to_be_a_lower_bound_under_ties() -> None:
    outcomes = [
        outcome(r, m, c, True) for r in range(3) for m in ("a", "b") for c in CASES
    ]
    ranking = analyse_reproducibility(outcomes).ranking_stability
    assert "lower bound" in ranking["note"]


# =========================================================================
# SEEDED STOCHASTIC
# =========================================================================


def test_a_seeded_noisy_model_gives_reproducible_numbers() -> None:
    """A genuinely random model, but the seed makes the ANALYSIS deterministic,
    so this test cannot flake."""

    def build(seed: int):
        rng = random.Random(seed)
        return [
            outcome(r, "m", c, rng.random() < 0.7) for r in range(5) for c in CASES
        ]

    first = analyse_reproducibility(build(0))
    second = analyse_reproducibility(build(0))

    assert (
        first.variance_sources["model_stochasticity"].value
        == second.variance_sources["model_stochasticity"].value
    )
    assert first.case_stability["stable_share"] == second.case_stability["stable_share"]


def test_a_noisy_model_shows_nonzero_variance_and_flip_prone_cases() -> None:
    rng = random.Random(0)
    outcomes = [
        outcome(r, "m", c, rng.random() < 0.5) for r in range(5) for c in CASES
    ]
    report = analyse_reproducibility(outcomes)

    model = report.variance_sources["model_stochasticity"]
    assert model.value is not None and model.value > 0.01
    # A coin-flip model over 5 runs: P(all identical) = 2 * 0.5^5 = 0.0625, so
    # almost every case should be unstable.
    assert report.case_stability["stable_share"] < 0.25

    # flip-prone needs min(passes, fails) >= 2 of 5, and for Bin(5, 0.5) that is
    # P(2 or 3 successes) = (C(5,2) + C(5,3)) / 32 = 20/32 = 0.625. Over 20 cases
    # the expectation is 12.5, and the observed count for this seed is 10 -- well
    # within binomial noise. An earlier version asserted > 10 and sat exactly on
    # the boundary, so the bound is now derived rather than eyeballed.
    assert report.case_stability["n_flip_prone"] >= 8


def test_near_identical_models_produce_unstable_rankings_under_noise() -> None:
    """Three models two percentage points apart. With sampling noise the
    ranking cannot be trusted, and the analyser must say so rather than
    reporting whichever order this run produced."""
    rng = random.Random(1)
    outcomes = [
        outcome(r, model, c, rng.random() < p)
        for r in range(5)
        for model, p in (("small", 0.60), ("medium", 0.62), ("large", 0.64))
        for c in [f"c{i}" for i in range(40)]
    ]
    ranking = analyse_reproducibility(outcomes).ranking_stability

    assert ranking["distinct_rankings"] > 1
    assert ranking["modal_ranking_share"] < 1.0
    assert ranking["kendalls_w"] < 0.9
    assert ranking["unstable_pairs"]


def test_bootstrap_is_seeded_so_sampling_variance_is_reproducible() -> None:
    outcomes = [outcome(0, "m", f"c{i}", i % 3 == 0) for i in range(60)]
    a = analyse_reproducibility(outcomes, seed=7).variance_sources["dataset_sampling"]
    b = analyse_reproducibility(outcomes, seed=7).variance_sources["dataset_sampling"]
    assert a.value == b.value


# =========================================================================
# Reporting contract and validation
# =========================================================================


def test_the_pooled_interval_declares_itself_too_narrow() -> None:
    """Pooling runs treats correlated observations as independent. Reported for
    reference, and labelled."""
    outcomes = [outcome(r, "m", c, True) for r in range(3) for c in CASES]
    pooled = analyse_reproducibility(outcomes).aggregate_stability["pooled_interval"]
    assert "not for inference" in pooled.interpretation
    assert "too narrow" in " ".join(pooled.limitations)


def test_the_design_is_stated() -> None:
    outcomes = [outcome(r, "m", c, True) for r in range(3) for c in CASES]
    report = analyse_reproducibility(outcomes)
    assert "3 run(s)" in report.design
    assert "20 case(s)" in report.design
    assert "1 judge(s)" in report.design


def test_as_dict_is_json_shaped() -> None:
    import json

    outcomes = []
    for run in range(2):
        for case in CASES:
            outcomes.append(outcome(run, "a", case, True, judge="j1"))
            outcomes.append(outcome(run, "b", case, run == 0, judge="j2"))
    payload = analyse_reproducibility(outcomes).as_dict()
    json.dumps(payload)
    assert set(payload["variance_sources"]) == {
        "model_stochasticity",
        "evaluator_stochasticity",
        "dataset_sampling",
    }


def test_rejects_empty_and_nonsense_input() -> None:
    with pytest.raises(ValueError, match="no outcomes"):
        analyse_reproducibility([])
    with pytest.raises(ValueError, match="flip_prone_share"):
        analyse_reproducibility([outcome(0, "m", "c", True)], flip_prone_share=0)


def test_kendalls_w_matches_a_hand_computed_case() -> None:
    """2 raters, 3 items, one adjacent swap.

    ranks:      A=[1,2,3]  B=[2,1,3]
    rank sums:  [3, 3, 6],  Rbar = 2 * 4 / 2 = 4
    S = 1 + 1 + 4 = 6
    W = 12 * 6 / (4 * (27 - 3)) = 72 / 96 = 0.75
    """
    assert kendalls_w([[1, 2, 3], [2, 1, 3]]) == pytest.approx(0.75)
