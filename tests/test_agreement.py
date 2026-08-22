"""Tests for the inter-rater agreement statistics.

Expected values are hand-computed from the definitions, not recorded from
output. A test that captures what the code currently says cannot detect a wrong
estimator, and these numbers decide whether the tool tells a user their judges
are reliable.
"""

from __future__ import annotations

import pytest

from evallint.checks.agreement import (
    cohens_kappa,
    fleiss_kappa,
    pairwise_agreement,
    raw_agreement,
)

# --- Cohen's kappa --------------------------------------------------------


def test_cohens_kappa_matches_a_hand_computed_case() -> None:
    """10 items, observed agreement 0.8, both raters split 5/5.

    p_e = (5/10)(5/10) + (5/10)(5/10) = 0.5
    kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6
    """
    a = ["y"] * 5 + ["n"] * 5
    b = ["y"] * 4 + ["n"] + ["n"] * 4 + ["y"]
    assert cohens_kappa(a, b) == pytest.approx(0.6)


def test_cohens_kappa_is_one_for_perfect_agreement_across_categories() -> None:
    assert cohens_kappa(["y", "n", "y"], ["y", "n", "y"]) == pytest.approx(1.0)


def test_cohens_kappa_is_undefined_when_one_category_is_used_throughout() -> None:
    """THE KAPPA PARADOX, and the common case here: two judges that both say
    "not ambiguous" for every case. Expected agreement is 1, so kappa is 0/0.

    None means undefined. Returning 1.0 would claim perfect reliability from
    two raters who never made a distinction, and returning 0.0 would claim they
    were no better than chance. Both are wrong.
    """
    assert cohens_kappa(["n"] * 8, ["n"] * 8) is None


def test_cohens_kappa_can_be_negative() -> None:
    """Worse than chance is a real result and must not be clamped."""
    kappa = cohens_kappa(["y", "y", "n", "n"], ["n", "n", "y", "y"])
    assert kappa is not None and kappa < 0


def test_cohens_kappa_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="same items"):
        cohens_kappa(["y"], ["y", "n"])


def test_cohens_kappa_with_no_items() -> None:
    assert cohens_kappa([], []) is None


# --- Fleiss' kappa -------------------------------------------------------


def test_fleiss_kappa_is_one_for_unanimous_use_of_both_categories() -> None:
    ratings = [["y", "y", "y"], ["n", "n", "n"], ["y", "y", "y"], ["n", "n", "n"]]
    assert fleiss_kappa(ratings) == pytest.approx(1.0)


def test_fleiss_kappa_hand_computed_on_a_split() -> None:
    """4 items, 3 raters.

    Per-item agreement P_i = (sum n_ij^2 - n) / (n(n-1)) with n = 3:
      [y,y,n] -> (4+1-3)/6 = 1/3
      [y,n,n] -> (1+4-3)/6 = 1/3
      [y,y,y] -> (9-3)/6   = 1
      [n,n,n] -> (9-3)/6   = 1
    P_bar = (1/3 + 1/3 + 1 + 1)/4 = 2/3
    marginals: y = 6/12 = 0.5, n = 6/12 = 0.5 -> P_e = 0.5
    kappa = (2/3 - 1/2) / (1 - 1/2) = 1/3
    """
    ratings = [["y", "y", "n"], ["y", "n", "n"], ["y", "y", "y"], ["n", "n", "n"]]
    assert fleiss_kappa(ratings) == pytest.approx(1 / 3)


def test_fleiss_kappa_is_undefined_when_every_rating_is_identical() -> None:
    assert fleiss_kappa([["n", "n"]] * 20) is None


def test_fleiss_kappa_needs_two_items() -> None:
    assert fleiss_kappa([["y", "n"]]) is None
    assert fleiss_kappa([]) is None


def test_fleiss_kappa_rejects_ragged_ratings() -> None:
    """Ragged data needs Krippendorff's alpha, and saying so beats silently
    computing something that is not Fleiss' kappa."""
    with pytest.raises(ValueError, match="Krippendorff"):
        fleiss_kappa([["y", "n"], ["y", "n", "y"]])


# --- raw and pairwise agreement -----------------------------------------


def test_raw_agreement_counts_only_unanimous_items() -> None:
    ratings = [["y", "y", "y"], ["y", "y", "n"], ["n", "n", "n"], ["y", "n", "n"]]
    assert raw_agreement(ratings) == pytest.approx(0.5)


def test_pairwise_agreement_distinguishes_a_split_from_a_near_miss() -> None:
    """The reason pairwise exists. Both items below fail unanimity, but two of
    three agreeing is not the same as a three-way split — and with only two
    categories, a 2-1 split is the worst possible outcome, so pairwise is 1/3.
    """
    near_miss = [["y", "y", "n"]]
    assert raw_agreement(near_miss) == 0.0
    assert pairwise_agreement(near_miss) == pytest.approx(1 / 3)

    unanimous = [["y", "y", "y"]]
    assert raw_agreement(unanimous) == 1.0
    assert pairwise_agreement(unanimous) == 1.0


def test_raw_agreement_is_high_on_skewed_data_which_is_why_kappa_exists() -> None:
    """19 of 20 cases unanimous "no", one unanimous "yes". Raw agreement is
    100%, which sounds excellent and says almost nothing — the raters barely
    made a distinction. This is exactly why raw agreement is never reported
    without a chance-corrected figure beside it.
    """
    skewed = [["n", "n"]] * 19 + [["y", "y"]]
    assert raw_agreement(skewed) == 1.0
    assert fleiss_kappa(skewed) == pytest.approx(1.0)  # here they do agree

    # But when the minority case is disputed, kappa collapses while raw
    # agreement stays high.
    disputed = [["n", "n"]] * 19 + [["y", "n"]]
    assert raw_agreement(disputed) == pytest.approx(0.95)
    kappa = fleiss_kappa(disputed)
    assert kappa is not None and kappa < 0.5


def test_agreement_with_no_items() -> None:
    assert raw_agreement([]) == 1.0
    assert pairwise_agreement([]) == 1.0


def test_single_rater_items_are_ignored_for_pairwise() -> None:
    assert pairwise_agreement([["y"], ["n"]]) == 1.0
