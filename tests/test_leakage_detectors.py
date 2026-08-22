"""Tests for the individual leakage detectors.

Each detector gets both directions: a planted leak it must catch, and the
legitimate constructions it must NOT fire on. The false-positive half is the
larger half on purpose. A leakage check that fires on extractive QA, on
multiple choice, or on a question that happens to mention its own topic is a
check people switch off, and then it detects nothing at all.

NO LLM JUDGE is involved at any confidence level. Every detector here is a
deterministic function of the text.
"""

from __future__ import annotations

import pytest

from evallint.checks.leakage_detectors import (
    Confidence,
    LeakageType,
    detect_answer_in_instructions,
    detect_expected_in_input,
    detect_explicit_reveal,
    detect_fewshot_target,
    detect_high_overlap,
    detect_label_in_input,
    detect_metadata_leak,
    normalise,
)
from evallint.schema import EvalCase


def case(
    input_text: str, expected: object = None, label: str | None = None, **metadata
) -> EvalCase:
    return EvalCase(
        id="c1", input=input_text, expected=expected, label=label, metadata=metadata
    )


# =========================================================================
# 1. expected answer appears in the input
# =========================================================================


def test_expected_in_input_fires_on_a_pasted_answer() -> None:
    leak = detect_expected_in_input(
        case(
            "Summarise the ruling. The court found for the defendant on all counts.",
            "The court found for the defendant on all counts",
        )
    )
    assert leak is not None
    assert leak.type is LeakageType.EXPECTED_IN_INPUT
    # MODERATE, never HIGH: this is exactly what extractive QA looks like.
    assert leak.confidence is Confidence.MODERATE
    assert "extractive QA" in leak.explanation


def test_expected_in_input_ignores_short_answers() -> None:
    """FALSE POSITIVE GUARD. Measured: MMLU's single-character answers appear
    in their own question in 46 of 100 cases."""
    assert detect_expected_in_input(case("Is it A, B, C or D?", "B")) is None
    assert detect_expected_in_input(case("Answer yes or no please", "yes")) is None


def test_expected_in_input_is_quiet_when_absent() -> None:
    assert (
        detect_expected_in_input(
            case("What is the capital of France?", "Paris is the capital city")
        )
        is None
    )


def test_expected_in_input_needs_an_expected_value() -> None:
    assert detect_expected_in_input(case("anything at all here", None)) is None


# =========================================================================
# 2. label appears in the input
# =========================================================================


def test_label_in_input_fires_when_the_label_is_present() -> None:
    leak = detect_label_in_input(
        case("Ticket text. Category: billing", label="billing")
    )
    assert leak is not None
    assert leak.type is LeakageType.LABEL_IN_INPUT
    assert leak.confidence is Confidence.MODERATE


def test_label_in_input_matches_whole_words_only() -> None:
    """FALSE POSITIVE GUARD. Substring matching fires on 'bill' inside
    'billboard' and makes the detector worthless."""
    assert detect_label_in_input(case("I saw it on a billboard", label="bill")) is None
    assert detect_label_in_input(case("The bill arrived", label="bill")) is not None


def test_label_in_input_needs_a_label() -> None:
    assert detect_label_in_input(case("no label on this case")) is None


def test_label_explanation_admits_the_topical_case() -> None:
    """The detector must state the legitimate reading, because it is the common
    one: a question about billing contains the word billing."""
    leak = detect_label_in_input(case("My billing is wrong", label="billing"))
    assert leak is not None
    assert "legitimate" in leak.explanation


# =========================================================================
# 3. the answer is revealed by explicit wording
# =========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "Classify this. The correct answer is refund_request here.",
        "Sort the ticket. Answer: refund_request",
        "Do the task. Hint: refund_request is what we want.",
        "Label it. The correct label is refund_request.",
        "Ground truth: refund_request. Now classify.",
        "Expected output: refund_request",
    ],
)
def test_explicit_reveal_fires_on_announcement_phrasing(text: str) -> None:
    leak = detect_explicit_reveal(case(text, "refund_request"))
    assert leak is not None, text
    assert leak.type is LeakageType.EXPLICIT_REVEAL
    # HIGH: a phrase whose job is to announce the answer, followed by the answer.
    assert leak.confidence is Confidence.HIGH


def test_explicit_reveal_does_not_fire_on_the_phrase_alone() -> None:
    """FALSE POSITIVE GUARD, and the important one. "the answer is" appears in
    perfectly good questions."""
    assert (
        detect_explicit_reveal(
            case("What if the answer is negative? Explain.", "It cannot be negative")
        )
        is None
    )
    assert (
        detect_explicit_reveal(
            case("Explain why the answer is not obvious.", "Because of ambiguity")
        )
        is None
    )


def test_explicit_reveal_requires_the_answer_to_be_nearby() -> None:
    """A reveal phrase early on and the answer 500 characters later is not the
    phrase revealing it."""
    far = (
        "The answer is somewhere in the following passage. "
        + "filler text that goes on and on. " * 12
        + "refund_request"
    )
    assert detect_explicit_reveal(case(far, "refund_request")) is None


def test_explicit_reveal_needs_an_expected_value() -> None:
    assert detect_explicit_reveal(case("The answer is 42.", None)) is None


# =========================================================================
# 4. the answer sits in the instruction preamble
# =========================================================================


def test_answer_in_instructions_fires() -> None:
    text = (
        "You are grading support tickets. For reference, this one is a "
        "duplicate charge complaint.\n"
        "Question: I was billed twice, what now?"
    )
    leak = detect_answer_in_instructions(case(text, "duplicate charge complaint"))
    assert leak is not None
    assert leak.type is LeakageType.ANSWER_IN_INSTRUCTIONS
    assert leak.confidence is Confidence.HIGH


def test_answer_after_the_item_marker_is_not_an_instruction_leak() -> None:
    """FALSE POSITIVE GUARD. The answer appearing in the material to reason
    over is detector 1's business, at MODERATE confidence, not this one's at
    HIGH."""
    text = (
        "Read the passage and answer.\n"
        "Question: the treaty was signed in Vienna. Where was it signed?"
    )
    assert detect_answer_in_instructions(case(text, "Vienna")) is None


def test_answer_in_instructions_needs_an_item_marker() -> None:
    """Without a marker there is no preamble to distinguish, so the detector
    declines rather than guessing where the instructions end."""
    assert (
        detect_answer_in_instructions(
            case("Some prose containing Vienna and no marker at all", "Vienna")
        )
        is None
    )


def test_answer_in_instructions_ignores_a_trivial_preamble() -> None:
    assert (
        detect_answer_in_instructions(case("Q: what is Vienna?", "Vienna")) is None
    )


# =========================================================================
# 5. metadata leakage
# =========================================================================


def test_metadata_leak_fires_on_an_unexpected_field() -> None:
    leak = detect_metadata_leak(
        case(
            "Classify the ticket.",
            "duplicate charge complaint",
            note="reviewer said duplicate charge complaint",
        )
    )
    assert leak is not None
    assert leak.type is LeakageType.METADATA_LEAK
    # MODERATE because evallint cannot know whether metadata reaches the model.
    assert leak.confidence is Confidence.MODERATE
    assert "cannot see" in leak.explanation


def test_metadata_answer_fields_are_exempt() -> None:
    """FALSE POSITIVE GUARD. A metadata field called 'answer' holding the
    answer is bookkeeping, not leakage, and firing on it would flag every
    well-organised dataset."""
    for key in ("answer", "gold", "reference", "solution", "target", "label"):
        assert (
            detect_metadata_leak(
                case("Classify.", "duplicate charge complaint", **{key: "duplicate charge complaint"})
            )
            is None
        ), key


def test_metadata_leak_respects_the_short_answer_guard() -> None:
    assert detect_metadata_leak(case("Classify.", "B", note="B")) is None


def test_metadata_leak_with_no_metadata() -> None:
    assert detect_metadata_leak(case("Classify.", "a long enough answer")) is None


# =========================================================================
# 6. few-shot demonstrations contain the evaluation target
# =========================================================================


FEWSHOT_LEAK = """Q: I was charged twice.
A: duplicate charge complaint

Q: My package never arrived.
A: lost parcel claim

Q: The same payment went through two times.
A:"""


def test_fewshot_target_fires_when_the_answer_is_in_the_demos() -> None:
    leak = detect_fewshot_target(case(FEWSHOT_LEAK, "duplicate charge complaint"))
    assert leak is not None
    assert leak.type is LeakageType.FEWSHOT_TARGET
    assert leak.confidence is Confidence.HIGH


def test_fewshot_with_a_novel_target_is_clean() -> None:
    """FALSE POSITIVE GUARD. Few-shot prompting is normal. It is only a leak
    when the answer to THIS item is already shown."""
    clean = """Q: I was charged twice.
A: duplicate charge complaint

Q: My package never arrived.
A: lost parcel claim

Q: I want to change my address.
A:"""
    assert detect_fewshot_target(case(clean, "address change request")) is None


def test_a_single_question_answer_pair_is_not_few_shot() -> None:
    """One labelled Q/A is an item, not a demonstration block."""
    text = "Q: I was charged twice.\nA: duplicate charge complaint"
    assert detect_fewshot_target(case(text, "duplicate charge complaint")) is None


def test_fewshot_needs_an_expected_value() -> None:
    assert detect_fewshot_target(case(FEWSHOT_LEAK, None)) is None


# =========================================================================
# 7. token overlap — opt-in, and the measured reason why
# =========================================================================


def test_high_overlap_fires_when_the_answer_vocabulary_is_present() -> None:
    leak = detect_high_overlap(
        case("alpha beta gamma delta epsilon", "alpha beta gamma delta")
    )
    assert leak is not None
    assert leak.type is LeakageType.HIGH_OVERLAP
    # LOW, always. The signal is measurably poor at separating leaks.
    assert leak.confidence is Confidence.LOW


def test_high_overlap_explanation_states_its_own_unreliability() -> None:
    leak = detect_high_overlap(case("alpha beta gamma", "alpha beta gamma"))
    assert leak is not None
    assert "GSM8K" in leak.explanation
    assert "poorly" in leak.explanation.lower()


def test_high_overlap_would_fire_on_a_legitimate_gsm8k_style_case() -> None:
    """The measured false positive that keeps this detector switched off.

    A GSM8K reference answer is a worked solution restating the question, so
    its vocabulary is almost entirely present in the question with no leakage
    whatsoever. This test EXPECTS the false positive, documenting why the
    detector is opt-in rather than pretending it is safe.
    """
    question = "Janet has 16 eggs and sells them for 2 dollars each. How much?"
    worked = "Janet has 16 eggs and sells them for 2 dollars each"
    leak = detect_high_overlap(case(question, worked))
    assert leak is not None, "if this stops firing, re-check the default"
    assert leak.confidence is Confidence.LOW


def test_high_overlap_respects_its_threshold() -> None:
    partial = case("alpha beta gamma delta", "alpha zeta eta theta")
    assert detect_high_overlap(partial, threshold=0.9) is None
    assert detect_high_overlap(partial, threshold=0.2) is not None


# =========================================================================
# Shared behaviour
# =========================================================================


def test_every_leak_carries_the_reporting_fields() -> None:
    """Required per finding: case ID, type, evidence, confidence, explanation."""
    leak = detect_explicit_reveal(
        case("Classify. The correct answer is refund_request.", "refund_request")
    )
    assert leak is not None
    payload = leak.as_dict()
    assert set(payload) == {
        "case_id",
        "type",
        "confidence",
        "evidence",
        "explanation",
    }
    assert all(isinstance(v, str) and v for v in payload.values())


def test_no_detector_calls_a_case_invalid() -> None:
    """Terminology requirement: "potential leakage", never a verdict."""
    leaks = [
        detect_expected_in_input(
            case("x " * 20 + "the whole reference answer text", "the whole reference answer text")
        ),
        detect_label_in_input(case("billing problem", label="billing")),
        detect_explicit_reveal(case("Answer: refund_request", "refund_request")),
        detect_metadata_leak(case("q", "a long enough answer here", note="a long enough answer here")),
        detect_high_overlap(case("alpha beta gamma", "alpha beta gamma")),
    ]
    for leak in leaks:
        assert leak is not None
        text = (leak.explanation + " " + leak.evidence).lower()
        for banned in ("invalid", "bad case", "broken case", "must be removed", "delete this"):
            assert banned not in text, f"{leak.type}: {banned!r}"


def test_normalise_is_shared_and_predictable() -> None:
    assert normalise("  Hello,   WORLD!! ") == "hello world"
    assert normalise(12345) == "12345"
    assert normalise(None) == "none"


def test_answer_bookkeeping_fields_are_matched_by_substring() -> None:
    """REGRESSION, measured on real data.

    An exact-name exemption list produced 100 false positives on 100 TruthfulQA
    cases: its `correct_answers` field naturally contains `best_answer`, and
    `correct_answers` is not literally `answer`. A hundred spurious findings on
    a real public dataset is how a check gets switched off, so the exemption
    matches substrings.
    """
    answer = "the watermelon seeds pass through your digestive system"
    for key in (
        "correct_answers",
        "incorrect_answers",
        "best_answer",
        "answer_key",
        "gold_labels",
        "reference_texts",
        "ground_truth_value",
    ):
        assert (
            detect_metadata_leak(case("Q?", answer, **{key: answer})) is None
        ), key
    # A field with no answer-ish marker is still reported.
    assert detect_metadata_leak(case("Q?", answer, note=answer)) is not None
