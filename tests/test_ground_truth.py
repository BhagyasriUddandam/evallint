"""Tests for the ground-truth quality analyser.

Two halves. The detectors get both directions, weighted towards false positives
— a check that flags every subjective question as broken is a check people turn
off. The judge layer gets its constraints asserted, because "never treat one
LLM judge as absolute truth" has to be enforced by code rather than intended.
"""

from __future__ import annotations

import pytest

from evallint.checks.ambiguity_detectors import (
    AmbiguityType,
    Confidence,
    detect_ambiguous_wording,
    detect_missing_criteria,
    detect_multiple_interpretations,
    detect_multiple_valid_answers,
    detect_reference_inconsistency,
    detect_subjective_question,
    detect_underspecified_reference,
)
from evallint.checks.base import Severity
from evallint.checks.ground_truth import GroundTruthCheck
from evallint.schema import EvalCase, EvalSet


def one(input_text: str, expected: object = None, cid: str = "c1") -> EvalCase:
    return EvalCase(id=cid, input=input_text, expected=expected)


def make_set(cases: list[EvalCase]) -> EvalSet:
    return EvalSet(cases=tuple(cases), source="t")


def clean(n: int, start: int = 0) -> list[EvalCase]:
    """Cases no detector should fire on."""
    return [
        EvalCase(
            id=f"ok{i}",
            input=f"What is {i} multiplied by four?",
            expected=f"{i * 4}",
        )
        for i in range(start, start + n)
    ]


# =========================================================================
# 1. multiple valid answers
# =========================================================================


def test_list_valued_reference_is_flagged_high() -> None:
    finding = detect_multiple_valid_answers(one("Name a prime under 10.", ["2", "3", "5"]))
    assert finding is not None
    assert finding.ambiguity_type is AmbiguityType.MULTIPLE_VALID_ANSWERS
    assert finding.confidence is Confidence.HIGH
    assert "scorer accepts ANY" in finding.recommended_action


def test_a_single_element_list_is_not_multiple_answers() -> None:
    assert detect_multiple_valid_answers(one("Q?", ["only one"])) is None


def test_prose_enumerating_alternatives_is_flagged() -> None:
    finding = detect_multiple_valid_answers(
        one("Q?", "Either Paris or Lyon is acceptable")
    )
    assert finding is not None
    assert finding.confidence is Confidence.MODERATE


def test_a_plain_answer_containing_or_is_not_flagged() -> None:
    """FALSE POSITIVE GUARD. "or" alone is not an enumeration of acceptable
    answers — "more or less thirty" is one answer."""
    assert detect_multiple_valid_answers(one("Q?", "more or less thirty")) is None


# =========================================================================
# 2. subjective questions
# =========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "Is this response helpful to the user?",
        "Is this helpful?",
        "How helpful was the assistant?",
        "Do you think the tone is right?",
        "Which is better, A or B?",
        "Rate the quality of this summary.",
        "What is the best approach here?",
    ],
)
def test_subjective_phrasing_is_detected(text: str) -> None:
    finding = detect_subjective_question(one(text, "Yes"))
    assert finding is not None, text
    assert finding.confidence is Confidence.MODERATE


@pytest.mark.parametrize(
    "text",
    [
        "Is this valid JSON?",
        "What is the capital of Peru?",
        "Is the total greater than 10?",
        "Explain why the bridge collapsed.",
        "Does this compile?",
    ],
)
def test_objective_questions_are_not_flagged_as_subjective(text: str) -> None:
    """FALSE POSITIVE GUARD. These have definite answers."""
    assert detect_subjective_question(one(text, "x")) is None, text


def test_subjective_finding_says_it_is_legitimate() -> None:
    """Requirement: do NOT label subjective cases as invalid."""
    finding = detect_subjective_question(one("Is this helpful?", "Yes"))
    assert finding is not None
    assert "not a broken case" in finding.recommended_action
    assert "legitimate" in finding.recommended_action


# =========================================================================
# 3. missing grading criteria
# =========================================================================


def test_absent_reference_is_flagged_high() -> None:
    finding = detect_missing_criteria(one("What is the capital of Peru?", None))
    assert finding is not None
    assert finding.confidence is Confidence.HIGH
    assert "cannot be scored automatically" in finding.recommended_action


@pytest.mark.parametrize("placeholder", ["TODO", "tbd", "N/A", "-", "?", "", "FIXME"])
def test_placeholder_references_are_flagged(placeholder: str) -> None:
    finding = detect_missing_criteria(one("Classify this.", placeholder))
    assert finding is not None, placeholder
    assert finding.confidence is Confidence.HIGH


def test_a_real_reference_is_not_a_placeholder() -> None:
    assert detect_missing_criteria(one("Classify this.", "billing")) is None
    # "none of the above" is a real answer that merely starts with "none".
    assert detect_missing_criteria(one("Which applies?", "none of the above")) is None


# =========================================================================
# 4. ambiguous wording
# =========================================================================


def test_dangling_pronoun_is_flagged_low() -> None:
    finding = detect_ambiguous_wording(one("It should be escalated.", "Escalate"))
    assert finding is not None
    assert finding.ambiguity_type is AmbiguityType.AMBIGUOUS_WORDING
    assert finding.confidence is Confidence.LOW


def test_a_pronoun_later_in_the_sentence_is_not_dangling() -> None:
    """FALSE POSITIVE GUARD. Only an OPENING pronoun has nothing to refer to."""
    assert (
        detect_ambiguous_wording(
            one("The ticket says it should be escalated.", "Escalate")
        )
        is None
    )


def test_one_vague_word_is_not_enough() -> None:
    """FALSE POSITIVE GUARD. Two or more, because a single "several" in an
    otherwise precise question is normal English."""
    assert detect_ambiguous_wording(one("Were several files changed?", "Yes")) is None
    finding = detect_ambiguous_wording(
        one("Some users recently saw several appropriate issues.", "Investigate")
    )
    assert finding is not None


# =========================================================================
# 5. reference inconsistency
# =========================================================================


def test_same_input_different_references_is_flagged_for_every_member() -> None:
    """The strongest structural signal: two cases cannot both be right."""
    findings = detect_reference_inconsistency(
        [
            one("What is the refund window?", "14 days", "a"),
            one("What is the refund window?", "30 days", "b"),
        ]
    )
    assert {f.case_id for f in findings} == {"a", "b"}
    assert all(f.confidence is Confidence.HIGH for f in findings)
    assert "consistency test" in findings[0].recommended_action


def test_inconsistency_matching_is_normalised() -> None:
    findings = detect_reference_inconsistency(
        [
            one("What is the REFUND window?", "14 days", "a"),
            one("what is the refund   window?", "30 days", "b"),
        ]
    )
    assert len(findings) == 2


def test_same_input_same_reference_is_not_inconsistent() -> None:
    """FALSE POSITIVE GUARD. That is a duplicate, which is the redundancy
    check's business, not a contradiction."""
    assert (
        detect_reference_inconsistency(
            [one("Q?", "same", "a"), one("Q?", "same", "b")]
        )
        == []
    )


def test_cases_without_a_reference_are_skipped() -> None:
    assert (
        detect_reference_inconsistency([one("Q?", None, "a"), one("Q?", "x", "b")])
        == []
    )


# =========================================================================
# 6. underspecified reference
# =========================================================================


def test_open_ended_prompt_with_a_terse_reference_is_flagged() -> None:
    finding = detect_underspecified_reference(
        one("Explain why the bridge collapsed.", "fatigue")
    )
    assert finding is not None
    assert finding.confidence is Confidence.MODERATE
    assert "rubric" in finding.recommended_action


def test_a_terse_reference_to_a_closed_question_is_fine() -> None:
    """FALSE POSITIVE GUARD. "48" is a complete answer to "what is 12x4"."""
    assert detect_underspecified_reference(one("What is 12 times 4?", "48")) is None


def test_an_open_ended_prompt_with_a_full_reference_is_fine() -> None:
    assert (
        detect_underspecified_reference(
            one(
                "Explain why the bridge collapsed.",
                "Metal fatigue in the eyebar chain, worsened by corrosion and "
                "an unusually heavy load that day",
            )
        )
        is None
    )


def test_list_references_are_left_to_the_other_detector() -> None:
    assert detect_underspecified_reference(one("Explain why.", ["a", "b"])) is None


# =========================================================================
# 7. multiple interpretations
# =========================================================================


def test_a_compound_question_is_flagged() -> None:
    finding = detect_multiple_interpretations(
        one("What is the deadline? And who approves it?", "Friday")
    )
    assert finding is not None
    assert finding.confidence is Confidence.MODERATE
    assert "One reference cannot answer two questions" in finding.recommended_action


def test_a_single_question_is_not_compound() -> None:
    assert detect_multiple_interpretations(one("What is the deadline?", "Friday")) is None


def test_ambiguous_coordination_is_flagged_low() -> None:
    finding = detect_multiple_interpretations(
        one("Should we notify billing and support or legal", "billing and support")
    )
    assert finding is not None
    assert finding.confidence is Confidence.LOW


# =========================================================================
# The check: dataset-level reporting
# =========================================================================


def test_dataset_level_reports_share_and_review_set() -> None:
    cases = clean(10) + [
        one("Classify this.", "TODO", "todo"),
        one("Is this helpful?", "Yes", "subj"),
    ]
    result = GroundTruthCheck().run(make_set(cases))

    assert result.stats["n_cases"] == 12
    assert result.stats["n_potentially_ambiguous"] == 2
    assert result.stats["potentially_ambiguous_share"] == pytest.approx(2 / 12)
    # HIGH-confidence findings need a human; the MODERATE subjective one does not.
    assert result.stats["human_review_case_ids"] == ["todo"]


def test_no_judges_means_no_llm_analysis_at_all() -> None:
    result = GroundTruthCheck().run(make_set(clean(5)))
    assert result.stats["judges_configured"] is False
    assert "n_judges" not in result.stats
    assert "no judges configured" in result.summary


def test_only_high_confidence_findings_warn() -> None:
    """A subjective question is information about how to read the score, not a
    defect. Warning about it is how a check gets ignored."""
    subjective = GroundTruthCheck().run(
        make_set(clean(5) + [one("Is this helpful?", "Yes", "subj")])
    )
    assert subjective.warnings == ()

    placeholder = GroundTruthCheck().run(
        make_set(clean(5) + [one("Classify.", "TODO", "todo")])
    )
    assert any(f.severity is Severity.WARNING for f in placeholder.findings)


def test_every_finding_carries_the_required_fields() -> None:
    """case_id, ambiguity_type, evidence, confidence, recommended_action."""
    result = GroundTruthCheck().run(
        make_set(clean(3) + [one("Classify.", "TODO", "todo")])
    )
    for payload in result.stats["findings"]:
        assert set(payload) == {
            "case_id",
            "ambiguity_type",
            "evidence",
            "confidence",
            "recommended_action",
        }
        assert all(isinstance(v, str) and v for v in payload.values())


def test_nothing_is_called_invalid() -> None:
    """Requirement, asserted rather than trusted.

    Checks for AFFIRMATIVE accusations, not bare words. Two earlier versions of
    this test failed on the tool's own denials -- "not a broken case", "A
    SUBJECTIVE CASE IS NOT AN INVALID CASE" -- because a substring search cannot
    tell an accusation from its negation. The negations are the point, so the
    test now requires them.
    """
    import re

    cases = clean(4) + [
        one("Is this response helpful?", "Yes", "subj"),
        one("Classify.", "TODO", "todo"),
        one("Explain why.", "fatigue", "terse"),
        one("Name a prime.", ["2", "3"], "multi"),
    ]
    result = GroundTruthCheck().run(make_set(cases))
    text = " ".join(f.message for f in result.findings)
    text += " " + " ".join(result.limitations)

    accusations = [
        r"\bis invalid\b",
        r"\bare invalid\b",
        r"\bis a bad case\b",
        r"\bis broken\b",
        r"\bare broken\b",
        r"\bdelete (?:this|these|the)\b",
        r"\bremove (?:this case|these cases)\b",
        r"\bshould be discarded\b",
    ]
    for pattern in accusations:
        assert not re.search(pattern, text, re.IGNORECASE), pattern

    # And the denials must be present, because that is the actual requirement:
    # a subjective case is legitimate, and the tool has to say so.
    lowered = text.lower()
    assert "potentially ambiguous" in lowered
    assert "not a broken case" in lowered
    assert "a subjective case is not an invalid case" in lowered
    assert "legitimate" in lowered


# =========================================================================
# The judge layer: the constraints, enforced
# =========================================================================


def flagging(*ids: str):
    return lambda case: case.id in ids


def test_a_single_judge_is_capped_at_low_and_labelled() -> None:
    """Requirement: never treat one LLM judge as absolute truth."""
    result = GroundTruthCheck(judges={"solo": flagging("ok1")}).run(make_set(clean(5)))

    assert result.stats["single_judge"] is True
    assert result.stats["judge_fleiss_kappa"] is None
    judged = [
        f for f in result.stats["findings"] if f["ambiguity_type"] == "judge_flagged"
    ]
    assert judged and all(f["confidence"] == "low" for f in judged)
    assert "single_judge_uncorroborated" in judged[0]["evidence"]
    assert "add a second judge" in judged[0]["recommended_action"]
    assert any("only ONE judge" in f.message for f in result.findings)


def test_agreeing_judges_reach_moderate_but_never_high() -> None:
    result = GroundTruthCheck(
        judges={"a": flagging("ok1"), "b": flagging("ok1")}
    ).run(make_set(clean(5)))

    judged = [
        f for f in result.stats["findings"] if f["ambiguity_type"] == "judge_flagged"
    ]
    assert [f["confidence"] for f in judged] == ["moderate"]
    # No judge finding is ever HIGH: a model's opinion is not structural evidence.
    assert all(f["confidence"] != "high" for f in judged)


def test_disputed_cases_go_to_review_not_to_findings() -> None:
    """Judge disagreement is evidence the case is hard to adjudicate, not
    evidence about its answer."""
    result = GroundTruthCheck(
        judges={
            "a": flagging("ok1", "ok2", "ok3"),
            "b": flagging("ok1", "ok2"),
            "c": flagging("ok1"),
        }
    ).run(make_set(clean(5)))

    assert result.stats["judge_disagreement_case_ids"] == ["ok2", "ok3"]
    reported = {
        f["case_id"]
        for f in result.stats["findings"]
        if f["ambiguity_type"] == "judge_flagged"
    }
    assert reported == {"ok1"}, "only unanimous cases are reported"
    assert set(result.stats["human_review_case_ids"]) >= {"ok2", "ok3"}


def test_judge_agreement_is_reported_with_a_chance_corrected_figure() -> None:
    result = GroundTruthCheck(
        judges={"a": flagging("ok1"), "b": flagging("ok2")}
    ).run(make_set(clean(6)))

    assert result.stats["judge_raw_agreement"] is not None
    assert result.stats["judge_pairwise_agreement"] is not None
    assert "judge_fleiss_kappa" in result.stats
    message = next(f.message for f in result.findings if "judge agreement" in f.message)
    assert "Fleiss kappa" in message
    assert "consensus, not correctness" in message


def test_kappa_undefined_is_reported_as_undefined() -> None:
    """All judges saying "not ambiguous" every time gives raw agreement 1.0 and
    an undefined kappa. Substituting a number there would invent reliability."""
    result = GroundTruthCheck(
        judges={"a": lambda c: False, "b": lambda c: False}
    ).run(make_set(clean(6)))

    assert result.stats["judge_raw_agreement"] == 1.0
    assert result.stats["judge_fleiss_kappa"] is None
    message = next(f.message for f in result.findings if "judge agreement" in f.message)
    assert "UNDEFINED" in message


def test_per_judge_flag_rates_are_reported() -> None:
    """Leniency differences between judges are visible, so a judge that flags
    everything cannot hide behind an agreement figure."""
    result = GroundTruthCheck(
        judges={"strict": lambda c: False, "loose": lambda c: True}
    ).run(make_set(clean(4)))

    assert result.stats["per_judge_flag_rate"] == {"strict": 0.0, "loose": 1.0}
    assert result.stats["n_judge_disagreements"] == 4


def test_a_failing_judge_does_not_take_down_the_run() -> None:
    """A user-supplied judge is user code. Its exception is recorded and the
    deterministic results survive."""

    def broken(case):
        raise RuntimeError("provider timeout")

    result = GroundTruthCheck(
        judges={"broken": broken, "fine": lambda c: False}
    ).run(make_set(clean(3) + [one("Classify.", "TODO", "todo")]))

    assert result.stats["judge_errors"]
    assert "provider timeout" in result.stats["judge_errors"][0]
    # The deterministic finding is still there.
    assert any(
        f["ambiguity_type"] == "missing_criteria" for f in result.stats["findings"]
    )


def test_limitations_state_that_agreement_is_not_correctness() -> None:
    result = GroundTruthCheck(judges={"a": lambda c: False}).run(make_set(clean(3)))
    joined = " ".join(result.limitations)
    assert "JUDGE AGREEMENT IS NOT CORRECTNESS" in joined
    assert "blind spots" in joined
    assert "single judge" in joined.lower()


def test_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="max_ambiguous_share"):
        GroundTruthCheck(max_ambiguous_share=0)


# =========================================================================
# Regressions found by running against real public datasets
# =========================================================================


def test_punctuation_differences_are_not_contradictions() -> None:
    """REGRESSION, measured on MMLU. Three questions differing only by an
    operator are mathematically distinct:

        Q(sqrt(2) + sqrt(3))  -> degree 1
        Q(sqrt(2), sqrt(3))   -> degree 1
        Q(sqrt(2)*sqrt(3))    -> degree 2

    The aggressive normaliser stripped punctuation, collapsed all three, and
    reported a HIGH-confidence, WARNING-severity claim that the dataset
    contradicts itself. It does not.
    """
    findings = detect_reference_inconsistency(
        [
            one("Find the degree of Q(sqrt(2) + sqrt(3)) over Q.", "1", "a"),
            one("Find the degree of Q(sqrt(2), sqrt(3)) over Q.", "1", "b"),
            one("Find the degree of Q(sqrt(2)*sqrt(3)) over Q.", "2", "c"),
        ]
    )
    assert findings == []


def test_case_and_whitespace_differences_are_still_contradictions() -> None:
    """The other direction: keeping punctuation must not lose the real thing
    this detector is for."""
    findings = detect_reference_inconsistency(
        [
            one("What is the refund window?", "14 days", "a"),
            one("what is the   REFUND window?", "30 days", "b"),
        ]
    )
    assert len(findings) == 2


def test_a_widespread_single_type_is_reported_once() -> None:
    """REGRESSION, measured on HellaSwag: it has no reference answers at all,
    so missing_criteria fired on 100 of 100 cases and produced 100 warnings.
    That is one fact about the file."""
    cases = [EvalCase(id=f"c{i}", input=f"Question {i}?") for i in range(20)]
    result = GroundTruthCheck().run(make_set(cases))

    assert result.stats["n_by_type"]["missing_criteria"] == 20
    collapsed = [f for f in result.findings if "share the SAME issue" in f.message]
    assert len(collapsed) == 1
    assert collapsed[0].case_ids == tuple(f"c{i}" for i in range(20))
    # And no per-case findings for that type.
    assert not [
        f for f in result.findings if "missing_criteria" in f.message and "SAME" not in f.message
    ]


def test_a_minority_type_is_still_reported_per_case() -> None:
    cases = clean(20) + [one("Classify.", "TODO", "todo")]
    result = GroundTruthCheck().run(make_set(cases))
    per_case = [f for f in result.findings if "missing_criteria" in f.message]
    assert len(per_case) == 1
    assert per_case[0].case_ids == ("todo",)
