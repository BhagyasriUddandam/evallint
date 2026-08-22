"""Deterministic detectors for under-specified ground truth.

Seven detection types, all deterministic. No model is required for any of them.
An optional LLM-assisted analyser lives in `ground_truth.py` and is off unless
judges are configured explicitly.

## A subjective question is not an invalid question

"Is this response helpful?" has no single correct answer, and an eval full of
such questions is not broken — it is measuring preference, which is a legitimate
thing to measure. What the finding says is that **scoring will depend on the
grader**, so the eval needs a rubric or must accept grader-dependent numbers.
Nothing here calls a case invalid, and a test asserts that.

## Confidence is ordinal

HIGH      Structural and hard to explain otherwise: two near-identical inputs
          with different reference answers, or a reference that is a placeholder.
MODERATE  A real linguistic signal whose interpretation depends on the domain:
          subjective phrasing, a compound question, a two-word reference for an
          open-ended prompt.
LOW       Weak lexical cues — vague quantifiers, an unresolved leading pronoun.
          Reported, but a reader should expect to overrule some of them.

Every finding carries a RECOMMENDED ACTION, and none of them is "delete".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ..schema import EvalCase

__all__ = [
    "AmbiguityFinding",
    "AmbiguityType",
    "Confidence",
    "detect_ambiguous_wording",
    "detect_missing_criteria",
    "detect_multiple_interpretations",
    "detect_multiple_valid_answers",
    "detect_reference_inconsistency",
    "detect_subjective_question",
    "detect_underspecified_reference",
]


class AmbiguityType(str, Enum):
    MULTIPLE_VALID_ANSWERS = "multiple_valid_answers"
    SUBJECTIVE_QUESTION = "subjective_question"
    MISSING_CRITERIA = "missing_criteria"
    AMBIGUOUS_WORDING = "ambiguous_wording"
    REFERENCE_INCONSISTENCY = "reference_inconsistency"
    UNDERSPECIFIED_REFERENCE = "underspecified_reference"
    MULTIPLE_INTERPRETATIONS = "multiple_interpretations"
    JUDGE_FLAGGED = "judge_flagged"


class Confidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class AmbiguityFinding:
    """One case where the reference may not be well enough defined.

    Never a claim that the case is invalid. `recommended_action` is always
    something to add or reconcile, never something to remove.
    """

    case_id: str
    ambiguity_type: AmbiguityType
    confidence: Confidence
    evidence: str
    recommended_action: str

    def as_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "ambiguity_type": self.ambiguity_type.value,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
        }


_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _normalise(text: object) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def _conservative(text: object) -> str:
    """Lowercase and collapse whitespace. PUNCTUATION IS KEPT.

    Used only for reference-inconsistency comparison, where the aggressive
    normaliser above is actively wrong. Measured on MMLU:

        Q(sqrt(2) + sqrt(3))  -> degree 1
        Q(sqrt(2), sqrt(3))   -> degree 1
        Q(sqrt(2)*sqrt(3))    -> degree 2

    Those are three DIFFERENT field extensions, and stripping punctuation
    collapses all three to "q sqrt 2 sqrt 3" -- producing a confident,
    high-severity claim that the dataset contradicts itself when it does not.
    Punctuation carries meaning in mathematics, code and logic, so the detector
    that accuses a dataset of contradicting itself keeps it.

    The cost: "What is 2+2?" and "What is 2 + 2" are no longer compared. A
    missed inconsistency is a gap; a fabricated one is a wrong answer stated
    confidently.
    """
    return " ".join(str(text).lower().split())


def _clip(text: object, limit: int = 100) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# --- 1. multiple valid answers ----------------------------------------------

#: Separators that indicate the reference itself lists alternatives.
_ALTERNATIVES = re.compile(
    r"\b(?:either|any of|one of|or equivalently|acceptable answers?|"
    r"also acceptable|or\s+(?:alternatively|similar))\b",
    re.IGNORECASE,
)


def detect_multiple_valid_answers(case: EvalCase) -> AmbiguityFinding | None:
    """The reference answer is itself a set of alternatives.

    A list-valued `expected`, or prose that enumerates acceptable answers. This
    is not a defect — it is honest about a question having several right answers
    — but a scorer comparing against it needs to know that, and a string
    comparison will mark two of three correct answers wrong.
    """
    if case.expected is None:
        return None

    if isinstance(case.expected, (list, tuple, set)):
        if len(case.expected) < 2:
            return None
        return AmbiguityFinding(
            case_id=case.id,
            ambiguity_type=AmbiguityType.MULTIPLE_VALID_ANSWERS,
            confidence=Confidence.HIGH,
            evidence=f"expected holds {len(case.expected)} alternatives: "
            f"{_clip(list(case.expected))}",
            recommended_action=(
                "confirm your scorer accepts ANY of these rather than comparing "
                "against one; a plain string comparison will mark the other "
                "correct answers wrong"
            ),
        )

    match = _ALTERNATIVES.search(str(case.expected))
    if match is None:
        return None
    return AmbiguityFinding(
        case_id=case.id,
        ambiguity_type=AmbiguityType.MULTIPLE_VALID_ANSWERS,
        confidence=Confidence.MODERATE,
        evidence=f"reference enumerates alternatives ({match.group(0)!r}): "
        f"{_clip(case.expected)}",
        recommended_action=(
            "split the alternatives into a list your scorer can match against, "
            "or state in the rubric which form is required"
        ),
    )


# --- 2. subjective questions ------------------------------------------------

#: Phrasings that ask for a judgement rather than a fact. Kept narrow on
#: purpose: broad matching on words like "good" fires on "is this good syntax",
#: which has a definite answer.
_SUBJECTIVE = re.compile(
    r"\b(?:"
    r"do you (?:think|feel|prefer)"
    r"|how (?:helpful|useful|good|well|polite|friendly)"
    # A bounded gap, so "is this response helpful" matches as well as "is this
    # helpful". An unbounded gap would join a subject at the start of a
    # paragraph to an adjective three sentences later.
    r"|is\s+(?:this|that|it|the)\b[^.?!]{0,40}?"
    r"\b(?:helpful|useful|good|better|appropriate|polite|reasonable)\b"
    r"|which (?:is|one is) better"
    r"|rate the"
    r"|in your opinion"
    r"|would you (?:say|prefer)"
    r"|best (?:way|approach|practice)"
    r")\b",
    re.IGNORECASE,
)


def detect_subjective_question(case: EvalCase) -> AmbiguityFinding | None:
    """The question asks for a judgement, so its answer depends on the grader.

    NOT a defect. Preference and helpfulness are legitimate things to evaluate.
    The finding says the score will be grader-dependent, which is information a
    reader needs when interpreting the number.
    """
    match = _SUBJECTIVE.search(str(case.input))
    if match is None:
        return None
    return AmbiguityFinding(
        case_id=case.id,
        ambiguity_type=AmbiguityType.SUBJECTIVE_QUESTION,
        confidence=Confidence.MODERATE,
        evidence=f"subjective phrasing {match.group(0)!r} in: {_clip(case.input)}",
        recommended_action=(
            "this is a legitimate thing to evaluate, not a broken case. Add a "
            "rubric stating what counts as correct, or accept that the score "
            "for this case is grader-dependent and report it as such"
        ),
    )


# --- 3. missing grading criteria -------------------------------------------

#: Values that stand in for an answer nobody has written yet.
_PLACEHOLDERS = frozenset(
    {"", "-", "--", "?", "??", "n a", "na", "tbd", "todo", "fixme", "none",
     "null", "nil", "xxx", "pending", "unknown"}
)


def detect_missing_criteria(case: EvalCase) -> AmbiguityFinding | None:
    """The reference is absent or a placeholder, so nothing defines correctness.

    HIGH confidence: there is no interpretation under which "TODO" is a
    reference answer.
    """
    if case.expected is None:
        return AmbiguityFinding(
            case_id=case.id,
            ambiguity_type=AmbiguityType.MISSING_CRITERIA,
            confidence=Confidence.HIGH,
            evidence="no expected value at all",
            recommended_action=(
                "add a reference answer or a rubric. Without one this case "
                "cannot be scored automatically, and any aggregate that "
                "includes it is counting an unscored case"
            ),
        )
    normalised = _normalise(case.expected)
    if normalised not in _PLACEHOLDERS:
        return None
    return AmbiguityFinding(
        case_id=case.id,
        ambiguity_type=AmbiguityType.MISSING_CRITERIA,
        confidence=Confidence.HIGH,
        evidence=f"reference is a placeholder: {str(case.expected)!r}",
        recommended_action=(
            "replace the placeholder with a real reference answer or a rubric"
        ),
    )


# --- 4. ambiguous wording ---------------------------------------------------

_VAGUE = re.compile(
    r"\b(?:some|several|a few|many|recently|soon|later|nearby|appropriate|"
    r"reasonable|suitable|proper|adequate|etc)\b",
    re.IGNORECASE,
)
#: A pronoun opening the input, with nothing before it to refer to.
_DANGLING_PRONOUN = re.compile(
    r"^\s*(?:it|they|them|this|that|these|those|he|she)\b", re.IGNORECASE
)


def detect_ambiguous_wording(case: EvalCase) -> AmbiguityFinding | None:
    """Vague quantifiers, or a pronoun with no antecedent in the input.

    LOW confidence, and deliberately so. These are lexical cues, not proof: a
    question containing "several" may be perfectly precise about what it wants.
    Expect to overrule some of these.
    """
    text = str(case.input)
    dangling = _DANGLING_PRONOUN.match(text)
    if dangling:
        return AmbiguityFinding(
            case_id=case.id,
            ambiguity_type=AmbiguityType.AMBIGUOUS_WORDING,
            confidence=Confidence.LOW,
            evidence=f"input opens with the pronoun {dangling.group(0).strip()!r} "
            f"and nothing precedes it: {_clip(text)}",
            recommended_action=(
                "name the thing the pronoun refers to, or confirm the context "
                "is supplied elsewhere in your harness"
            ),
        )
    vague = _VAGUE.findall(text)
    if len(vague) < 2:
        return None
    return AmbiguityFinding(
        case_id=case.id,
        ambiguity_type=AmbiguityType.AMBIGUOUS_WORDING,
        confidence=Confidence.LOW,
        evidence=f"vague terms {sorted(set(w.lower() for w in vague))}: {_clip(text)}",
        recommended_action=(
            "quantify the vague terms, or state in a rubric what range of "
            "answers counts as correct"
        ),
    )


# --- 5. reference inconsistency (set level) --------------------------------


def detect_reference_inconsistency(
    cases: Sequence[EvalCase],
) -> list[AmbiguityFinding]:
    """Near-identical inputs carrying DIFFERENT reference answers.

    The strongest ground-truth signal available without a model, and it is
    structural: two cases cannot both be right. Either one reference is wrong,
    or the pair is a deliberate consistency test — and until the user says
    which, the set cannot be scored coherently.

    Comparison collapses case and whitespace but KEEPS PUNCTUATION, because
    stripping it produced three false contradictions on MMLU where the inputs
    differed only by "+" versus "," versus "*" -- mathematically distinct
    questions. See _conservative.
    """
    groups: dict[str, list[EvalCase]] = {}
    for case in cases:
        if case.expected is None:
            continue
        # Conservative normalisation: see _conservative for the MMLU false
        # positives that made this necessary.
        groups.setdefault(_conservative(case.input), []).append(case)

    findings: list[AmbiguityFinding] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        answers = {_conservative(c.expected) for c in members}
        if len(answers) < 2:
            continue
        shown = sorted(answers)[:3]
        for case in members:
            findings.append(
                AmbiguityFinding(
                    case_id=case.id,
                    ambiguity_type=AmbiguityType.REFERENCE_INCONSISTENCY,
                    confidence=Confidence.HIGH,
                    evidence=(
                        f"{len(members)} cases share this input but carry "
                        f"{len(answers)} different references, e.g. {shown}"
                    ),
                    recommended_action=(
                        "reconcile the references, or mark the group as an "
                        "intentional consistency test so the disagreement is "
                        "recorded rather than looking like an error"
                    ),
                )
            )
    return findings


# --- 6. underspecified reference ------------------------------------------

#: Verbs that ask for an extended answer.
_OPEN_ENDED = re.compile(
    r"\b(?:explain|describe|discuss|elaborate|justify|compare|analyse|analyze|"
    r"summarise|summarize|why|how come|walk me through)\b",
    re.IGNORECASE,
)

#: An open-ended prompt answered in this many words or fewer cannot convey what
#: a good answer looks like. Three is deliberately small: it admits "because it
#: rusts" as a legitimate terse answer while catching bare labels.
MAX_TERSE_REFERENCE_WORDS = 3


def detect_underspecified_reference(case: EvalCase) -> AmbiguityFinding | None:
    """An open-ended prompt with a reference too thin to define correctness."""
    if case.expected is None:
        return None
    if isinstance(case.expected, (list, tuple, set)):
        return None
    match = _OPEN_ENDED.search(str(case.input))
    if match is None:
        return None
    words = _normalise(case.expected).split()
    if not words or len(words) > MAX_TERSE_REFERENCE_WORDS:
        return None
    return AmbiguityFinding(
        case_id=case.id,
        ambiguity_type=AmbiguityType.UNDERSPECIFIED_REFERENCE,
        confidence=Confidence.MODERATE,
        evidence=(
            f"open-ended prompt ({match.group(0)!r}) with a {len(words)}-word "
            f"reference: {str(case.expected)!r}"
        ),
        recommended_action=(
            "expand the reference, or add a rubric listing the points a correct "
            "answer must contain. A grader cannot tell a good long answer from "
            "a bad one against a reference this short"
        ),
    )


# --- 7. multiple plausible interpretations --------------------------------

_COORDINATION = re.compile(r"\band\s*/\s*or\b|\bor\b.*\band\b|\band\b.*\bor\b", re.IGNORECASE)


def detect_multiple_interpretations(case: EvalCase) -> AmbiguityFinding | None:
    """A compound question, or coordination that can be read more than one way.

    Two questions in one input have two answers, and a single reference can only
    hold one of them. "A and B or C" is genuinely ambiguous about grouping.
    """
    text = str(case.input)
    questions = text.count("?")
    if questions >= 2:
        return AmbiguityFinding(
            case_id=case.id,
            ambiguity_type=AmbiguityType.MULTIPLE_INTERPRETATIONS,
            confidence=Confidence.MODERATE,
            evidence=f"{questions} questions in one input: {_clip(text)}",
            recommended_action=(
                "split into separate cases, or state in the reference which "
                "question is being scored. One reference cannot answer two "
                "questions"
            ),
        )
    match = _COORDINATION.search(text)
    if match is None:
        return None
    return AmbiguityFinding(
        case_id=case.id,
        ambiguity_type=AmbiguityType.MULTIPLE_INTERPRETATIONS,
        confidence=Confidence.LOW,
        evidence=f"ambiguous coordination {match.group(0)!r}: {_clip(text)}",
        recommended_action=(
            "punctuate or rephrase the coordination so the grouping is "
            "unambiguous, or record both readings as acceptable"
        ),
    )
