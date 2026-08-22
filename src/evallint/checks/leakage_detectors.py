"""The individual leakage detectors, separated so each can be tested alone.

Seven deterministic detectors plus one optional semantic one. NO LLM JUDGE is
used anywhere, at any confidence level — a judge would make the result depend on
a model's opinion, and two findings in this project were already retracted for
exactly that reason.

## Confidence is ordinal, not a probability

Each detector reports HIGH / MODERATE / LOW. These are **ordinal labels with a
stated basis, not probabilities**. Attaching a number like 0.87 to "this looks
like leakage" would invent precision that does not exist, which is the failure
this project reports in eval sets.

What the labels mean:

  HIGH      The evidence is hard to explain innocently. An explicit reveal
            phrase followed by the exact reference answer, or the reference
            answer sitting inside the instruction preamble, or the evaluation
            target appearing in the case's own few-shot demonstrations.
  MODERATE  The observation is certain but its interpretation is not. The
            reference answer appears somewhere in the input — which is exactly
            what extractive QA looks like — or in a metadata field evallint
            cannot know is shown to the model.
  LOW       A statistical signal with known poor separation. Reported only
            when explicitly enabled.

Every finding carries the raw evidence, so a reader can overrule the label.

## Why token overlap is off by default

Measured on GSM8K before this module existed: overlap between the reference
answer and the question has a median of 0.57 and a MAXIMUM OF 1.00 with no
leakage anywhere, because GSM8K reference answers are worked solutions that
restate the question. A planted leak also scores 1.00. No threshold separates
them, so the overlap detector is opt-in and labelled LOW.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..schema import EvalCase

__all__ = [
    "DEFAULT_MIN_ANSWER_CHARS",
    "Confidence",
    "LeakageType",
    "PotentialLeak",
    "detect_answer_in_instructions",
    "detect_expected_in_input",
    "detect_explicit_reveal",
    "detect_fewshot_target",
    "detect_high_overlap",
    "detect_label_in_input",
    "detect_metadata_leak",
    "normalise",
]

DEFAULT_MIN_ANSWER_CHARS = 12


class LeakageType(str, Enum):
    EXPECTED_IN_INPUT = "expected_in_input"
    LABEL_IN_INPUT = "label_in_input"
    EXPLICIT_REVEAL = "explicit_reveal"
    ANSWER_IN_INSTRUCTIONS = "answer_in_instructions"
    METADATA_LEAK = "metadata_leak"
    FEWSHOT_TARGET = "fewshot_target"
    HIGH_OVERLAP = "high_overlap"
    SEMANTIC_ECHO = "semantic_echo"


class Confidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class PotentialLeak:
    """One potential leak. Never a claim that the case is invalid.

    `evidence` is the actual matched text, truncated for display but taken
    verbatim from the data, so a reader can check the finding rather than
    trusting it.
    """

    case_id: str
    type: LeakageType
    confidence: Confidence
    evidence: str
    explanation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "type": self.type.value,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
            "explanation": self.explanation,
        }


_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def normalise(text: object) -> str:
    """Case-fold, collapse whitespace, drop punctuation."""
    return " ".join(_WORD.findall(str(text).lower()))


def _clip(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- 1. expected answer appears in the input ---------------------------------


def detect_expected_in_input(
    case: EvalCase, min_answer_chars: int = DEFAULT_MIN_ANSWER_CHARS
) -> PotentialLeak | None:
    """The reference answer appears verbatim in the input.

    MODERATE, never HIGH, and that is deliberate: this is exactly what a
    legitimate extractive-QA or reading-comprehension case looks like. The
    observation is certain; the interpretation is not.
    """
    if case.expected is None:
        return None
    answer = normalise(case.expected)
    if len(answer) < min_answer_chars or not answer:
        return None
    if answer not in normalise(case.input):
        return None
    return PotentialLeak(
        case_id=case.id,
        type=LeakageType.EXPECTED_IN_INPUT,
        confidence=Confidence.MODERATE,
        evidence=f"expected ({len(answer)} chars) found in input: {_clip(answer)}",
        explanation=(
            "the reference answer appears verbatim in the input, so this case "
            "can be answered by copying rather than by the capability under "
            "test. If the answer is MEANT to be present — extractive QA, span "
            "selection, reading comprehension — this is the task working, not "
            "a leak"
        ),
    )


# --- 2. label appears in the input -------------------------------------------


def detect_label_in_input(case: EvalCase) -> PotentialLeak | None:
    """The class label appears as a whole word in the input.

    MODERATE at most, because a label naturally occurs in text about its own
    topic: a case labelled 'billing' that mentions billing is normal, not a
    leak. The check that makes this usable lives in the parent check, which
    compares the RATE across the set — a label present in most inputs is
    topical, and only reported once as such.
    """
    if case.label is None:
        return None
    label = normalise(case.label)
    if not label:
        return None
    # Whole-word match. Substring matching would fire on 'bill' inside
    # 'billboard' and make the detector useless.
    pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)")
    if not pattern.search(normalise(case.input)):
        return None
    return PotentialLeak(
        case_id=case.id,
        type=LeakageType.LABEL_IN_INPUT,
        confidence=Confidence.MODERATE,
        evidence=f"label {case.label!r} appears as a word in the input",
        explanation=(
            "the class label appears in the input text. This is often "
            "legitimate — a question about billing naturally contains the word "
            "billing — so treat it as a leak only if the label was added by an "
            "annotator rather than being part of the original text"
        ),
    )


# --- 3. the answer is revealed by explicit wording ---------------------------

#: Phrases that announce an answer. Matched only when the reference answer
#: actually follows, because these phrases occur innocently on their own
#: ("what if the answer is negative?").
_REVEAL_PHRASES = (
    r"the (?:correct )?answer is",
    r"the answer should be",
    r"answer\s*[:=]",
    r"correct answer\s*[:=]",
    r"the (?:correct )?label is",
    r"label\s*[:=]",
    r"solution\s*[:=]",
    r"the solution is",
    r"hint\s*[:=]",
    r"note\s*[:=]\s*the answer",
    r"\(\s*correct\s*\)",
    r"ground truth\s*[:=]",
    r"expected(?: output| answer)?\s*[:=]",
)
_REVEAL = re.compile("|".join(_REVEAL_PHRASES), re.IGNORECASE)

#: How far after a reveal phrase the answer may appear and still count as
#: revealed by it. 80 characters is about one clause: far enough for "the
#: correct answer is, of course, Paris", short enough that an unrelated later
#: mention does not get attributed to the phrase.
REVEAL_WINDOW_CHARS = 80


def detect_explicit_reveal(case: EvalCase) -> PotentialLeak | None:
    """An explicit reveal phrase followed by the reference answer.

    HIGH confidence. A phrase whose job is to announce the answer, immediately
    followed by the actual answer, is difficult to explain as anything but
    leakage. The phrase alone is NOT reported: "what if the answer is negative"
    is a perfectly good question.
    """
    if case.expected is None:
        return None
    answer = normalise(case.expected)
    if not answer:
        return None
    raw = str(case.input)
    for match in _REVEAL.finditer(raw):
        window = raw[match.end() : match.end() + REVEAL_WINDOW_CHARS]
        if answer and answer in normalise(window):
            return PotentialLeak(
                case_id=case.id,
                type=LeakageType.EXPLICIT_REVEAL,
                confidence=Confidence.HIGH,
                evidence=(
                    f"{match.group(0)!r} followed by the reference answer: "
                    f"{_clip(match.group(0) + window)}"
                ),
                explanation=(
                    "the input contains a phrase that announces the answer, "
                    "immediately followed by the reference answer itself. A "
                    "model does not need the capability under test to answer "
                    "this — it needs to read"
                ),
            )
    return None


# --- 4. the answer sits in the instruction preamble -------------------------

#: Markers that separate an instruction preamble from the actual item. Ordered
#: so the earliest match wins.
_ITEM_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:question|input|prompt|q|task|item|text)\s*[:.\)]",
    re.IGNORECASE,
)


def detect_answer_in_instructions(case: EvalCase) -> PotentialLeak | None:
    """The reference answer appears in the instruction preamble.

    HIGH confidence. An instruction block is addressed to the model about HOW to
    do the task; the answer to this particular item has no business there, and
    unlike the body of the input it is not plausibly the material to reason
    over.
    """
    if case.expected is None:
        return None
    answer = normalise(case.expected)
    if not answer:
        return None
    raw = str(case.input)
    marker = _ITEM_MARKERS.search(raw)
    if marker is None:
        return None
    preamble = raw[: marker.start()]
    if len(normalise(preamble)) < 10 or answer not in normalise(preamble):
        return None
    return PotentialLeak(
        case_id=case.id,
        type=LeakageType.ANSWER_IN_INSTRUCTIONS,
        confidence=Confidence.HIGH,
        evidence=f"answer found in the instruction preamble: {_clip(preamble)}",
        explanation=(
            "the reference answer appears BEFORE the item marker, i.e. in the "
            "instructions rather than in the material to reason over. "
            "Instructions describe how to do the task; this one contains the "
            "answer to it"
        ),
    )


# --- 5. metadata leakage -----------------------------------------------------

#: Substrings marking a metadata key as answer bookkeeping. A copy of the
#: reference answer in such a field is organisation, not leakage.
#:
#: SUBSTRINGS, not exact names, and that is a measured correction. An exact-name
#: list produced 100 false positives on 100 TruthfulQA cases, because its
#: `correct_answers` field naturally contains `best_answer` and the key name did
#: not match `answer` exactly. A hundred spurious findings on a real public
#: dataset is how a check gets switched off.
#:
#: The cost of matching loosely: a genuine leak in a field called something like
#: `answer_context` will be missed. That is the right way round — a missed leak
#: is a gap, a hundred false positives is a broken check.
_ANSWER_KEY_MARKERS = (
    "answer",
    "expected",
    "gold",
    "ground_truth",
    "groundtruth",
    "label",
    "reference",
    "solution",
    "target",
    "truth",
)


def _is_answer_field(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _ANSWER_KEY_MARKERS)


def detect_metadata_leak(
    case: EvalCase, min_answer_chars: int = DEFAULT_MIN_ANSWER_CHARS
) -> PotentialLeak | None:
    """The reference answer appears in a metadata field that is not about answers.

    MODERATE, and the reason is a genuine limit: evallint cannot know whether
    your harness serialises metadata into the prompt. If it does not, this is
    harmless. If it does, it is a leak. Only the user knows which.
    """
    if case.expected is None or not case.metadata:
        return None
    answer = normalise(case.expected)
    if len(answer) < min_answer_chars or not answer:
        return None
    for key in sorted(case.metadata):
        if _is_answer_field(key):
            continue
        value = normalise(case.metadata[key])
        if answer and answer in value:
            return PotentialLeak(
                case_id=case.id,
                type=LeakageType.METADATA_LEAK,
                confidence=Confidence.MODERATE,
                explanation=(
                    "the reference answer appears in a metadata field that is "
                    "not an answer field. Whether this leaks depends on "
                    "something evallint cannot see: if your harness puts "
                    "metadata into the prompt it is a leak, and if it does not "
                    "it is harmless"
                ),
                evidence=f"metadata[{key!r}] contains the answer: {_clip(value)}",
            )
    return None


# --- 6. few-shot demonstrations contain the evaluation target ---------------

#: Two or more of these pairs indicate a few-shot block.
_DEMO_PAIR = re.compile(
    r"(?:^|\n)\s*(?:q|question|input|user)\s*[:.\)]"
    r".*?"
    r"(?:^|\n)\s*(?:a|answer|output|assistant)\s*[:.\)]",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_ITEM = re.compile(
    r"(?:^|\n)\s*(?:q|question|input|user)\s*[:.\)]", re.IGNORECASE
)

#: Fewer demonstration pairs than this is not a few-shot block, it is one item
#: with a labelled question and answer.
MIN_DEMO_PAIRS = 2


def detect_fewshot_target(case: EvalCase) -> PotentialLeak | None:
    """The evaluation target appears inside the case's own few-shot examples.

    HIGH confidence, and the least ambiguous detector here. Demonstrations
    exist to show the format; containing the answer to the very item being
    asked makes the item free. This is almost always a template bug.
    """
    if case.expected is None:
        return None
    answer = normalise(case.expected)
    if not answer:
        return None
    raw = str(case.input)
    if len(_DEMO_PAIR.findall(raw)) < MIN_DEMO_PAIRS:
        return None
    starts = [m.start() for m in _FINAL_ITEM.finditer(raw)]
    if len(starts) < MIN_DEMO_PAIRS + 1:
        return None
    # Everything before the LAST question marker is demonstration material.
    demos = raw[: starts[-1]]
    if answer not in normalise(demos):
        return None
    return PotentialLeak(
        case_id=case.id,
        type=LeakageType.FEWSHOT_TARGET,
        confidence=Confidence.HIGH,
        evidence=f"answer appears in the demonstrations: {_clip(demos)}",
        explanation=(
            "this input contains few-shot demonstrations, and the answer to the "
            "final question already appears among them. The demonstrations "
            "give the answer away, so the item measures reading rather than the "
            "capability under test"
        ),
    )


# --- 7. suspicious token overlap (opt-in) -----------------------------------

#: Overlap at or above which the opt-in detector fires. Deliberately very high
#: and still not reliable: on GSM8K, legitimate reference answers reach 1.00
#: because they restate the question.
DEFAULT_OVERLAP_THRESHOLD = 0.9


def detect_high_overlap(
    case: EvalCase, threshold: float = DEFAULT_OVERLAP_THRESHOLD
) -> PotentialLeak | None:
    """Share of the reference answer's words that also appear in the input.

    LOW confidence, off by default, and the docstring is the justification.
    Measured on GSM8K: median overlap 0.57, MAXIMUM 1.00, with no leakage
    present anywhere in the set, because reference answers there are worked
    solutions that restate the question. A planted leak also scores 1.00. No
    threshold separates the two, so this detector cannot support a conclusion
    and is enabled only on request.
    """
    if case.expected is None:
        return None
    answer_words = set(normalise(case.expected).split())
    if not answer_words:
        return None
    input_words = set(normalise(case.input).split())
    overlap = len(answer_words & input_words) / len(answer_words)
    if overlap < threshold:
        return None
    return PotentialLeak(
        case_id=case.id,
        type=LeakageType.HIGH_OVERLAP,
        confidence=Confidence.LOW,
        evidence=(
            f"{overlap:.0%} of the answer's words appear in the input "
            f"({len(answer_words & input_words)} of {len(answer_words)})"
        ),
        explanation=(
            "a high share of the answer's vocabulary also appears in the input. "
            "This signal separates leaks from non-leaks POORLY: on GSM8K it "
            "reaches 100% with no leakage at all, because the reference answer "
            "restates the question. Treat it as a prompt to read the case, "
            "nothing more"
        ),
    )
