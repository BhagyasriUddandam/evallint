"""Labelled fixtures with known problems, for measuring evallint's detectors.

## The split rule, which is the point of this file

Every generator takes a :class:`Split`. ``DEV`` is for looking at, tuning
against and debugging. ``TEST`` is for the numbers that get reported, and
**nothing may be tuned on it**.

The two splits differ in *both* seed and surface content: DEV draws from a
customer-support vocabulary, TEST from a healthcare-admin and transit
vocabulary, with different paraphrase templates. Reseeding one generator would
not be a held-out set for anything except the exact strings; different content
pools make it a genuinely different sample.

**This is still weaker than an independent benchmark**, and the limitation is
real: both splits come from the same author and the same generator design, so a
detector could pass both while failing on data neither anticipates. That is why
`benchmarks/README.md` records the false-positive counts measured on real public
datasets separately — those are the only numbers here not written by the person
who wrote the detectors.

## What "labelled" means, per category

The honest answer differs by category, and this matters more than the metrics:

* **Objective by construction.** Exact duplicates (byte-identical input),
  ceiling/floor/separating cases (the scorer is a fixture, so which models pass
  is a fact), statistical noise (models are constructed equal or unequal).
  A label error here would be a bug in this file, not a judgement call.
* **Author judgement.** Semantic duplicates, ambiguous references. Two
  paraphrases being "the same scenario" is my opinion. A different annotator
  would draw some of these differently, and the reported precision is therefore
  precision *against my labels*, not against truth.
* **Definitional.** Leakage: the label is "the reference answer appears verbatim
  in the input", which is exactly what the detector looks for. So high scores
  here measure implementation correctness, NOT whether verbatim containment is a
  good proxy for leakage. It is not, and the detector says so itself.

Negatives are deliberately adversarial. A fixture full of obvious positives
measures nothing about precision, so each one carries near-misses designed to
trip the detector: paraphrases that are genuinely different questions, answers
mentioned by topic but not verbatim, questions that read as subjective but have
one defensible answer.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from evallint.schema import EvalCase, EvalSet

__all__ = [
    "Fixture",
    "Split",
    "ambiguity_fixture",
    "discrimination_fixture",
    "exact_duplicate_fixture",
    "imbalance_fixtures",
    "judge_fixture",
    "leakage_fixture",
    "noise_trials",
    "semantic_duplicate_fixture",
]


class Split(StrEnum):
    """DEV may be tuned on. TEST may not, and carries the reported numbers."""

    DEV = "dev"
    TEST = "test"


#: Base seeds. Different per split so no trial is shared, and fixed so every
#: run of the benchmark measures the same data.
_SEED = {Split.DEV: 20260101, Split.TEST: 20260822}


@dataclass(frozen=True, slots=True)
class Fixture:
    """One labelled dataset.

    ``positives`` and ``positive_pairs`` are the ground truth. Everything in the
    set that is not a positive is a negative, so precision is measurable rather
    than assumed.
    """

    name: str
    split: Split
    eval_set: EvalSet
    label_basis: str
    positives: frozenset[str] = frozenset()
    positive_pairs: frozenset[frozenset[str]] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cases(self) -> int:
        return len(self.eval_set)

    @property
    def negatives(self) -> frozenset[str]:
        return frozenset(c.id for c in self.eval_set) - self.positives


# --------------------------------------------------------------------------
# Content pools. Disjoint across splits, so TEST is not DEV with a new seed.
# --------------------------------------------------------------------------

_POOL = {
    Split.DEV: {
        "topics": [
            "a duplicate charge on the March invoice",
            "a parcel that never reached the delivery address",
            "resetting a password without access to the old email",
            "cancelling an annual subscription before renewal",
            "updating the billing address on a saved card",
            "a discount code that the checkout page rejected",
            "transferring a licence to a new employee",
            "downloading an invoice for the finance team",
        ],
        "paraphrases": [
            "I need help with {topic}.",
            "Could someone look into {topic}?",
            "Having trouble with {topic} — what should I do?",
            "What is the process for {topic}?",
        ],
        "distinct": [
            "How long does standard delivery take to rural addresses?",
            "Which currencies can I be invoiced in?",
            "Does the enterprise plan include phone support?",
            "Where do I find the data processing agreement?",
            "Can I pause my subscription for one month?",
            "What happens to my data if I close the account?",
            "Is there an audit log for administrator actions?",
            "How many seats does the team plan include?",
        ],
    },
    Split.TEST: {
        "topics": [
            "rescheduling a physiotherapy appointment after a cancellation",
            "a library book renewed twice but still marked overdue",
            "a monthly transit pass that stopped working at the gate",
            "registering a new dependant on a health record",
            "correcting a misspelled name on a vaccination card",
            "a reserved study room that was double-booked",
            "transferring a season ticket to a different route",
            "requesting printed copies of imaging results",
        ],
        "paraphrases": [
            "I am trying to sort out {topic}.",
            "Who handles {topic}?",
            "Something went wrong with {topic} — how is it fixed?",
            "What are the steps for {topic}?",
        ],
        "distinct": [
            "How far in advance can a specialist appointment be booked?",
            "What identification is needed for a replacement pass?",
            "Are interlibrary loans available to visiting members?",
            "How long are radiology images retained?",
            "Can a carer collect a prescription on someone's behalf?",
            "Which stations have step-free access at both ends?",
            "What is the fine for a book returned two weeks late?",
            "Is there a concessionary fare for part-time students?",
        ],
    },
}


def _rng(split: Split, salt: int) -> random.Random:
    """A seeded generator per (split, fixture), so adding a fixture cannot
    change the data of an existing one."""
    return random.Random(_SEED[split] + salt)


# --------------------------------------------------------------------------
# 1. Exact duplicates — objective label
# --------------------------------------------------------------------------


def exact_duplicate_fixture(split: Split) -> Fixture:
    """Byte-identical inputs, plus near-misses that must NOT be called exact.

    The negatives are the interesting part: a trailing full stop, a changed
    capital, and one extra space. Those are *normalised* duplicates, not exact
    ones, and a detector that calls them exact has lost the distinction between
    "the same bytes" and "the same after cleaning".
    """
    pool = _POOL[split]
    cases: list[EvalCase] = []
    pairs: set[frozenset[str]] = set()

    for i in range(4):
        text = pool["paraphrases"][0].format(topic=pool["topics"][i])
        answer = f"Follow procedure {i} in the handbook."
        cases.append(EvalCase(id=f"exact_{i}a", input=text, expected=answer))
        cases.append(EvalCase(id=f"exact_{i}b", input=text, expected=answer))
        pairs.add(frozenset({f"exact_{i}a", f"exact_{i}b"}))

    # Near-misses. Same meaning, different bytes.
    base = pool["paraphrases"][1].format(topic=pool["topics"][5])
    cases.append(EvalCase(id="nearmiss_plain", input=base, expected="See policy 9."))
    cases.append(EvalCase(id="nearmiss_period", input=base + ".", expected="See policy 9."))
    cases.append(EvalCase(id="nearmiss_case", input=base.upper(), expected="See policy 9."))
    cases.append(EvalCase(id="nearmiss_space", input=base.replace(" ", "  ", 1),
                          expected="See policy 9."))

    for i, text in enumerate(pool["distinct"]):
        cases.append(EvalCase(id=f"unique_{i}", input=text, expected=f"answer {i}"))

    return Fixture(
        name="exact_duplicates",
        split=split,
        eval_set=EvalSet(cases=tuple(cases), source=f"exact_{split}.jsonl"),
        label_basis="objective: byte-identical input and expected",
        positives=frozenset(cid for pair in pairs for cid in pair),
        positive_pairs=frozenset(pairs),
        metadata={
            "n_planted_pairs": len(pairs),
            "near_misses": [
                "nearmiss_plain", "nearmiss_period", "nearmiss_case",
                "nearmiss_space",
            ],
            "near_miss_note": "these four are NORMALISED duplicates of each "
            "other, not exact ones. A detector calling them exact has lost the "
            "distinction it exists to make.",
        },
    )


# --------------------------------------------------------------------------
# 2. Semantic duplicates — author judgement, and labelled as such
# --------------------------------------------------------------------------


def semantic_duplicate_fixture(split: Split) -> Fixture:
    """Paraphrase families, plus same-topic questions that are NOT duplicates.

    The hard negatives are pairs sharing a topic and most of their vocabulary
    while asking different things ("can I cancel" vs "what happens to my data
    if I cancel"). A cosine threshold has no way to know that distinction
    matters, which is exactly what this fixture measures.
    """
    pool = _POOL[split]
    cases: list[EvalCase] = []
    pairs: set[frozenset[str]] = set()

    # Paraphrase families: every within-family pair is a planted positive.
    for family, topic in enumerate(pool["topics"][:3]):
        ids = []
        for variant, template in enumerate(pool["paraphrases"]):
            cid = f"para_{family}_{variant}"
            ids.append(cid)
            cases.append(
                EvalCase(
                    id=cid,
                    input=template.format(topic=topic),
                    expected=f"Resolution path for topic {family}.",
                    label=f"family_{family}",
                )
            )
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs.add(frozenset({ids[a], ids[b]}))

    # Hard negatives: high lexical overlap, different question.
    topic = pool["topics"][4]
    cases.append(
        EvalCase(id="hard_a", input=f"Can I change {topic}?",
                 expected="Yes, from the settings page.", label="hard")
    )
    cases.append(
        EvalCase(id="hard_b", input=f"What happens to my records if I change {topic}?",
                 expected="Previous records are retained for six years.", label="hard")
    )
    cases.append(
        EvalCase(id="hard_c", input=f"How long does it take to change {topic}?",
                 expected="Two working days.", label="hard")
    )

    for i, text in enumerate(pool["distinct"]):
        cases.append(EvalCase(id=f"unique_{i}", input=text,
                              expected=f"answer {i}", label="unique"))

    return Fixture(
        name="semantic_duplicates",
        split=split,
        eval_set=EvalSet(cases=tuple(cases), source=f"semantic_{split}.jsonl"),
        label_basis="AUTHOR JUDGEMENT: paraphrases of one scenario. A different "
        "annotator would draw some of these differently, so precision here is "
        "precision against these labels rather than against truth.",
        positives=frozenset(cid for pair in pairs for cid in pair),
        positive_pairs=frozenset(pairs),
        metadata={
            "n_planted_pairs": len(pairs),
            "hard_negatives": ["hard_a", "hard_b", "hard_c"],
            "hard_negative_note": "same topic, most of the same words, "
            "different question. Any pair among these counted as a duplicate is "
            "a false positive.",
        },
    )


# --------------------------------------------------------------------------
# 3. Class imbalance — dataset-level, objective label
# --------------------------------------------------------------------------


def imbalance_fixtures(split: Split) -> list[tuple[EvalSet, bool, str]]:
    """(dataset, is_imbalanced, why) triples.

    The label is arithmetic: a dataset is imbalanced if its largest class holds
    more than 10 times the smallest, or any class falls below the 10% share the
    default check reports on. Both are the check's own documented criteria, so
    this measures implementation rather than judgement -- stated plainly.
    """
    rng = _rng(split, 30)
    pool = _POOL[split]
    out: list[tuple[EvalSet, bool, str]] = []

    def build(counts: dict[str, int], name: str) -> EvalSet:
        cases = []
        n = 0
        for label, count in counts.items():
            for _ in range(count):
                cases.append(
                    EvalCase(
                        id=f"{name}_{n}",
                        input=f"{rng.choice(pool['distinct'])} (variant {n})",
                        expected=f"answer {n}",
                        label=label,
                    )
                )
                n += 1
        return EvalSet(cases=tuple(cases), source=f"{name}.jsonl")

    out.append((build({"a": 10, "b": 10, "c": 10}, "balanced_even"), False,
                "three equal classes"))
    out.append((build({"a": 12, "b": 10, "c": 9}, "balanced_mild"), False,
                "mild variation, ratio 1.3:1, every class above 10%"))
    out.append((build({"a": 40, "b": 3, "c": 2}, "imbalanced_ratio"), True,
                "ratio 20:1 and two classes below 10%"))
    out.append((build({"a": 30, "b": 30, "c": 1}, "imbalanced_tiny"), True,
                "one class of 1, which cannot produce a stable accuracy"))
    out.append((build({"a": 50}, "single_class"), True,
                "one class only: no per-class comparison is possible"))
    return out


# --------------------------------------------------------------------------
# 4. Leakage — definitional label
# --------------------------------------------------------------------------


def leakage_fixture(split: Split) -> Fixture:
    """The answer sitting in the prompt, plus four kinds of near-miss.

    The label is definitional -- "the reference answer appears verbatim in the
    input" is what the detector looks for -- so a high score here measures
    implementation correctness and NOT whether verbatim containment is a good
    proxy for leakage. It is not: extractive QA legitimately contains the
    answer, and the detector says so in its own limitations.
    """
    pool = _POOL[split]
    cases: list[EvalCase] = []
    positives: set[str] = set()

    treatments = [
        "escalate to the billing team within one working day",
        "issue a credit note against the original invoice",
        "verify identity with two forms of documentation",
        "reassign the licence through the admin console",
    ]
    for i, answer in enumerate(treatments):
        cid = f"leak_{i}"
        positives.add(cid)
        cases.append(
            EvalCase(
                id=cid,
                input=f"Policy states: {answer}. What is the correct step for "
                f"{pool['topics'][i]}?",
                expected=answer,
                # Deliberately a label that does NOT appear in the input. An
                # earlier version used "policy", which the input began with, so
                # 5 of 9 labelled cases contained their own label -- over the
                # 0.5 widespread share, collapsing the type a second time.
                label="tier_one",
            )
        )

    # Near-miss 1: the topic is discussed, the answer is not present verbatim.
    cases.append(
        EvalCase(
            id="clean_topic",
            input="The billing team handles credit notes and refunds. What is "
            "the correct step for a duplicate charge?",
            expected="issue a credit note against the original invoice",
            label="escalation",
        )
    )
    # Near-miss 2: a short answer that appears incidentally. Below the
    # detector's character floor on purpose -- a two-character string turning up
    # in a long prompt is a coincidence, not evidence.
    cases.append(
        EvalCase(id="clean_short", input="Is the audit log retained? Answer yes or no.",
                 expected="no", label="retention")
    )
    # Near-miss 3: heavy word overlap, different answer.
    cases.append(
        EvalCase(
            id="clean_overlap",
            input="Verify identity with one form of documentation, then proceed. "
            "What does the policy require?",
            expected="verify identity with two forms of documentation",
            label="identity",
        )
    )
    # Near-miss 4: the label word appears naturally in the input, which is
    # normal for a topic-labelled dataset and must not be called leakage.
    cases.append(
        EvalCase(id="clean_label", input="My billing address changed last month. "
                 "How do I update it?", expected="Edit it under payment settings.",
                 label="billing")
    )

    # Labels on the remaining cases too. Without them the label-in-input
    # detector sees a single labelled case, hits its widespread-collapse path at
    # 100%, and is never actually exercised -- the fixture would have measured
    # five detectors while claiming to measure six.
    for i, text in enumerate(pool["distinct"][:4]):
        cases.append(EvalCase(id=f"clean_{i}", input=text, label="general",
                              expected=f"unrelated answer number {i} follows here"))

    return Fixture(
        name="leakage",
        split=split,
        eval_set=EvalSet(cases=tuple(cases), source=f"leakage_{split}.jsonl"),
        label_basis="DEFINITIONAL: the reference answer appears verbatim in the "
        "input. This measures implementation, not whether verbatim containment "
        "is a good proxy for leakage.",
        positives=frozenset(positives),
        metadata={
            "near_misses": ["clean_topic", "clean_short", "clean_overlap",
                            "clean_label"],
            "near_miss_note": "clean_label carries the label 'billing' and the "
            "word 'billing' in its input. The label-in-input detector is "
            "DESIGNED to flag that at reduced confidence, and its own message "
            "says the pattern is often legitimate. So a false positive there is "
            "expected behaviour, not a defect -- which is why this benchmark "
            "reports precision stratified by the detector's own confidence.",
        },
    )


# --------------------------------------------------------------------------
# 5. Ambiguous references — author judgement
# --------------------------------------------------------------------------


#: Split-specific ambiguity content. An earlier version hardcoded these, so
#: DEV and TEST had byte-identical positives and TEST was not held out for this
#: category at all -- caught by a check in tests/test_benchmark.py, which now
#: enforces the separation for every fixture rather than trusting it.
_AMBIGUOUS = {
    Split.DEV: {
        "list": ("List some ways to reduce a monthly bill.", "Switch to annual billing"),
        "alternatives": ("Which department handles this request?",
                         "Billing or Support, either is acceptable"),
        "placeholder": ("What is the escalation deadline?", "TODO"),
        "subjective": ("What is the best way to organise a support queue?",
                       "By priority"),
        "contradiction": ("How many days does a refund take?",
                          "5 working days", "10 working days"),
        "clean_keyword": (
            "According to the handbook, what is the best-practice retention "
            "period for audit logs?", "Seven years"),
        "clean_or": ("Is the audit log retained for five or seven years?",
                     "Seven years"),
        "clean_list": ("Name the two forms of identification required.",
                       "Passport and utility bill"),
        "clean_agree": ("What is the account lockout threshold?", "Five attempts"),
    },
    Split.TEST: {
        "list": ("Give a few options for renewing a lapsed membership.",
                 "Renew online with a card"),
        "alternatives": ("Which desk issues a replacement card?",
                         "Reception or the accessibility office, either works"),
        "placeholder": ("What is the maximum loan period?", "TBD"),
        "subjective": ("What is the most sensible order to book appointments in?",
                       "Earliest first"),
        "contradiction": ("How long is a specialist referral valid?",
                          "Three months", "Twelve months"),
        "clean_keyword": (
            "Per the access policy, what is the recommended best-practice "
            "notice period for cancelling a booking?", "Forty-eight hours"),
        "clean_or": ("Is a referral valid for three or twelve months?",
                     "Three months"),
        "clean_list": ("Name the two documents needed to register a dependant.",
                       "Birth certificate and proof of address"),
        "clean_agree": ("What is the daily borrowing limit?", "Six items"),
    },
}


def ambiguity_fixture(split: Split) -> Fixture:
    """Under-specified ground truth, plus questions that only look subjective.

    Author judgement, and the hardest category to label honestly: "what is the
    best way to X" with a single reference answer is under-specified in my
    reading, and a reasonable person could disagree. The negatives include
    questions containing subjective-sounding words with one defensible answer,
    which is where a keyword rule goes wrong.
    """
    pool = _AMBIGUOUS[split]
    cases: list[EvalCase] = []
    positives: set[str] = set()

    for kind in ("list", "alternatives", "placeholder", "subjective"):
        question, answer = pool[kind]
        cid = f"amb_{kind}"
        positives.add(cid)
        cases.append(EvalCase(id=cid, input=question, expected=answer))

    # Two cases, same question, contradictory references. Structural: they
    # cannot both be right, so no judgement is needed for this one.
    question, first, second = pool["contradiction"]
    for suffix, answer in (("a", first), ("b", second)):
        cid = f"amb_contradict_{suffix}"
        positives.add(cid)
        cases.append(EvalCase(id=cid, input=question, expected=answer))

    # --- negatives, each designed to trip a keyword rule -------------------
    question, answer = pool["clean_keyword"]
    cases.append(EvalCase(id="clean_best_practice", input=question, expected=answer))
    question, answer = pool["clean_or"]
    cases.append(EvalCase(id="clean_or_question", input=question, expected=answer))
    question, answer = pool["clean_list"]
    cases.append(EvalCase(id="clean_complete_list", input=question, expected=answer))
    question, answer = pool["clean_agree"]
    cases.append(EvalCase(id="clean_agree_a", input=question, expected=answer))
    cases.append(EvalCase(id="clean_agree_b", input=question, expected=answer))

    for i, text in enumerate(_POOL[split]["distinct"][:4]):
        cases.append(EvalCase(id=f"clean_{i}", input=text,
                              expected=f"A single specific answer, number {i}."))

    return Fixture(
        name="ambiguity",
        split=split,
        eval_set=EvalSet(cases=tuple(cases), source=f"ambiguity_{split}.jsonl"),
        label_basis="AUTHOR JUDGEMENT: the reference does not uniquely determine "
        "a correct answer. A reasonable annotator could disagree on several. The "
        "contradiction pair is the exception -- two different answers to one "
        "question is structural, not a matter of opinion.",
        positives=frozenset(positives),
        metadata={
            "near_misses": ["clean_best_practice", "clean_or_question",
                            "clean_complete_list", "clean_agree_a",
                            "clean_agree_b"],
        },
    )


# --------------------------------------------------------------------------
# 6, 7, 9. Ceiling / floor / separating — objective by construction
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiscriminationFixture:
    """An eval set plus a scorer whose behaviour is a known fact.

    Because the scorer is a fixture, "which models pass this case" is not a
    judgement: it is written down. So the expected CaseClass for every case is
    objective, and a disagreement is a detector error rather than a labelling
    dispute. This is the strongest category in the benchmark.
    """

    eval_set: EvalSet
    scorer: Callable[[Any, Any], bool]
    models: tuple[str, ...]
    expected_class: dict[str, str]
    split: Split


def discrimination_fixture(split: Split) -> DiscriminationFixture:
    """20 cases: 6 ceiling, 5 floor, 6 separating, 3 non-monotonic."""
    pool = _POOL[split]
    plan: dict[str, str] = {}
    cases: list[EvalCase] = []

    def add(prefix: str, n: int, kind: str) -> None:
        for i in range(n):
            cid = f"{prefix}_{i}"
            plan[cid] = kind
            cases.append(
                EvalCase(
                    id=cid,
                    input=f"{pool['distinct'][i % len(pool['distinct'])]} "
                    f"[{prefix} variant {i}]",
                    expected=f"answer {cid}",
                    label=kind,
                )
            )

    add("ceiling", 6, "ceiling")
    add("floor", 5, "floor")
    add("sep", 6, "separating")
    add("nonmono", 3, "non_monotonic")

    models = ("weak", "middle", "strong")

    def scorer(case: EvalCase, model: Any) -> bool:
        kind = plan[case.id]
        if kind == "ceiling":
            return True  # every model passes
        if kind == "floor":
            return False  # every model fails
        if kind == "separating":
            return model in ("middle", "strong")  # weak fails, others pass
        # non-monotonic: the MIDDLE model fails where both others pass, which
        # cannot happen if the models are truly ordered by capability.
        return model in ("weak", "strong")

    return DiscriminationFixture(
        eval_set=EvalSet(cases=tuple(cases), source=f"discrimination_{split}.jsonl"),
        scorer=scorer,
        models=models,
        expected_class=plan,
        split=split,
    )


# --------------------------------------------------------------------------
# 8. Judge instability — constructed agreement rate
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgeFixture:
    """Judges whose agreement rate is a known parameter.

    Chance-corrected agreement has a closed form here, so the measurement can be
    checked against arithmetic rather than against an opinion about what counts
    as "unstable".
    """

    observations: list[Any]
    agreement_rate: float
    n_items: int
    n_judges: int
    label: str


def judge_fixture(
    split: Split, *, agreement_rate: float, n_items: int = 60, n_judges: int = 3
) -> JudgeFixture:
    """Build judges that agree on a known fraction of items.

    On an agreeing item every judge returns the same verdict. On a disagreeing
    item each judge answers independently at random, so the DESIGN parameter is
    the share of items on which agreement is forced, not the raw agreement rate.
    """
    from evallint.checks import JudgeObservation

    rng = _rng(split, int(agreement_rate * 1000) + 80)
    observations = []
    for item in range(n_items):
        forced = rng.random() < agreement_rate
        shared = rng.random() < 0.5
        for judge in range(n_judges):
            verdict = shared if forced else rng.random() < 0.5
            observations.append(
                JudgeObservation(
                    judge=f"judge_{judge}", item=f"item_{item}", repeat=0,
                    verdict=verdict,
                )
            )
    return JudgeFixture(
        observations=observations,
        agreement_rate=agreement_rate,
        n_items=n_items,
        n_judges=n_judges,
        label=f"forced-agreement share {agreement_rate:.2f}",
    )


# --------------------------------------------------------------------------
# 10. Statistical noise — the null hypothesis, constructed
# --------------------------------------------------------------------------


def noise_trials(
    split: Split,
    *,
    n_trials: int = 200,
    n_cases: int = 60,
    true_difference: float = 0.0,
    base_rate: float = 0.5,
) -> Iterator[dict[str, dict[str, bool]]]:
    """Independent trials of two models with a KNOWN true difference.

    With ``true_difference = 0`` the two models are genuinely equal, so any
    "real and meaningful difference" verdict is a false positive. Over many
    trials that rate should sit near alpha, which is a property arithmetic can
    check -- no opinion required. With a non-zero difference the same loop
    measures power.
    """
    rng = _rng(split, int(true_difference * 1000) + 100)
    for _ in range(n_trials):
        a: dict[str, bool] = {}
        b: dict[str, bool] = {}
        for case in range(n_cases):
            cid = f"case_{case}"
            a[cid] = rng.random() < base_rate
            b[cid] = rng.random() < base_rate + true_difference
        yield {"model_a": a, "model_b": b}
