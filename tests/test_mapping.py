"""Tests for the field-mapping layer.

The schemas in REAL_SCHEMAS are not invented. They are the exact column lists
of five widely-used public eval datasets, taken from the HuggingFace
datasets-server, and every one of them failed to load before this layer
existed. Pinning them here means a future change to ALIASES that breaks a real
dataset fails the suite instead of being discovered by a user.
"""

from __future__ import annotations

import json

import pytest

from evallint.io import LoadError, load, load_with_mapping
from evallint.mapping import (
    AmbiguousMappingError,
    FieldMapping,
    MappingError,
    apply_mapping,
    resolve_mapping,
)

# Verbatim column lists from the datasets-server, 2026-08.
REAL_SCHEMAS = {
    "truthfulqa": [
        "type", "category", "question", "best_answer",
        "correct_answers", "incorrect_answers", "source",
    ],
    "gsm8k": ["question", "answer"],
    "mmlu": ["question", "subject", "choices", "answer"],
    "hellaswag": [
        "ind", "activity_label", "ctx_a", "ctx_b", "ctx",
        "endings", "source_id", "split", "split_type", "label",
    ],
    "mtbench": ["question_id", "category", "turns", "reference"],
}


# --- the good case: real schemas resolve without help ----------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("gsm8k", {"input": "question", "expected": "answer"}),
        (
            "mmlu",
            {"input": "question", "expected": "answer", "label": "subject"},
        ),
    ],
)
def test_real_schemas_resolve_unaided(name, expected):
    mapping = resolve_mapping(REAL_SCHEMAS[name])
    assert mapping.resolved == expected


def test_hellaswag_resolves_and_uses_ind_as_id():
    mapping = resolve_mapping(REAL_SCHEMAS["hellaswag"])
    assert mapping.resolved["input"] == "ctx"
    assert mapping.resolved["id"] == "ind"


def test_mtbench_id_and_label_resolve():
    # 'turns' is not an alias for input, so MT-Bench needs an explicit map --
    # which is correct: its prompt is a LIST of turns, not one input string.
    mapping = resolve_mapping(REAL_SCHEMAS["mtbench"], {"input": "turns"})
    assert mapping.resolved["id"] == "question_id"
    assert mapping.resolved["label"] == "category"
    assert mapping.resolved["expected"] == "reference"


# --- the important case: refusing to guess --------------------------------

def test_truthfulqa_is_ambiguous_rather_than_guessed():
    """TruthfulQA has both 'category' and 'type'. Picking one would be a coin
    flip, and a wrong pick produces a plausible report about the wrong data."""
    with pytest.raises(AmbiguousMappingError) as exc:
        resolve_mapping(REAL_SCHEMAS["truthfulqa"])
    message = str(exc.value)
    assert "category" in message and "type" in message
    assert "--map label=" in message  # tells the user how to resolve it


def test_ambiguity_is_resolved_by_an_override():
    mapping = resolve_mapping(REAL_SCHEMAS["truthfulqa"], {"label": "category"})
    assert mapping.resolved["label"] == "category"
    assert mapping.resolved["input"] == "question"
    # best_answer beats nothing else here, but it must not silently become
    # 'correct_answers' (a list) either.
    assert mapping.resolved["expected"] == "best_answer"


def test_ambiguous_input_also_refuses():
    with pytest.raises(AmbiguousMappingError):
        resolve_mapping(["question", "prompt", "answer"])


# --- the cautionary case: a canonical name that means something else -------

def test_canonical_label_is_used_but_alternatives_are_reported():
    """HellaSwag's 'label' is the index of the correct ending, NOT a class.

    evallint cannot know that. What it can do is use the canonical name and
    say that 'activity_label' was also there, which is how a user catches it.
    Silently consuming 'label' as a class would be a misinterpretation
    presented as a result.
    """
    mapping = resolve_mapping(REAL_SCHEMAS["hellaswag"])
    assert mapping.resolved["label"] == "label"
    assert "label" not in mapping.inferred  # taken as-is, not guessed
    assert "activity_label" in mapping.alternatives["label"]
    explanation = " ".join(mapping.explain())
    assert "WARNING" in explanation
    assert "activity_label" in explanation


def test_canonical_names_are_never_remapped():
    mapping = resolve_mapping(["id", "input", "expected", "label"])
    assert mapping.resolved == {
        "id": "id", "input": "input", "expected": "expected", "label": "label",
    }
    assert mapping.is_identity()
    assert mapping.explain()  # still explains, even when nothing was inferred


def test_a_displaced_canonical_column_cannot_overwrite_the_mapped_value():
    """REGRESSION. `--map label=activity_label` on HellaSwag was silently
    ignored: the leftover 'label' column fell through the extras loop and
    overwrote the value taken from 'activity_label'. No error, just the wrong
    column reported as the class -- the exact failure this module exists to
    prevent, and it survived because no test mapped around an existing
    canonical name.
    """
    mapping = resolve_mapping(
        REAL_SCHEMAS["hellaswag"], {"label": "activity_label"}
    )
    assert mapping.resolved["label"] == "activity_label"
    out = apply_mapping(
        {"ctx": "A man is...", "activity_label": "Roof shingle removal",
         "label": "2", "ind": 24},
        mapping,
    )
    assert out["label"] == "Roof shingle removal"
    assert out["unmapped_label"] == "2"  # kept, not dropped


def test_displacement_is_reported_not_silent():
    mapping = resolve_mapping(
        REAL_SCHEMAS["hellaswag"], {"label": "activity_label"}
    )
    explanation = " ".join(mapping.explain())
    assert "'label' was NOT used as label" in explanation
    assert "unmapped_label" in explanation


def test_displacement_avoids_colliding_with_a_real_column():
    mapping = resolve_mapping(
        ["question", "label", "activity_label", "unmapped_label"],
        {"label": "activity_label"},
    )
    out = apply_mapping(
        {"question": "q", "label": "2", "activity_label": "cls",
         "unmapped_label": "mine"},
        mapping,
    )
    assert out["label"] == "cls"
    assert out["unmapped_label"] == "mine"  # the user's own column is untouched
    assert "2" in out.values()  # the displaced value is still somewhere


def test_no_displacement_when_the_canonical_column_is_the_one_used():
    mapping = resolve_mapping(REAL_SCHEMAS["hellaswag"])
    assert mapping.displaced == {}


def test_displaced_column_reaches_metadata_end_to_end(tmp_path):
    path = _write(tmp_path, [
        {"ctx": "A man is on a roof.", "activity_label": "Roof shingle removal",
         "label": 2, "ind": 24},
    ])
    eval_set, _ = load_with_mapping(path, {"label": "activity_label"})
    case = eval_set.cases[0]
    assert case.label == "Roof shingle removal"
    assert case.metadata["unmapped_label"] == 2


# --- errors are actionable ------------------------------------------------

def test_missing_input_names_the_available_columns():
    with pytest.raises(MappingError) as exc:
        resolve_mapping(["foo", "bar"])
    assert "foo" in str(exc.value) and "bar" in str(exc.value)


def test_override_to_a_nonexistent_column_is_rejected():
    with pytest.raises(MappingError) as exc:
        resolve_mapping(["question"], {"input": "prompt"})
    assert "no column named 'prompt'" in str(exc.value)
    assert "question" in str(exc.value)  # lists what IS available


def test_override_to_an_unknown_field_is_rejected():
    with pytest.raises(MappingError) as exc:
        resolve_mapping(["question"], {"inpt": "question"})
    assert "inpt" in str(exc.value)


def test_two_fields_cannot_take_the_same_column():
    mapping = resolve_mapping(["question", "answer"], {"input": "question"})
    assert mapping.resolved["input"] == "question"
    assert mapping.resolved["expected"] == "answer"


def test_matching_is_case_insensitive():
    mapping = resolve_mapping(["Question", "Answer"])
    # The ORIGINAL casing is preserved, because that is the key in the data.
    assert mapping.resolved["input"] == "Question"


# --- apply_mapping --------------------------------------------------------

def test_apply_mapping_renames_and_keeps_extras():
    mapping = resolve_mapping(["question", "answer", "difficulty"])
    out = apply_mapping(
        {"question": "2+2?", "answer": "4", "difficulty": "easy"}, mapping
    )
    assert out["input"] == "2+2?"
    assert out["expected"] == "4"
    assert out["difficulty"] == "easy"  # unmapped columns survive


def test_apply_mapping_tolerates_a_ragged_record():
    mapping = resolve_mapping(["question", "answer"])
    out = apply_mapping({"question": "2+2?"}, mapping)
    assert out == {"input": "2+2?"}


# --- end to end through load() -------------------------------------------

def _write(tmp_path, rows, name="cases.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_load_infers_aliases(tmp_path):
    path = _write(tmp_path, [
        {"question": "What is 2+2?", "answer": "4", "subject": "math"},
        {"question": "Capital of France?", "answer": "Paris", "subject": "geo"},
    ])
    eval_set = load(path)
    assert len(eval_set) == 2
    assert eval_set.cases[0].input == "What is 2+2?"
    assert eval_set.cases[0].expected == "4"
    assert eval_set.cases[0].label == "math"


def test_load_reports_the_mapping_it_used(tmp_path):
    path = _write(tmp_path, [{"question": "q", "answer": "a"}])
    _, mapping = load_with_mapping(path)
    assert mapping.inferred == {"input": "question", "expected": "answer"}
    assert not mapping.is_identity()


def test_load_accepts_an_explicit_map(tmp_path):
    path = _write(tmp_path, [{"q": "2+2?", "a": "4", "topic": "math"}])
    eval_set, mapping = load_with_mapping(
        path, {"input": "q", "expected": "a", "label": "topic"}
    )
    assert eval_set.cases[0].input == "2+2?"
    assert eval_set.cases[0].label == "math"
    assert mapping.resolved["input"] == "q"


def test_load_surfaces_ambiguity_as_a_load_error(tmp_path):
    path = _write(tmp_path, [
        {"question": "q", "answer": "a", "category": "c", "type": "t"},
    ])
    with pytest.raises(LoadError) as exc:
        load(path)
    assert "category" in str(exc.value) and "type" in str(exc.value)


def test_load_unions_columns_across_ragged_records(tmp_path):
    """JSONL in the wild is ragged. If row 1 lacks 'subject' but row 2 has it,
    the label must still be found -- reading only row 1 would lose it."""
    path = _write(tmp_path, [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2", "subject": "math"},
    ])
    eval_set, mapping = load_with_mapping(path)
    assert mapping.resolved.get("label") == "subject"
    assert eval_set.cases[0].label is None
    assert eval_set.cases[1].label == "math"


def test_identity_schema_still_loads_unchanged(tmp_path):
    """The bundled example set must be unaffected by all of the above."""
    eval_set, mapping = load_with_mapping("examples/sample_evalset.jsonl")
    assert len(eval_set) == 20
    assert mapping.is_identity()
    assert eval_set.cases[0].id == "billing_001"


def test_explain_is_empty_for_an_empty_mapping():
    assert FieldMapping({}).explain() == []
