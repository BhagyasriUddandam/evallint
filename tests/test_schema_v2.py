"""Tests for schema version 2: chat, system prompts, rubrics, tools.

Three things are being proven, in descending order of importance.

**1. Nothing that worked before changed.** The whole feature is worthless if it
breaks a file someone already has, so the first section fixes v1 behaviour in
place: the four-field record, the constructor with four arguments, the aliases,
and above all the exact TEXT each check will see. `input` is what eleven checks
read, so a v1 case whose derived text shifted by one character would silently
change findings.

**2. Each new field carries its data and refuses bad data.** Every feature gets
a pair: it works when written correctly, and it is rejected with a locating
message when it is not.

**3. Version 2 is strict where version 1 is lenient**, which is the only thing
declaring a version does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evallint.io import LoadError, load, load_with_mapping
from evallint.schema import (
    SCHEMA_VERSION,
    Criterion,
    EvalCase,
    Message,
    Role,
    Rubric,
    SchemaError,
    ToolCall,
    ToolSpec,
    derive_input,
)
from evallint.validation import SchemaValidationError, parse_cases

V1_RECORDS = [
    {"id": "a1", "input": "What is 2 + 2?", "expected": "4", "label": "math"},
    {"id": "a2", "input": "Capital of France?", "expected": "Paris", "label": "geo"},
]


def write_jsonl(path: Path, records: list[dict], *, version: int | None = None) -> Path:
    lines = []
    if version is not None:
        lines.append(json.dumps({"evallint_schema": version}))
    lines.extend(json.dumps(r) for r in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, records: list[dict], *, version: int | None = None) -> Path:
    payload: dict = {"cases": records}
    if version is not None:
        payload = {"evallint_schema": version, **payload}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ==========================================================================
# 1. Backward compatibility. If any test in this section fails, the feature
#    is not shippable regardless of what the rest of the file says.
# ==========================================================================


def test_v1_file_loads_unchanged_and_reports_version_1(tmp_path):
    """A file with no version declaration is version 1, not "probably 2"."""
    eval_set = load(write_jsonl(tmp_path / "v1.jsonl", V1_RECORDS))
    assert len(eval_set) == 2
    assert eval_set.schema_version == 1
    assert eval_set.chat_cases == 0
    assert [c.input for c in eval_set] == ["What is 2 + 2?", "Capital of France?"]
    assert [c.expected for c in eval_set] == ["4", "Paris"]
    assert [c.label for c in eval_set] == ["math", "geo"]
    assert all(not c.input_is_derived for c in eval_set)
    assert eval_set.load_notes == ()


def test_v1_constructor_still_takes_exactly_four_arguments():
    """Every new field defaults, so old construction code keeps compiling."""
    case = EvalCase(id="x", input="q", expected="a", label="cls")
    assert case.messages == ()
    assert case.system is None
    assert case.acceptable == ()
    assert case.rubric is None
    assert case.tools == ()
    assert case.expected_tool_calls == ()
    assert case.expected_schema is None
    assert case.is_chat is False


def test_chat_form_of_a_simple_case_gives_byte_identical_input(tmp_path):
    """The compatibility hinge, tested directly.

    `input` is what eleven checks read. A one-turn conversation must flatten to
    exactly the text the four-field form would have supplied, or migrating a
    file would change findings that have nothing to do with chat.
    """
    plain = load(write_jsonl(tmp_path / "plain.jsonl", V1_RECORDS))
    chat = load(
        write_jsonl(
            tmp_path / "chat.jsonl",
            [
                {
                    "id": r["id"],
                    "messages": [{"role": "user", "content": r["input"]}],
                    "expected": r["expected"],
                    "label": r["label"],
                }
                for r in V1_RECORDS
            ],
            version=2,
        )
    )
    assert [c.input for c in plain] == [c.input for c in chat]
    # ... and the derivation is recorded rather than hidden.
    assert all(not c.input_is_derived for c in plain)
    assert all(c.input_is_derived for c in chat)


def test_unknown_keys_are_still_metadata_under_version_1(tmp_path):
    """v1 leniency, kept deliberately. Real datasets have columns like
    'labels' and 'inputs', and turning those into errors would break files
    this package promises to keep reading."""
    eval_set = load(
        write_jsonl(
            tmp_path / "extra.jsonl",
            [{"id": "a", "input": "q", "labels": ["x"], "inputs": "y", "notes": "z"}],
        )
    )
    assert eval_set.cases[0].metadata == {
        "labels": ["x"],
        "inputs": "y",
        "notes": "z",
    }


def test_existing_aliases_are_untouched_by_the_new_ones(tmp_path):
    """Adding 'messages' and 'system' aliases must not disturb the old five."""
    path = tmp_path / "gsm8k.jsonl"
    write_jsonl(path, [{"question": "2+2?", "answer": "4"}])
    eval_set, mapping = load_with_mapping(path)
    assert mapping.resolved["input"] == "question"
    assert mapping.resolved["expected"] == "answer"
    assert eval_set.cases[0].input == "2+2?"


# ==========================================================================
# 2. Feature by feature: it works, and bad input is refused with a location.
# ==========================================================================


# ---- 1. chat / multi-turn ------------------------------------------------


def test_multi_turn_conversation_is_preserved_and_flattened(tmp_path):
    eval_set = load(
        write_json(
            tmp_path / "chat.json",
            [
                {
                    "id": "c1",
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                        {"role": "user", "content": "Refund?"},
                    ],
                }
            ],
            version=2,
        )
    )
    case = eval_set.cases[0]
    assert case.is_chat
    assert [m.role for m in case.messages] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert case.input == "user: Hi\nassistant: Hello\nuser: Refund?"
    assert eval_set.chat_cases == 1


def test_flattening_is_stable_across_a_round_trip(tmp_path):
    """as_dict -> file -> load must give the same text, or migration is unsafe."""
    original = EvalCase(
        id="c1",
        input="ignored, will be re-derived",
        messages=(
            Message(role=Role.USER, content="Book a flight"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=(ToolCall(name="book", arguments={"to": "LIS"}, id="t1"),),
            ),
            Message(role=Role.TOOL, content="done", tool_call_id="t1", name="book"),
        ),
        input_is_derived=True,
    )
    path = write_json(tmp_path / "rt.json", [original.as_dict()], version=2)
    reloaded = load(path).cases[0]
    assert reloaded.input == derive_input(original.messages)
    assert reloaded.messages == original.messages


def test_a_bad_role_is_rejected_with_its_path_and_a_suggestion(tmp_path):
    path = write_jsonl(
        tmp_path / "bad.jsonl",
        [{"id": "c1", "messages": [{"role": "assistent", "content": "x"}]}],
        version=2,
    )
    with pytest.raises(LoadError) as exc:
        load(path)
    text = str(exc.value)
    assert "messages[0].role" in text
    assert "'assistent' is not a valid role" in text
    assert "did you mean 'assistant'?" in text


def test_a_file_with_no_prompt_column_names_both_options(tmp_path):
    """Caught by the mapping layer, which can also list the columns present."""
    path = write_jsonl(tmp_path / "none.jsonl", [{"id": "c1", "expected": "4"}])
    with pytest.raises(LoadError) as exc:
        load(path)
    text = str(exc.value)
    assert "no column could be used as 'input'" in text
    assert "or 'messages' for a chat dataset" in text
    assert "Columns found: id, expected" in text


def test_one_case_missing_its_prompt_in_a_ragged_file_names_both_options(tmp_path):
    """Here the column exists, so the mapping succeeds and the per-case parser
    is what reports it -- pointing at the one line that is wrong."""
    path = write_jsonl(
        tmp_path / "ragged.jsonl",
        [{"id": "a", "input": "q"}, {"id": "b", "expected": "4"}],
    )
    with pytest.raises(LoadError) as exc:
        load(path)
    text = str(exc.value)
    assert "line 2: missing required field 'input'" in text
    assert "'messages' for a multi-turn case" in text


def test_a_conversation_of_plain_strings_is_read_as_user_turns_and_reported(tmp_path):
    """The MT-Bench shape. Interpreted, then declared -- see validation.py."""
    path = write_jsonl(
        tmp_path / "mt.jsonl",
        [{"question_id": 81, "turns": ["Write a post.", "Now rewrite it."]}],
    )
    eval_set = load(path)
    case = eval_set.cases[0]
    assert [m.role for m in case.messages] == [Role.USER, Role.USER]
    assert case.input == "user: Write a post.\nuser: Now rewrite it."
    assert any("successive USER turns" in n for n in eval_set.load_notes)


def test_repeated_load_notes_collapse_to_one_line(tmp_path):
    """A real MT-Bench file has 80 cases. It must not produce 80 notes."""
    path = write_jsonl(
        tmp_path / "many.jsonl",
        [{"id": f"c{i}", "turns": ["a", "b"]} for i in range(40)],
    )
    notes = load(path).load_notes
    assert len(notes) == 1
    assert "40 cases" in notes[0]


# ---- 2. system prompts ---------------------------------------------------


def test_system_prompt_is_kept_out_of_the_derived_input(tmp_path):
    """The least obvious rule in derive_input, and the one with teeth.

    A system prompt is usually identical across every case. Folding it into
    each `input` would give every case a long shared prefix, so the redundancy
    check would report near-duplicates throughout and the effective-size
    estimate would collapse -- both as artefacts of the flattening.
    """
    shared = "You are a terse assistant. " * 20
    eval_set = load(
        write_json(
            tmp_path / "sys.json",
            [
                {"id": "a", "system": shared, "messages": [{"role": "user", "content": "2+2?"}]},
                {"id": "b", "system": shared, "messages": [{"role": "user", "content": "3+3?"}]},
            ],
            version=2,
        )
    )
    assert [c.system for c in eval_set] == [shared, shared]
    assert [c.input for c in eval_set] == ["2+2?", "3+3?"]
    # The two inputs share nothing, which is the point: without this rule they
    # would be ~97% identical text.
    assert not set(eval_set.cases[0].input.split()) & set(
        eval_set.cases[1].input.split()
    )


def test_a_system_turn_inside_messages_still_appears_in_the_input(tmp_path):
    """`system` the FIELD is excluded; a system TURN the user put in the
    conversation is their choice and is flattened like any other turn."""
    eval_set = load(
        write_json(
            tmp_path / "systurn.json",
            [
                {
                    "id": "a",
                    "messages": [
                        {"role": "system", "content": "Be terse."},
                        {"role": "user", "content": "2+2?"},
                    ],
                }
            ],
            version=2,
        )
    )
    assert eval_set.cases[0].input == "system: Be terse.\nuser: 2+2?"


def test_non_string_system_is_rejected(tmp_path):
    path = write_jsonl(
        tmp_path / "s.jsonl", [{"id": "a", "input": "q", "system": {"text": "x"}}], version=2
    )
    with pytest.raises(LoadError) as exc:
        load(path)
    assert ".system" in str(exc.value)
    assert "must be a string" in str(exc.value)


# ---- 3. metadata ---------------------------------------------------------


def test_metadata_survives_alongside_every_new_field(tmp_path):
    eval_set = load(
        write_json(
            tmp_path / "meta.json",
            [
                {
                    "id": "a",
                    "input": "q",
                    "system": "s",
                    "acceptable": ["b"],
                    "rubric": ["accuracy"],
                    "ticket": "T-1",
                    "reviewer": "bh",
                }
            ],
            version=2,
        )
    )
    assert eval_set.cases[0].metadata == {"ticket": "T-1", "reviewer": "bh"}


# ---- 4. multiple acceptable answers --------------------------------------


def test_acceptable_answers_lead_with_expected_and_deduplicate():
    case = EvalCase(
        id="a",
        input="Capital of the Netherlands?",
        expected="Amsterdam",
        acceptable=("The Hague", "Amsterdam"),
    )
    assert case.acceptable_answers() == ("Amsterdam", "The Hague")


def test_a_bare_string_acceptable_is_one_answer_not_its_characters(tmp_path):
    """Iterating a string would turn "yes" into ("y", "e", "s") -- a wrong
    answer set produced silently, which is the worst available outcome."""
    eval_set = load(
        write_jsonl(
            tmp_path / "acc.jsonl",
            [{"id": "a", "input": "q", "expected": "no", "acceptable": "yes"}],
            version=2,
        )
    )
    assert eval_set.cases[0].acceptable == ("yes",)
    assert eval_set.cases[0].acceptable_answers() == ("no", "yes")


def test_unhashable_acceptable_answers_are_kept():
    """Structured answers cannot go in a set. Dropping one to keep the
    de-duplication tidy would discard a valid answer."""
    case = EvalCase(
        id="a", input="q", expected={"x": 1}, acceptable=({"x": 2}, {"x": 1})
    )
    assert len(case.acceptable_answers()) == 3


def test_an_object_in_acceptable_is_flagged_as_probably_wrapped_wrong(tmp_path):
    path = write_jsonl(
        tmp_path / "acc.jsonl",
        [{"id": "a", "input": "q", "acceptable": {"a": 1}}],
        version=2,
    )
    with pytest.raises(LoadError) as exc:
        load(path)
    assert "must be a list of answers" in str(exc.value)


# ---- 5. structured expected outputs -------------------------------------


def test_structured_expected_and_schema_are_preserved(tmp_path):
    eval_set = load(
        write_json(
            tmp_path / "struct.json",
            [
                {
                    "id": "a",
                    "input": "Extract the total",
                    "expected": {"total": 42.5, "currency": "EUR"},
                    "expected_schema": {
                        "type": "object",
                        "required": ["total", "currency"],
                    },
                }
            ],
            version=2,
        )
    )
    case = eval_set.cases[0]
    assert case.expected == {"total": 42.5, "currency": "EUR"}
    assert case.expected_schema["required"] == ["total", "currency"]


def test_expected_schema_must_be_an_object(tmp_path):
    path = write_jsonl(
        tmp_path / "es.jsonl",
        [{"id": "a", "input": "q", "expected_schema": ["object"]}],
        version=2,
    )
    with pytest.raises(LoadError) as exc:
        load(path)
    assert "expected_schema" in str(exc.value)
    assert "JSON Schema object" in str(exc.value)


# ---- 6. rubrics ----------------------------------------------------------


def test_rubric_keeps_raw_weights_and_derives_shares():
    """Weights of 3 and 1 mean the same as 0.75 and 0.25. The user's numbers
    are stored; the shares are computed, so the file and the report agree."""
    rubric = Rubric(
        criteria=(Criterion("accuracy", weight=3), Criterion("tone", weight=1))
    )
    assert [c.weight for c in rubric.criteria] == [3.0, 1.0]
    assert rubric.weight_shares() == {"accuracy": 0.75, "tone": 0.25}


def test_rubric_accepts_the_bare_list_and_bare_name_shorthands(tmp_path):
    eval_set = load(
        write_json(
            tmp_path / "rub.json",
            [{"id": "a", "input": "q", "rubric": ["accuracy", {"name": "tone", "weight": 2}]}],
            version=2,
        )
    )
    rubric = eval_set.cases[0].rubric
    assert [c.name for c in rubric.criteria] == ["accuracy", "tone"]
    assert rubric.weight_shares() == {"accuracy": pytest.approx(1 / 3), "tone": pytest.approx(2 / 3)}


def test_rubric_with_all_zero_weights_is_refused():
    """No criterion can contribute, so no aggregate is definable. Falling back
    to an unweighted mean would invent a scoring rule the user did not ask for.
    """
    with pytest.raises(SchemaError, match="sum to zero"):
        Rubric(criteria=(Criterion("a", weight=0), Criterion("b", weight=0)))


def test_duplicate_criterion_names_are_refused():
    """Two criteria with one name cannot be reported separately and their
    weights would silently add."""
    with pytest.raises(SchemaError, match="must be unique"):
        Rubric(criteria=(Criterion("accuracy"), Criterion("accuracy", weight=5)))


def test_negative_criterion_weight_is_located_in_the_file(tmp_path):
    path = write_jsonl(
        tmp_path / "rub.jsonl",
        [{"id": "a", "input": "q", "rubric": {"criteria": [{"name": "x", "weight": -2}]}}],
        version=2,
    )
    with pytest.raises(LoadError) as exc:
        load(path)
    assert "rubric.criteria[0]" in str(exc.value)
    assert "must not be negative" in str(exc.value)


def test_rubric_scale_must_be_ordered():
    with pytest.raises(SchemaError, match="low must be below high"):
        Rubric(criteria=(Criterion("a"),), scale=(5, 1))


# ---- 7. tool calls -------------------------------------------------------


def test_tool_calls_and_specs_load_including_provider_shapes(tmp_path):
    """Providers nest a call under 'function' and serialise arguments as a JSON
    string. A transcript copied out of an API response should load as-is."""
    eval_set = load(
        write_json(
            tmp_path / "tools.json",
            [
                {
                    "id": "a",
                    "messages": [
                        {"role": "user", "content": "Weather in Oslo?"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "t1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Oslo"}',
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": [
                        {"name": "get_weather", "input_schema": {"type": "object"}}
                    ],
                    "expected_tool_calls": [
                        {"name": "get_weather", "arguments": {"city": "Oslo"}}
                    ],
                }
            ],
            version=2,
        )
    )
    case = eval_set.cases[0]
    call = case.messages[1].tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Oslo"}
    assert call.id == "t1"
    assert case.tools[0].parameters == {"type": "object"}
    assert case.expected_tool_calls[0].signature() == 'get_weather(city="Oslo")'


def test_tool_call_signature_is_key_order_independent():
    """Two calls differing only in key order must compare equal as text, or a
    duplicate check would count them as different scenarios."""
    a = ToolCall(name="f", arguments={"x": 1, "y": 2})
    b = ToolCall(name="f", arguments={"y": 2, "x": 1})
    assert a.signature() == b.signature()


def test_only_an_assistant_can_call_a_tool():
    with pytest.raises(SchemaError, match="only an assistant message"):
        Message(role=Role.USER, content="hi", tool_calls=(ToolCall(name="f"),))


def test_an_expected_call_to_an_unavailable_tool_is_refused():
    """The case could only ever fail, so no scoring run could produce a useful
    number from it. An error, not a finding."""
    with pytest.raises(SchemaError, match="not in 'tools'"):
        EvalCase(
            id="a",
            input="q",
            tools=(ToolSpec(name="search"),),
            expected_tool_calls=(ToolCall(name="lookup"),),
        )


def test_expected_calls_without_a_tools_list_are_allowed():
    """Not every dataset declares the tool inventory, and requiring it would
    reject files that are merely terse rather than wrong."""
    case = EvalCase(
        id="a", input="q", expected_tool_calls=(ToolCall(name="lookup"),)
    )
    assert case.expected_tool_calls[0].name == "lookup"


def test_an_assistant_turn_may_be_empty_when_it_calls_a_tool():
    message = Message(role=Role.ASSISTANT, tool_calls=(ToolCall(name="f"),))
    assert message.content == ""


def test_an_otherwise_empty_turn_is_refused():
    """It would contribute nothing to the flattened input, silently shrinking
    the case rather than failing."""
    with pytest.raises(SchemaError, match="carries no information"):
        Message(role=Role.USER, content="   ")


def test_duplicate_tool_names_are_refused():
    with pytest.raises(SchemaError, match="tool names must be unique"):
        EvalCase(
            id="a", input="q", tools=(ToolSpec(name="s"), ToolSpec(name="s"))
        )


# ==========================================================================
# 3. Versioning, and the one thing it changes
# ==========================================================================


def test_declaring_version_2_turns_a_demotion_into_an_error(tmp_path):
    """The same file, twice, differing only in the header line."""
    record = {"id": "a", "input": "q", "messages": "not a conversation"}

    lenient = load(write_jsonl(tmp_path / "lenient.jsonl", [record]))
    assert lenient.cases[0].metadata == {"messages": "not a conversation"}
    assert any("kept as metadata" in n for n in lenient.load_notes)

    with pytest.raises(LoadError) as exc:
        load(write_jsonl(tmp_path / "strict.jsonl", [record], version=2))
    assert "must be a list of turns" in str(exc.value)


def test_a_demoted_field_really_is_in_metadata(tmp_path):
    """The note says "kept as metadata". If the parser had already popped the
    key, that message would be wrong in the one direction that matters: the
    user would stop looking for their data."""
    eval_set = load(
        write_jsonl(tmp_path / "d.jsonl", [{"id": "a", "input": "q", "tools": 7}])
    )
    assert eval_set.cases[0].metadata == {"tools": 7}


def test_near_miss_keys_are_flagged_only_under_version_2(tmp_path):
    record = {"id": "a", "input": "q", "sytem": "oops", "expected_schemas": {}}

    assert load(write_jsonl(tmp_path / "l.jsonl", [record])).cases[0].metadata == {
        "sytem": "oops",
        "expected_schemas": {},
    }

    with pytest.raises(LoadError) as exc:
        load(write_jsonl(tmp_path / "s.jsonl", [record], version=2))
    text = str(exc.value)
    assert "did you mean 'system'?" in text
    assert "did you mean 'expected_schema'?" in text


def test_an_unsupported_version_is_refused_by_name(tmp_path):
    """A version-3 file misread under version-2 rules would produce a confident
    report about data that was parsed wrong."""
    with pytest.raises(LoadError) as exc:
        load(write_jsonl(tmp_path / "future.jsonl", V1_RECORDS, version=3))
    text = str(exc.value)
    assert "declares schema version 3" in text
    assert "supported versions: 1, 2" in text


def test_a_non_integer_version_is_refused(tmp_path):
    with pytest.raises(LoadError) as exc:
        load(write_json(tmp_path / "v.json", V1_RECORDS, version="2.0"))
    assert "must be an integer version" in str(exc.value)


def test_a_case_is_never_mistaken_for_a_version_header(tmp_path):
    """A JSONL header is recognised only when it declares a version and carries
    nothing that could make it a case. Otherwise a real case would vanish."""
    path = tmp_path / "h.jsonl"
    path.write_text(
        json.dumps({"evallint_schema": 2, "id": "a", "input": "q"}) + "\n",
        encoding="utf-8",
    )
    eval_set = load(path)
    assert len(eval_set) == 1
    # Not consumed as a header, so the version stays 1 and the key is metadata.
    assert eval_set.schema_version == 1
    assert eval_set.cases[0].metadata == {"evallint_schema": 2}


def test_the_json_envelope_carries_the_version(tmp_path):
    eval_set = load(write_json(tmp_path / "e.json", V1_RECORDS, version=2))
    assert eval_set.schema_version == SCHEMA_VERSION


# ==========================================================================
# 4. Error collection: one run should be enough to fix a whole file
# ==========================================================================


def test_every_problem_in_a_file_is_reported_in_one_run(tmp_path):
    records = [
        {"id": "a", "input": "q", "system": 1},
        {"id": "b", "input": "q", "acceptable": {"x": 1}},
        {"id": "c", "input": "q", "expected_schema": []},
        {"id": "d", "messages": [{"role": "nobody", "content": "x"}]},
    ]
    with pytest.raises(LoadError) as exc:
        load(write_jsonl(tmp_path / "many.jsonl", records, version=2))
    text = str(exc.value)
    assert "4 schema problems in 4 cases" in text
    # Records start at line 2: the version header is line 1, and these are real
    # file line numbers rather than record ordinals. An earlier version of this
    # assertion counted records, which would have sent someone to the wrong line.
    for line in ("line 2.system", "line 3.acceptable", "line 4.expected_schema",
                 "line 5.messages[0].role"):
        assert line in text


def test_a_consequence_is_not_reported_on_top_of_its_cause(tmp_path):
    """A case whose `messages` failed to parse also has no `input`. Reporting
    both makes one mistake look like two."""
    with pytest.raises(LoadError) as exc:
        load(
            write_jsonl(
                tmp_path / "c.jsonl",
                [{"id": "a", "messages": [{"role": "bad", "content": "x"}]}],
                version=2,
            )
        )
    text = str(exc.value)
    assert "1 schema problem" in text
    assert "missing required field 'input'" not in text


def test_the_case_count_survives_a_dotted_file_name(tmp_path):
    """Regression: the affected-case count was derived by splitting the issue
    path on '.', so a file named `x.jsonl` collapsed every issue into one
    bucket and the count was silently always 1."""
    records = [{"id": f"c{i}", "input": "q", "system": i} for i in range(3)]
    with pytest.raises(LoadError) as exc:
        load(write_jsonl(tmp_path / "dotted.name.jsonl", records, version=2))
    assert "3 schema problems in 3 cases" in str(exc.value)


def test_only_the_first_twenty_issues_print_but_all_are_carried(tmp_path):
    records = [{"id": f"c{i}", "input": "q", "system": i} for i in range(25)]
    mapped = [(f"line {i}", r) for i, r in enumerate(records, start=1)]
    with pytest.raises(SchemaValidationError) as exc:
        parse_cases(mapped, strict=True, source="big.jsonl")
    assert len(exc.value.issues) == 25
    assert "... and 5 more" in str(exc.value)


# ==========================================================================
# 5. CSV, which can express all of this only through JSON in a cell
# ==========================================================================


def test_a_conversation_in_a_csv_cell_is_parsed(tmp_path):
    path = tmp_path / "chat.csv"
    path.write_text(
        'id,conversation,answer,category\n'
        'c1,"[{""role"":""user"",""content"":""Hi""},'
        '{""role"":""assistant"",""content"":""Hello""}]",ok,support\n',
        encoding="utf-8",
    )
    eval_set, mapping = load_with_mapping(path)
    case = eval_set.cases[0]
    assert mapping.resolved["messages"] == "conversation"
    assert case.input == "user: Hi\nassistant: Hello"
    assert case.label == "support"


def test_an_unparseable_json_cell_is_demoted_rather_than_dropped(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text(
        'id,input,messages\nc1,q,"[{""role"": ""user""]"\n', encoding="utf-8"
    )
    eval_set = load(path)
    assert "messages" in eval_set.cases[0].metadata
    assert any("does not parse" in n for n in eval_set.load_notes)
