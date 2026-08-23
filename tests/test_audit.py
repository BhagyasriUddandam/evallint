"""Tests for the unified audit report.

The first section is the one that matters most: **there is no overall score, and
no field that could be mistaken for one.** That is the requirement the whole
design hangs on, so it is asserted structurally (no score-like key anywhere in
the JSON) rather than by reading the output.

After that: the tier rules are enforced rather than documented, a section that
did not run can never read as a pass, the four derived views cannot disagree with
the sections they summarise, and all three renderers carry what they claim to.
"""

from __future__ import annotations

import json
import re

import pytest

from evallint.audit import (
    ANALYSIS_SECTIONS,
    SECTION_ORDER,
    AuditFinding,
    AuditReport,
    AuditSection,
    SectionStatus,
    Tier,
    render_html,
    render_terminal,
    run_audit,
    to_json,
)
from evallint.checks import (
    EvaluatorReliability,
    GroundTruthCheck,
    ImbalanceCheck,
    JudgeObservation,
    LeakageCheck,
    RunOutcome,
    analyse_reproducibility,
    compare_models,
)
from evallint.checks.base import Severity
from evallint.checks.effective_size import estimate_from_clusters
from evallint.schema import EvalCase, EvalSet


def tiny_set(n: int = 12) -> EvalSet:
    return EvalSet(
        cases=tuple(
            EvalCase(
                id=f"c{i}",
                input=f"Question number {i} about topic {i % 3}?",
                expected=f"answer {i}",
                label=("alpha" if i < n - 2 else "beta"),
            )
            for i in range(n)
        ),
        source="tiny.jsonl",
    )


def file_only_audit(eval_set: EvalSet | None = None) -> AuditReport:
    """What the CLI can produce: the checks that need no model."""
    s = eval_set or tiny_set()
    return run_audit(
        s,
        imbalance=ImbalanceCheck().run(s),
        leakage=LeakageCheck().run(s),
        ground_truth=GroundTruthCheck().run(s),
    )


def full_audit() -> AuditReport:
    """Everything, including the three analyses a CLI cannot run."""
    s = tiny_set()
    ids = [c.id for c in s]
    outcomes = {
        "weak": {c: i % 2 == 0 for i, c in enumerate(ids)},
        "strong": {c: i >= 2 for i, c in enumerate(ids)},
    }
    judges = [
        JudgeObservation(judge=f"j{j}", item=c, repeat=0, verdict=bool((k + j) % 2))
        for j in range(2)
        for k, c in enumerate(ids)
    ]
    runs = [
        RunOutcome(run=f"r{r}", model=m, case=c, passed=bool((k + r) % 2))
        for r in range(4)
        for m in ("weak", "strong")
        for k, c in enumerate(ids)
    ]
    return run_audit(
        s,
        imbalance=ImbalanceCheck().run(s),
        leakage=LeakageCheck().run(s),
        ground_truth=GroundTruthCheck().run(s),
        effective_size=estimate_from_clusters([3, 2], len(s), threshold=0.85),
        model_comparison=compare_models(outcomes),
        evaluator_reliability=EvaluatorReliability().analyse(judges),
        reproducibility=analyse_reproducibility(runs),
    )


# ==========================================================================
# 1. No single score. The requirement the design exists to satisfy.
# ==========================================================================


def test_no_key_in_the_json_could_be_mistaken_for_an_overall_score() -> None:
    """Structural, not textual. The word "score" appears in prose that DENIES
    having one, and in `pass_rate`-adjacent statistics from the analysers; what
    must not exist is a FIELD a consumer could read as a grade."""
    blob = to_json(full_audit())
    keys = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', blob))
    offenders = sorted(
        k
        for k in keys
        if any(
            word in k.lower()
            for word in ("grade", "rating", "overall", "trust", "quality_score")
        )
    )
    assert not offenders, f"score-like keys present: {offenders}"


def test_the_report_has_no_numeric_top_level_summary_attribute() -> None:
    """Nobody can add `report.score` later without this failing."""
    report = full_audit()
    for name in ("score", "grade", "rating", "overall", "quality", "trust"):
        assert not hasattr(report, name), f"AuditReport gained a `{name}`"


def test_the_executive_summary_never_grades_the_eval() -> None:
    text = " ".join(full_audit().executive_summary()).lower()
    for verdict in (
        "good", "bad", "poor", "excellent", "healthy", "passes", "failed",
        "high quality", "low quality", "trustworthy",
    ):
        assert verdict not in text, f"executive summary says {verdict!r}"


def test_evidence_coverage_replaces_the_score_without_compressing_it() -> None:
    coverage = file_only_audit().evidence_coverage
    assert coverage["analyses_available"] == len(ANALYSIS_SECTIONS)
    # It names them rather than counting them into a percentage.
    assert isinstance(coverage["analyses_with_evidence"], list)
    assert isinstance(coverage["analyses_without_evidence"], dict)
    for reason in coverage["analyses_without_evidence"].values():
        assert reason and len(reason) > 20, "a gap must say what it would take"


# ==========================================================================
# 2. The tier rules are enforced, not documented
# ==========================================================================


def _finding(**kw) -> AuditFinding:
    base = dict(
        section="dataset_statistics",
        severity=Severity.INFO,
        tier=Tier.OBSERVED,
        evidence="e",
        explanation="x",
        limitation="l",
        recommended_action="a",
    )
    return AuditFinding(**{**base, **kw})


def test_an_observed_finding_may_not_carry_a_confidence() -> None:
    """A census is exact. Attaching "HIGH" implies a doubt that does not exist,
    and that is how a count starts being argued with."""
    with pytest.raises(ValueError, match="implies a doubt"):
        _finding(tier=Tier.OBSERVED, confidence="high")


def test_a_heuristic_finding_must_carry_a_confidence() -> None:
    """A threshold-derived signal without one reads as a fact."""
    with pytest.raises(ValueError, match="reads as a fact"):
        _finding(tier=Tier.HEURISTIC, confidence=None)


@pytest.mark.parametrize(
    "missing", ["evidence", "explanation", "limitation", "recommended_action"]
)
def test_every_finding_must_carry_all_seven_fields(missing: str) -> None:
    with pytest.raises(ValueError, match=f"no {missing}"):
        _finding(**{missing: "   "})


def test_recommendations_are_ordered_weakest_tier_first() -> None:
    """So nobody acts on a heuristic thinking it was a census."""
    report = AuditReport(
        source="s", n_cases=1, schema_version=1, evallint_version="test",
        sections=(
            AuditSection(
                key="dataset_statistics", title="t", status=SectionStatus.RAN,
                summary="s", limitations=("l",),
                findings=(
                    _finding(recommended_action="count-based action"),
                    _finding(
                        tier=Tier.HEURISTIC, confidence="low",
                        recommended_action="threshold-based action",
                    ),
                ),
            ),
        ),
    )
    assert [f.tier for f in report.recommendations] == [
        Tier.HEURISTIC,
        Tier.OBSERVED,
    ]


def test_identical_recommended_actions_are_deduplicated() -> None:
    report = AuditReport(
        source="s", n_cases=1, schema_version=1, evallint_version="test",
        sections=(
            AuditSection(
                key="dataset_statistics", title="t", status=SectionStatus.RAN,
                summary="s", limitations=("l",),
                findings=(
                    _finding(recommended_action="same"),
                    _finding(recommended_action="same"),
                    _finding(recommended_action="different"),
                ),
            ),
        ),
    )
    assert len(report.recommendations) == 2


# ==========================================================================
# 3. "Not run" can never read as a pass
# ==========================================================================


def test_a_not_run_section_must_say_what_it_would_take() -> None:
    with pytest.raises(ValueError, match="no reason"):
        AuditSection(
            key="discrimination", title="t", status=SectionStatus.NOT_RUN,
            summary="Not assessed.",
        )


def test_a_not_run_section_cannot_carry_findings() -> None:
    with pytest.raises(ValueError, match="did not run but reports"):
        AuditSection(
            key="discrimination", title="t", status=SectionStatus.NOT_RUN,
            summary="s", not_run_reason="needs a scorer",
            findings=(_finding(section="discrimination"),),
        )


def test_an_empty_audit_reports_nothing_assessed_rather_than_nothing_wrong() -> None:
    """The whole point. Passing no analyses must not produce a clean report."""
    report = run_audit(tiny_set())
    assert len(report.not_run) == len(ANALYSIS_SECTIONS) - 1  # stats always runs
    summary = " ".join(report.executive_summary())
    assert "NOT ASSESSED" in summary
    assert "unsupported in those respects" in summary


def test_no_findings_is_reported_as_evidence_about_what_ran_only() -> None:
    clean = EvalSet(
        cases=tuple(
            EvalCase(id=f"c{i}", input=f"Distinct question {i}?", expected=f"a{i}")
            for i in range(30)
        ),
        source="clean.jsonl",
    )
    report = run_audit(clean, ground_truth=GroundTruthCheck().run(clean))
    text = " ".join(report.executive_summary())
    assert "not a clean bill of health" in text


def test_the_three_model_dependent_analyses_are_not_run_from_a_file_alone() -> None:
    keys = {s.key for s in file_only_audit().not_run}
    assert {"discrimination", "evaluator_reliability", "reproducibility"} <= keys


def test_a_partial_check_makes_its_section_partial() -> None:
    """A degraded analysis must not report as a complete one."""
    from evallint.checks.base import CheckResult

    degraded = CheckResult(
        check="redundancy", summary="ran, semantic level skipped",
        limitations=("l",), partial=("embeddings not installed",),
    )
    section = run_audit(tiny_set(), redundancy=degraded).section("redundancy")
    assert section.status is SectionStatus.PARTIAL
    assert "embeddings not installed" in section.not_run_reason


# ==========================================================================
# 4. The derived views cannot disagree with the sections
# ==========================================================================


def test_the_eleven_requested_sections_are_all_present() -> None:
    d = full_audit().as_dict()
    section_keys = {s["key"] for s in d["sections"]}
    for key, _title in SECTION_ORDER:
        assert key in d or key in section_keys, f"section {key} is missing"


def test_critical_and_warnings_partition_the_findings_exactly() -> None:
    """No finding lost, none double-counted, none reworded between views."""
    report = full_audit()
    assert len(report.critical) + len(report.informational) == len(report.findings)
    assert not set(report.critical) & set(report.informational)
    assert {f.evidence for f in report.critical} <= {
        f.evidence for f in report.findings
    }


def test_critical_findings_are_exactly_the_warning_severity_ones() -> None:
    report = full_audit()
    assert all(f.severity is Severity.WARNING for f in report.critical)
    assert all(f.severity is Severity.INFO for f in report.informational)


def test_critical_findings_are_ordered_by_cases_affected() -> None:
    counts = [f.n_cases for f in full_audit().critical]
    assert counts == sorted(counts, reverse=True)


# ==========================================================================
# 5. Findings derived from the absence of evidence
# ==========================================================================


def test_a_refused_judge_metric_becomes_a_finding() -> None:
    """The most important adapter behaviour. A metric that could not be computed
    means there is no chance-corrected evidence about the judge, and an absent
    reliability measure is not a passing one."""
    report = full_audit()
    findings = report.section("evaluator_reliability").findings
    refusals = [f for f in findings if "was not computed" in f.evidence]
    assert refusals, "no refused metric surfaced as a finding"
    for finding in refusals:
        # A refusal is a fact about the design, so it is OBSERVED and carries no
        # confidence -- there is no doubt that it was not computed.
        assert finding.tier is Tier.OBSERVED
        assert finding.confidence is None
        assert "not a passing one" in finding.explanation


def test_an_unidentifiable_variance_source_becomes_a_finding() -> None:
    """Unmeasured is not zero. Reporting 0.0 would claim a stability that was
    never measured."""
    findings = full_audit().section("reproducibility").findings
    unidentifiable = [f for f in findings if "not identifiable" in f.evidence]
    assert unidentifiable
    assert all(f.tier is Tier.OBSERVED for f in unidentifiable)
    assert any("rather than zero" in f.explanation for f in unidentifiable)


def test_an_underpowered_comparison_is_a_finding_but_a_real_difference_is_not() -> None:
    """A detected difference is a RESULT, not a defect. Only the verdicts that
    mean "this eval cannot answer the question" become findings, and the verdict
    is the analyser's own label so no threshold is invented here."""
    ids = [f"c{i}" for i in range(6)]
    underpowered = compare_models(
        {
            "a": {c: i < 3 for i, c in enumerate(ids)},
            "b": {c: i < 4 for i, c in enumerate(ids)},
        }
    )
    section = run_audit(tiny_set(), model_comparison=underpowered).section(
        "statistical_reliability"
    )
    verdicts = {p.verdict for p in underpowered.pairs}
    if verdicts & {"underpowered", "no_information"}:
        assert section.findings
        assert "absence of evidence" in section.findings[0].limitation
    else:  # pragma: no cover - guards the fixture, not the code
        pytest.skip(f"fixture produced {verdicts}, not an underpowered verdict")


def test_a_clear_difference_produces_no_statistical_finding() -> None:
    """The quiet half of the pair: 40 cases with a large, consistent gap should
    not be flagged as a problem with the eval."""
    ids = [f"c{i}" for i in range(40)]
    clear = compare_models(
        {
            "weak": {c: False for c in ids},
            "strong": {c: True for c in ids},
        }
    )
    section = run_audit(tiny_set(), model_comparison=clear).section(
        "statistical_reliability"
    )
    assert not section.findings
    assert section.status is SectionStatus.RAN


def test_the_design_effect_finding_states_the_interval_inflation() -> None:
    estimate = estimate_from_clusters([4, 3], 20, threshold=0.85)
    findings = run_audit(tiny_set(), effective_size=estimate).section(
        "statistical_reliability"
    ).findings
    assert findings
    assert "too narrow" in findings[0].explanation
    assert "not a statistical effective sample size" in findings[0].limitation


# ==========================================================================
# 6. Leakage has a stated home
# ==========================================================================


def test_leakage_findings_appear_under_ground_truth_and_the_grouping_is_stated() -> None:
    """The eleven requested sections have no leakage entry, so it shares the
    ground-truth section. Both answer "can this case be scored as intended".
    Dropping the analyser instead would have lost information the CLI reports.
    """
    # Two of eight leak, not all eight. The leakage check deliberately collapses
    # a 100%-affected type into one "this is probably the intended task"
    # finding, so an all-leaking fixture would exercise that path instead of the
    # per-case one. And the answers are over 12 characters because the detector
    # ignores shorter ones -- a short string appearing in the input is a
    # coincidence waiting to happen rather than evidence.
    leaky = EvalSet(
        cases=tuple(
            EvalCase(
                id=f"c{i}",
                input=(
                    f"The recommended treatment is antihistamine {i} daily. "
                    "What is the recommended treatment?"
                    if i < 2
                    else f"What is the recommended treatment for condition {i}?"
                ),
                expected=f"antihistamine {i} daily",
            )
            for i in range(8)
        ),
        source="leaky.jsonl",
    )
    section = run_audit(leaky, leakage=LeakageCheck().run(leaky)).section(
        "ground_truth"
    )
    assert "leakage" in section.summary.lower()
    assert section.findings
    assert all(f.section == "ground_truth" for f in section.findings)


def test_ground_truth_is_partial_when_only_one_of_the_two_ran() -> None:
    s = tiny_set()
    section = run_audit(s, leakage=LeakageCheck().run(s)).section("ground_truth")
    assert section.status is SectionStatus.PARTIAL
    assert "references was not analysed" in section.not_run_reason


# ==========================================================================
# 7. Renderers
# ==========================================================================


def test_json_is_parseable_and_keeps_every_case_id() -> None:
    report = full_audit()
    d = json.loads(to_json(report))
    for finding, original in zip(d["critical_findings"], report.critical, strict=True):
        assert finding["affected_cases"] == list(original.case_ids)
        assert finding["n_affected_cases"] == original.n_cases


def test_terminal_truncates_case_ids_but_never_the_not_run_list() -> None:
    """Concision is for case ids and statistics. The not-run list is the part a
    reader is most likely to mistake for a pass, so it is always complete."""
    from io import StringIO

    from rich.console import Console

    from evallint.io import load

    # The example set has a 4-case class, so max_cases=2 must truncate. tiny_set
    # does not: its largest finding covers 2 cases, so nothing would be cut and
    # the test would have passed for the wrong reason.
    report = file_only_audit(load("examples/sample_evalset.jsonl"))
    buffer = StringIO()
    render_terminal(report, console=Console(file=buffer, width=100), max_cases=2)
    text = " ".join(buffer.getvalue().split())

    assert "more)" in text  # case ids were truncated
    for section in report.not_run:
        assert section.title in text


def test_terminal_detail_flag_adds_the_three_prose_fields() -> None:
    from io import StringIO

    from rich.console import Console

    def render(detail: bool) -> str:
        buffer = StringIO()
        render_terminal(
            file_only_audit(), console=Console(file=buffer, width=100), detail=detail
        )
        return " ".join(buffer.getvalue().split())

    assert "recommended action:" not in render(False)
    assert "recommended action:" in render(True)


def test_terminal_stays_concise() -> None:
    """A bound, so a future section cannot quietly turn the concise view into
    the full one. Chosen as roughly two screens at 100 columns."""
    from io import StringIO

    from rich.console import Console

    buffer = StringIO()
    render_terminal(full_audit(), console=Console(file=buffer, width=100))
    assert len(buffer.getvalue().splitlines()) < 120


def test_html_is_self_contained_with_expandable_sections() -> None:
    markup = render_html(full_audit())
    assert markup.startswith("<!DOCTYPE html>")
    assert markup.count("<details>") >= len(ANALYSIS_SECTIONS)
    # No CDN, no script tag, no external anything: an audit artefact should
    # still open in five years, and a report that fails to render because a CDN
    # moved is worse than a plain one.
    assert "<script" not in markup
    assert "http://" not in markup and "https://" not in markup


def test_html_includes_case_level_detail() -> None:
    report = full_audit()
    markup = render_html(report)
    some_case = report.critical[0].case_ids[0]
    assert f"<code>{some_case}</code>" in markup


def test_html_escapes_case_ids_and_evidence() -> None:
    """A case id comes from a user's file, so it is untrusted input. An
    unescaped one would make the report an injection vector for anyone who
    opens it."""
    hostile = EvalSet(
        cases=(
            EvalCase(
                id="<script>alert('x')</script>",
                input="A question that is quite long so it is not a duplicate?",
                expected="a",
                label="only",
            ),
            EvalCase(id="b", input="Another distinct question here?", label="only"),
        ),
        source="<img src=x onerror=alert(1)>",
    )
    markup = render_html(
        run_audit(hostile, imbalance=ImbalanceCheck().run(hostile))
    )
    assert "<script>alert" not in markup
    assert "&lt;script&gt;" in markup
    assert "<img src=x" not in markup


def test_html_states_that_there_is_no_score() -> None:
    markup = render_html(file_only_audit()).lower()
    assert "no overall quality score" in markup


def test_every_renderer_survives_an_audit_with_nothing_assessed() -> None:
    """The degenerate input, which is also the most likely first run."""
    from io import StringIO

    from rich.console import Console

    report = run_audit(tiny_set())
    json.loads(to_json(report))
    assert render_html(report).endswith("</html>")
    render_terminal(report, console=Console(file=StringIO(), width=100))
