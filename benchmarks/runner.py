"""Run the benchmark and emit results.

    python -m benchmarks.runner --split test
    python -m benchmarks.runner --split dev --semantic     # needs [embeddings]
    python -m benchmarks.runner --split test --json results/test.json

## The split rule is enforced here, not just documented

Running against DEV prints a banner saying the numbers must not be reported, and
tags the output `"reportable": false`. TEST is the reportable split, and nothing
in this repository tunes a threshold against it -- every default in `evallint`
was chosen before these fixtures existed, on real datasets and on hand-built
unit-test cases, which is the honest version of "held out": the fixtures came
after the thresholds, so they cannot have influenced them.

The one place to be careful in future: if a contributor changes a threshold
because a TEST number looked bad, TEST stops being held out and the guarantee is
gone. `benchmarks/README.md` says so, and the results file records which split
produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evallint.checks import (
    CompareFields,
    DiscriminationCheck,
    EvaluatorReliability,
    GroundTruthCheck,
    ImbalanceCheck,
    LeakageCheck,
    RedundancyCheck,
    compare_models,
)

from .fixtures import (
    Fixture,
    Split,
    ambiguity_fixture,
    discrimination_fixture,
    exact_duplicate_fixture,
    imbalance_fixtures,
    judge_fixture,
    leakage_fixture,
    noise_trials,
    semantic_duplicate_fixture,
)
from .metrics import Confusion, environment, measure, score_sets

__all__ = ["ALL_DETECTORS", "run_benchmark"]


def _all_pairs(fixture: Fixture) -> set[frozenset[str]]:
    """Every unordered case pair. The universe for pair-level detectors.

    n(n-1)/2 pairs, which is what makes the false-positive rate meaningful: a
    detector clustering everything together would score perfect recall and a
    visibly ruinous FPR.
    """
    ids = [c.id for c in fixture.eval_set]
    return {
        frozenset({ids[i], ids[j]})
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    }


def _clustered_pairs(stats: dict[str, Any], levels: set[str] | None = None) -> set[
    frozenset[str]
]:
    """Pairs the redundancy check placed in the same cluster.

    Clusters are connected components, so every within-cluster pair counts as
    predicted-redundant even when the detector never compared those two
    directly. That is the correct reading of what the tool claims: it reports the
    cluster, and a user acts on the cluster.
    """
    pairs: set[frozenset[str]] = set()
    for cluster in stats.get("clusters", ()):
        if levels is not None and cluster["level"] not in levels:
            continue
        ids = cluster["case_ids"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add(frozenset({ids[i], ids[j]}))
    return pairs


# --------------------------------------------------------------------------
# 1. Exact duplicates
# --------------------------------------------------------------------------


def bench_exact_duplicates(split: Split, **_: Any) -> dict[str, Any]:
    fixture = exact_duplicate_fixture(split)
    check = RedundancyCheck(semantic=False, compare=CompareFields.INPUT_EXPECTED)
    m = measure(lambda: check.run(fixture.eval_set), n_items=fixture.n_cases)

    universe = _all_pairs(fixture)
    predicted = _clustered_pairs(m.result.stats, levels={"exact"})
    confusion = score_sets(predicted, fixture.positive_pairs, universe)

    # The near-misses must NOT appear in an exact-level cluster. They are
    # normalised duplicates, and conflating the two loses the distinction.
    near = set(fixture.metadata["near_misses"])
    leaked = sorted(
        "|".join(sorted(pair))
        for pair in predicted
        if pair <= near
    )
    return {
        "unit": "case pair",
        "label_basis": fixture.label_basis,
        "n_cases": fixture.n_cases,
        "confusion": confusion.as_dict(),
        "timing": m.as_dict(),
        "near_miss_check": {
            "description": "four normalised-but-not-exact duplicates must not "
            "be clustered at the exact level",
            "violations": leaked,
            "passed": not leaked,
        },
    }


# --------------------------------------------------------------------------
# 2. Semantic duplicates
# --------------------------------------------------------------------------


def bench_semantic_duplicates(
    split: Split, *, semantic: bool = False, threshold: float = 0.85, **_: Any
) -> dict[str, Any]:
    fixture = semantic_duplicate_fixture(split)
    check = RedundancyCheck(
        semantic=semantic, threshold=threshold, compare=CompareFields.INPUT
    )
    m = measure(
        lambda: check.run(fixture.eval_set),
        n_items=fixture.n_cases,
        repeats=1 if semantic else 3,
    )
    universe = _all_pairs(fixture)
    predicted = _clustered_pairs(m.result.stats)
    confusion = score_sets(predicted, fixture.positive_pairs, universe)

    hard = set(fixture.metadata["hard_negatives"])
    hard_fp = sorted(
        "|".join(sorted(pair)) for pair in predicted if pair <= hard
    )
    return {
        "unit": "case pair",
        "label_basis": fixture.label_basis,
        "n_cases": fixture.n_cases,
        "semantic_level_ran": semantic and m.result.stats.get("semantic_available"),
        "threshold": threshold,
        "levels_run": m.result.stats.get("levels_run"),
        "confusion": confusion.as_dict(),
        "timing": m.as_dict(),
        "hard_negative_check": {
            "description": "three same-topic, different-question cases; any "
            "pair among them is a false positive",
            "false_positives": hard_fp,
        },
        "caveat": (
            "deterministic levels only — pass --semantic to include the "
            "embedding level, which is where recall on paraphrases comes from"
            if not semantic
            else None
        ),
        # The precision/recall trade-off across thresholds, reported as a CURVE
        # rather than a single number. This is a sensitivity analysis, NOT a way
        # to pick the threshold: 0.85 was chosen on real datasets long before
        # these fixtures existed, and selecting it here from the test split
        # would destroy the only thing that makes the split held out.
        "threshold_sensitivity": (
            _threshold_curve(fixture, universe) if semantic else {}
        ),
    }


def _threshold_curve(
    fixture: Fixture, universe: set[frozenset[str]]
) -> dict[str, Any]:
    curve = {}
    for tau in (0.75, 0.80, 0.85, 0.90, 0.95):
        check = RedundancyCheck(
            semantic=True, threshold=tau, compare=CompareFields.INPUT
        )
        stats = check.run(fixture.eval_set).stats
        confusion = score_sets(
            _clustered_pairs(stats), fixture.positive_pairs, universe
        )
        curve[f"{tau:.2f}"] = {
            "precision": confusion.as_dict()["precision"],
            "recall": confusion.as_dict()["recall"],
            "f1": confusion.as_dict()["f1"],
            "false_positive_rate": confusion.as_dict()["false_positive_rate"],
        }
    return curve


# --------------------------------------------------------------------------
# 3. Class imbalance
# --------------------------------------------------------------------------


def bench_imbalance(split: Split, **_: Any) -> dict[str, Any]:
    datasets = imbalance_fixtures(split)
    check = ImbalanceCheck()
    per_dataset = []
    tp = fp = fn = tn = 0
    total_cases = 0

    for eval_set, is_imbalanced, why in datasets:
        total_cases += len(eval_set)
        m = measure(lambda es=eval_set: check.run(es), n_items=len(eval_set))
        flagged = bool(m.result.warnings)
        per_dataset.append(
            {
                "source": eval_set.source,
                "labelled_imbalanced": is_imbalanced,
                "flagged": flagged,
                "correct": flagged == is_imbalanced,
                "why_labelled": why,
                "summary": m.result.summary,
            }
        )
        if is_imbalanced and flagged:
            tp += 1
        elif is_imbalanced:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1

    return {
        "unit": "dataset",
        "label_basis": "arithmetic, against the check's own documented criteria: "
        "ratio above 10:1 or a class below a 10% share",
        "n_datasets": len(datasets),
        "n_cases_total": total_cases,
        # F1 over five datasets would be a number with no interpretation.
        "confusion": Confusion(tp=tp, fp=fp, fn=fn, tn=tn).as_dict(with_f1=False),
        "per_dataset": per_dataset,
    }


# --------------------------------------------------------------------------
# 4. Leakage    5. Ambiguity
# --------------------------------------------------------------------------


def _case_level(
    fixture: Fixture,
    runner: Callable[[], Any],
    *,
    stats_key: str,
) -> dict[str, Any]:
    """Score a per-case detector, overall and stratified by its own confidence.

    Predictions come from the STRUCTURED per-case records in `stats`, not from
    parsing the finding prose: both checks publish `case_id` and `confidence`
    there, and a regex over a human-readable message would break the first time
    someone rewords it.

    Dataset-level findings are excluded. Both checks emit one when a type
    affects most of the set, and it names every case -- counting it would make
    recall trivially 1.0 and precision meaningless.

    The stratification matters because these detectors GRADE their own output.
    A moderate-confidence finding the tool itself describes as "often
    legitimate" is not the same claim as a high-confidence one, so scoring them
    together hides the detector's own hedging.
    """
    m = measure(runner, n_items=fixture.n_cases)
    universe = {c.id for c in fixture.eval_set}
    records = m.result.stats.get(stats_key, ())

    by_confidence: dict[str, set[str]] = {}
    by_type: dict[str, set[str]] = {}
    predicted: set[str] = set()
    for record in records:
        cid = record["case_id"]
        predicted.add(cid)
        by_confidence.setdefault(str(record.get("confidence", "unstated")), set()).add(cid)
        kind = str(
            record.get("type") or record.get("ambiguity_type") or "unstated"
        )
        by_type.setdefault(kind, set()).add(cid)

    confusion = score_sets(predicted, fixture.positives, universe)
    near = set(fixture.metadata.get("near_misses", ()))

    strata = {}
    for level, flagged in sorted(by_confidence.items()):
        strata[level] = {
            "n_flagged": len(flagged),
            **score_sets(flagged, fixture.positives, universe).as_dict(),
        }
    # Cumulative: what precision looks like if you act only on the strongest
    # findings, which is what a user with limited review time actually does.
    order = [lvl for lvl in ("high", "moderate", "low") if lvl in by_confidence]
    cumulative = {}
    accumulated: set[str] = set()
    for level in order:
        accumulated |= by_confidence[level]
        cumulative[f"{level}_and_above"] = score_sets(
            accumulated, fixture.positives, universe
        ).as_dict()

    # Per-DETECTOR-TYPE precision, which is the actionable breakdown. On the
    # leakage fixture the confidence stratification does NOT separate the true
    # from the false positives -- both come out "moderate" -- while the type
    # does: `expected_in_input` is exact, `label_in_input` carries every false
    # positive. A contributor deciding whether a detector earns its keep needs
    # the second number, not the first.
    per_type = {
        kind: {"n_flagged": len(flagged),
               **score_sets(flagged, fixture.positives, universe).as_dict()}
        for kind, flagged in sorted(by_type.items())
    }

    return {
        "unit": "case",
        "label_basis": fixture.label_basis,
        "n_cases": fixture.n_cases,
        "confusion": confusion.as_dict(),
        "by_detector_type": per_type,
        "by_confidence": strata,
        "cumulative_by_confidence": cumulative,
        "timing": m.as_dict(),
        "near_miss_check": {
            "n_near_misses": len(near),
            "false_positives_among_them": sorted(predicted & near),
        },
        "missed": sorted(fixture.positives - predicted),
    }


def bench_leakage(split: Split, **_: Any) -> dict[str, Any]:
    fixture = leakage_fixture(split)
    check = LeakageCheck()
    return _case_level(
        fixture, lambda: check.run(fixture.eval_set), stats_key="potential_leaks"
    )


def bench_ambiguity(split: Split, **_: Any) -> dict[str, Any]:
    fixture = ambiguity_fixture(split)
    check = GroundTruthCheck()
    return _case_level(
        fixture, lambda: check.run(fixture.eval_set), stats_key="findings"
    )


# --------------------------------------------------------------------------
# 6, 7, 9. Ceiling / floor / separating
# --------------------------------------------------------------------------


def bench_discrimination(split: Split, **_: Any) -> dict[str, Any]:
    fixture = discrimination_fixture(split)
    check = DiscriminationCheck(fixture.scorer, list(fixture.models))
    m = measure(
        lambda: check.run(fixture.eval_set), n_items=len(fixture.eval_set), repeats=1
    )
    observed = m.result.stats["case_classes"]
    universe = set(fixture.expected_class)

    per_class = {}
    for kind in ("ceiling", "floor", "separating", "non_monotonic"):
        actual = {cid for cid, k in fixture.expected_class.items() if k == kind}
        predicted = {cid for cid, k in observed.items() if k == kind}
        per_class[kind] = score_sets(predicted, actual, universe).as_dict()

    exact = sum(
        1 for cid, kind in fixture.expected_class.items() if observed.get(cid) == kind
    )
    return {
        "unit": "case, classified into one of four classes",
        "label_basis": "OBJECTIVE BY CONSTRUCTION: the scorer is a fixture, so "
        "which models pass which case is written down rather than judged",
        "n_cases": len(fixture.eval_set),
        "models": list(fixture.models),
        "exact_class_agreement": round(exact / len(fixture.expected_class), 4),
        "per_class": per_class,
        "timing": m.as_dict(),
        "confusions": [
            {"case": cid, "labelled": kind, "reported": observed.get(cid)}
            for cid, kind in sorted(fixture.expected_class.items())
            if observed.get(cid) != kind
        ],
    }


# --------------------------------------------------------------------------
# 8. Judge instability
# --------------------------------------------------------------------------


def bench_judge_stability(split: Split, **_: Any) -> dict[str, Any]:
    """Does the measured agreement track the constructed agreement?

    Not a classification problem, so no F1. What is checkable is monotonicity:
    as the forced-agreement share rises, chance-corrected agreement must rise
    too. A measure that did not would be reporting noise.
    """
    rows = []
    module = EvaluatorReliability()
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        fixture = judge_fixture(split, agreement_rate=rate)
        m = measure(
            lambda f=fixture: module.analyse(f.observations),
            n_items=fixture.n_items,
            repeats=1,
        )
        metrics = m.result.metrics
        kappa = metrics.get("fleiss_kappa")
        raw = metrics.get("raw_agreement")
        rows.append(
            {
                "forced_agreement_share": rate,
                "raw_agreement": _value(raw),
                "fleiss_kappa": _value(kappa),
                "kappa_undefined_reason": getattr(kappa, "unmet_reason", None),
                "n_items": fixture.n_items,
                "n_judges": fixture.n_judges,
                "timing": m.as_dict(),
            }
        )

    measured = [r["fleiss_kappa"] for r in rows]
    comparable = [v for v in measured if v is not None]
    monotonic = all(
        a <= b + 1e-9 for a, b in zip(comparable, comparable[1:], strict=False)
    )
    return {
        "unit": "judge panel",
        "label_basis": "CONSTRUCTED: the share of items on which agreement is "
        "forced is a design parameter",
        "series": rows,
        "monotonicity_check": {
            "description": "chance-corrected agreement must not fall as the "
            "constructed agreement rises",
            "passed": monotonic,
            "kappa_series": comparable,
        },
        "note": "at share 1.0 every judge uses one category on some items and "
        "kappa may be undefined — the kappa paradox. An undefined value is the "
        "correct output there, not a failure.",
    }


# --------------------------------------------------------------------------
# 10. Statistical noise
# --------------------------------------------------------------------------


def bench_statistical_noise(
    split: Split, *, trials: int = 200, **_: Any
) -> dict[str, Any]:
    """False-positive rate under a true null, and power under a real effect.

    The strongest check in the benchmark, because the target is arithmetic
    rather than opinion: with two genuinely equal models, a test at alpha = 0.05
    should call a real difference about 5% of the time. Materially above that is
    a broken test; far below it means the test is conservative, which is worth
    knowing too.
    """
    out: dict[str, Any] = {
        "unit": "trial (a pair of models over 60 paired cases)",
        "label_basis": "CONSTRUCTED: the true difference is a parameter",
        "alpha": 0.05,
        "arms": [],
    }
    for difference in (0.0, 0.05, 0.20):
        verdicts: dict[str, int] = {}
        significant = 0
        n = 0
        started_items = 0
        m = None
        for outcomes in noise_trials(
            split, n_trials=trials, true_difference=difference
        ):
            n += 1
            started_items += len(outcomes["model_a"])
            if m is None:
                m = measure(
                    lambda o=outcomes: compare_models(o, bootstrap_resamples=200),
                    n_items=len(outcomes["model_a"]),
                    repeats=1,
                )
                report = m.result
            else:
                report = compare_models(outcomes, bootstrap_resamples=200)
            pair = report.pairs[0]
            verdicts[pair.verdict] = verdicts.get(pair.verdict, 0) + 1
            significant += bool(pair.statistically_significant)

        arm = {
            "true_difference": difference,
            "n_trials": n,
            "significant_share": round(significant / n, 4),
            "verdicts": dict(sorted(verdicts.items())),
            "timing_one_trial": m.as_dict() if m else None,
        }
        if difference == 0.0:
            arm["interpretation"] = (
                "models are genuinely equal, so this share is the FALSE "
                "POSITIVE RATE and should sit near alpha = 0.05"
            )
            arm["within_tolerance_of_alpha"] = (
                arm["significant_share"] <= 0.12
            )
        else:
            arm["interpretation"] = (
                f"models differ by {difference:.0%}, so this share is POWER at "
                "60 cases — low power is a property of the sample size, not a bug"
            )
        out["arms"].append(arm)
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

ALL_DETECTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "exact_duplicates": bench_exact_duplicates,
    "semantic_duplicates": bench_semantic_duplicates,
    "class_imbalance": bench_imbalance,
    "leakage": bench_leakage,
    "ambiguous_references": bench_ambiguity,
    "ceiling_floor_discrimination": bench_discrimination,
    "judge_instability": bench_judge_stability,
    "statistical_noise": bench_statistical_noise,
}


def run_benchmark(
    split: Split = Split.TEST,
    *,
    only: list[str] | None = None,
    semantic: bool = False,
    trials: int = 200,
) -> dict[str, Any]:
    """Run every detector benchmark and return the results as a plain dict."""
    selected = only or list(ALL_DETECTORS)
    unknown = [name for name in selected if name not in ALL_DETECTORS]
    if unknown:
        raise ValueError(
            f"unknown detector(s) {unknown}. Available: {sorted(ALL_DETECTORS)}"
        )

    results = {
        name: ALL_DETECTORS[name](split, semantic=semantic, trials=trials)
        for name in selected
    }
    return {
        "split": str(split),
        # The flag exists so a stale dev run cannot be mistaken for a reportable
        # result by anything reading the file.
        "reportable": split is Split.TEST,
        "semantic_level_requested": semantic,
        "environment": environment(),
        "detectors": results,
        "limitations": LIMITATIONS,
    }


LIMITATIONS = [
    "The fixtures and the detectors have the same author. A detector can pass "
    "both splits and still fail on data neither anticipates; the real-dataset "
    "false-positive counts in benchmarks/README.md are the only numbers here "
    "not written by the person who wrote the detectors.",
    "DEV and TEST differ in seed and in surface content, which makes TEST a "
    "held-out sample from the SAME generator design — weaker than an "
    "independent benchmark. It protects against overfitting to specific "
    "strings, not against a shared blind spot in how the fixtures were "
    "conceived.",
    "Labels for semantic duplicates and ambiguous references are author "
    "judgement. Precision on those is precision against these labels, not "
    "against truth, and a second annotator would move some of them.",
    "The leakage label is definitional: it is the same rule the detector "
    "applies. A high score there measures implementation correctness and says "
    "nothing about whether verbatim containment is a good proxy for leakage.",
    "Fixture sizes are tens of cases, so a single misclassification moves "
    "precision by several points. Treat differences of a few percent as noise.",
    "python_peak_kib comes from tracemalloc and excludes numpy and torch "
    "buffers, so the embedding path is undercounted by orders of magnitude.",
    "Runtime is wall clock on one machine. Comparable between two runs there, "
    "not across machines, and not a claim about performance in general.",
    "No F1 is reported for class imbalance, judge instability or statistical "
    "noise: the first has a five-item universe, and the other two are not "
    "classification problems. An F1 for them would be uninterpretable.",
    "The benchmark measures the detectors, not the audit report that composes "
    "them. Those have their own tests.",
]


def _value(metric: Any) -> float | None:
    return None if metric is None else getattr(metric, "value", None)


def _render(payload: dict[str, Any]) -> str:
    lines = [
        f"evallint detector benchmark — split={payload['split']}",
        f"evallint {payload['environment']['evallint_version']} on "
        f"{payload['environment']['platform']}",
        "",
    ]
    if not payload["reportable"]:
        lines += [
            "!!! DEV SPLIT — these numbers are for tuning and debugging and",
            "!!! MUST NOT be reported. Run with --split test for reportable",
            "!!! results.",
            "",
        ]
    header = f"{'detector':<30} {'unit':<12} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7} {'FNR':>7} {'ms/item':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, result in payload["detectors"].items():
        confusion = result.get("confusion")
        timing = result.get("timing", {})
        if confusion:
            lines.append(
                f"{name:<30} {result['unit'][:12]:<12} "
                f"{_fmt(confusion['precision'])} {_fmt(confusion['recall'])} "
                f"{_fmt(confusion['f1'])} "
                f"{_fmt(confusion['false_positive_rate'])} "
                f"{_fmt(confusion['false_negative_rate'])} "
                f"{_fmt(timing.get('ms_per_item'), 9, 3)}"
            )
        else:
            lines.append(f"{name:<30} {result['unit'][:12]:<12} " + "  see JSON".rjust(45))
    lines.append("")

    disc = payload["detectors"].get("ceiling_floor_discrimination")
    if disc:
        lines.append(
            f"case-class agreement: {disc['exact_class_agreement']:.1%} "
            f"({len(disc['confusions'])} disagreement(s))"
        )
    noise = payload["detectors"].get("statistical_noise")
    if noise:
        for arm in noise["arms"]:
            label = (
                "false-positive rate"
                if arm["true_difference"] == 0.0
                else f"power at {arm['true_difference']:.0%}"
            )
            lines.append(
                f"{label}: {arm['significant_share']:.1%} over "
                f"{arm['n_trials']} trials"
            )
    judge = payload["detectors"].get("judge_instability")
    if judge:
        lines.append(
            f"judge agreement monotonic: "
            f"{judge['monotonicity_check']['passed']}"
        )
    lines.append("")
    lines.append("LIMITATIONS")
    lines.extend(f"  · {line}" for line in payload["limitations"])
    return "\n".join(lines)


def _fmt(value: Any, width: int = 7, places: int = 3) -> str:
    if value is None:
        return "n/a".rjust(width)
    return f"{value:.{places}f}".rjust(width)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.runner",
        description="Measure evallint's detectors against labelled fixtures.",
    )
    parser.add_argument(
        "--split", choices=[s.value for s in Split], default=Split.TEST.value,
        help="dev for tuning (not reportable), test for reported numbers",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="DETECTOR",
        help=f"run a subset. Available: {', '.join(sorted(ALL_DETECTORS))}",
    )
    parser.add_argument(
        "--semantic", action="store_true",
        help="include the embedding level of redundancy detection; needs "
        "`pip install evallint[embeddings]` and downloads a model",
    )
    parser.add_argument(
        "--trials", type=int, default=200,
        help="trials per arm for the statistical-noise benchmark",
    )
    parser.add_argument("--json", type=Path, help="also write the full results here")
    args = parser.parse_args(argv)

    payload = run_benchmark(
        Split(args.split), only=args.only, semantic=args.semantic,
        trials=args.trials,
    )
    print(_render(payload))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
