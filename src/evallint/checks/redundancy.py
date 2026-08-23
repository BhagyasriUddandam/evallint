"""Semantic redundancy analysis — how many distinct scenarios an eval contains.

Supersedes the narrower `DuplicateCheck`, which compares input text with a
single cosine threshold. That check is still exported and unchanged; this one is
what the CLI runs.

## Five levels, four of which need no threshold at all

The instruction "do not rely on one cosine threshold as ground truth" is taken
literally: only ONE of the five detectors uses cosine similarity. The others
are deterministic and exact, so they cannot be tuned into the wrong answer.

    level        mechanism                               certainty
    exact        byte-identical comparison text          DUPLICATE, exact
    normalized   identical after case/space/punct fold   DUPLICATE, exact
    template     identical after masking numbers and     DUPLICATE, exact
                 quoted spans -- catches "what is 5+3"
                 vs "what is 12+7", which cosine may
                 rate as merely similar
    scenario     cosine >= threshold AND the same         SIMILAR, heuristic
                 expected answer
    semantic     cosine >= threshold                      SIMILAR, heuristic

Levels are ordered by strength. A pair is reported at the strongest level that
matches it, so an exact duplicate is never merely "semantic".

## Duplicate is not the same claim as similar

  DUPLICATE  exact / normalized / template. Established by comparison, not by
             a threshold. Moving any parameter cannot change the answer.
  SIMILAR    scenario / semantic. Depends on an embedding model and a
             threshold, both of which are choices. Reported as heuristic.

The distinction is in the output, the stats, and the severity: a deterministic
duplicate is a WARNING, a heuristic similarity is INFO unless it carries enough
aggregate weight to matter on its own.

## Nothing here says a case is invalid

Similar cases are frequently deliberate. Consistency tests use near-identical
inputs on purpose. Template families are how a benchmark covers an operation
across many operands. Regression suites repeat a scenario that once broke.
What this check reports is **how much weight one scenario carries in the
aggregate score** — a fact about the arithmetic of averaging, not a judgement
about any case. The word "invalid" does not appear in any finding, and a test
asserts that.

## Threshold provenance

`threshold` defaults to 0.85, measured on the bundled 20-case example where
unrelated cases top out at cosine 0.70 and the weakest planted paraphrase sits
at 0.81. That is ONE small set, which is weak evidence, and it is why four of
the five detectors do not depend on it. The check also reports how many
similarity pairs it would find at 0.80, 0.85, 0.90 and 0.95 — if the count is
stable across that range the threshold is not load-bearing for your data, and
if it swings you have been told so rather than left to assume.

## Works without the embeddings extra

The three deterministic levels need no model, so this check runs and reports
useful results in a core install. `levels_run` states which detectors were
available, so an absent embedder is visible rather than silently narrowing the
audit.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:  # annotations only: numpy ships with the [embeddings] extra
    import numpy as np

from ..schema import EvalCase, EvalSet
from .base import Check, CheckResult, Finding, Severity

class SemanticSetTooLargeError(RuntimeError):
    """The similarity matrix for this many cases would not fit in memory.

    Deliberately NOT a subclass of ImportError: nothing is missing, and a caller
    branching on install problems must not confuse the two. `run()` catches it
    explicitly so the deterministic levels still report, and the run is marked
    partial rather than clean.
    """


__all__ = [
    "DEFAULT_THRESHOLD",
    "HIGH_WEIGHT_SHARE",
    "MEDIUM_WEIGHT_SHARE",
    "MIN_SKELETON_TOKENS",
    "CompareFields",
    "RedundancyCheck",
    "RedundancyLevel",
    "SENSITIVITY_THRESHOLDS",
]

#: Measured on the bundled example: unrelated cases top out at 0.70, the
#: weakest planted paraphrase sits at 0.81. One small set is weak evidence,
#: which is why only one of five detectors depends on it and why the
#: sensitivity curve below is always reported.
DEFAULT_THRESHOLD = 0.85

#: Thresholds the sensitivity curve is evaluated at, to show whether the choice
#: of threshold is load-bearing for a given dataset.
SENSITIVITY_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)

#: A skeleton with fewer than this many non-placeholder tokens is too generic to
#: be evidence of a shared template: "what is <NUM>" would collapse every
#: arithmetic question in a benchmark into one family. Four is the smallest
#: value at which the bundled example and the four real public datasets produce
#: no template false positives.
MIN_SKELETON_TOKENS = 4

#: Most pairs one hash bucket may materialise before falling back to a star.
#: 200_000 pairs is roughly a 640-member family and about 20 MiB of dict, which
#: is the point where the exact pair set stops being worth its memory: nobody
#: acts on the 200_001st pair of a template family, and cluster membership is
#: identical either way.
MAX_BUCKET_PAIRS = 200_000

#: Most cases the semantic level will build a similarity matrix for. The matrix
#: is n^2 float32, so 20000 cases is 1.6 GB and 50000 is 10 GB -- an OOM kill
#: rather than a slow run. Refusing with the arithmetic is more useful than
#: dying, and the deterministic levels still run.
MAX_SEMANTIC_CASES = 15_000

#: Weight-share bands for reporting aggregate risk. CONVENTIONS, not derived
#: quantities, and reported alongside the raw share so a reader can apply their
#: own. The rationale for 10%: a cluster controlling a tenth of the score can
#: move the headline by more than the difference most model comparisons claim,
#: so it can decide a ranking on its own. The principled way to set this is
#: against the eval's own resolution -- the discrimination check's minimum
#: detectable effect -- and that link is not yet automated.
HIGH_WEIGHT_SHARE = 0.10
MEDIUM_WEIGHT_SHARE = 0.05


class RedundancyLevel(str, Enum):
    """How two cases were found to be redundant, strongest first."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    TEMPLATE = "template"
    SCENARIO = "scenario"
    SEMANTIC = "semantic"

    @property
    def is_duplicate(self) -> bool:
        """True for the deterministic levels, where no threshold is involved."""
        return self in (
            RedundancyLevel.EXACT,
            RedundancyLevel.NORMALIZED,
            RedundancyLevel.TEMPLATE,
        )


#: Strength order. A pair is reported at the first level that matches.
LEVEL_ORDER = (
    RedundancyLevel.EXACT,
    RedundancyLevel.NORMALIZED,
    RedundancyLevel.TEMPLATE,
    RedundancyLevel.SCENARIO,
    RedundancyLevel.SEMANTIC,
)


class CompareFields(str, Enum):
    """Which fields form the comparison text.

    INPUT reproduces the older input-only behaviour and remains available on
    request. INPUT_EXPECTED is the default: two cases with the same question
    and different reference answers are a contradiction rather than a
    duplicate, and collapsing them hides that.
    """

    INPUT = "input"
    INPUT_EXPECTED = "input+expected"
    ALL = "input+expected+metadata"


LIMITATIONS = (
    "SIMILAR IS NOT INVALID. Near-identical cases are often deliberate: "
    "consistency tests reuse an input on purpose, template families cover an "
    "operation across many operands, and regression suites repeat a scenario "
    "that once broke. This check reports how much weight one scenario carries "
    "in an averaged score. It does not tell you to delete anything.",
    "The exact, normalized and template levels are deterministic and cannot be "
    "changed by tuning. The scenario and semantic levels depend on an embedding "
    "model AND a threshold, both of which are choices — treat them as evidence "
    "to look at, not as measurements.",
    "Template detection masks numbers and quoted spans only. It does NOT "
    "recognise named entities, so 'flight from Paris to Rome' and 'flight from "
    "Berlin to Madrid' are not detected as one template. Nor does it know "
    "whether a template family is deliberate coverage or accidental bulk.",
    "Similarity is computed on whichever fields you selected. With the default "
    "input+expected, two cases with the same question and DIFFERENT answers are "
    "not grouped — which is correct, because that is a ground-truth "
    "contradiction rather than redundancy, but it means this check will not "
    "report it. Compare on 'input' alone if you want those grouped.",
    "Clusters are connected components: if A is redundant with B and B with C, "
    "all three are grouped even when A and C are not themselves close. The "
    "minimum pairwise similarity for each cluster is reported so this is "
    "visible.",
    "Weight-share risk bands (10% high, 5% medium) are conventions with no "
    "statistical derivation. The raw share is reported so you can apply your "
    "own. The principled anchor is your eval's own resolution — a cluster that "
    "can move the score by more than the smallest difference the eval can "
    "detect is the one that matters — and this check does not compute that.",
    "Inputs longer than the embedding model's token limit are truncated before "
    "encoding, so two long cases differing only in their endings can look "
    "identical to the semantic levels.",
    "Every pair is compared, so semantic analysis costs time and memory "
    "proportional to the square of the case count.",
)

Embedder = Callable[[Sequence[str]], "np.ndarray"]

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
_QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{1,80})[\"'“”‘’]")


def normalise(text: str) -> str:
    """Case-fold, collapse whitespace, drop punctuation.

    Deliberately aggressive: this level exists to catch the same text written
    with different spacing, capitalisation or trailing punctuation, which is
    what a spreadsheet round trip or a copy-paste produces.
    """
    return " ".join(_WORD.findall(str(text).lower()))


def skeleton(text: str) -> str | None:
    """Mask numbers and quoted spans, leaving the template.

    "What is 5 + 3?" and "What is 12 + 7?" both become "what is <num> <num>",
    so they are recognised as one template. Cosine similarity may or may not
    rate those as duplicates depending on the model; the skeleton comparison is
    exact and cannot be tuned.

    Returns None when the skeleton has fewer than MIN_SKELETON_TOKENS real
    tokens, because a skeleton that is mostly placeholders would group unrelated
    cases: without the floor, "what is <num>" collapses every arithmetic
    question in a benchmark into a single family.
    """
    masked = _QUOTED.sub(" <quoted> ", str(text))
    masked = _NUMBER.sub(" <num> ", masked)
    tokens = normalise(masked).split()
    real = [t for t in tokens if t not in ("num", "quoted")]
    if len(real) < MIN_SKELETON_TOKENS:
        return None
    return " ".join(tokens)


class _Union:
    """Union-find, so heterogeneous redundancy edges merge into one cluster."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, a: int) -> int:
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            buckets.setdefault(self.find(i), []).append(i)
        return [sorted(g) for g in buckets.values() if len(g) > 1]


class RedundancyCheck(Check):
    """Finds redundant cases at five levels and reports aggregate weighting."""

    name = "redundancy"
    description = "Redundant cases, and how much weight each scenario carries."

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        compare: CompareFields | str = CompareFields.INPUT_EXPECTED,
        embedder: Embedder | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        semantic: bool = True,
        high_weight_share: float = HIGH_WEIGHT_SHARE,
        medium_weight_share: float = MEDIUM_WEIGHT_SHARE,
    ) -> None:
        """
        Args:
            threshold: Cosine at or above which two cases are SIMILAR. Affects
                only the two heuristic levels; the three deterministic ones
                ignore it.
            compare: Which fields form the comparison text. Defaults to
                input+expected. Pass ``"input"`` for the older input-only
                behaviour.
            embedder: Any callable mapping texts to a 2-D array. Injected for
                the same reason as the scorer: no provider lock-in, and tests
                need no model.
            semantic: Set False to run only the deterministic levels — no
                model, no download, no quadratic cost.
            high_weight_share / medium_weight_share: Aggregate-weight bands.
                Conventions; the raw share is always reported.
        """
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if not 0 < medium_weight_share <= high_weight_share <= 1:
            raise ValueError(
                "weight shares must satisfy 0 < medium <= high <= 1, got "
                f"medium={medium_weight_share}, high={high_weight_share}"
            )
        self.threshold = threshold
        self.compare = CompareFields(compare)
        self.model_name = model_name
        self.semantic = semantic
        self.high_weight_share = high_weight_share
        self.medium_weight_share = medium_weight_share
        self._embedder = embedder
        self._injected = embedder is not None

    # -- comparison text ------------------------------------------------
    def comparison_text(self, case: EvalCase) -> str:
        """The text used for comparison, per the selected fields."""
        parts = [str(case.input)]
        if self.compare in (CompareFields.INPUT_EXPECTED, CompareFields.ALL):
            parts.append("" if case.expected is None else str(case.expected))
        if self.compare is CompareFields.ALL and case.metadata:
            # Sorted for determinism: dict order must not change the verdict.
            parts.extend(
                f"{k}={case.metadata[k]}" for k in sorted(case.metadata)
            )
        return "\n".join(parts)

    def run(self, eval_set: EvalSet) -> CheckResult:
        cases = list(eval_set)
        n = len(cases)
        texts = [self.comparison_text(c) for c in cases]

        levels_run = [
            RedundancyLevel.EXACT.value,
            RedundancyLevel.NORMALIZED.value,
            RedundancyLevel.TEMPLATE.value,
        ]
        stats: dict[str, Any] = {
            "n_cases": n,
            "compare": self.compare.value,
            "threshold": self.threshold,
            "levels_run": levels_run,
            "semantic_available": False,
            "embedder": None,
            "n_clusters": 0,
            "n_cases_in_clusters": 0,
            "n_redundant_cases": 0,
            "effective_n_cases": n,
            "n_by_level": {level.value: 0 for level in LEVEL_ORDER},
            "n_duplicate_clusters": 0,
            "n_similar_clusters": 0,
            "max_cluster_weight_share": 0.0,
            "threshold_sensitivity": {},
            "truncated_buckets": [],
            "clusters": [],
        }

        if n < 2:
            return CheckResult(
                check=self.name,
                summary=f"{n} case — nothing to compare",
                stats=stats,
                limitations=LIMITATIONS,
            )

        # --- pair relations, strongest level wins -------------------------
        pair_level: dict[tuple[int, int], RedundancyLevel] = {}
        truncated: list[dict[str, Any]] = []

        def record(i: int, j: int, level: RedundancyLevel) -> None:
            key = (i, j) if i < j else (j, i)
            existing = pair_level.get(key)
            if existing is None or LEVEL_ORDER.index(level) < LEVEL_ORDER.index(
                existing
            ):
                pair_level[key] = level

        for level, keyfn in (
            (RedundancyLevel.EXACT, lambda idx: texts[idx]),
            (RedundancyLevel.NORMALIZED, lambda idx: normalise(texts[idx])),
            (RedundancyLevel.TEMPLATE, lambda idx: skeleton(texts[idx])),
        ):
            buckets: dict[Any, list[int]] = {}
            for idx in range(n):
                key = keyfn(idx)
                if key:  # skeleton() returns None when too generic
                    buckets.setdefault(key, []).append(idx)
            for members in buckets.values():
                size = len(members)
                n_pairs = size * (size - 1) // 2
                if n_pairs <= MAX_BUCKET_PAIRS:
                    for a in range(size):
                        for b in range(a + 1, size):
                            record(members[a], members[b], level)
                    continue
                # A single bucket can be the whole dataset: mask the numbers out
                # of a templated eval and 5000 cases collapse to one skeleton,
                # which is 12.5 million pairs and about 1.5 GiB. Measured, not
                # hypothetical -- and at 20000 cases it is an OOM kill.
                #
                # Clusters are CONNECTED COMPONENTS, so linking every member to
                # the first produces the identical component with size-1 edges
                # instead of size*(size-1)/2. Cluster membership is therefore
                # exactly preserved; what is lost is the per-pair detail, and
                # that loss is recorded rather than silent.
                first = members[0]
                for other in members[1:]:
                    record(first, other, level)
                truncated.append(
                    {
                        "level": level.value,
                        "bucket_size": size,
                        "pairs_not_materialised": n_pairs - (size - 1),
                    }
                )

        similarity = None
        if self.semantic:
            try:
                similarity = self._similarity_matrix(texts)
            except (ImportError, SemanticSetTooLargeError) as exc:
                # ONLY unavailability and an unaffordable matrix degrade
                # gracefully -- two named, expected conditions. A bare `except
                # Exception` here swallowed a KeyError from a mis-built test
                # embedder and silently reported a narrower audit as a complete
                # one -- exactly the failure this project exists to report. Any
                # other exception propagates, and the CLI records the run as
                # incomplete.
                stats["semantic_skipped_reason"] = str(exc)
            else:
                levels_run.extend(
                    [RedundancyLevel.SCENARIO.value, RedundancyLevel.SEMANTIC.value]
                )
                stats["semantic_available"] = True
                stats["embedder"] = "custom" if self._injected else self.model_name
                expected_keys = [
                    normalise("" if c.expected is None else c.expected) for c in cases
                ]
                for i in range(n):
                    for j in range(i + 1, n):
                        if similarity[i, j] >= self.threshold:
                            same_answer = expected_keys[i] == expected_keys[j]
                            record(
                                i,
                                j,
                                RedundancyLevel.SCENARIO
                                if same_answer
                                else RedundancyLevel.SEMANTIC,
                            )
                import numpy as np

                stats["threshold_sensitivity"] = {
                    f"{t:.2f}": int(
                        np.count_nonzero(np.triu(similarity >= t, k=1))
                    )
                    for t in SENSITIVITY_THRESHOLDS
                }

        for level in pair_level.values():
            stats["n_by_level"][level.value] += 1

        # --- clusters ------------------------------------------------------
        union = _Union(n)
        for (i, j) in pair_level:
            union.union(i, j)
        clusters = union.groups()

        # Group the recorded pairs by cluster in ONE pass. The previous version
        # looped over every ordered pair of members per cluster, which is
        # quadratic a second time and dominates on a large family.
        levels_by_root: dict[int, list[RedundancyLevel]] = {}
        for (i, j), level in pair_level.items():
            levels_by_root.setdefault(union.find(i), []).append(level)

        cluster_records: list[dict[str, Any]] = []
        for position, members in enumerate(clusters, start=1):
            levels = levels_by_root.get(union.find(members[0]), [])
            strongest = min(levels, key=LEVEL_ORDER.index)
            # Similarity summaries need the full pair set, so they are computed
            # only when the cluster is small enough for that to be cheap. A
            # 5000-member cluster would be 12.5 million matrix reads for three
            # numbers nobody can act on.
            sims = (
                [
                    float(similarity[a, b])
                    for a in members
                    for b in members
                    if a < b
                ]
                if similarity is not None
                and len(members) * (len(members) - 1) // 2 <= MAX_BUCKET_PAIRS
                else []
            )
            weight_share = len(members) / n
            cluster_records.append(
                {
                    "cluster": position,
                    "case_ids": [cases[i].id for i in members],
                    "size": len(members),
                    "level": strongest.value,
                    "certainty": "duplicate" if strongest.is_duplicate else "similar",
                    "levels_present": sorted({lv.value for lv in levels}),
                    "mean_similarity": (
                        round(sum(sims) / len(sims), 4) if sims else None
                    ),
                    "min_similarity": round(min(sims), 4) if sims else None,
                    "max_similarity": round(max(sims), 4) if sims else None,
                    "weight_share": weight_share,
                    "excess_weight_share": (len(members) - 1) / n,
                    "weighting_risk": self._risk(weight_share),
                }
            )

        stats["truncated_buckets"] = truncated
        redundant = sum(r["size"] - 1 for r in cluster_records)
        stats.update(
            n_clusters=len(cluster_records),
            n_cases_in_clusters=sum(r["size"] for r in cluster_records),
            n_redundant_cases=redundant,
            effective_n_cases=n - redundant,
            n_duplicate_clusters=sum(
                1 for r in cluster_records if r["certainty"] == "duplicate"
            ),
            n_similar_clusters=sum(
                1 for r in cluster_records if r["certainty"] == "similar"
            ),
            max_cluster_weight_share=max(
                (r["weight_share"] for r in cluster_records), default=0.0
            ),
            clusters=cluster_records,
        )

        partial: tuple[str, ...] = ()
        if self.semantic and not stats["semantic_available"]:
            # Ran, but with 2 of 5 detectors missing. Declared so the CLI does
            # not report a clean gate on a narrowed audit. An explicit
            # semantic=False is NOT partial: that is the caller choosing scope.
            partial = (
                "semantic and scenario levels did not run: "
                + stats.get("semantic_skipped_reason", "no embedder available"),
            )

        return CheckResult(
            check=self.name,
            summary=self._summary(stats),
            findings=tuple(self._findings(stats)),
            stats=stats,
            limitations=LIMITATIONS,
            partial=partial,
        )

    def _risk(self, weight_share: float) -> str:
        if weight_share >= self.high_weight_share:
            return "HIGH"
        if weight_share >= self.medium_weight_share:
            return "MEDIUM"
        return "LOW"

    def _summary(self, stats: dict) -> str:
        n, k = stats["n_cases"], stats["n_clusters"]
        if not k:
            scope = (
                "at all five levels"
                if stats["semantic_available"]
                else "at the three deterministic levels (no embedder)"
            )
            return f"{n} cases, no redundancy found {scope}"
        return (
            f"{n} cases, {k} redundant cluster{'' if k == 1 else 's'} covering "
            f"{stats['n_cases_in_clusters']} cases — about "
            f"{stats['effective_n_cases']} distinct scenarios "
            f"({stats['n_duplicate_clusters']} duplicate, "
            f"{stats['n_similar_clusters']} similar)"
        )

    def _findings(self, stats: dict) -> list[Finding]:
        findings: list[Finding] = []
        if not stats["semantic_available"] and self.semantic:
            findings.append(
                Finding(
                    "the semantic levels did not run, so this covers exact, "
                    "normalized and template redundancy only. Install "
                    "evallint[embeddings] or pass embedder= to include "
                    "paraphrase detection"
                    + (
                        f" ({stats['semantic_skipped_reason']})"
                        if "semantic_skipped_reason" in stats
                        else ""
                    ),
                    severity=Severity.INFO,
                )
            )

        for record in stats["clusters"]:
            findings.append(self._cluster_finding(record))

        sensitivity = stats["threshold_sensitivity"]
        if sensitivity:
            counts = list(sensitivity.values())
            spread = max(counts) - min(counts)
            note = (
                "stable, so the threshold is not load-bearing for this data"
                if spread <= max(1, min(counts))
                else "UNSTABLE, so the number of similar pairs here depends "
                "heavily on the threshold you chose"
            )
            findings.append(
                Finding(
                    "similar pairs by threshold — "
                    + ", ".join(f"{t}: {c}" for t, c in sensitivity.items())
                    + f". That is {note}",
                    severity=Severity.INFO,
                )
            )
        return findings

    def _cluster_finding(self, record: dict) -> Finding:
        """One finding per cluster, in the requested shape.

        Deterministic duplicates are a WARNING because the redundancy is
        certain. Heuristic similarity is INFO unless the cluster carries enough
        aggregate weight that it could move the headline on its own — at which
        point the weighting is worth attention regardless of how the grouping
        was found. Neither wording says the cases are invalid.
        """
        size = record["size"]
        level = record["level"]
        certainty = record["certainty"]
        risk = record["weighting_risk"]

        descriptor = {
            "exact": "identical",
            "normalized": "identical after normalising case, spacing and punctuation",
            "template": "built from the same template with different numbers or quoted values",
            "scenario": "the same scenario worded differently, with the same expected answer",
            "semantic": "semantically similar",
        }[level]

        mean = record["mean_similarity"]
        similarity_note = (
            f" Mean pairwise similarity {mean:.2f}." if mean is not None else ""
        )
        span = ""
        if record["min_similarity"] is not None and size > 2:
            span = (
                f" Range {record['min_similarity']:.2f}-"
                f"{record['max_similarity']:.2f}."
            )

        message = (
            f"cluster {record['cluster']}: {size} cases are {descriptor} "
            f"[{certainty.upper()}].{similarity_note}{span} They receive "
            f"{record['weight_share']:.0%} of the aggregate weight where one "
            f"scenario would receive {1 / max(1, size) * record['weight_share']:.0%}"
            f" — benchmark weighting risk {risk}. This may be intentional; "
            f"consistency tests and template families are legitimate"
        )

        severity = (
            Severity.WARNING
            if certainty == "duplicate" or risk == "HIGH"
            else Severity.INFO
        )
        return Finding(message, severity=severity, case_ids=tuple(record["case_ids"]))

    # -- embeddings ------------------------------------------------------
    def _similarity_matrix(self, texts: list[str]) -> np.ndarray:
        from .duplicates import MissingEmbeddingsError

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised in a subprocess
            raise MissingEmbeddingsError(
                    "semantic redundancy needs numpy, which ships with the "
                    "same optional extra as the embedding model because the "
                    "similarity matrix is built with it.\n\n"
                    "    pip install 'evallint[embeddings]'\n\n"
                    "The exact, normalized and template levels run without it."
            ) from exc

        n = len(texts)
        if n > MAX_SEMANTIC_CASES:
            gib = (n * n * 4) / 1024**3
            raise SemanticSetTooLargeError(
                f"semantic redundancy needs an {n}x{n} similarity matrix, "
                f"which is {gib:.1f} GiB of float32 — more than this is willing "
                f"to allocate (limit {MAX_SEMANTIC_CASES} cases). The exact, "
                "normalized and template levels still run. To compare a set "
                "this large, sample it first or raise MAX_SEMANTIC_CASES "
                "knowing the cost."
            )
        vectors = np.asarray(self._get_embedder()(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(
                f"embedder must return one vector per text: expected "
                f"({len(texts)}, d), got {vectors.shape}"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vectors / norms
        matrix = np.clip(unit @ unit.T, -1.0, 1.0)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from .duplicates import MissingEmbeddingsError

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise MissingEmbeddingsError(
                    "semantic redundancy needs sentence-transformers, which is "
                    "an optional extra because it pulls in torch (~2GB).\n\n"
                    "    pip install 'evallint[embeddings]'\n\n"
                    "The exact, normalized and template levels run without it. "
                    "Or pass your own embedder, or set semantic=False."
                ) from exc
            model = SentenceTransformer(self.model_name)
            self._embedder = lambda texts: model.encode(
                list(texts), show_progress_bar=False
            )
        return self._embedder
