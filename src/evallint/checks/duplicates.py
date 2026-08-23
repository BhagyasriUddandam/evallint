"""Check 2 — duplicate and near-duplicate detection.

The failure this catches: an eval advertises 500 cases, but 80 of them are the
same scenario reworded. The set looks bigger and broader than it is, and one
scenario quietly gets weighted 5x in the aggregate score.

How it works: embed every case's input with a local sentence-transformers
model, compute pairwise cosine similarity, and group cases linked above a
threshold into clusters (connected components).

The embedder is injectable. That keeps the tests fast and offline — they pass a
deterministic fake instead of downloading a model — and it lets a user swap in
their own domain-tuned embeddings. Same reasoning as the injected scoring
function in the discrimination check.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:  # annotations only: numpy ships with the [embeddings] extra
    import numpy as np

from ..schema import EvalSet
from .base import Check, CheckResult, Finding

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "DuplicateCheck",
    "Embedder",
    "MissingEmbeddingsError",
]


class MissingEmbeddingsError(ImportError):
    """The default embedder was requested but its optional extra is absent.

    Subclasses ImportError so `except ImportError` still catches it, while
    giving callers something specific to branch on.
    """

Embedder = Callable[[Sequence[str]], "np.ndarray"]

DEFAULT_MODEL = "all-MiniLM-L6-v2"
# 0.85 is a convention, but it was measured rather than guessed. On the bundled
# 20-case example, genuinely different cases top out at cosine 0.70 while the
# weakest planted duplicate pair sits at 0.81 — so any threshold in 0.70-0.81
# separates them directly, and the exact value is not load-bearing.
#
# 0.85 is deliberately ABOVE that gap, i.e. stricter than the data requires.
# One consequence is worth knowing: billing_002 and billing_003 (0.81) are not
# linked directly at 0.85; they land in the same cluster only transitively, via
# billing_001. All three planted clusters are still recovered with no false
# positives, but that one depends on connected components rather than on a
# direct edge. See LIMITATIONS.
DEFAULT_THRESHOLD = 0.85

LIMITATIONS = (
    "Similarity is computed on the 'input' field alone. Two cases with the same "
    "input but different 'expected' values may be deliberate (testing "
    "consistency, or a known ambiguity) — read a cluster before deleting from it.",
    "The threshold is a heuristic, not a decision boundary. It was checked "
    "against one small example set where duplicates and non-duplicates separate "
    "cleanly; a domain with formulaic phrasing (SQL, legal boilerplate, "
    "templated prompts) will push unrelated cases above it.",
    "Clusters are connected components: if A is similar to B and B to C, all "
    "three are grouped even when A and C are not themselves similar. The "
    "pairwise range reported for each cluster shows when this has happened.",
    "Inputs longer than the model's token limit (256 word pieces for "
    f"{DEFAULT_MODEL}) are truncated before embedding, so two long cases that "
    "differ only in their endings can look identical to this check.",
    "Every pair is compared, so time and memory grow with the square of the "
    "case count. Fine for a few thousand cases, not for a corpus.",
)


class DuplicateCheck(Check):
    """Finds semantically near-identical cases and reports them as clusters."""

    name = "duplicates"
    description = "Near-duplicate cases found by embedding similarity."

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        embedder: Embedder | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.model_name = model_name
        self._embedder = embedder
        self._injected = embedder is not None

    def run(self, eval_set: EvalSet) -> CheckResult:
        cases = list(eval_set)
        n_cases = len(cases)
        stats: dict = {
            "n_cases": n_cases,
            "threshold": self.threshold,
            "embedder": "custom" if self._injected else self.model_name,
            "n_clusters": 0,
            "n_cases_in_clusters": 0,
            "n_redundant_cases": 0,
            "effective_n_cases": n_cases,
            "max_similarity": None,
            "clusters": [],
        }

        if n_cases < 2:
            return CheckResult(
                check=self.name,
                summary=f"{n_cases} case — nothing to compare",
                stats=stats,
                limitations=LIMITATIONS,
            )

        similarity = self._similarity_matrix([c.input for c in cases])
        # The diagonal is every case matching itself. Blank it out so it can
        # never be reported as a duplicate pair.
        import numpy as np

        np.fill_diagonal(similarity, 0.0)
        stats["max_similarity"] = round(float(similarity.max()), 4)

        clusters = _connected_components(similarity >= self.threshold)
        findings = [
            self._cluster_finding([cases[i] for i in cluster], similarity, cluster)
            for cluster in clusters
        ]

        redundant = sum(len(c) - 1 for c in clusters)
        stats.update(
            n_clusters=len(clusters),
            n_cases_in_clusters=sum(len(c) for c in clusters),
            n_redundant_cases=redundant,
            effective_n_cases=n_cases - redundant,
            clusters=[
                {
                    "case_ids": [cases[i].id for i in cluster],
                    "min_similarity": round(_pairwise_min(similarity, cluster), 4),
                    "max_similarity": round(_pairwise_max(similarity, cluster), 4),
                }
                for cluster in clusters
            ],
        )

        if clusters:
            summary = (
                f"{n_cases} cases, {len(clusters)} near-duplicate cluster"
                f"{'' if len(clusters) == 1 else 's'} covering "
                f"{stats['n_cases_in_clusters']} cases — about "
                f"{stats['effective_n_cases']} distinct scenarios "
                f"({redundant / n_cases:.0%} redundant)"
            )
        else:
            summary = (
                f"{n_cases} cases, no near-duplicates above "
                f"cosine {self.threshold:.2f}"
            )

        return CheckResult(
            check=self.name,
            summary=summary,
            findings=tuple(findings),
            stats=stats,
            limitations=LIMITATIONS,
        )

    def _cluster_finding(self, members, similarity, indices) -> Finding:
        case_ids = tuple(c.id for c in members)
        # The ids live in Finding.case_ids, not in the message. The report
        # prints them from there, so repeating them here would double them up
        # in the terminal output.
        if len({_normalise(c.input) for c in members}) == 1:
            # Identical text is a certainty, not a heuristic. Say so, because
            # the user can act on it without reading the cases.
            message = f"{len(case_ids)} cases have identical input text"
        else:
            low = _pairwise_min(similarity, indices)
            high = _pairwise_max(similarity, indices)
            span = f"{low:.2f}" if low == high else f"{low:.2f}-{high:.2f}"
            message = (
                f"{len(case_ids)} near-identical cases (pairwise cosine {span})"
            )
        return Finding(message, case_ids=case_ids)

    def _similarity_matrix(self, texts: list[str]) -> np.ndarray:
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

        # float32, not float64. The output is an n x n matrix, so precision is
        # the single biggest lever on how large an eval set fits in memory:
        # at n=10,000 the matrix alone is 800MB at float64 and 400MB at float32.
        # float32 carries ~7 significant digits, and this is compared against a
        # threshold quoted to two decimal places (0.85) and reported to four, so
        # the precision is irrelevant while the memory is not.
        vectors = np.asarray(self._get_embedder()(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(
                f"embedder must return one vector per text: expected "
                f"({len(texts)}, d), got {vectors.shape}"
            )
        # Normalise here rather than trusting the embedder to do it, so cosine
        # similarity is just a dot product and a custom embedder cannot break
        # the maths by returning unnormalised vectors.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vectors / norms
        return np.clip(unit @ unit.T, -1.0, 1.0)

    def _get_embedder(self) -> Embedder:
        """Build the default model on first use, not at import time.

        Two reasons this is lazy rather than a module-level import:
          - Loading sentence-transformers costs seconds and ~90MB of model
            weights. A user running only the imbalance check never pays it.
          - sentence-transformers is an OPTIONAL extra, because it pulls in
            torch. Importing it at module scope would make `import evallint`
            fail for everyone who installed the core package, which is most
            users. Here, the cost lands only on the code path that needs it.
        """
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise MissingEmbeddingsError(
                    "The duplicate check needs sentence-transformers, which is "
                    "an optional extra because it pulls in torch (~2GB).\n\n"
                    "    pip install 'evallint[embeddings]'\n\n"
                    "Alternatively, pass your own embedder — "
                    "DuplicateCheck(embedder=my_fn) takes any callable mapping "
                    "a sequence of strings to a 2-D array, so no extra is "
                    "needed. Or skip this check entirely: "
                    "`evallint --skip-duplicates PATH`."
                ) from exc

            model = SentenceTransformer(self.model_name)
            self._embedder = lambda texts: model.encode(
                list(texts), show_progress_bar=False
            )
        return self._embedder


def _connected_components(adjacency: np.ndarray) -> list[list[int]]:
    import numpy as np

    """Group linked cases. Only groups of 2+ are returned."""
    seen = np.zeros(adjacency.shape[0], dtype=bool)
    components = []
    for start in range(adjacency.shape[0]):
        if seen[start]:
            continue
        seen[start] = True
        stack, group = [start], []
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbour in np.flatnonzero(adjacency[node]):
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(int(neighbour))
        if len(group) > 1:
            components.append(sorted(group))
    return components


def _pairs(indices: Sequence[int]) -> list[tuple[int, int]]:
    return [
        (indices[a], indices[b])
        for a in range(len(indices))
        for b in range(a + 1, len(indices))
    ]


def _pairwise_min(similarity: np.ndarray, indices: Sequence[int]) -> float:
    import numpy as np

    return min(float(similarity[i, j]) for i, j in _pairs(indices))


def _pairwise_max(similarity: np.ndarray, indices: Sequence[int]) -> float:
    import numpy as np

    return max(float(similarity[i, j]) for i, j in _pairs(indices))


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())
