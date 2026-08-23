"""The core install must work with nothing but click and rich.

numpy was 25 MiB of a 32 MiB core install, and the only use on the
no-embeddings path was one median. It now lives in the `[embeddings]` extra, and
these tests keep it there: a stray module-level `import numpy` anywhere in the
package would silently put 25 MiB back on every install.

Run in a SUBPROCESS with numpy blocked, not by monkeypatching in-session, for
two reasons: numpy is already imported by the time the test suite starts (the
embeddings extra is installed in dev), and a module-scope import is only
observable at first import.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Injected ahead of the real import machinery so `import numpy` fails the way
#: it would on a core-only install.
BLOCK_NUMPY = textwrap.dedent(
    """
    import sys

    class _Blocker:
        def find_module(self, name, path=None):
            return self.find_spec(name, path)
        def find_spec(self, name, path=None, target=None):
            if name == "numpy" or name.startswith("numpy."):
                raise ImportError("No module named 'numpy'")
            return None

    sys.meta_path.insert(0, _Blocker())
    for mod in [m for m in sys.modules if m.startswith("numpy")]:
        del sys.modules[mod]
    """
)


def run_without_numpy(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BLOCK_NUMPY + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_the_package_imports_without_numpy() -> None:
    result = run_without_numpy(
        """
        import evallint
        import evallint.audit, evallint.cli, evallint.io, evallint.migrate
        import evallint.report, evallint.schema, evallint.validation
        import evallint.checks
        print("OK", len(evallint.__all__))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK")


@pytest.mark.parametrize(
    "check", ["ImbalanceCheck", "LeakageCheck", "GroundTruthCheck"]
)
def test_each_deterministic_check_runs_without_numpy(check: str) -> None:
    result = run_without_numpy(
        f"""
        from evallint import load
        from evallint.checks import {check}
        r = {check}().run(load("examples/sample_evalset.jsonl"))
        assert r.summary
        print("OK", len(r.findings))
        """
    )
    assert result.returncode == 0, result.stderr


def test_deterministic_redundancy_runs_without_numpy() -> None:
    """The three deterministic levels are pure Python; only the similarity
    matrix needs numpy."""
    result = run_without_numpy(
        """
        from evallint import load
        from evallint.checks import RedundancyCheck
        r = RedundancyCheck(semantic=False).run(load("examples/sample_evalset.jsonl"))
        assert r.stats["levels_run"] == ["exact", "normalized", "template"]
        assert not r.partial, r.partial
        print("OK", r.stats["n_clusters"])
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_unified_report_renders_without_numpy() -> None:
    result = run_without_numpy(
        """
        from evallint import load
        from evallint.audit import run_audit, render_html, to_json
        from evallint.checks import ImbalanceCheck, LeakageCheck
        s = load("examples/sample_evalset.jsonl")
        rep = run_audit(s, imbalance=ImbalanceCheck().run(s),
                        leakage=LeakageCheck().run(s))
        assert to_json(rep) and render_html(rep).endswith("</html>")
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_cli_runs_without_numpy() -> None:
    result = run_without_numpy(
        """
        from click.testing import CliRunner
        from evallint.cli import main
        r = CliRunner().invoke(
            main,
            ["examples/sample_evalset.jsonl", "--skip-duplicates", "--no-config",
             "--format", "json"],
        )
        assert r.exit_code == 0, r.output
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr


def test_asking_for_semantic_without_numpy_gives_install_instructions() -> None:
    """Not a bare "No module named 'numpy'", which tells a user nothing about
    why a dataset linter wants numpy."""
    result = run_without_numpy(
        """
        from evallint import load
        from evallint.checks import RedundancyCheck
        r = RedundancyCheck(semantic=True).run(load("examples/sample_evalset.jsonl"))
        assert r.partial, "must be marked partial"
        assert "evallint[embeddings]" in r.partial[0], r.partial[0]
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr


def test_numpy_is_not_a_core_dependency() -> None:
    """Pins the pyproject side, so the extra cannot drift back into core."""
    import tomllib
    from pathlib import Path

    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    core = " ".join(project["dependencies"])
    assert "numpy" not in core, f"numpy is back in core dependencies: {core}"
    assert "numpy" in " ".join(project["optional-dependencies"]["embeddings"])
    # Core stays small. Three would already be one too many.
    assert len(project["dependencies"]) == 2, project["dependencies"]
