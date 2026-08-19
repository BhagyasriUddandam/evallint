# Publishing evallint

Releases go out through GitHub Actions using **PyPI Trusted Publishing** (OIDC).
There is no API token anywhere in the repo or in repository secrets — GitHub
mints a short-lived identity token scoped to this exact repository, workflow
file, and environment, and PyPI verifies it. Nothing long-lived exists to leak
or rotate.

## One-time setup (do this before the first run)

Register a **pending publisher** on each index. This is the only manual step.

| Field | TestPyPI value | PyPI value |
|---|---|---|
| PyPI Project Name | `evallint` | `evallint` |
| Owner | `BhagyasriUddandam` | `BhagyasriUddandam` |
| Repository name | `evallint` | `evallint` |
| Workflow name | `publish.yml` | `publish.yml` |
| Environment name | `testpypi` | `pypi` |

- TestPyPI: <https://test.pypi.org/manage/account/publishing/>
- PyPI: <https://pypi.org/manage/account/publishing/>

Every field must match exactly. A mismatch fails with a bare `403` that does
not say which field was wrong, so copy them rather than retyping.

Then create the two GitHub environments (Settings → Environments): `testpypi`
and `pypi`. Adding a required reviewer to `pypi` gives you a manual gate before
anything reaches the real index — recommended, since a PyPI version number can
never be reused.

> **The repository must be named `evallint` on GitHub.** The trusted publisher
> is bound to the repo name, and `pyproject.toml`'s project URLs already point
> at `.../evallint`. Rename it before the first publish.

## Releasing

**1. Dry run to TestPyPI.** Actions → `publish` → Run workflow → target
`testpypi`. This exercises the whole pipeline — version guard, tests, build,
metadata validation, OIDC upload — without touching the real index.

Then install what it published:

```bash
uv venv /tmp/t
uv pip install --python /tmp/t/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ evallint
/tmp/t/bin/evallint examples/sample_evalset.jsonl
```

The `--extra-index-url` is required: TestPyPI does not mirror numpy, click or
rich, so resolution fails without a fallback to real PyPI.

**2. Real release.** Bump `version` in `pyproject.toml`, add a `CHANGELOG.md`
entry, commit, then create a GitHub Release tagged `vX.Y.Z`. Publishing to PyPI
runs automatically on release.

The `build` job refuses to continue if the tag and the `pyproject.toml` version
disagree. A release tagged `v0.2.0` that ships `0.1.0` is unfixable after
upload, because PyPI never permits re-uploading a version — only yanking it.

## What CI checks on every push

`test.yml` runs three things, and the split is deliberate:

- **`core`** — ubuntu + macOS × Python 3.12/3.13, installing only what
  `pip install evallint` gives a user. It asserts torch is **absent**, so if
  `sentence-transformers` ever migrates back into `[project.dependencies]` CI
  fails instead of users discovering a 2 GB download. Roughly one second of
  test time per job.
- **`full`** — one job with the `[embeddings]` extra, so the two end-to-end
  embedding tests actually execute. It **fails if any test was skipped**: a
  silent skip would leave the embedding path untested while CI still reported
  green, which is precisely the class of quiet failure this project exists to
  catch.
- **`build`** — builds the sdist and wheel, validates metadata with
  `twine check --strict`, then installs the built **wheel** into a clean venv
  and runs it. That catches missing package data (`py.typed`), a broken console
  script, or an import that only works from the repository root — none of which
  a source-tree test run would notice.

## Gotcha worth knowing

Neither `uv sync` nor `uv run` is used in the `core` job. `uv sync` installs the
dev dependency group, and **`uv run` auto-syncs it** — and that group contains
`sentence-transformers`. Either one would pull torch into a job whose entire
purpose is proving torch is unnecessary, and the no-torch assertion would then
fail against its own setup. The job calls `.venv/bin/python` directly instead.
