# Repository Guidelines

## Project Structure & Module Organization

- `src/geo_fixability/` contains the Python package. Current core logic lives in `src/geo_fixability/mapping.py`.
- `README.md` documents project intent and longer-term roadmap.
- `pyproject.toml` defines package metadata and runtime dependencies; `uv.lock` pins versions for reproducible installs.

## Build, Test, and Development Commands

- Install dependencies (editable): `python -m pip install -e .`
    - Uses `pyproject.toml` and is the simplest way to work on the package locally.
- If you use `uv`, sync the locked environment: `uv sync`
- There is no CLI entry point yet. Run modules directly from Python, e.g.:
    - `python -c "from geo_fixability.mapping import generate_field_spectral; print(generate_field_spectral((64,64)).shape)"`

## Coding Style & Naming Conventions

- Python 3.13+ is required (see `pyproject.toml`).
- Follow PEP 8 with 4-space indentation and descriptive, lowercase-with-underscores function names (e.g., `generate_field_spectral`).
- Keep module-level docstrings and short, focused functions; prefer explicit parameter names for scientific routines.

## Testing Guidelines

- Automated tests are not set up yet.
- When adding tests, place them under `tests/` and use `pytest`-style naming such as `test_mapping.py` and `test_generate_field_spectral`.
- Document any validation plots or numeric tolerances in the test docstring.

## Commit & Pull Request Guidelines

- The Git history currently contains only an initial commit, so no established commit convention exists.
- Use concise, imperative commit messages (e.g., "Add GRF generator"), and group related changes per commit.
- PRs should include:
    - A short summary of changes
    - How you validated (commands run or "not tested")
    - Any new dependencies or data requirements

## Security & Configuration Notes

- Do not commit large datasets; store generated or downloaded data outside the repo or under a clearly ignored path.
- Keep dependency changes in `pyproject.toml` and regenerate the lock file as needed.

## PR Protocol

1. After completing the task, run the project's test suite.
2. If tests pass, use gh pr create --fill to create a draft PR.
3. Ensure the PR description includes "Closes #X" to link the issue.

## General git tools

You have acces to the basic git tool and commands as well as the gh tool for interacting with GitHub.