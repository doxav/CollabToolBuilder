# Repository Guidelines

## Project Structure & Module Organization
- `learn.py`: main loop for human–LLM collaborative function development.
- `functions/`, `primitives/`, `pipelines/`, `prompts/`: core building blocks used by the loop.
- `utils/`: shared helpers; `config.py` for runtime configuration (e.g., ES host/IP).
- `frontend/`: local UI (`index.html`, JS/CSS) for interacting with the loop.
- `tests/`: pytest-based suite; see `pytest.ini` for root settings.
- `requirements.txt`, `setup.py`, `Dockerfile`, `run_docker.sh`: installation and container tooling.

## Build, Test, and Development Commands
- Create env and install: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install -e .`
- Run the loop locally: `python learn.py` (edit `config.py` to point to your Elastic/Vector DB).
- Open the UI: open `frontend/index.html` in a browser.
- Run tests: `pytest -q` (use `-k name` to filter, `-x` to stop on first failure).
- Docker UI quickstart: `bash run_docker.sh` (see README for manual variants).

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, max line length ~100 characters.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Prefer type hints and concise docstrings on public functions.
- Keep modules focused; place shared logic in `utils/`; keep prompts and domain text in `prompts/`.

## Testing Guidelines
- Framework: pytest (see root `pytest.ini` and `tests/pytest.ini`).
- Layout: tests live under `tests/`, files named `test_*.py`, functions `def test_*`.
- Use fixtures/parametrize (see existing tests) and avoid external network calls; mock I/O and services.
- Run `pytest -q` before opening a PR; target meaningful coverage for changed code.

## Trace Optimizers & Dynamic Config
- Trace adapters must not pass execution context into the optimizer. Respect the Trace interfaces:
  - `backward(node, feedback)` receives only feedback text; no context dict.
  - `step()` is called without context by default. If a custom optimizer implements a richer signature (e.g., `step(context=..., targets=...)`), pass only supported args.
- Online (H1) vs Offline (H2):
  - H1: Build feedback from the immediate run (user diffs, metrics, annotations) and call `backward(...)` before `step()`; apply returned edits to the current call (`invoke_kwargs`, `system_prompt`, etc.).
  - H2: Aggregate multiple past records into feedback, call `backward(...)` then `step()`; apply persistent edits to `config.*` and `dynamic_llm_config.*` (stored under `dynamic_llm_defaults`).
- Targets define trainables: For a `type: trace` modification, `targets` constrain which parameters are trainable/optimizable for that call. Non‑targets are set non‑trainable.
- Dynamic config patching: Use the shared dotted‑path helpers in `utils/llm_utils`:
  - `set_in_dict_by_path(root, "a.b.c", value)`
  - `get_from_dict_by_path(root, "a.b.c", default)`
  Always apply `dynamic_llm_config_patch` with dotted‑path writes so nested sections (e.g., `annotations_help_trace_POST.rules.confidence.threshold`) update the real structure.
- Offline CI note: In restricted environments where a real optimizer cannot call an LLM, the adapter may synthesize simple proposals from declared bounds/current values for the declared targets. Prefer enabling real optimizers in production.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject; optional body for context (e.g., “fix: handle empty prompt”).
- Scope small and focused; update docs when behavior or UI changes.
- PRs: clear description, linked issues, reproduction steps, relevant logs/screenshots.
- CI readiness: lint locally if used, ensure tests pass (`pytest -q`), and verify `learn.py` minimal run.

## Security & Configuration Tips
- Do not commit secrets; use environment variables or `.env` (dotenv supported).
- Update `config.py` with local service endpoints (Elastic/Chroma/Neo4j) as needed.
- Large assets or checkpoints belong outside the repo or in `data/` with .gitignore entries.
