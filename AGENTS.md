# Repository Guidelines

## Project Structure & Module Organization
- `rag_test_script.py`: Main RAG pipeline (S3 → chunk → embed → query).
- `chroma_db/`: Persistent Chroma vector store (gitignored; safe to delete locally).
- `questions.txt`: Demo queries loaded at runtime (optional; has fallbacks).
- `.env` / `env.sample`: Environment variables (API keys, S3 config).
- `scratch/`: Ad‑hoc experiments and temporary assets (gitignored).

## Build, Test, and Development Commands
- Setup environment:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run demo locally:
  - `OPENAI_API_KEY=... S3_BUCKET_NAME=... python rag_test_script.py`
  - Or create `.env` from `env.sample` and run `python rag_test_script.py`.
- Regenerate dependencies lock (optional): `pip freeze > requirements.lock`

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indentation, UTF‑8.
- Use `snake_case` for files, functions, and variables; `UpperCamelCase` for classes.
- Prefer type hints (`typing`) and f‑strings. Keep functions small and single‑purpose.
- Follow existing logging style (concise prints with clear progress indicators).
- Keep persistent paths configurable (e.g., `persist_directory`), defaulting to `./chroma_db`.

## Testing Guidelines
- Framework: `pytest` (not bundled). Install with `pip install pytest`.
- Focus on pure units first:
  - `chunk_text`: boundary handling, overlap, minimal length filtering.
  - `get_collection_info`: counts and source aggregation (use a temp persistent dir).
- Mock external services for unit tests (e.g., monkeypatch `boto3`, `openai`).
- Example: `tests/test_chunking.py`, `tests/test_collection.py`; run `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: short, imperative, and scoped. Examples:
  - `feat: add batch embedding to speed up ingestion`
  - `fix: prevent infinite loop in chunking`
- PRs include:
  - Purpose and scope, screenshots or logs if output changed.
  - How to run locally (env vars, sample commands).
  - Any data or S3 prerequisites and backward‑compatibility notes.

## Security & Configuration Tips
- Never commit secrets. `.env`, `chroma_db/`, and temporary files are gitignored.
- Use `env.sample` as the template; keep real keys in `.env` only.
- Limit AWS credentials to the minimum S3 permissions required.
- Avoid uploading real PDFs to the repo; source from S3 during runs.
