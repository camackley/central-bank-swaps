# Central Bank Swaps — Agentic Scraping Pipeline

## What This Project Is

An agentic web scraping system that collects central bank swap line press releases, classifies them using LLMs, and extracts structured swap data into DuckDB. Built for an academic economics thesis.

**PRD:** https://www.notion.so/318184b536e580f5bd29fefdbe4885d6
**Dev Plan:** https://www.notion.so/319184b536e581ad9cede2b7723e4552

## Development Strategy

**Trunk-Based Development + TDD (Red → Green → Refactor)**

- All work merges to `main` via short-lived branches: `feat/<slice-id>`, `fix/<slice-id>`
- Every slice starts with a failing test, then minimum code to pass, then refactor
- The trunk is always green — never merge a red suite

## Project Structure

```
central-bank-swaps/
├── src/cbs/
│   ├── config/        # YAML loader, tracing, env config
│   ├── db/            # DuckDB schema, repos (press_release_repo, swap_repo, run_manager)
│   ├── llm/           # LLM provider abstraction (LangChain wrapper)
│   ├── scraper/       # PinchTab browser, agentic navigator, HTML/PDF extractors
│   └── pipeline/      # translator, classifier, extractor, orchestrator, backfill, incremental
├── tests/
│   ├── unit/          # Fast, no network, mocked LLM/browser
│   ├── integration/   # @pytest.mark.integration — may hit real services
│   ├── benchmarks/    # Accuracy measurement (precision/recall)
│   ├── fixtures/      # html/, pdf/, llm_responses/, config/
│   └── conftest.py    # DuckDB in-memory fixture (fresh per test)
├── config/
│   └── banks.yaml     # Central bank URLs + per-bank settings
├── pyproject.toml
└── .env.example
```

## Tech Stack

- **Python 3.11+**, **DuckDB**, **LangChain + LangSmith**, **PinchTab** (browser automation)
- **LLM providers:** Anthropic / OpenAI / Gemini — swappable via config, no provider-specific logic in business code
- **Testing:** pytest, ruff (lint + format), mypy
- **PDF extraction:** PyMuPDF (primary), pdfplumber (fallback)

## Data Model (4 tables in DuckDB)

- `press_releases` — URL (UNIQUE), title, body, translation, classification, source type (html/pdf)
- `swaps` — FK to press_releases. **Two rows per bilateral swap** (one per direction/currency side)
- `scraping_runs` — Run metadata (backfill vs incremental, counts, cost, errors)
- `bank_scraping_status` — Per-bank status per run (pending/in_progress/completed/failed) for resumability

## Pipeline Flow

Navigate site (PinchTab) → Extract press release (HTML or PDF) → Translate if non-English (LLM) → Classify swap-related? (LLM) → Extract structured swap data (LLM) → Store in DuckDB

## Commands

```bash
pytest tests/unit/                    # Unit tests only (fast, no network)
pytest tests/integration/             # Integration tests (may need services)
pytest                                # All tests
ruff check src/ tests/                # Lint
ruff format --check src/ tests/       # Format check
mypy src/                             # Type check
```

## Slice Execution Rules

1. Read the dev plan in Notion to find the current slice
2. Write failing tests first (Red)
3. Write minimum code to pass (Green)
4. Refactor while green
5. All CI checks must pass before merging: pytest + ruff check + ruff format + mypy

## Key Conventions

- **Pydantic models** for all data representations and config validation
- **Thin adapters** around third-party tools (PinchTab, LLM) — mock the adapter, not the library
- **Unit tests never hit the network** — use fixtures in `tests/fixtures/`
- **DuckDB in-memory** for all DB tests — fresh connection per test via conftest fixture
- **URL-based deduplication** — press releases are unique by URL
- **Per-bank resumability** — failed runs skip completed banks on retry
- **LangSmith traces everything** — every LLM call, navigation step, classification, extraction
- Integration tests use `@pytest.mark.integration`

## Top 10 Dev Banks

Fed (US), ECB (Eurozone), BoJ (Japan), PBoC (China), BoE (UK), SNB (Switzerland), BoC (Canada), RBA (Australia), RBI (India), Banco de la República (Colombia)

## Non-Goals

- No user-facing dashboard or API — economists consume DuckDB directly via Jupyter
- No private/commercial banks — central banks only
- No real-time scraping — weekly cron is sufficient
- No complex infra — lightweight deployment
