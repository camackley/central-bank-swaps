# Central Bank Swaps

An agentic web scraping pipeline that collects central bank swap line press releases, classifies them using LLMs, and extracts structured swap data into DuckDB. Built to power academic economics research on bilateral central bank swap agreements.

## Why This Exists

Researchers tracking central bank swap lines currently face a painful manual process: visiting dozens of central bank websites, hunting for press releases, copying data into spreadsheets, and translating content from multiple languages. This project automates the entire workflow — from discovering press releases to producing a queryable analytical database ready for Jupyter notebooks.

## How It Works

The pipeline processes each central bank through five stages:

```
Navigate site → Extract press release → Translate (if needed) → Classify (swap?) → Extract structured data
     ↓                  ↓                      ↓                     ↓                      ↓
  PinchTab          HTML / PDF              LLM call             LLM call              LLM call
  browser           extraction              to English          yes / no             → DuckDB
```

- **Navigate**: An LLM-driven agent explores central bank websites using PinchTab (browser automation), or goes directly to a known press releases URL if configured.
- **Extract**: Pulls clean text, title, date, and language from HTML pages or PDFs.
- **Translate**: Non-English press releases are translated via LLM.
- **Classify**: Determines whether a press release is about a swap agreement.
- **Extract swaps**: For swap-related releases, extracts structured data (parties, currencies, amounts, dates, terms) into the database. Each bilateral swap produces two directional rows — one per currency side.

The system supports two modes:
- **Backfill**: Historical scrape going back to 2008 (the financial crisis era when swap lines became prominent).
- **Incremental**: Weekly cron job that picks up only new press releases.

Per-bank resumability ensures that if a run fails partway through, completed banks are never re-processed.

## Data Model

Four tables in DuckDB:

| Table | Purpose |
|---|---|
| `press_releases` | URL, title, body (original + English), classification, source type |
| `swaps` | Structured swap data linked to press releases (two rows per bilateral agreement) |
| `scraping_runs` | Run metadata — type, counts, cost, errors |
| `bank_scraping_status` | Per-bank status per run for resumability |

## Tech Stack

- **Python 3.11+**
- **DuckDB** — analytical database, zero infrastructure, consumed directly from Python/Jupyter
- **LangChain + LangSmith** — LLM orchestration and observability (every call is traced)
- **PinchTab** — token-efficient browser automation with anti-bot features
- **LLM providers** — Anthropic, OpenAI, or Google Gemini — swappable via config
- **PyMuPDF / pdfplumber** — PDF text extraction

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An API key for at least one LLM provider (Anthropic, OpenAI, or Google)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd central-bank-swaps

# Install dependencies (using uv)
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

### Configuration

1. **Environment variables** — copy the example and fill in your keys:

```bash
cp .env.example .env
```

```env
# At least one LLM provider key is required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...

# Optional: enable LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=central-bank-swaps
```

2. **Bank configuration** — the target central banks are defined in `config/banks.yaml`. The top 10 development banks are pre-configured:

| Bank | Country |
|---|---|
| Federal Reserve | US |
| European Central Bank | Eurozone |
| Bank of Japan | Japan |
| People's Bank of China | China |
| Bank of England | UK |
| Swiss National Bank | Switzerland |
| Bank of Canada | Canada |
| Reserve Bank of Australia | Australia |
| Reserve Bank of India | India |
| Banco de la República | Colombia |

Adding a new bank requires only a YAML entry — no code changes.

## Development

### Running Tests

```bash
# Unit tests (fast, no network, mocked LLM/browser)
pytest tests/unit/

# Integration tests (may hit real services)
pytest tests/integration/

# All tests
pytest
```

### Linting & Formatting

```bash
ruff check src/ tests/          # Lint
ruff format --check src/ tests/  # Format check
ruff format src/ tests/          # Auto-format
```

### Type Checking

```bash
mypy src/
```

## Project Structure

```
central-bank-swaps/
├── src/cbs/
│   ├── config/        # YAML bank loader, tracing, env config
│   ├── db/            # DuckDB schema and data access layer
│   ├── llm/           # LLM provider abstraction (LangChain wrapper)
│   ├── scraper/       # Browser automation, HTML/PDF extractors
│   └── pipeline/      # Translator, classifier, extractor, orchestrator
├── tests/
│   ├── unit/          # Fast, deterministic, no network
│   ├── integration/   # May hit real services (@pytest.mark.integration)
│   ├── fixtures/      # Sample HTML, PDFs, LLM responses, configs
│   └── conftest.py    # DuckDB in-memory fixture (fresh per test)
├── config/
│   └── banks.yaml     # Central bank URLs and per-bank settings
├── pyproject.toml
└── .env.example
```

## Current Status

The project is in active development. The foundational infrastructure is in place:

- Project skeleton and tooling
- DuckDB schema (all 4 tables)
- Bank configuration loader with Pydantic validation
- LLM provider abstraction (swappable Anthropic / OpenAI / Gemini)
- LangSmith tracing (opt-in)

The pipeline stages (scraping, translation, classification, extraction) and the backfill/incremental orchestrators are the next milestones.

## License

This project is part of an academic thesis and is not currently published under an open-source license.
