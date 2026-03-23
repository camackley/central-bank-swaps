# Central Bank Swaps

An agentic scraping pipeline that collects central bank swap line press releases, classifies them using LLMs, and extracts structured swap data into DuckDB. Built for academic economics research.

## Architecture

The pipeline is an **LLM-driven agent** that autonomously navigates central bank websites, discovers press releases, and processes them through a multi-stage pipeline:

```
                         Playwright MCP
                        (Claude controls
                         headless browser)
                              |
                              v
    +---------------------------------------------------------+
    |                    NAVIGATION LAYER                      |
    |                                                          |
    |  Strategy 1: Playwright MCP (preferred)                  |
    |    Claude uses browser_navigate + browser_snapshot        |
    |    to read the accessibility tree (~2-5 KB per page)     |
    |    and extract press release URLs directly.              |
    |                                                          |
    |  Strategy 2: HTML extraction (fallback)                  |
    |    Python Playwright navigates, gets rendered HTML,       |
    |    sends cleaned HTML to LLM for URL extraction.         |
    |                                                          |
    |  Strategy 3: Discovery agent (no known URL)              |
    |    ReAct agent navigates from homepage to find the       |
    |    press releases section autonomously.                  |
    +---------------------------------------------------------+
                              |
                     List of press release URLs
                              |
                              v
    +---------------------------------------------------------+
    |                   PROCESSING LAYER                       |
    |                                                          |
    |  For each press release URL:                             |
    |                                                          |
    |  1. Fetch content (Playwright for HTML, httpx for PDF)   |
    |  2. Extract text (BeautifulSoup / PyMuPDF)               |
    |  3. Detect language                                      |
    |  4. Translate to English (if needed)          [LLM call] |
    |  5. Classify: is this about a swap agreement? [LLM call] |
    |  6. Extract structured swap data              [LLM call] |
    |  7. Store in DuckDB (2 rows per bilateral swap)          |
    +---------------------------------------------------------+
                              |
                              v
                     DuckDB (per-run file)
                   ready for Jupyter analysis
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 18+ (for Playwright MCP)
- Claude Code CLI authenticated (for `claude-code` provider)

### Install

```bash
git clone <repo-url>
cd central-bank-swaps

# Install Python dependencies
uv sync --dev

# Install Playwright browsers
uv run playwright install chromium

# Install Playwright MCP (already configured in .mcp.json)
npx @playwright/mcp@latest --help   # verify it's accessible
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Environment variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key |
| `OPENAI_API_KEY` | Yes* | OpenAI API key |
| `GOOGLE_API_KEY` | Yes* | Google AI API key |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (`true`) |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `CBS_BROWSER_PROFILE` | No | Path to persistent browser profile for bot-detection cookies |

*At least one LLM provider key, or use `claude-code` provider with Claude Code CLI.

### Run the Pipeline

```bash
# Full pipeline — all 10 banks, backfill mode
uv run python -u -m cbs.pipeline \
  --provider claude-code \
  --model claude-opus-4-6 \
  --max-pages 2

# Single bank
uv run python -u -m cbs.pipeline \
  --banks fed \
  --provider claude-code \
  --model claude-opus-4-6 \
  --max-pages 2

# Multiple specific banks
uv run python -u -m cbs.pipeline \
  --banks england japan canada \
  --provider claude-code \
  --model claude-opus-4-6 \
  --max-pages 2

# Incremental mode (only new press releases)
uv run python -u -m cbs.pipeline \
  --mode incremental \
  --provider claude-code \
  --model claude-opus-4-6

# Resume a failed run
uv run python -u -m cbs.pipeline \
  --resume <run-uuid>

# With persistent browser profile (helps with bot detection)
CBS_BROWSER_PROFILE=$HOME/.cbs/browser_profile \
  uv run python -u -m cbs.pipeline \
  --provider claude-code \
  --model claude-opus-4-6 \
  --max-pages 2
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config/banks.yaml` | Bank configuration file |
| `--banks` | all | Filter banks by name/country (partial match) |
| `--provider` | `anthropic` | LLM provider: `claude-code`, `anthropic`, `openai`, `google-genai` |
| `--model` | `claude-sonnet-4-6` | LLM model name |
| `--max-pages` | `5` | Max listing pages to paginate per bank |
| `--mode` | `backfill` | `backfill` (full history) or `incremental` (new only) |
| `--resume` | — | Resume an existing run by UUID |
| `--schedule` | — | Run every N days (blocks forever) |
| `--classify-model` | — | Override model for classification stage |
| `--extract-model` | — | Override model for extraction stage |
| `--translate-model` | — | Override model for translation stage |

### Output

Each run creates a DuckDB file in `runs/run_<uuid>.duckdb` with 4 tables:

| Table | Description |
|-------|-------------|
| `press_releases` | URL, title, body, translation, classification, source type |
| `swaps` | Structured swap data (2 rows per bilateral swap, one per currency side) |
| `scraping_runs` | Run metadata (mode, counts, errors) |
| `bank_scraping_status` | Per-bank status for resumability |

Open the `.duckdb` file in DBeaver, Jupyter, or any DuckDB client to query the data.

## How the Agent Navigates (Playwright MCP)

The navigation layer uses **Playwright MCP** — Microsoft's Model Context Protocol server that gives Claude direct browser control.

When the pipeline processes a bank:

1. **Claude receives a prompt** like "navigate to this URL and extract press release URLs"
2. **Claude calls MCP tools** (`browser_navigate`, `browser_snapshot`, `browser_click`) to control a headless Chromium
3. **The accessibility tree** (structured YAML, ~2-5 KB) tells Claude what's on the page — links, headings, buttons, pagination
4. **Claude identifies press releases** from the page structure and returns a JSON array of URLs
5. **For pagination**, Claude clicks "Next" using element refs from the snapshot and repeats

This approach works because:
- The accessibility tree is tiny (~2-5 KB vs ~150 KB of raw HTML)
- Claude reasons about page structure semantically, not by parsing HTML
- React/SPA content is captured (Playwright waits for `networkidle`)
- Pagination works naturally (Claude clicks, snapshots, extracts)

**Anti-detection**: The MCP browser uses a custom config (`playwright-mcp-config.json`) with realistic user agent, viewport, and `navigator.webdriver` override to avoid bot detection.

**Fallback**: If MCP fails (timeout, site blocks), the pipeline falls back to HTML extraction via the Python Playwright adapter.

## Adding a New Bank

Add an entry to `config/banks.yaml`:

```yaml
- name: Banco Central do Brasil
  country: Brazil
  homepage_url: https://www.bcb.gov.br
  press_releases_url: https://www.bcb.gov.br/en/pressdetail  # optional
  page_load_timeout: 60
  wait_strategy: networkidle        # networkidle | domcontentloaded | load
  wait_for_selector: "article"      # optional CSS selector to wait for
```

If `press_releases_url` is set, the agent navigates there directly. Otherwise, it starts from the homepage and discovers the press releases section autonomously.

## Development

```bash
# Unit tests (fast, no network)
uv run pytest tests/unit/

# Lint + format
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy src/
```

## Project Structure

```
central-bank-swaps/
├── src/cbs/
│   ├── config/          # Bank config loader, tracing, env
│   ├── db/              # DuckDB schema, repositories, run manager
│   ├── llm/             # LLM provider abstraction + ClaudeCodeChatModel
│   ├── scraper/         # PlaywrightBrowserAdapter, navigator, extractors
│   └── pipeline/        # Orchestrator, bank processor, translator,
│                        #   classifier, extractor, backfill, incremental
├── tests/
│   ├── unit/            # 290+ tests, no network
│   ├── integration/     # @pytest.mark.integration
│   └── fixtures/        # Sample HTML, PDFs, LLM responses
├── config/
│   └── banks.yaml       # Central bank definitions
├── .mcp.json            # Playwright MCP server config
├── playwright-mcp-config.json   # Anti-detection browser settings
├── playwright-mcp-init.js       # webdriver property override
└── docs/
    └── ORIGINAL_README.md       # Previous version of this README
```

## License

Academic thesis project. Not currently published under an open-source license.
