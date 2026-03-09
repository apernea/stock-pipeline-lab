# Stock Price Prediction Pipeline

A **stock price prediction pipeline** that demonstrates:

- **Observer pattern** — alert subscribers when predictions cross thresholds.
- **Factory pattern** — swap between models (linear regression, random forest, LSTM) via configuration.
- **Async data fetching** — pull price data from Alpha Vantage using `httpx`.
- **Data pipeline** — fetch → preprocess → train → predict → store.
- **Persistence** — store results in PostgreSQL.
- **Containerization** — run the full pipeline inside Docker as a scheduled background service.

> **Status:** Initial scaffold / skeleton — core structure and patterns set up, logic mostly stubbed.

---

## Project Goals

- **Educational:**
  - Design a small but realistic ML pipeline end-to-end.
  - Practice design patterns (Observer, Factory) in a real context.
  - Structure a Python project that can grow over time.
- **Operational:**
  - Run the prediction loop on a **5-minute schedule** in the background.
  - Package and run everything with **Docker Compose** (pipeline + PostgreSQL).
  - Make it easy to add new models or alert channels without changing core logic.

---

## High-Level Architecture

```
Alpha Vantage API
       │
       ▼
  ┌──────────┐    ┌────────────────┐    ┌──────────────┐    ┌───────────┐
  │  Fetcher  │───▶│ Preprocessing  │───▶│ Model (Factory)│───▶│ Predictor │
  │ (postgre) │    │                │    │               │    │           │
  └──────────┘    └────────────────┘    └──────────────┘    └─────┬─────┘
                                                                  │
                                                    ┌─────────────┼─────────────┐
                                                    ▼             ▼             ▼
                                              ┌──────────┐ ┌──────────┐ ┌────────────┐
                                              │ Storage   │ │ Observer │ │  Console   │
                                              │ (Postgres)│ │ (Subject)│ │  Logger    │
                                              └──────────┘ └──────────┘ └────────────┘
```

### Components

- **Fetcher (`provider/postgre.py`)** — Async data retrieval from Alpha Vantage via `httpx`, with results stored in PostgreSQL.
- **Preprocessing** — Cleans and normalizes raw price data. Creates features (returns, rolling averages, etc.) for models.
- **Model Factory** (Factory pattern) — Constructs models by name from `config/models.yaml`. Hides framework details (scikit-learn, PyTorch) behind a common `ModelInterface`.
- **Observer** (Observer pattern) — When a prediction crosses a configured threshold, the subject notifies its observers. Starts with console logging; email/webhook observers can be added later alongside a web interface.
- **Storage** — Repository layer backed by PostgreSQL (via SQLAlchemy) for raw data snapshots, model metadata, predictions, and alerts.
- **Runner** — Background loop that fetches fresh data, runs predictions, stores results, and triggers alerts every 5 minutes.

---

## Tech Stack

| Category | Choice |
|----------|--------|
| Language | Python 3.11+ |
| Data | `pandas`, `numpy` |
| Classical ML | `scikit-learn` (linear regression, random forest) |
| Deep learning | `PyTorch` (LSTM) |
| HTTP | `httpx` (async, HTTP/2) |
| Database | PostgreSQL via `SQLAlchemy` + `psycopg` |
| Validation | `pydantic` |
| Config | `pyyaml` |
| Container | Docker + Docker Compose |
| Data provider | Alpha Vantage |

---

## Repository Structure

```text
stock-pipeline-lab/
├── .container/
│   └── Dockerfile                    # Container image definition
├── config/
│   └── models.yaml                   # Model definitions and hyperparameters
├── src/
│   └── pipeline/
│       ├── __init__.py
│       ├── config.py                 # Settings: API keys, DB URL, thresholds
│       ├── preprocessing.py          # Data cleaning & feature engineering
│       ├── observer.py               # Observer pattern implementation
│       ├── pipeline.py               # Orchestrates fetch → preprocess → train → predict → store
│       ├── runner.py                 # Background loop / CLI entrypoint
│       ├── models/
│       │   ├── __init__.py
│       │   ├── factory.py            # Factory pattern for model selection
│       │   └── model.py              # Common model interface
│       └── provider/
│           ├── __init__.py
│           ├── postgre.py            # Alpha Vantage fetcher + PostgreSQL storage
│           └── common/
│               ├── __init__.py
│               └── database.py       # Base DB connection and session management
├── tests/
├── docker-compose.yml                # Pipeline + PostgreSQL services
├── pyproject.toml                    # Dependencies and project metadata
├── Makefile                          # Build, run, and test shortcuts
└── README.md
```

---

## Running the Project

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- An [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key)

### 1. Clone and set up

```bash
git clone https://github.com/alexpernea/stock-pipeline-lab.git
cd stock-pipeline-lab
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[postgres,dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Alpha Vantage API key and database URL
```

### 3. Run the pipeline once

```bash
python -m pipeline.runner --mode oneshot
```

This will:
1. Load configuration and connect to PostgreSQL.
2. Fetch latest stock data from Alpha Vantage.
3. Preprocess data and prepare features.
4. Train or load a model from the factory.
5. Generate predictions and store them.
6. Notify observers if thresholds are crossed.

### 4. Run on a schedule

```bash
python -m pipeline.runner --mode schedule
```

Fetches data and runs predictions every 5 minutes.

### 5. Run with Docker Compose

```bash
docker compose up --build
```

This starts both PostgreSQL and the pipeline service. Pass your API key via `.env` or environment variables.

---

## Design Patterns

### Factory Pattern

`ModelFactory` creates models by name using definitions from `config/models.yaml`:

| Key | Implementation |
|-----|---------------|
| `linear_regression` | scikit-learn `LinearRegression` |
| `random_forest` | scikit-learn `RandomForestRegressor` |
| `lstm` | PyTorch LSTM network |

All models implement `ModelInterface` with `train()` and `predict()` methods. Swap models by changing `default_model` in `models.yaml` — no pipeline code changes needed.

### Observer Pattern

- **Subject** — The prediction manager holds current predictions and a list of observers. When a prediction exceeds a threshold (e.g. predicted daily return > 5%), it notifies all observers.
- **Observers** implement `update(prediction_event)`:
  - `ConsoleObserver` — logs to stdout (current).
  - `EmailObserver`, `WebhookObserver` — planned for later, alongside a web interface.

---

## Configuration & Secrets

Environment variables loaded in `config.py`:

| Variable | Description |
|----------|-------------|
| `ALPHA_VANTAGE_API_KEY` | API key for data fetching |
| `DATABASE_URL` | PostgreSQL connection string |

A `.env.example` file is provided with non-secret defaults.

---

## Roadmap

- [x] Project scaffold and pattern stubs
- [x] Model configuration (`models.yaml`)
- [ ] Implement `ModelInterface` and `ModelFactory`
- [ ] Async data fetching from Alpha Vantage
- [ ] Preprocessing and feature engineering
- [ ] PostgreSQL storage layer
- [ ] Observer pattern with console logging
- [ ] Pipeline orchestration
- [ ] Background runner with scheduling
- [ ] Docker and Docker Compose setup
- [ ] LSTM model implementation
- [ ] Web interface for alerts and monitoring

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

## License

[MIT](LICENSE)
