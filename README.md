# Stock Price Prediction Pipeline

A **stock price prediction pipeline** that demonstrates:

- **Observer pattern** — alert subscribers when predictions cross thresholds.
- **Factory pattern** — swap between models (linear regression, random forest, LSTM) via configuration.
- **Async data fetching** — pull price data from Alpha Vantage using `httpx`.
- **Data pipeline** — fetch → preprocess → train → predict → store.
- **Persistence** — store results in PostgreSQL.
- **Containerization** — run the full pipeline inside Docker as a scheduled background service.

> **Status:** Core infrastructure complete — data fetching, storage, preprocessing, and LSTM model are implemented. Observer pattern, pipeline orchestration, and runner are the remaining pieces.

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

- **APIProvider (`provider/api_provider.py`)** — Async Alpha Vantage client via `httpx`. Fetches daily OHLCV data and news sentiment, returning structured dicts ready for storage.
- **PostgreSQLBackend (`provider/postgre.py`)** — Async insert/query layer over an `asyncpg` connection pool. Stores and retrieves stock data, sentiment data, and joined training sets.
- **PreprocessingPipeline (`preprocessing/preprocessing.py`)** — Adds technical indicators (SMA, RSI, volatility, returns), shifts the target to next-day close, splits chronologically, and scales with `MinMaxScaler`.
- **Model Factory** (Factory pattern) — `ModelFactory` looks up model classes by name from a generic `Registry`. All models implement `ModelInterface` (`train`, `predict`, `save`, `load`, `summary`). LSTM is fully implemented; linear regression and random forest are stubbed.
- **Observer** (Observer pattern) — *Not yet implemented.* When a prediction crosses a configured threshold the subject will notify observers. Starts with console logging; email/webhook observers planned.
- **Runner** — *Not yet implemented.* Background loop that fetches fresh data, runs predictions, stores results, and triggers alerts every 5 minutes.

---

## Tech Stack

| Category | Choice |
|----------|--------|
| Language | Python 3.11+ |
| Data | `pandas`, `numpy` |
| Classical ML | `scikit-learn` (linear regression, random forest) |
| Deep learning | `PyTorch` (LSTM) |
| HTTP | `httpx` (async, HTTP/2) |
| Database | PostgreSQL via `asyncpg` |
| Validation | `pydantic` / `pydantic-settings` |
| Config | `.env` via `pydantic-settings` |
| Container | Docker + Docker Compose |
| Data provider | Alpha Vantage |

---

## Repository Structure

```text
stock-pipeline-lab/
├── .container/
│   └── Dockerfile                        # Container image definition
├── config/
│   └── models.yaml                       # Model hyperparameter definitions
├── src/
│   └── pipeline/
│       ├── config.py                     # ✅ Settings via pydantic-settings (.env)
│       ├── interfaces/
│       │   ├── model.py                  # ✅ ModelInterface ABC (train/predict/save/load/summary)
│       │   └── database.py               # ✅ DatabaseInterface ABC (connect/execute/fetch)
│       ├── utils/
│       │   └── registry.py               # ✅ Generic Registry for factory pattern
│       ├── preprocessing/
│       │   └── preprocessing.py          # ✅ PreprocessingPipeline (features, split, scaling)
│       ├── models/
│       │   ├── factory.py                # ✅ ModelFactory (Registry-backed, create/load)
│       │   ├── lstm.py                   # ✅ LSTMModel (PyTorch, fully implemented)
│       │   ├── linear_reg.py             # 🔲 LinearRegressionModel (stub)
│       │   └── random_forest.py          # 🔲 RandomForestModel (stub)
│       ├── provider/
│       │   ├── api_provider.py           # ✅ APIProvider — Alpha Vantage (stock + sentiment)
│       │   ├── postgre.py                # ✅ PostgreSQLBackend (insert/get stock & sentiment)
│       │   └── common/
│       │       └── database.py           # ✅ DatabaseProvider (asyncpg connection pool)
│       ├── observer.py                   # 🔲 Observer pattern (not yet created)
│       ├── pipeline.py                   # 🔲 Pipeline orchestration (not yet created)
│       └── runner.py                     # 🔲 Background loop / CLI entrypoint (not yet created)
├── tests/
│   ├── test_api_provider.py              # ✅ Tests for APIProvider
│   └── test_postgre.py                   # ✅ Tests for PostgreSQLBackend
├── docker-compose.yml                    # Pipeline + PostgreSQL services
├── pyproject.toml                        # Dependencies and project metadata
├── Makefile                              # Build, run, and test shortcuts
└── README.md
```

**Legend:** ✅ Implemented &nbsp;|&nbsp; 🔲 Not yet implemented

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

`ModelFactory` resolves model classes from a generic `Registry`. Each model class self-registers via a `@model_registry.register(name)` decorator:

| Key | Implementation | Status |
|-----|---------------|--------|
| `lstm` | PyTorch LSTM (two stacked layers) | ✅ Implemented |
| `linear_reg` | scikit-learn `LinearRegression` | 🔲 Stub |
| `random_forest` | scikit-learn `RandomForestRegressor` | 🔲 Stub |

All models implement `ModelInterface`: `train()`, `predict()`, `save()`, `load()`, `summary()`. Swap models by passing a different key to `ModelFactory.create()`.

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

### Done
- [x] Project scaffold and configuration (`config.py`, `models.yaml`)
- [x] `ModelInterface` and `ModelFactory` with generic `Registry`
- [x] Async data fetching from Alpha Vantage (`APIProvider`)
- [x] PostgreSQL storage layer — asyncpg pool + `PostgreSQLBackend`
- [x] Preprocessing and feature engineering (`PreprocessingPipeline`)
- [x] LSTM model — train, predict, save, load (PyTorch)
- [x] Tests for API provider and PostgreSQL backend

### Next Steps

1. **Implement `LinearRegressionModel` and `RandomForestModel`** (`models/linear_reg.py`, `models/random_forest.py`)
   Follow the same pattern as `LSTMModel`: use scikit-learn internally, pickle for save/load, register in `model_registry`.

2. **Register models in the factory** (`models/__init__.py` or each model file)
   Apply `@model_registry.register("linear_reg")` etc. so `ModelFactory.create()` works end-to-end.

3. **Implement the Observer pattern** (`pipeline/observer.py`)
   Create `Observer` ABC, `Subject` mixin, and a `ConsoleObserver`. The subject fires when a prediction crosses a configured threshold.

4. **Implement pipeline orchestration** (`pipeline/pipeline.py`)
   Wire together: `APIProvider` → `PostgreSQLBackend` → `PreprocessingPipeline` → `ModelFactory` → prediction → `Subject.notify()`.

5. **Implement the background runner** (`pipeline/runner.py`)
   CLI entrypoint with `--mode oneshot|schedule`. Use `asyncio` + `apscheduler` (or a simple sleep loop) for the 5-minute schedule.

6. **Add DB schema / migrations**
   The `stock_data` and `sentiment_data` tables are referenced in queries but no schema file or migration tool exists yet. Add an `init.sql` or Alembic setup.

7. **Docker Compose wiring**
   Ensure the pipeline service waits for PostgreSQL to be ready and passes `DATABASE_URL` / `API_KEY` from `.env`.

8. **Expand test coverage**
   Unit tests for `PreprocessingPipeline`, `LSTMModel`, and `ModelFactory`; integration tests against a test DB.

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

## License

[MIT](LICENSE)
