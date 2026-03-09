Stock Price Prediction Pipeline
================================

This project is a **stock price prediction pipeline** designed to teach and demonstrate:

- **Observer pattern**: alert subscribers when predictions cross thresholds.
- **Factory pattern**: easily swap between models (e.g. linear regression, random forest, LSTM).
- **Async data fetching**: pull data from external APIs (e.g. Yahoo Finance / Alpha Vantage).
- **Data pipeline**: fetch → preprocess → train → predict → store.
- **Persistence**: store results in SQLite (default) or PostgreSQL.
- **Containerization**: run the full pipeline inside Docker as a background service.

> Status: **Initial scaffold / skeleton** (core structure and patterns set up, logic mostly stubbed).

---

## Project Goals

- **Educational**:
  - Understand how to design a small but realistic ML pipeline.
  - Practice design patterns (Observer, Factory) in a real context.
  - Learn how to structure a Python project that can grow over time.
- **Operational**:
  - Run the prediction loop in the **background** on a schedule.
  - Package and run everything inside **Docker** with one command.
  - Make it easy to later switch from **SQLite** to **PostgreSQL**.

---

## High-Level Architecture

- **`async data fetcher`**:
  - Uses `asyncio` + `aiohttp` (or library wrappers later) to fetch price data.
  - Supports multiple data providers (e.g. Yahoo Finance, Alpha Vantage) behind a clean interface.

- **`preprocessing`**:
  - Cleans and normalizes raw price data.
  - Creates features (returns, rolling averages, etc.) for models.

- **`model factory`** (Factory pattern):
  - Central place to construct models by name (e.g. `"linear_regression"`, `"random_forest"`, `"lstm"`).
  - Hides framework details (e.g. scikit-learn vs. PyTorch) behind a common interface.

- **`prediction + observer`** (Observer pattern):
  - When a prediction crosses a configured threshold (e.g. price change, confidence level),
    the subject notifies its observers (e.g. console logger, email, webhook, etc.).

- **`storage`**:
  - Uses a simple repository layer to store:
    - Raw data snapshots
    - Model metadata
    - Predictions and alerts
  - Starts with **SQLite** (file-based, simple to run anywhere).
  - Designed so you can swap to PostgreSQL later with minimal changes.

- **`background runner`**:
  - A loop / scheduler that:
    1. Fetches fresh data.
    2. Updates or trains models if needed.
    3. Generates predictions.
    4. Stores results and triggers alerts.
  - Intended to run inside Docker (e.g. `docker run ...`).

---

## Tech Stack (proposed)

- **Language**: Python 3.11+
- **Core libraries** (planned, may evolve):
  - `pandas` for data manipulation.
  - `scikit-learn` for classical models (linear regression, random forest).
  - (Optional later) `pytorch` or `tensorflow` / `keras` for LSTM.
  - `aiohttp` or `httpx` for async HTTP requests.
  - `SQLAlchemy` for database access (SQLite and PostgreSQL).
  - `pydantic` (optional) for data validation / settings.
- **Database**:
  - Default: **SQLite** (local file, no extra services needed).
  - Optional later: **PostgreSQL**, likely via a `docker-compose` setup.
- **Container**:
  - `Dockerfile` for building an image that:
    - Installs dependencies.
    - Copies project code.
    - Runs a background loop entrypoint.

---

## Repository Structure (initial)

Planned layout (subject to refinement as we build):

```text
design-pattern/
  README.md
  pyproject.toml   # Python dependencies (to be added)
  Dockerfile                        # Container image definition (to be added)
  src/
    stock_pipeline/
      __init__.py
      config.py                     # Settings for APIs, DB, thresholds, etc.
      data_fetcher.py               # Async fetching from Yahoo/Alpha Vantage
      preprocessing.py              # Data cleaning & feature engineering
      models/
        __init__.py
        factory.py                  # Factory pattern for model selection
        base.py                     # Common model interface
      observer.py                   # Observer pattern implementation
      storage.py                    # DB connection + repositories
      pipeline.py                   # Orchestrates fetch → preprocess → train → predict → store
      runner.py                     # Background loop / CLI entrypoint
```

At this stage, most modules are stubs with clear responsibilities and TODOs, so we can incrementally implement behavior.

---

## Running the Project (planned)

Until the core logic is implemented, commands will mostly be placeholders. The eventual flow will look like this:

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline once

```bash
python -m stock_pipeline.runner
```

This will:

1. Load configuration.
2. Fetch latest stock data.
3. Preprocess data and prepare features.
4. Train or load a model from the factory.
5. Generate predictions and store them.
6. Notify observers if thresholds are crossed.

### 4. Run inside Docker

Once the `Dockerfile` is fully set up:

```bash
docker build -t stock-pipeline .
docker run --rm stock-pipeline
```

In a later step, we can:

- Add environment variables for API keys and DB URLs.
- Add a `docker-compose.yml` to run with PostgreSQL.

---

## Design Patterns Overview

### Observer Pattern

- **Subject**:
  - A prediction or alert manager that holds:
    - Current predictions.
    - A list of observers.
  - When new predictions exceed some threshold (e.g. predicted daily return > 5%), it notifies observers.

- **Observers**:
  - Implement a common interface (e.g. `update(prediction_event)`):
    - `ConsoleObserver` – log to stdout.
    - `EmailObserver` – send an email (later).
    - `WebhookObserver` – POST to a URL (later).

The goal is to let you add/remove alert channels without changing the core prediction logic.

### Factory Pattern

- Central `ModelFactory` to create models given:
  - A model name or identifier.
  - Configuration (hyperparameters, etc.).

Example mapping (to be implemented):

- `"linear_regression"` → scikit-learn `LinearRegression`.
- `"random_forest"` → scikit-learn `RandomForestRegressor`.
- `"lstm"` → PyTorch/TensorFlow model (later).

This lets you swap models simply by changing configuration rather than editing pipeline code.

---

## Background Execution

The `runner` module will eventually:

- Use `asyncio` or a simple loop with `time.sleep` to:
  - Periodically fetch new data and run predictions.
  - Persist results and emit alerts.
- Be designed as a long-running process, ideal for running inside a Docker container.

Possible modes (future work):

- **oneshot**: run the full pipeline once and exit.
- **schedule**: run every N minutes.

---

## Configuration & Secrets

Planned approach:

- Use environment variables (e.g. `ALPHA_VANTAGE_API_KEY`, `DATABASE_URL`) loaded in `config.py`.
- Provide a `.env.example` with non-secret defaults.

---

## Open Questions / Your Preferences

To tailor the implementation to your preferences, please answer these when convenient:

1. **Language**: Are you happy with **Python** for this project, or do you prefer another language (e.g. TypeScript/Node, Java, Go)?
2. **ML stack**:
   - Are you primarily interested in **classical models** (linear regression, random forest) with scikit-learn first, and add LSTM later?
   - Or do you want LSTM / deep learning to be a first-class citizen from the beginning?
3. **Database**:
   - Do you want to start with **SQLite only**, or should we immediately integrate **PostgreSQL** via `docker-compose`?
4. **Data provider**:
   - Do you already have an **Alpha Vantage API key**, or should we start with **Yahoo Finance** (which may have fewer barriers to entry)?
5. **Scheduling**:
   - How often do you want the background pipeline to run (e.g. every 1 min, 5 min, 15 min, once per hour)?
6. **Alert channels**:
   - Initially, is **console logging** enough for alerts, or do you want email / webhooks wired from the start?

Once you confirm these, I’ll:

- Finalize the dependency list.
- Add the project skeleton under `src/stock_pipeline/`.
- Implement the Observer and Factory pattern scaffolding.
- Add a functional `Dockerfile` and (optionally) `docker-compose.yml`.

