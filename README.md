# Stock Price Prediction Pipeline

An end-to-end stock prediction pipeline built around a rolling-window LSTM model, async data fetching, PostgreSQL persistence, and an observer-based alert system.

---

## What it does

1. **Fetches** daily OHLCV prices and news sentiment from Alpha Vantage
2. **Engineers** 20+ technical and sentiment features (MACD, Bollinger Bands, ATR, OBV, RSI, sentiment momentum, and more)
3. **Trains** a stacked LSTM model on a configurable rolling window of recent trading days
4. **Predicts** next-day close price, return %, and direction signal
5. **Stores** predictions in PostgreSQL alongside actual closes once they're available
6. **Scores** every prediction nightly — computing MAE and direction accuracy per model
7. **Alerts** observers when predictions cross thresholds or direction flips

---

## Architecture

```
Alpha Vantage API
       │
       ▼
  APIProvider (httpx, async)
       │  stock + sentiment rows
       ▼
  PostgreSQLBackend (asyncpg pool)
       │  get_training_data()
       ▼
  PreprocessingPipeline
       │  scaled feature matrix (rolling window)
       ▼
  LSTMModel (PyTorch)
       │  predicted_close, predicted_return, direction
       ▼
  PostgreSQLBackend.insert_prediction()
       │
       ▼
  Subject.notify(PredictionEvent)
       ├── ConsoleObserver       → structured log line
       ├── ThresholdObserver     → warning when |return| ≥ threshold
       └── DirectionChangeObserver → warning on direction flip

  [nightly] score_predictions.py
       └── fills actual_close, mae, direction_correct from stock_data
```

---

## Repository Structure

```text
stocklab/
├── .container/
│   ├── Dockerfile                        # (empty — not yet containerised)
│   └── docker-compose.yml                # PostgreSQL service
├── config/
│   └── models.yaml                       # default_model setting
├── migrations/
│   ├── 001_create_stock_data.sql
│   ├── 002_create_sentiment_data.sql
│   └── 003_create_predictions.sql        # predictions + auto-scoring trigger
├── scripts/
│   ├── migrate.py                        # idempotent SQL migration runner
│   └── score_predictions.py             # nightly actual-vs-predicted scorer
├── src/pipeline/
│   ├── config.py                         # pydantic-settings (.env loader)
│   ├── observer.py                       # Observer ABC, Subject, 3 concrete observers
│   ├── pipeline.py                       # StockPipeline — full orchestration + rolling window
│   ├── runner.py                         # CLI entry point (oneshot / schedule modes)
│   ├── interfaces/
│   │   ├── model.py                      # ModelInterface ABC
│   │   └── database.py                   # DatabaseInterface ABC
│   ├── models/
│   │   ├── factory.py                    # ModelFactory + ModelType constants
│   │   ├── lstm.py                       # LSTMModel (PyTorch, fully implemented)
│   │   ├── linear_reg.py                 # stub
│   │   └── random_forest.py              # stub
│   ├── preprocessing/
│   │   └── preprocessing.py             # PreprocessingPipeline (20+ features)
│   ├── provider/
│   │   ├── api_provider.py               # Alpha Vantage client (stock + sentiment)
│   │   ├── postgre.py                    # PostgreSQLBackend (CRUD for all tables)
│   │   ├── fetcher.py                    # standalone fetch-and-store helper
│   │   └── common/database.py            # DatabaseProvider (asyncpg pool)
│   └── utils/
│       └── registry.py                   # generic Registry for factory pattern
├── tests/
│   ├── test_api_provider.py
│   └── test_postgre.py
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Tech Stack

| Category | Choice |
|----------|--------|
| Language | Python 3.11+ |
| Package manager | `uv` |
| Data | `pandas`, `numpy` |
| Deep learning | `PyTorch` (LSTM) |
| ML utilities | `scikit-learn` (preprocessing, scalers) |
| HTTP | `httpx` (async, HTTP/2) |
| Database | PostgreSQL via `asyncpg` |
| Config | `pydantic-settings` + `.env` |
| Data source | Alpha Vantage |
| Container | Docker + Docker Compose |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (for PostgreSQL)
- An [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key)

### 1. Install dependencies

```bash
git clone https://github.com/alexpernea/stocklab.git
cd stocklab
make install
```

This runs `uv sync --extra all --extra dev`, which installs all dependencies (including PyTorch and asyncpg) and registers the `pipeline` CLI command.

### 2. Configure environment

```bash
cp .env.example .env
# Fill in DATABASE_URL and API_KEY
```

### 3. Start PostgreSQL

```bash
make docker-up
```

### 4. Run migrations

```bash
make migrate
```

### 5. Run the pipeline

```bash
# Fetch, train, and predict for IBM (oneshot)
make run SYMBOLS=IBM

# Multiple symbols
make run SYMBOLS="IBM AAPL MSFT"

# Schedule mode — re-runs every 5 minutes
make run SYMBOLS=IBM MODE=schedule

# Force retrain even if a model is already saved
make retrain SYMBOLS=IBM

# Or call the CLI directly
uv run pipeline IBM AAPL --mode schedule --interval 300 --alert-threshold 0.03
```

### 6. Score predictions (run after market close)

```bash
make score
```

Fills `actual_close`, `actual_return`, `mae`, and `direction_correct` for any predictions whose `target_date` has passed.

---

## Features Engineered

| Group | Features |
|-------|----------|
| Price structure | `log_return`, `daily_range`, `price_position`, `gap` |
| Moving averages | `sma_5`, `sma_20`, `ema_12`, `ema_20` |
| MACD | `macd`, `macd_signal`, `macd_histogram` |
| Bollinger Bands | `bb_upper`, `bb_lower`, `bb_pct_b`, `bb_bandwidth` |
| Momentum | `rsi_14`, `roc_10`, `close_lag_24` |
| Volatility | `atr_14`, `volatility_20` |
| Volume | `obv`, `volume_ratio_20` |
| Sentiment | `avg_sentiment`, `article_count`, `sentiment_std`, `avg_relevance`, `sentiment_momentum`, `sentiment_dispersion` |

---

## Rolling Window Training

The model always trains on the most recent `--training-window` trading days (default: 504 ≈ 2 years). When new data is fetched, the window slides forward automatically — the oldest rows fall off and the newest are included. Retraining only triggers when the latest date in `stock_data` is ahead of the model's recorded training cutoff.

```bash
make run SYMBOLS=IBM TRAINING_WINDOW=252   # 1-year window
make run SYMBOLS=IBM TRAINING_WINDOW=0     # all available data
```

---

## Observer Pattern

`StockPipeline` extends `Subject`. After every prediction, it fires a `PredictionEvent` to all attached observers:

| Observer | Behaviour |
|----------|-----------|
| `ConsoleObserver` | Logs every prediction with direction arrow and return % |
| `ThresholdObserver` | `WARNING` when `\|predicted_return\|` ≥ threshold |
| `DirectionChangeObserver` | `WARNING` when direction flips from previous run |

Adding a new alert channel (email, Slack, webhook) is a new class that implements `Observer.update()` — no changes to the pipeline.

---

## Makefile Reference

| Target | What it does |
|--------|-------------|
| `make install` | `uv sync --extra all --extra dev` |
| `make run SYMBOLS=...` | Run the pipeline (oneshot by default) |
| `make retrain SYMBOLS=...` | Force full retrain then predict |
| `make migrate` | Apply pending SQL migrations |
| `make score` | Score unscored predictions against actuals |
| `make test` | Run the test suite |
| `make lint` | `ruff check` |
| `make format` | `black` |
| `make docker-up` | Start PostgreSQL container |
| `make docker-down` | Stop containers |

---

## Predictions Schema

Each prediction stores:

- `predicted_close`, `predicted_return`, `direction` — model outputs
- `confidence`, `lower_bound`, `upper_bound` — optional uncertainty fields
- `actual_close`, `actual_return` — filled nightly by the scoring job
- `mae`, `direction_correct` — auto-computed by a DB trigger when actuals land
- `model_name`, `model_version`, `horizon_days` — for multi-model, multi-horizon support

---

## License

[MIT](LICENSE)
