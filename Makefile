.PHONY: install sync lint format test migrate score run docker-build docker-up docker-down

# ── Setup ────────────────────────────────────────────────────────────────────

# Install everything (postgres + lstm + dev) and register the `pipeline` command
install:
	uv sync --extra all --extra dev

# Install only core + dev (no torch/asyncpg) — faster for CI or non-ML work
install-dev:
	uv sync --extra dev

# ── Quality ──────────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/

format:
	uv run black src/ tests/

test:
	uv run pytest tests/ -v

# ── Pipeline ─────────────────────────────────────────────────────────────────

# Run the pipeline for one or more symbols (oneshot by default)
#   make run SYMBOLS="IBM AAPL"
#   make run SYMBOLS="IBM" MODE=schedule INTERVAL=300
SYMBOLS          ?= IBM
MODE             ?= oneshot
INTERVAL         ?= 300
MODEL            ?= lstm
TRAINING_WINDOW  ?= 504

run:
	uv run pipeline $(SYMBOLS) --mode $(MODE) --interval $(INTERVAL) --model $(MODEL) --training-window $(TRAINING_WINDOW)

# Force a full retrain then predict
retrain:
	uv run pipeline $(SYMBOLS) --force-retrain

# Apply pending database migrations
migrate:
	uv run python scripts/migrate.py

# Score unscored predictions against actual closes from stock_data
score:
	uv run python scripts/score_predictions.py

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -f .container/Dockerfile -t stock-pipeline .

docker-up:
	docker compose -f .container/docker-compose.yml up -d

docker-down:
	docker compose -f .container/docker-compose.yml down
