.PHONY: install sync dev lint format test clean docker-build docker-up docker-down

# Install the project in editable mode with all optional dependencies
install:
	uv pip install -e ".[postgres,lstm,dev]"

# Sync dependencies from pyproject.toml (fast, uses uv resolver)
sync:
	uv sync --all-extras

# Install dev dependencies only
dev:
	uv pip install -e ".[dev]"

# Run linter
lint:
	ruff check src/ tests/

# Auto-format code
format:
	black src/ tests/

# Run tests
test:
	pytest tests/ -v

# Remove build artifacts and caches
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Build Docker image
docker-build:
	docker build -f .container/Dockerfile -t stock-pipeline .

# Start pipeline + PostgreSQL
docker-up:
	docker compose up --build -d

# Stop all services
docker-down:
	docker compose down
