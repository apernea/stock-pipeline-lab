"""Data fetcher: orchestrates API collection and database storage."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from pipeline.interfaces.database import DatabaseInterface
from pipeline.provider.api_provider import APIProvider
from pipeline.provider.postgre import PostgreSQLBackend


def _credentials_from_uri(uri: str) -> DatabaseInterface.Credentials:
    parsed = urlparse(uri)
    return DatabaseInterface.Credentials(
        db_user=parsed.username,
        db_pass=parsed.password,
        db_host=parsed.hostname,
        db_port=parsed.port or 5432,
        db_name=parsed.path.lstrip("/"),
    )


async def fetch_and_store(
    symbol: str,
    database_url: str,
    api_base_url: str,
    api_key: str,
    outputsize: str = "compact",
) -> dict[str, int]:
    """Fetch stock and sentiment data from the API and persist to the database.

    Args:
        symbol: Ticker symbol (e.g. "IBM").
        database_url: PostgreSQL connection URI.
        api_base_url: Alpha Vantage base URL.
        api_key: Alpha Vantage API key.
        outputsize: "compact" (100 days) or "full" (20+ years).

    Returns:
        Dict with "stock_rows" and "sentiment_rows" counts inserted.
    """
    credentials = _credentials_from_uri(database_url)

    async with APIProvider(api_base_url, api_key) as api:
        stock_rows = await api.fetch_stock_data(symbol, outputsize=outputsize)
        sentiment_rows = await api.fetch_sentiment_data(symbol)

    async with PostgreSQLBackend(credentials) as db:
        stock_inserted = await db.insert_stock_data(symbol, stock_rows)
        sentiment_inserted = await db.insert_sentiment_data(symbol, sentiment_rows)

    logging.info(
        f"[{symbol}] Stored {stock_inserted}/{len(stock_rows)} stock rows, "
        f"{sentiment_inserted}/{len(sentiment_rows)} sentiment articles"
    )

    return {"stock_rows": stock_inserted, "sentiment_rows": sentiment_inserted}
