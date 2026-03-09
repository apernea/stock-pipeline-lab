"""Tests for PostgreSQLBackend using a mocked DatabaseProvider."""

import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from pipeline.interfaces.database import DatabaseInterface
from pipeline.provider.postgre import PostgreSQLBackend


@pytest.fixture
def credentials():
    return DatabaseInterface.Credentials(
        db_user="test",
        db_pass="test",
        db_host="localhost",
        db_port=5432,
        db_name="test_db",
    )


@pytest.fixture
def backend(credentials):
    return PostgreSQLBackend(credentials)


STOCK_ROWS = [
    {
        "date": datetime.date(2026, 3, 6),
        "open": 256.44,
        "high": 259.40,
        "low": 252.21,
        "close": 258.85,
        "volume": 6234402,
    },
    {
        "date": datetime.date(2026, 3, 5),
        "open": 249.32,
        "high": 260.38,
        "low": 249.00,
        "close": 256.55,
        "volume": 9899962,
    },
]

SENTIMENT_ARTICLES = [
    {
        "published_at": datetime.datetime(2026, 3, 9, 14, 50, 28),
        "title": "IBM beats expectations",
        "source": "Benzinga",
        "overall_sentiment_score": 0.25,
        "overall_sentiment_label": "Somewhat_Bullish",
        "ticker_relevance_score": 0.85,
        "ticker_sentiment_score": 0.30,
        "ticker_sentiment_label": "Somewhat_Bullish",
    },
]


# ── Insert Tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_stock_data(backend):
    backend.execute = AsyncMock(return_value="INSERT 0 1")

    inserted = await backend.insert_stock_data("IBM", STOCK_ROWS)

    assert inserted == 2
    assert backend.execute.call_count == 2


@pytest.mark.asyncio
async def test_insert_stock_data_skips_duplicates(backend):
    # First row inserted, second is a duplicate
    backend.execute = AsyncMock(side_effect=["INSERT 0 1", "INSERT 0 0"])

    inserted = await backend.insert_stock_data("IBM", STOCK_ROWS)

    assert inserted == 1


@pytest.mark.asyncio
async def test_insert_stock_data_empty(backend):
    inserted = await backend.insert_stock_data("IBM", [])
    assert inserted == 0


@pytest.mark.asyncio
async def test_insert_sentiment_data(backend):
    backend.execute = AsyncMock(return_value="INSERT 0 1")

    count = await backend.insert_sentiment_data("IBM", SENTIMENT_ARTICLES)

    assert count == 1
    assert backend.execute.call_count == 1

    # Verify the query contains the right values
    call_args = backend.execute.call_args
    assert call_args[0][1] == "IBM"
    assert call_args[0][2] == datetime.datetime(2026, 3, 9, 14, 50, 28)
    assert call_args[0][3] == "IBM beats expectations"


@pytest.mark.asyncio
async def test_insert_sentiment_data_empty(backend):
    count = await backend.insert_sentiment_data("IBM", [])
    assert count == 0


# ── Query Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_stock_data_returns_dataframe(backend):
    backend.fetch = AsyncMock(
        return_value=[
            {
                "date": datetime.date(2026, 3, 5),
                "open": 249.32,
                "high": 260.38,
                "low": 249.00,
                "close": 256.55,
                "volume": 9899962,
            },
            {
                "date": datetime.date(2026, 3, 6),
                "open": 256.44,
                "high": 259.40,
                "low": 252.21,
                "close": 258.85,
                "volume": 6234402,
            },
        ]
    )

    df = await backend.get_stock_data("IBM")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    # Should be sorted ascending by date
    assert df.index[0] < df.index[1]


@pytest.mark.asyncio
async def test_get_stock_data_empty(backend):
    backend.fetch = AsyncMock(return_value=[])

    df = await backend.get_stock_data("IBM")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


@pytest.mark.asyncio
async def test_get_stock_data_with_limit(backend):
    backend.fetch = AsyncMock(return_value=[])

    await backend.get_stock_data("IBM", limit=10)

    query = backend.fetch.call_args[0][0]
    assert "LIMIT" in query


@pytest.mark.asyncio
async def test_get_sentiment_data_returns_dataframe(backend):
    backend.fetch = AsyncMock(
        return_value=[
            {
                "date": datetime.date(2026, 3, 9),
                "avg_sentiment": 0.30,
                "article_count": 3,
                "sentiment_std": 0.05,
                "avg_relevance": 0.85,
            },
        ]
    )

    df = await backend.get_sentiment_data("IBM")

    assert isinstance(df, pd.DataFrame)
    assert "avg_sentiment" in df.columns
    assert "article_count" in df.columns
    assert len(df) == 1


@pytest.mark.asyncio
async def test_get_sentiment_data_empty(backend):
    backend.fetch = AsyncMock(return_value=[])

    df = await backend.get_sentiment_data("IBM")

    assert df.empty
    assert "avg_sentiment" in df.columns


# ── Training Data Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_training_data_joins_stock_and_sentiment(backend):
    stock_df = pd.DataFrame(
        {
            "open": [256.44],
            "high": [259.40],
            "low": [252.21],
            "close": [258.85],
            "volume": [6234402],
        },
        index=pd.to_datetime(["2026-03-06"]),
    )
    stock_df.index.name = "date"

    sentiment_df = pd.DataFrame(
        {
            "avg_sentiment": [0.30],
            "article_count": [3],
            "sentiment_std": [0.05],
            "avg_relevance": [0.85],
        },
        index=pd.to_datetime(["2026-03-06"]),
    )
    sentiment_df.index.name = "date"

    with (
        patch.object(
            backend, "get_stock_data", new_callable=AsyncMock, return_value=stock_df
        ),
        patch.object(
            backend,
            "get_sentiment_data",
            new_callable=AsyncMock,
            return_value=sentiment_df,
        ),
    ):
        df = await backend.get_training_data("IBM")

    assert "close" in df.columns
    assert "avg_sentiment" in df.columns
    assert len(df) == 1


@pytest.mark.asyncio
async def test_get_training_data_fills_missing_sentiment(backend):
    stock_df = pd.DataFrame(
        {
            "open": [256.44],
            "high": [259.40],
            "low": [252.21],
            "close": [258.85],
            "volume": [6234402],
        },
        index=pd.to_datetime(["2026-03-06"]),
    )
    stock_df.index.name = "date"

    empty_sentiment = pd.DataFrame(
        columns=["avg_sentiment", "article_count", "sentiment_std", "avg_relevance"]
    )

    with (
        patch.object(
            backend, "get_stock_data", new_callable=AsyncMock, return_value=stock_df
        ),
        patch.object(
            backend,
            "get_sentiment_data",
            new_callable=AsyncMock,
            return_value=empty_sentiment,
        ),
    ):
        df = await backend.get_training_data("IBM")

    # Should return stock data even without sentiment
    assert len(df) == 1
    assert "close" in df.columns
