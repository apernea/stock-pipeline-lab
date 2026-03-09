"""Tests for APIProvider data transformation logic."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.provider.api_provider import APIProvider


@pytest.fixture
def api():
    return APIProvider(base_url="https://fake.api/query", api_key="test_key")


DAILY_RESPONSE = {
    "Meta Data": {
        "1. Information": "Daily Prices",
        "2. Symbol": "IBM",
    },
    "Time Series (Daily)": {
        "2026-03-06": {
            "1. open": "256.4400",
            "2. high": "259.3999",
            "3. low": "252.2100",
            "4. close": "258.8500",
            "5. volume": "6234402",
        },
        "2026-03-05": {
            "1. open": "249.3200",
            "2. high": "260.3800",
            "3. low": "249.0000",
            "4. close": "256.5500",
            "5. volume": "9899962",
        },
    },
}

SENTIMENT_RESPONSE = {
    "items": "2",
    "feed": [
        {
            "title": "IBM beats expectations",
            "time_published": "20260309T145028",
            "source": "Benzinga",
            "overall_sentiment_score": 0.25,
            "overall_sentiment_label": "Somewhat_Bullish",
            "ticker_sentiment": [
                {
                    "ticker": "IBM",
                    "relevance_score": "0.85",
                    "ticker_sentiment_score": "0.30",
                    "ticker_sentiment_label": "Somewhat_Bullish",
                }
            ],
        },
        {
            "title": "Tech stocks rally",
            "time_published": "20260309T120000",
            "source": "Reuters",
            "overall_sentiment_score": 0.10,
            "overall_sentiment_label": "Neutral",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.90",
                    "ticker_sentiment_score": "0.15",
                    "ticker_sentiment_label": "Neutral",
                }
            ],
        },
    ],
}


@pytest.mark.asyncio
async def test_fetch_stock_data_transforms_correctly(api):
    with patch.object(api, "_get", new_callable=AsyncMock, return_value=DAILY_RESPONSE):
        rows = await api.fetch_stock_data("IBM")

    assert len(rows) == 2

    row = next(r for r in rows if r["date"] == datetime.date(2026, 3, 6))
    assert row["open"] == 256.44
    assert row["high"] == 259.3999
    assert row["low"] == 252.21
    assert row["close"] == 258.85
    assert row["volume"] == 6234402


@pytest.mark.asyncio
async def test_fetch_stock_data_empty_response(api):
    with patch.object(api, "_get", new_callable=AsyncMock, return_value={}):
        rows = await api.fetch_stock_data("IBM")

    assert rows == []


@pytest.mark.asyncio
async def test_fetch_stock_data_types(api):
    with patch.object(api, "_get", new_callable=AsyncMock, return_value=DAILY_RESPONSE):
        rows = await api.fetch_stock_data("IBM")

    for row in rows:
        assert isinstance(row["date"], datetime.date)
        assert isinstance(row["open"], float)
        assert isinstance(row["high"], float)
        assert isinstance(row["low"], float)
        assert isinstance(row["close"], float)
        assert isinstance(row["volume"], int)


@pytest.mark.asyncio
async def test_fetch_sentiment_data_transforms_correctly(api):
    with patch.object(
        api, "_get", new_callable=AsyncMock, return_value=SENTIMENT_RESPONSE
    ):
        articles = await api.fetch_sentiment_data("IBM")

    # Only the IBM article should be returned, not the AAPL one
    assert len(articles) == 1

    article = articles[0]
    assert article["title"] == "IBM beats expectations"
    assert article["source"] == "Benzinga"
    assert article["published_at"] == datetime.datetime(2026, 3, 9, 14, 50, 28)
    assert article["overall_sentiment_score"] == 0.25
    assert article["ticker_relevance_score"] == 0.85
    assert article["ticker_sentiment_score"] == 0.30
    assert article["ticker_sentiment_label"] == "Somewhat_Bullish"


@pytest.mark.asyncio
async def test_fetch_sentiment_data_skips_irrelevant_tickers(api):
    with patch.object(
        api, "_get", new_callable=AsyncMock, return_value=SENTIMENT_RESPONSE
    ):
        articles = await api.fetch_sentiment_data("MSFT")

    assert articles == []


@pytest.mark.asyncio
async def test_fetch_sentiment_data_empty_feed(api):
    with patch.object(api, "_get", new_callable=AsyncMock, return_value={"feed": []}):
        articles = await api.fetch_sentiment_data("IBM")

    assert articles == []


@pytest.mark.asyncio
async def test_api_error_raises(api):
    mock_response = MagicMock()
    mock_response.json.return_value = {"Error Message": "Invalid API call"}
    mock_response.raise_for_status.return_value = None
    api._client.get = AsyncMock(return_value=mock_response)

    with pytest.raises(ValueError, match="API error"):
        await api.fetch_stock_data("IBM")


@pytest.mark.asyncio
async def test_api_rate_limit_raises(api):
    mock_response = MagicMock()
    mock_response.json.return_value = {"Information": "API rate limit reached"}
    mock_response.raise_for_status.return_value = None
    api._client.get = AsyncMock(return_value=mock_response)

    with pytest.raises(ValueError, match="API limit"):
        await api.fetch_stock_data("IBM")
