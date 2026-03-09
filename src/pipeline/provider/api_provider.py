"""API provider for fetching stock and sentiment data from Alpha Vantage."""

from __future__ import annotations

import datetime
import logging

import httpx


class APIProvider:
    """Fetches stock and sentiment data from Alpha Vantage.

    Returns data as lists of dicts ready for PostgreSQLBackend.insert_* methods.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self._client = httpx.AsyncClient(http2=True, timeout=30.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> APIProvider:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _get(self, **params) -> dict:
        params["apikey"] = self.api_key
        response = await self._client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API error: {data['Error Message']}")
        if "Information" in data:
            raise ValueError(f"API limit: {data['Information']}")

        return data

    async def fetch_stock_data(
        self, symbol: str, outputsize: str = "compact"
    ) -> list[dict]:
        """Fetch daily OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "IBM").
            outputsize: "compact" (100 days) or "full" (20+ years).

        Returns:
            List of dicts with keys: date, open, high, low, close, volume.
        """
        data = await self._get(
            function="TIME_SERIES_DAILY",
            symbol=symbol,
            outputsize=outputsize,
        )

        time_series = data.get("Time Series (Daily)", {})
        rows = []
        for date_str, values in time_series.items():
            rows.append(
                {
                    "date": datetime.date.fromisoformat(date_str),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["5. volume"]),
                }
            )

        logging.debug(f"Fetched {len(rows)} daily rows for {symbol}")
        return rows

    async def fetch_sentiment_data(self, symbol: str) -> list[dict]:
        """Fetch news sentiment data for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "IBM").

        Returns:
            List of dicts with keys matching PostgreSQLBackend.insert_sentiment_data.
        """
        data = await self._get(
            function="NEWS_SENTIMENT",
            tickers=symbol,
        )

        articles = []
        for item in data.get("feed", []):
            # Find the ticker-specific sentiment
            ticker_sentiment = None
            for ts in item.get("ticker_sentiment", []):
                if ts["ticker"] == symbol:
                    ticker_sentiment = ts
                    break

            if ticker_sentiment is None:
                continue

            # Parse "20260309T145028" -> datetime
            published_at = datetime.datetime.strptime(
                item["time_published"], "%Y%m%dT%H%M%S"
            )

            articles.append(
                {
                    "published_at": published_at,
                    "title": item["title"],
                    "source": item.get("source"),
                    "overall_sentiment_score": item.get("overall_sentiment_score"),
                    "overall_sentiment_label": item.get("overall_sentiment_label"),
                    "ticker_relevance_score": float(
                        ticker_sentiment["relevance_score"]
                    ),
                    "ticker_sentiment_score": float(
                        ticker_sentiment["ticker_sentiment_score"]
                    ),
                    "ticker_sentiment_label": ticker_sentiment.get(
                        "ticker_sentiment_label"
                    ),
                }
            )

        logging.debug(f"Fetched {len(articles)} sentiment articles for {symbol}")
        return articles
