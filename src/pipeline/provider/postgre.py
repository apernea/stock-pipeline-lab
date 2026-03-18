"""PostgreSQL-based data provider for accessing DB linked to this project, made for retrieving and storing data into it."""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.interfaces.database import DatabaseInterface
from pipeline.provider.common.database import DatabaseProvider


class PostgreSQLBackend(DatabaseProvider):
    """PostgreSQL backend for storing and retrieving stock and sentiment data."""

    def __init__(self, credentials: DatabaseInterface.Credentials):
        super().__init__(credentials)

    async def get_symbols(self) -> list[str]:
        """Return all distinct ticker symbols present in stock_data."""
        rows = await self.fetch("SELECT DISTINCT symbol FROM stock_data ORDER BY symbol")
        return [r["symbol"] for r in rows]

    async def insert_stock_data(self, symbol: str, rows: list[dict]) -> int:
        """Bulk insert OHLCV rows into stock_data, skipping duplicates.

        Args:
            symbol: Ticker symbol (e.g. "IBM").
            rows: List of dicts with keys: date, open, high, low, close, volume.

        Returns:
            Number of rows inserted.
        """
        if not rows:
            return 0

        query = """
            INSERT INTO stock_data (symbol, date, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (symbol, date) DO NOTHING
        """

        inserted = 0
        for row in rows:
            result = await self.execute(
                query,
                symbol,
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            )
            if result == "INSERT 0 1":
                inserted += 1

        logging.debug(f"Inserted {inserted}/{len(rows)} stock rows for {symbol}")
        return inserted

    async def get_stock_data(
        self, symbol: str, limit: int | None = None
    ) -> pd.DataFrame:
        """Retrieve OHLCV data for a symbol as a DataFrame.

        Args:
            symbol: Ticker symbol.
            limit: Maximum number of rows (most recent first).

        Returns:
            DataFrame indexed by date with open, high, low, close, volume columns.
        """
        query = """
            SELECT date, open, high, low, close, volume
            FROM stock_data
            WHERE symbol = $1
            ORDER BY date DESC
        """
        args: list = [symbol]

        if limit is not None:
            query += " LIMIT $2"
            args.append(limit)

        rows = await self.fetch(query, *args)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df

    async def insert_sentiment_data(self, symbol: str, articles: list[dict]) -> int:
        """Bulk insert sentiment data for each symbol available in DB.

        Args:
            symbol: Ticker symbol.
            articles: List of dicts with keys matching the sentiment_data columns:
                published_at, title, source, overall_sentiment_score,
                overall_sentiment_label, ticker_relevance_score,
                ticker_sentiment_score, ticker_sentiment_label.

        Returns:
            Number of rows inserted.
        """
        if not articles:
            return 0

        query = """
            INSERT INTO sentiment_data (
                symbol, published_at, title, source,
                overall_sentiment_score, overall_sentiment_label,
                ticker_relevance_score, ticker_sentiment_score, ticker_sentiment_label
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        count = 0
        for article in articles:
            await self.execute(
                query,
                symbol,
                article["published_at"],
                article["title"],
                article.get("source"),
                article.get("overall_sentiment_score"),
                article.get("overall_sentiment_label"),
                article.get("ticker_relevance_score"),
                article.get("ticker_sentiment_score"),
                article.get("ticker_sentiment_label"),
            )
            count += 1

        logging.debug(f"Inserted {count} sentiment rows for {symbol}")
        return count

    async def get_sentiment_data(
        self, symbol: str, limit: int | None = None
    ) -> pd.DataFrame:
        """Retrieve sentiment data for a symbol, aggregated by date.

        Aggregates per-article scores into daily averages, counts, and std dev
        so the result aligns with stock_data on date.

        Args:
            symbol: Ticker symbol.
            limit: Maximum number of days to return.

        Returns:
            DataFrame indexed by date with avg_sentiment, article_count,
            sentiment_std, and avg_relevance columns.
        """
        query = """
            SELECT
                published_at::date AS date,
                AVG(ticker_sentiment_score) AS avg_sentiment,
                COUNT(*) AS article_count,
                STDDEV(ticker_sentiment_score) AS sentiment_std,
                AVG(ticker_relevance_score) AS avg_relevance
            FROM sentiment_data
            WHERE symbol = $1
            GROUP BY published_at::date
            ORDER BY date DESC
        """
        args: list = [symbol]

        if limit is not None:
            query += " LIMIT $2"
            args.append(limit)

        rows = await self.fetch(query, *args)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "avg_sentiment",
                    "article_count",
                    "sentiment_std",
                    "avg_relevance",
                ]
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df

    async def insert_prediction(self, prediction: dict) -> int:
        """Insert a single prediction row, skipping if it already exists.

        Args:
            prediction: Dict with keys:
                symbol, model_name, model_version, prediction_date, target_date,
                horizon_days, predicted_close, predicted_return, direction,
                and optionally: confidence, lower_bound, upper_bound.

        Returns:
            1 if inserted, 0 if the row already existed (unique conflict).
        """
        query = """
            INSERT INTO predictions (
                symbol, model_name, model_version,
                prediction_date, target_date, horizon_days,
                predicted_close, predicted_return, direction,
                confidence, lower_bound, upper_bound
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT ON CONSTRAINT uq_prediction DO NOTHING
        """
        result = await self.execute(
            query,
            prediction["symbol"],
            prediction["model_name"],
            prediction.get("model_version", "unknown"),
            prediction["prediction_date"],
            prediction["target_date"],
            prediction.get("horizon_days", 1),
            prediction["predicted_close"],
            prediction["predicted_return"],
            prediction["direction"],
            prediction.get("confidence"),
            prediction.get("lower_bound"),
            prediction.get("upper_bound"),
        )
        return 1 if result == "INSERT 0 1" else 0

    async def get_predictions(
        self,
        symbol: str,
        model_name: str | None = None,
        horizon_days: int = 1,
        unscored_only: bool = False,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve predictions for a symbol, ordered by target_date descending.

        Args:
            symbol: Ticker symbol.
            model_name: Filter to a specific model. None returns all models.
            horizon_days: Prediction horizon to filter on (default: 1).
            unscored_only: If True, return only rows where actual_close is NULL.
            limit: Maximum number of rows to return.

        Returns:
            DataFrame indexed by target_date with prediction and scoring columns.
        """
        conditions = ["symbol = $1", "horizon_days = $2"]
        args: list = [symbol, horizon_days]

        if model_name is not None:
            args.append(model_name)
            conditions.append(f"model_name = ${len(args)}")

        if unscored_only:
            conditions.append("actual_close IS NULL")

        where = " AND ".join(conditions)
        query = f"""
            SELECT
                target_date, prediction_date, model_name, model_version,
                horizon_days, predicted_close, predicted_return, direction,
                confidence, lower_bound, upper_bound,
                actual_close, actual_return, mae, direction_correct,
                created_at
            FROM predictions
            WHERE {where}
            ORDER BY target_date DESC
        """

        if limit is not None:
            args.append(limit)
            query += f" LIMIT ${len(args)}"

        rows = await self.fetch(query, *args)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["target_date"] = pd.to_datetime(df["target_date"])
        df.set_index("target_date", inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df

    async def get_training_data(
        self, symbol: str, limit: int | None = None
    ) -> pd.DataFrame:
        """Retrieve stock + sentiment data joined by date for model training.

        Args:
            symbol: Ticker symbol.
            limit: Maximum number of rows.

        Returns:
            DataFrame indexed by date with OHLCV + sentiment features.
        """
        stock_df = await self.get_stock_data(symbol, limit)
        sentiment_df = await self.get_sentiment_data(symbol, limit)

        if sentiment_df.empty:
            return stock_df

        combined = stock_df.join(sentiment_df, how="left")
        combined["avg_sentiment"] = combined["avg_sentiment"].fillna(0.0)
        combined["article_count"] = combined["article_count"].fillna(0)
        combined["sentiment_std"] = combined["sentiment_std"].fillna(0.0)
        combined["avg_relevance"] = combined["avg_relevance"].fillna(0.0)
        return combined
