"""Preprocessing pipeline for stock + sentiment data.

Takes the raw DataFrame from PostgreSQLBackend.get_training_data() and produces
feature matrices (X) and target vectors (y) ready for model training/prediction.

Target: next-day close price.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class TrainTestSplit:
    """Container for chronologically split and scaled data."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: MinMaxScaler
    dates_train: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex


class PreprocessingPipeline:
    """Transforms raw training data into model-ready features.

    Usage:
        pp = PreprocessingPipeline()
        featured_df = pp.add_features(raw_df)
        split = pp.prepare(raw_df, train_ratio=0.8)
    """

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive technical indicators and pass through sentiment features.

        Args:
            df: DataFrame indexed by date with columns:
                open, high, low, close, volume,
                avg_sentiment, article_count, sentiment_std, avg_relevance.

        Returns:
            DataFrame with original + engineered feature columns.
        """
        df = df.copy()

        # --- Lagged price ---
        df["close_lag_24"] = df["close"].shift(24)

        # --- Returns ---
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # --- Price structure ---
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]
        hl_range = df["high"] - df["low"]
        df["price_position"] = (df["close"] - df["low"]) / hl_range.where(hl_range != 0, np.nan)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # --- Moving averages ---
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        # --- MACD ---
        df["macd"] = df["ema_12"] - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # --- Bollinger Bands ---
        std_20 = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["sma_20"] + 2 * std_20
        df["bb_lower"] = df["sma_20"] - 2 * std_20
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / bb_range.where(bb_range != 0, np.nan)
        df["bb_bandwidth"] = bb_range / df["sma_20"]

        # --- Momentum ---
        df["rsi_14"] = PreprocessingPipeline._compute_rsi(df["close"], period=14)
        df["roc_10"] = df["close"].pct_change(periods=10)

        # --- Volatility ---
        df["atr_14"] = PreprocessingPipeline._compute_atr(df, period=14)
        df["volatility_20"] = df["log_return"].rolling(window=20).std()

        # --- Volume ---
        df["obv"] = PreprocessingPipeline._compute_obv(df)
        vol_ma_20 = df["volume"].rolling(window=20).mean()
        df["volume_ratio_20"] = df["volume"] / vol_ma_20.where(vol_ma_20 != 0, np.nan)

        # --- Sentiment-derived ---
        if "avg_sentiment" in df.columns:
            df["sentiment_momentum"] = df["avg_sentiment"].rolling(window=3).mean()
            df["sentiment_dispersion"] = df["sentiment_std"].rolling(window=3).mean()

        return df

    @staticmethod
    def add_target(df: pd.DataFrame) -> pd.DataFrame:
        """Add next-day close as the prediction target.

        The last row will have NaN target (no future data) and is dropped.
        """
        df = df.copy()
        df["target"] = df["close"].shift(-1)
        return df

    @staticmethod
    def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN values introduced by rolling windows and target shift."""
        return df.dropna()

    def prepare(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> TrainTestSplit:
        """Full preprocessing: features -> target -> clean -> split -> scale.

        Args:
            df: Raw DataFrame from get_training_data().
            train_ratio: Fraction of data used for training (rest is test).

        Returns:
            TrainTestSplit with scaled X/y arrays and metadata.
        """
        df = self.add_features(df)
        df = self.add_target(df)
        df = self.drop_incomplete_rows(df)

        # Separate features and target
        feature_cols = [
            col for col in df.columns if col not in ("target",)
        ]
        X = df[feature_cols]
        y = df["target"].values

        # Chronological split — no shuffling
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        dates_train = X_train.index
        dates_test = X_test.index

        # Scale features — fit on training data only
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return TrainTestSplit(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_cols,
            scaler=scaler,
            dates_train=dates_train,
            dates_test=dates_test,
        )

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute the Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range — measures volatility across high/low/close."""
        prev_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def _compute_obv(df: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume — volume accumulates in price direction."""
        direction = np.sign(df["close"].diff()).fillna(0)
        return (direction * df["volume"]).cumsum()
