"""End-to-end stock prediction pipeline.

Orchestrates the full cycle for a given symbol:
    fetch → store → preprocess → train → predict → store prediction

Training uses a rolling window: only the most recent `training_window` trading
days are used. When new data is fetched, the window slides forward automatically
and the model is retrained on the updated slice — the oldest rows fall off the
back as new ones are added at the front.

The pipeline is stateful: once a model and scaler are loaded or trained, they
are reused across repeated calls to predict() without reloading from disk.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from pipeline.interfaces.model import ModelInterface
from pipeline.models.factory import ModelFactory, ModelType
from pipeline.observer import PredictionEvent, Subject
from pipeline.preprocessing.preprocessing import PreprocessingPipeline
from pipeline.provider.api_provider import APIProvider
from pipeline.provider.postgre import PostgreSQLBackend
from sklearn.preprocessing import MinMaxScaler


class StockPipeline(Subject):
    """Connects every component into a single runnable workflow.

    Args:
        db: Connected PostgreSQLBackend instance.
        api: Connected APIProvider instance.
        model_type: Model to use (default: lstm). See ModelType constants.
        model_dir: Directory for saving and loading model artefacts.
        training_window: Number of most-recent trading days to train on.
            The window slides forward each time new data is fetched, so the
            model always reflects the latest market behaviour.
            Default: 504 (~2 trading years). Use None for all available data.
    """

    # Minimum rows fetched at inference time — enough for all rolling windows
    # (EMA-26 + close_lag_24 + ATR-14 + LSTM window_size ≈ 75 minimum; 200 is safe)
    INFERENCE_LOOKBACK = 200

    def __init__(
        self,
        db: PostgreSQLBackend,
        api: APIProvider,
        model_type: str = ModelType.LSTM,
        model_dir: str = "models",
        training_window: int | None = 504,
    ):
        self.db = db
        self.api = api
        self.model_type = model_type
        self.model_dir = model_dir
        self.training_window = training_window

        super().__init__()
        self._model: ModelInterface | None = None
        self._scaler: MinMaxScaler | None = None
        self._feature_names: list[str] | None = None
        self._training_cutoff: date | None = None
        self._preprocessing = PreprocessingPipeline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, symbol: str, outputsize: str = "compact") -> dict[str, int]:
        """Pull latest stock and sentiment data from the API and persist to DB.

        Args:
            symbol: Ticker symbol (e.g. "IBM").
            outputsize: "compact" (100 days) or "full" (20+ years).

        Returns:
            Dict with "stock_rows" and "sentiment_rows" counts inserted.
        """
        stock_rows = await self.api.fetch_stock_data(symbol, outputsize=outputsize)
        sentiment_rows = await self.api.fetch_sentiment_data(symbol)

        stock_inserted = await self.db.insert_stock_data(symbol, stock_rows)
        sentiment_inserted = await self.db.insert_sentiment_data(symbol, sentiment_rows)

        logging.info(
            f"[{symbol}] Stored {stock_inserted}/{len(stock_rows)} stock rows, "
            f"{sentiment_inserted}/{len(sentiment_rows)} sentiment articles"
        )
        return {"stock_rows": stock_inserted, "sentiment_rows": sentiment_inserted}

    async def train(self, symbol: str, force: bool = False) -> None:
        """Fetch the training window, preprocess, fit the model, and save artefacts.

        The training window always covers the most recent `training_window` trading
        days in the database. Calling train() after new data has been fetched slides
        the window forward and drops the equivalent number of oldest rows.

        Skips retraining if:
          - A saved model exists, AND
          - No new data has arrived since the model was last trained, AND
          - force=False

        Args:
            symbol: Ticker symbol.
            force: Retrain unconditionally, even if the model is current.
        """
        if not force and not await self._needs_retraining(symbol):
            logging.info(f"[{symbol}] Model is current (trained up to {self._training_cutoff}). Skipping retrain.")
            if self._model is None:
                self._load(symbol)
            return

        raw_df = await self.db.get_training_data(symbol, limit=self.training_window)
        if raw_df.empty:
            raise ValueError(f"[{symbol}] No training data in database. Run fetch() first.")

        window_start = raw_df.index[0].date()
        window_end = raw_df.index[-1].date()
        logging.info(
            f"[{symbol}] Training window: {window_start} → {window_end} "
            f"({len(raw_df)} rows)"
        )

        split = self._preprocessing.prepare(raw_df)
        prefix = self._model_prefix(symbol)

        self._model = ModelFactory.create(self.model_type)
        self._model.train(split.X_train, split.y_train)
        self._model.save(self.model_dir, prefix)

        # Persist scaler, feature names, and training cutoff together
        artefact = {
            "scaler": split.scaler,
            "feature_names": split.feature_names,
            "training_cutoff": window_end,
            "training_window": self.training_window,
        }
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.model_dir) / f"{prefix}_scaler.pkl", "wb") as f:
            pickle.dump(artefact, f, pickle.HIGHEST_PROTOCOL)

        self._scaler = split.scaler
        self._feature_names = split.feature_names
        self._training_cutoff = window_end
        logging.info(f"[{symbol}] Training complete. Artefacts saved to '{self.model_dir}/'.")

    async def predict(self, symbol: str, horizon_days: int = 1) -> dict:
        """Generate a prediction for the next trading day and store it.

        Loads artefacts from disk if the model is not already in memory.

        Args:
            symbol: Ticker symbol.
            horizon_days: Number of days ahead to predict (default: 1).

        Returns:
            The prediction record dict that was written to the database.
        """
        if self._model is None or self._scaler is None:
            self._load(symbol)

        raw_df = await self.db.get_training_data(symbol, limit=self.INFERENCE_LOOKBACK)
        if raw_df.empty:
            raise ValueError(f"[{symbol}] No data available for inference.")

        featured_df = self._preprocessing.add_features(raw_df)
        featured_df = featured_df.dropna()

        if len(featured_df) == 0:
            raise ValueError(f"[{symbol}] Insufficient data after feature engineering.")

        # Align columns to training order; fill any missing columns with 0
        X_df = featured_df.reindex(columns=self._feature_names, fill_value=0.0)
        X = self._scaler.transform(X_df)

        predictions = self._model.predict(X)
        if len(predictions) == 0:
            raise ValueError(f"[{symbol}] Model returned no predictions.")

        predicted_close = float(predictions[-1])
        last_close = float(featured_df["close"].iloc[-1])
        predicted_return = (predicted_close - last_close) / last_close
        direction = int(np.sign(predicted_return))

        prediction_date = featured_df.index[-1].date()
        target_date = prediction_date + timedelta(days=horizon_days)

        record = {
            "symbol": symbol,
            "model_name": self.model_type,
            "model_version": "1.0",
            "prediction_date": prediction_date,
            "target_date": target_date,
            "horizon_days": horizon_days,
            "predicted_close": predicted_close,
            "predicted_return": predicted_return,
            "direction": direction,
        }

        await self.db.insert_prediction(record)

        self.notify(PredictionEvent(
            symbol=symbol,
            prediction_date=prediction_date,
            target_date=target_date,
            horizon_days=horizon_days,
            predicted_close=predicted_close,
            predicted_return=predicted_return,
            direction=direction,
        ))

        return record

    async def run(self, symbol: str, force_retrain: bool = False) -> dict:
        """Full pipeline cycle: fetch → train (if needed) → predict.

        After fetching, the training window automatically advances. If the
        latest stock_data date is ahead of the model's training cutoff, the
        model is retrained on the new window before predicting.

        Args:
            symbol: Ticker symbol.
            force_retrain: Retrain unconditionally regardless of data freshness.

        Returns:
            The prediction record dict.
        """
        logging.info(f"[{symbol}] Pipeline run started.")
        await self.fetch(symbol)
        await self.train(symbol, force=force_retrain)
        return await self.predict(symbol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_prefix(self, symbol: str) -> str:
        return f"{self.model_type}_{symbol.lower()}"

    def _load(self, symbol: str) -> None:
        """Load model and scaler artefacts from disk into instance state."""
        prefix = self._model_prefix(symbol)
        scaler_path = Path(self.model_dir) / f"{prefix}_scaler.pkl"

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"No saved artefacts for '{symbol}' in '{self.model_dir}/'. "
                "Run train() first."
            )

        self._model = ModelFactory.load(self.model_type, self.model_dir, prefix)

        with open(scaler_path, "rb") as f:
            artefact = pickle.load(f)

        self._scaler = artefact["scaler"]
        self._feature_names = artefact["feature_names"]
        self._training_cutoff = artefact.get("training_cutoff")
        logging.info(
            f"[{symbol}] Model loaded (trained up to {self._training_cutoff})."
        )

    async def _needs_retraining(self, symbol: str) -> bool:
        """Return True if new data has arrived since the model was last trained.

        Compares the latest date in stock_data against the training cutoff stored
        in the scaler artefact. Also returns True if no artefacts exist yet.
        """
        scaler_path = Path(self.model_dir) / f"{self._model_prefix(symbol)}_scaler.pkl"
        if not scaler_path.exists():
            return True

        # Read cutoff from disk without fully loading the model
        if self._training_cutoff is None:
            with open(scaler_path, "rb") as f:
                artefact = pickle.load(f)
            self._training_cutoff = artefact.get("training_cutoff")

        if self._training_cutoff is None:
            return True

        row = await self.db.fetchrow(
            "SELECT MAX(date) AS latest FROM stock_data WHERE symbol = $1", symbol
        )
        latest_date = row["latest"] if row else None

        if latest_date is None:
            return False

        # Convert to date if asyncpg returns a datetime.date already; handle both
        if hasattr(latest_date, "date"):
            latest_date = latest_date.date()

        needs = latest_date > self._training_cutoff
        if needs:
            logging.info(
                f"[{symbol}] New data detected (stock_data up to {latest_date}, "
                f"model trained up to {self._training_cutoff}). Retraining."
            )
        return needs
