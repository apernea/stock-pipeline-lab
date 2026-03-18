"""LSTM model for next-day stock close price prediction using PyTorch."""

from pipeline.interfaces import ModelInterface
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _LSTMNetwork(nn.Module):
    """PyTorch LSTM network for sequence-to-one regression."""

    def __init__(self, n_features: int, units: int, dropout: float):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units, units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU()
        self.fc = nn.Linear(units // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])  # take last timestep
        return self.fc(x)


class LSTMModel(ModelInterface):
    """LSTM neural network for stock price prediction.

    Uses sliding-window sequences of historical features to predict
    the next-day close price.
    """

    UNITS = 50
    EPOCHS = 20
    BATCH_SIZE = 64
    WINDOW_SIZE = 24
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001

    def __init__(
        self,
        units: int = UNITS,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.model: _LSTMNetwork | None = None
        self.device = torch.device("cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if _TORCH_AVAILABLE else None

    @staticmethod
    def create_sequences(
        X: np.ndarray, y: np.ndarray, window_size: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reshape 2D scaled data into 3D sequences for LSTM.

        Takes a sliding window of `window_size` consecutive rows to form
        each sample. The target for each sample is the y value at the
        end of the window.

        Args:
            X: Scaled feature array of shape (samples, features).
            y: Target array of shape (samples,).
            window_size: Number of past days per sequence.

        Returns:
            Tuple of (X_seq, y_seq) where:
                X_seq has shape (samples - window_size, window_size, features)
                y_seq has shape (samples - window_size,)
        """
        X_seq, y_seq = [], []
        for i in range(window_size, len(X)):
            X_seq.append(X[i - window_size : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def process(self, df) -> pd.DataFrame:
        pass

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LSTM model on pre-scaled feature/target arrays.

        Args:
            X: Scaled feature array of shape (samples, features).
            y: Target array of shape (samples,).

        Raises:
            ImportError: If torch is not installed.
            ValueError: If not enough data after sequencing.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("Training requires PyTorch. Install with: pip install 'stock-price-prediction-pipeline[lstm]'")

        X_seq, y_seq = self.create_sequences(X, y, self.window_size)

        if len(X_seq) == 0:
            raise ValueError(
                f"Insufficient data: need more than {self.window_size} samples, got {len(X)}"
            )

        n_features = X_seq.shape[2]
        self.model = _LSTMNetwork(n_features, self.units, self.DROPOUT_RATE).to(self.device)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32, device=self.device).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.L1Loss()  # MAE

        logging.info(
            f"Training LSTM — sequences: {X_seq.shape[0]}, "
            f"window: {self.window_size}, features: {n_features}, device: {self.device}"
        )

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= len(dataset)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch + 1}/{self.epochs} — MAE: {epoch_loss:.4f}")

        logging.info("Training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-day close prices from pre-scaled features.

        Args:
            X: Scaled feature array of shape (samples, features).

        Returns:
            1D array of predictions. Length is ``len(X) - window_size``.
            Returns an empty array if the model is not trained or torch
            is unavailable.
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return np.array([])

        dummy_y = np.zeros(len(X))
        X_seq, _ = self.create_sequences(X, dummy_y, self.window_size)

        if len(X_seq) == 0:
            return np.array([])

        self.model.eval()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def save(self, directory: str, prefix: str) -> None:
        """Save the trained model and hyperparameters to disk.

        Creates ``<prefix>_model.pt`` and ``<prefix>_params.pkl``
        inside *directory*.

        Args:
            directory: Target directory path.
            prefix: Filename prefix for saved artefacts.
        """
        if not _TORCH_AVAILABLE or self.model is None:
            logging.error("No trained model to save")
            return

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / f"{prefix}_model.pt")

        params = {
            "units": self.units,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "n_features": self.model.lstm1.input_size,
        }
        with open(path / f"{prefix}_params.pkl", "wb") as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

        logging.info(f"LSTM model saved to {path}")

    def load(self, directory: str, prefix: str) -> None:
        """Load a previously saved model and hyperparameters from disk.

        Args:
            directory: Directory containing saved artefacts.
            prefix: Filename prefix used during save.
        """
        if not _TORCH_AVAILABLE:
            logging.error("Cannot load model: torch is not installed")
            return

        path = Path(directory)
        params_path = path / f"{prefix}_params.pkl"
        model_path = path / f"{prefix}_model.pt"

        if not model_path.exists():
            logging.error(f"Model file not found: {model_path}")
            return

        if params_path.exists():
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            self.units = params.get("units", self.units)
            self.epochs = params.get("epochs", self.epochs)
            self.batch_size = params.get("batch_size", self.batch_size)
            self.window_size = params.get("window_size", self.window_size)
            n_features = params["n_features"]
        else:
            logging.error(f"Params file not found: {params_path}")
            return

        self.model = _LSTMNetwork(n_features, self.units, self.DROPOUT_RATE).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        logging.info(f"LSTM model loaded from {path}")

    def summary(self) -> str | None:
        """Return a string summary of the model architecture.

        Returns:
            Model summary string, or None if no model is available.
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return None

        return str(self.model)
