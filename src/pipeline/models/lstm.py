from pipeline.interfaces import ModelInterface

import numpy as np

class LSTMModel(ModelInterface):
    def __init__(self):
        super().__init__()

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

    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, directory, prefix):
        raise NotImplementedError

    def load(self, directory, prefix):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

