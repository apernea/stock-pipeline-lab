from pipeline.interfaces import ModelInterface


class RandomForestModel(ModelInterface):
    def __init__(self, n_estimators: int = 100, max_depth: int = 5):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth

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