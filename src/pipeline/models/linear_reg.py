from pipeline.interfaces import ModelInterface


class LinearRegressionModel(ModelInterface):
    def __init__(self, fit_intercept: bool = True):
        super().__init__()
        self.fit_intercept = fit_intercept

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