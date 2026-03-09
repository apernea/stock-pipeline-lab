from pipeline.interfaces import ModelInterface


class RandomForestModel(ModelInterface):
    def __init__(self):
        super().__init__()

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