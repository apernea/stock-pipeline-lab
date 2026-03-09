from abc import ABC, abstractmethod


class ModelInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save(self, directory, prefix):
        raise NotImplementedError

    @abstractmethod
    def load(self, directory, prefix):
        raise NotImplementedError

    @abstractmethod
    def summary(self):
        raise NotImplementedError
