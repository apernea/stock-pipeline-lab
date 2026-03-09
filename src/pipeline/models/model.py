from dataclasses import dataclass
from abc import ABC, abstractmethod


class ModelInterface:
    def __init__(self):
        pass

    def show(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError

    @abstractmethod
    def summary(self):
        raise NotImplementedError
