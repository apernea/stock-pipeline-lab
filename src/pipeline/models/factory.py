from dataclasses import dataclass


@dataclass
class ModelInterface():
    def __init__(self):
        pass

    def show(self):
        raise NotImplementedError

@dataclass
class ModelFactory():
    raise NotImplementedError