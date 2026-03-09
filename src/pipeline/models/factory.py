from pipeline.interfaces.model import ModelInterface
from pipeline.utils.registry import Registry

model_registry = Registry[ModelInterface]()


class ModelType:
    LSTM = "lstm"
    LINEAR_REGRESSION = "linear_reg"
    RANDOM_FOREST = "random_forest"


class ModelFactory:
    @staticmethod
    def create(model_type: str) -> ModelInterface:
        cls = model_registry.get(model_type)
        return cls()

    @staticmethod
    def load(model_type: str, directory: str, prefix: str) -> ModelInterface:
        """Loads a previously saved model by type name."""
        cls = model_registry.get(model_type)
        instance = cls()
        instance.load(directory, prefix)
        return instance
