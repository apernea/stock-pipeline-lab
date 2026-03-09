from typing import TypeVar, Generic

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self) -> None:
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str | None = None):
        """Class decorator. Registers cls under `name` (default: cls.__name__)."""

        def decorator(cls: type[T]) -> type[T]:
            key = name or cls.__name__
            if key in self._entries:
                raise ValueError(f"'{key}' is already registered in {self!r}")
            self._entries[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Return the registered class or raise KeyError."""
        if name not in self._entries:
            available = ", ".join(self._entries)
            raise KeyError(f"'{name}' not registered. Available: [{available}]")
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        entries = ", ".join(self._entries)
        return f"Registry([{entries}])"