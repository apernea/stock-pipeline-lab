from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class DatabaseInterface(ABC):
    """Base interface for async database providers."""

    @dataclass
    class Credentials:
        db_user: str
        db_pass: str
        db_host: str
        db_port: int
        db_name: str

    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, query: str, *args: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        raise NotImplementedError

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
