"""Base database provider with asyncpg connection pooling."""

from __future__ import annotations

import asyncpg

from pipeline.interfaces.database import DatabaseInterface


class DatabaseProvider(DatabaseInterface):
    """PostgreSQL provider using asyncpg with connection pooling."""

    def __init__(self, credentials: DatabaseInterface.Credentials, pool_size: int = 5):
        self.credentials = credentials
        self.pool_size = pool_size
        self._pool: asyncpg.Pool | None = None

    @staticmethod
    def from_uri(uri: str) -> DatabaseProvider:
        """Create a provider from a PostgreSQL connection URI.

        Expected format: postgresql://user:pass@host:port/dbname
        """
        from urllib.parse import urlparse

        parsed = urlparse(uri)

        assert parsed.username, "Missing database user"
        assert parsed.password, "Missing database password"
        assert parsed.hostname, "Missing database host"

        credentials = DatabaseInterface.Credentials(
            db_user=parsed.username,
            db_pass=parsed.password,
            db_host=parsed.hostname,
            db_port=parsed.port or 5432,
            db_name=parsed.path.lstrip("/"),
        )
        return DatabaseProvider(credentials)

    async def connect(self) -> None:
        if self._pool is not None:
            return

        creds = self.credentials
        self._pool = await asyncpg.create_pool(
            user=creds.db_user,
            password=creds.db_pass,
            host=creds.db_host,
            port=creds.db_port,
            database=creds.db_name,
            min_size=1,
            max_size=self.pool_size,
        )

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "Not connected. Use 'async with' or call connect() first."
            )
        return self._pool

    async def execute(self, query: str, *args) -> str:
        pool = self._ensure_pool()
        return await pool.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[dict]:
        pool = self._ensure_pool()
        rows = await pool.fetch(query, *args)
        return [dict(r) for r in rows]

    async def fetchrow(self, query: str, *args) -> dict | None:
        pool = self._ensure_pool()
        row = await pool.fetchrow(query, *args)
        return dict(row) if row else None
