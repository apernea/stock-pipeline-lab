"""Run SQL migration scripts against the database in order."""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from pipeline.config import settings


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


async def run_migrations(database_url: str) -> None:
    conn = await asyncpg.connect(database_url)

    # Track which migrations have already been applied
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT now()
        )
    """)

    applied = {row["name"] for row in await conn.fetch("SELECT name FROM _migrations")}

    # Collect and sort migration files
    scripts = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not scripts:
        print("No migration files found.")
        await conn.close()
        return

    count = 0
    for script in scripts:
        if script.name in applied:
            print(f"  skip  {script.name} (already applied)")
            continue

        sql = script.read_text().strip()
        if not sql:
            print(f"  skip  {script.name} (empty)")
            continue

        await conn.execute(sql)
        await conn.execute("INSERT INTO _migrations (name) VALUES ($1)", script.name)
        print(f"  apply {script.name}")
        count += 1

    if count == 0:
        print("All migrations already applied.")
    else:
        print(f"Applied {count} migration(s).")

    await conn.close()


def main() -> None:
    asyncio.run(run_migrations(settings.database_url))


if __name__ == "__main__":
    main()
