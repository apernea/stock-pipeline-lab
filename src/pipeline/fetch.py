"""CLI script to fetch stock and sentiment data into the database."""

import asyncio
import logging
import sys

from pipeline.config import settings
from pipeline.provider.fetcher import fetch_and_store
from pipeline.provider.postgre import PostgreSQLBackend
from pipeline.provider.common.database import DatabaseProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def main(symbols: list[str] | None = None, outputsize: str = "compact") -> None:
    if not symbols:
        async with PostgreSQLBackend(DatabaseProvider.from_uri(settings.database_url).credentials) as db:
            symbols = await db.get_symbols()

    if not symbols:
        logging.error("No symbols found in the database. Pass symbols as arguments, e.g.: python -m pipeline.fetch IBM AAPL")
        return

    logging.info(f"Fetching data for {len(symbols)} symbol(s): {', '.join(symbols)}")

    for symbol in symbols:
        result = await fetch_and_store(
            symbol=symbol,
            database_url=settings.database_url,
            api_base_url=settings.api_base_url,
            api_key=settings.api_key,
            outputsize=outputsize,
        )
        print(f"[{symbol}] stock_rows={result['stock_rows']} sentiment_rows={result['sentiment_rows']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    asyncio.run(main(symbols=args or None))
