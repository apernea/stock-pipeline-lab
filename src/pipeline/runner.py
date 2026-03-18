"""CLI entry point for the stock prediction pipeline.

Usage:
    # Run once for one or more symbols
    python -m pipeline.runner IBM AAPL --mode oneshot

    # Run on a schedule (default: every 5 minutes)
    python -m pipeline.runner IBM --mode schedule --interval 300

    # Force model retraining
    python -m pipeline.runner IBM --force-retrain

    # Choose a different model
    python -m pipeline.runner IBM --model linear_reg
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from urllib.parse import urlparse

from pipeline.config import settings
from pipeline.interfaces.database import DatabaseInterface
from pipeline.models.factory import ModelType
from pipeline.observer import ConsoleObserver, DirectionChangeObserver, ThresholdObserver
from pipeline.pipeline import StockPipeline
from pipeline.provider.api_provider import APIProvider
from pipeline.provider.postgre import PostgreSQLBackend


def _credentials_from_uri(uri: str) -> DatabaseInterface.Credentials:
    parsed = urlparse(uri)
    return DatabaseInterface.Credentials(
        db_user=parsed.username,
        db_pass=parsed.password,
        db_host=parsed.hostname,
        db_port=parsed.port or 5432,
        db_name=parsed.path.lstrip("/"),
    )


def _build_pipeline(
    db: PostgreSQLBackend,
    api: APIProvider,
    model_type: str,
    training_window: int | None,
    alert_threshold: float,
) -> StockPipeline:
    pipeline = StockPipeline(db, api, model_type=model_type, training_window=training_window)
    pipeline.attach(ConsoleObserver())
    pipeline.attach(ThresholdObserver(threshold=alert_threshold))
    pipeline.attach(DirectionChangeObserver())
    return pipeline


async def _run_once(
    symbols: list[str],
    model_type: str,
    force_retrain: bool,
    training_window: int | None,
    alert_threshold: float,
) -> None:
    credentials = _credentials_from_uri(settings.database_url)
    async with PostgreSQLBackend(credentials) as db:
        async with APIProvider(settings.api_base_url, settings.api_key) as api:
            pipeline = _build_pipeline(db, api, model_type, training_window, alert_threshold)
            for symbol in symbols:
                try:
                    await pipeline.run(symbol, force_retrain=force_retrain)
                except Exception as exc:
                    logging.error(f"[{symbol}] Pipeline failed: {exc}")


async def _run_scheduled(
    symbols: list[str],
    model_type: str,
    interval_seconds: int,
    force_retrain: bool,
    training_window: int | None,
    alert_threshold: float,
) -> None:
    credentials = _credentials_from_uri(settings.database_url)
    async with PostgreSQLBackend(credentials) as db:
        async with APIProvider(settings.api_base_url, settings.api_key) as api:
            pipeline = _build_pipeline(db, api, model_type, training_window, alert_threshold)
            first_run = True
            while True:
                for symbol in symbols:
                    try:
                        await pipeline.run(
                            symbol, force_retrain=force_retrain and first_run
                        )
                    except Exception as exc:
                        logging.error(f"[{symbol}] Pipeline failed: {exc}")
                first_run = False
                logging.info(f"Next run in {interval_seconds}s. Press Ctrl+C to stop.")
                await asyncio.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        metavar="SYMBOL",
        help="One or more ticker symbols to process (e.g. IBM AAPL MSFT)",
    )
    parser.add_argument(
        "--mode",
        choices=["oneshot", "schedule"],
        default="oneshot",
        help="Run once or loop on a schedule",
    )
    parser.add_argument(
        "--model",
        default=ModelType.LSTM,
        metavar="MODEL",
        help="Model type: lstm | linear_reg | random_forest",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Seconds between runs in schedule mode",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the model even if a saved model already exists",
    )
    parser.add_argument(
        "--training-window",
        type=int,
        default=504,
        metavar="DAYS",
        help="Rolling training window in trading days (default: 504 ≈ 2 years). "
             "Pass 0 to train on all available data.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.02,
        metavar="FLOAT",
        help="Predicted return threshold for threshold alerts (default: 0.02 = 2%%)",
    )

    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols]
    training_window = args.training_window if args.training_window > 0 else None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "oneshot":
        asyncio.run(_run_once(symbols, args.model, args.force_retrain, training_window, args.alert_threshold))
    else:
        asyncio.run(_run_scheduled(symbols, args.model, args.interval, args.force_retrain, training_window, args.alert_threshold))


if __name__ == "__main__":
    main()
