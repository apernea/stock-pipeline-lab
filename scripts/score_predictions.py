"""Score unscored predictions by back-filling actuals from stock_data.

Runs a single UPDATE that joins predictions against stock_data for both the
target_date (actual close) and the prediction_date (baseline close for return).
The DB trigger on the predictions table auto-computes mae and direction_correct.

Run after market close each trading day:
    python -m scripts.score_predictions

Or via Makefile:
    make score
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg

from pipeline.config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")


SCORE_QUERY = """
    UPDATE predictions AS p
    SET
        actual_close  = s_target.close,
        actual_return = (s_target.close - s_base.close) / s_base.close
    FROM
        stock_data AS s_target,
        stock_data AS s_base
    WHERE
        p.actual_close IS NULL
        AND p.target_date <= CURRENT_DATE
        AND s_target.symbol = p.symbol
        AND s_target.date   = p.target_date
        AND s_base.symbol   = p.symbol
        AND s_base.date     = p.prediction_date
"""

SUMMARY_QUERY = """
    SELECT
        model_name,
        model_version,
        horizon_days,
        COUNT(*)                                          AS total,
        ROUND(AVG(mae)::numeric, 4)                       AS avg_mae,
        ROUND(
            100.0 * SUM(CASE WHEN direction_correct THEN 1 ELSE 0 END)
            / NULLIF(COUNT(direction_correct), 0), 1
        )                                                 AS direction_accuracy_pct,
        MIN(target_date)                                  AS from_date,
        MAX(target_date)                                  AS to_date
    FROM predictions
    WHERE actual_close IS NOT NULL
    GROUP BY model_name, model_version, horizon_days
    ORDER BY model_name, horizon_days
"""


async def score(database_url: str) -> None:
    conn = await asyncpg.connect(database_url)

    try:
        result = await conn.execute(SCORE_QUERY)
        # asyncpg returns "UPDATE N" as the status string
        scored = int(result.split()[-1])
        logging.info(f"Scored {scored} prediction(s).")

        if scored > 0:
            rows = await conn.fetch(SUMMARY_QUERY)
            if rows:
                logging.info("\nModel performance summary:")
                header = f"  {'model':<20} {'ver':<10} {'days':>4}  {'n':>5}  {'MAE':>8}  {'dir%':>6}  {'period'}"
                logging.info(header)
                logging.info("  " + "-" * (len(header) - 2))
                for r in rows:
                    logging.info(
                        f"  {r['model_name']:<20} {r['model_version']:<10} "
                        f"{r['horizon_days']:>4}  {r['total']:>5}  "
                        f"{r['avg_mae'] or 'n/a':>8}  "
                        f"{r['direction_accuracy_pct'] or 'n/a':>5}%  "
                        f"{r['from_date']} → {r['to_date']}"
                    )
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(score(settings.database_url))


if __name__ == "__main__":
    main()
