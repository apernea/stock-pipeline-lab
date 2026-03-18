"""Observer pattern for prediction events.

Observers attach to a StockPipeline (Subject) and are notified each time a
prediction is generated. This decouples alerting and monitoring logic from the
pipeline itself.

Usage:
    pipeline = StockPipeline(db, api)
    pipeline.attach(ConsoleObserver())
    pipeline.attach(ThresholdObserver(threshold=0.02))
    pipeline.attach(DirectionChangeObserver())

    await pipeline.run("IBM")  # observers fire automatically after each prediction
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


@dataclass
class PredictionEvent:
    """Carries a completed prediction to every attached observer."""

    symbol: str
    prediction_date: date
    target_date: date
    horizon_days: int
    predicted_close: float
    predicted_return: float
    direction: int  # 1 = up, -1 = down, 0 = neutral


class Observer(ABC):
    """Base class for all prediction observers."""

    @abstractmethod
    def update(self, event: PredictionEvent) -> None:
        raise NotImplementedError


class Subject:
    """Mixin that gives a class an observer list and notify logic."""

    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def attach(self, observer: Observer) -> None:
        """Register an observer to receive prediction events."""
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """Remove a previously registered observer."""
        self._observers.remove(observer)

    def notify(self, event: PredictionEvent) -> None:
        """Dispatch a prediction event to all registered observers."""
        for observer in self._observers:
            observer.update(event)


# ---------------------------------------------------------------------------
# Concrete observers
# ---------------------------------------------------------------------------

class ConsoleObserver(Observer):
    """Logs every prediction with direction arrow and return percentage."""

    def update(self, event: PredictionEvent) -> None:
        arrow = "↑" if event.direction > 0 else ("↓" if event.direction < 0 else "→")
        logging.info(
            f"[{event.symbol}] {event.prediction_date} → {event.target_date} "
            f"({event.horizon_days}d): predicted close {event.predicted_close:.2f} "
            f"{arrow} {event.predicted_return:+.2%}"
        )


class ThresholdObserver(Observer):
    """Fires a warning when |predicted_return| meets or exceeds a threshold.

    Args:
        threshold: Minimum absolute return to trigger the alert (default: 0.02 = 2%).
    """

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = threshold

    def update(self, event: PredictionEvent) -> None:
        if abs(event.predicted_return) >= self.threshold:
            bias = "BULLISH" if event.direction > 0 else "BEARISH"
            logging.warning(
                f"[THRESHOLD] {event.symbol}: {bias} signal "
                f"{event.predicted_return:+.2%} exceeds ±{self.threshold:.1%} "
                f"(target: {event.target_date})"
            )


class DirectionChangeObserver(Observer):
    """Fires a warning when the predicted direction flips from the previous run.

    Tracks the last known direction per symbol so it works correctly across
    multiple symbols on the same pipeline instance.
    """

    def __init__(self) -> None:
        self._last_direction: dict[str, int] = {}

    def update(self, event: PredictionEvent) -> None:
        last = self._last_direction.get(event.symbol)

        if last is not None and last != event.direction and event.direction != 0:
            prev_arrow = "↑" if last > 0 else ("↓" if last < 0 else "→")
            curr_arrow = "↑" if event.direction > 0 else ("↓" if event.direction < 0 else "→")
            logging.warning(
                f"[DIRECTION FLIP] {event.symbol}: {prev_arrow} → {curr_arrow} "
                f"({event.predicted_return:+.2%}, target: {event.target_date})"
            )

        self._last_direction[event.symbol] = event.direction
