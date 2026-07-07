"""Wall-clock budget manager: degrade gracefully instead of missing the 10-minute wall.

Degrade ladder (mode consumed by solvers):
  FULL    -> policy as configured
  REDUCED -> n_samples -> 1, skip cross-checks/repairs that need extra generations
  PANIC   -> one fast local attempt, accept unverified or escalate immediately
             (remote is ~10x faster than CPU-local: under time pressure tokens buy time)
"""

from __future__ import annotations

import time
from collections import deque
from enum import StrEnum

from src.core import settings


class Mode(StrEnum):
    FULL = "full"
    REDUCED = "reduced"
    PANIC = "panic"


class Budget:
    def __init__(
        self,
        total_s: float = settings.RUN_BUDGET_S,
        concurrency: int = settings.TASK_CONCURRENCY,
        est_task_cost_s: float = settings.EST_TASK_COST_S,
    ) -> None:
        self._t0 = time.monotonic()
        self._total_s = total_s
        self._concurrency = max(1, concurrency)
        self._prior_cost = est_task_cost_s
        self._observed: deque[float] = deque(maxlen=20)

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def remaining(self) -> float:
        return max(0.0, self._total_s - self.elapsed())

    def observe_task(self, duration_s: float) -> None:
        self._observed.append(duration_s)

    def _avg_cost(self) -> float:
        if not self._observed:
            return self._prior_cost
        return sum(self._observed) / len(self._observed)

    def mode(self, remaining_tasks: int) -> Mode:
        if remaining_tasks <= 0:
            return Mode.FULL
        usable = self.remaining() - settings.FINAL_FLUSH_RESERVE_S
        est_needed = remaining_tasks * self._avg_cost() / self._concurrency
        if est_needed <= 0:
            return Mode.FULL
        pressure = usable / est_needed
        if pressure >= settings.PRESSURE_REDUCED:
            return Mode.FULL
        if pressure >= settings.PRESSURE_PANIC:
            return Mode.REDUCED
        return Mode.PANIC

    def task_deadline_s(self) -> float:
        """Per-task allowance: the 25s ceiling, or less if the run budget is nearly spent."""
        return max(1.0, min(settings.TASK_CEILING_S,
                            self.remaining() - settings.FINAL_FLUSH_RESERVE_S))
