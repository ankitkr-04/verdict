"""Per-task run ledger (JSONL) + aggregate token/route counters.

Feeds the offline eval loop and the end-of-run Fireworks-usage guarantee. Logging must
never take the run down: all I/O failures are swallowed after the first warning.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from src.core.schemas import SolveResult

log = logging.getLogger("verdict.ledger")


class Ledger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._t0 = time.time()
        self._io_ok = True
        self.routes: Counter[str] = Counter()
        self.remote_calls = 0
        self.remote_prompt_tokens = 0
        self.remote_completion_tokens = 0
        self.tasks = 0
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("", encoding="utf-8")
        except OSError as e:
            self._io_ok = False
            log.warning("ledger disabled (cannot write %s): %r", path, e)

    def record(self, r: SolveResult) -> dict:
        """Append one row; returns it so Metrics can aggregate the same data."""
        self.tasks += 1
        self.routes[r.route.value] += 1
        self.remote_calls += r.remote_calls
        self.remote_prompt_tokens += r.remote_prompt_tokens
        self.remote_completion_tokens += r.remote_completion_tokens
        row = {
            "task_id": r.task_id,
            "category": r.category.value,
            "route": r.route.value,
            "mode": r.mode,
            "confidence": round(r.confidence, 4),
            "mean_logprob": round(r.best_logprob, 5) if r.best_logprob is not None else None,
            "remote_calls": r.remote_calls,
            "remote_prompt_tokens": r.remote_prompt_tokens,
            "remote_completion_tokens": r.remote_completion_tokens,
            "local_calls": r.local_calls,
            "local_prompt_tokens": r.local_prompt_tokens,
            "local_completion_tokens": r.local_completion_tokens,
            "wall_ms": r.wall_ms,
            "local_ms": r.local_ms,
            "remote_ms": r.remote_ms,
            "dispatch_ms": r.dispatch_ms,
            "queue_ms": r.queue_ms,
            "t_s": round(r.finished_s, 1),
            "detail": r.detail,
        }
        if not self._io_ok:
            return row
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            self._io_ok = False
            log.warning("ledger writes disabled: %r", e)
        return row

    @property
    def remote_tokens(self) -> int:
        return self.remote_prompt_tokens + self.remote_completion_tokens

    def summary(self) -> dict:
        return {
            "tasks": self.tasks,
            "routes": dict(self.routes),
            "remote_calls": self.remote_calls,
            "remote_prompt_tokens": self.remote_prompt_tokens,
            "remote_completion_tokens": self.remote_completion_tokens,
            "remote_tokens_total": self.remote_tokens,
            "wall_s": round(time.time() - self._t0, 1),
        }

    def print_summary(self) -> None:
        print(json.dumps(self.summary(), indent=2), file=sys.stderr)
