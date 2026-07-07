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

    def record(self, r: SolveResult) -> None:
        self.tasks += 1
        self.routes[r.route.value] += 1
        self.remote_calls += r.remote_calls
        self.remote_prompt_tokens += r.remote_prompt_tokens
        self.remote_completion_tokens += r.remote_completion_tokens
        if not self._io_ok:
            return
        row = {
            "task_id": r.task_id,
            "category": r.category.value,
            "route": r.route.value,
            "confidence": round(r.confidence, 4),
            "remote_calls": r.remote_calls,
            "remote_prompt_tokens": r.remote_prompt_tokens,
            "remote_completion_tokens": r.remote_completion_tokens,
            "local_calls": r.local_calls,
            "wall_ms": r.wall_ms,
            "detail": r.detail,
        }
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            self._io_ok = False
            log.warning("ledger writes disabled: %r", e)

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
