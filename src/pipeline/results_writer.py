"""Crash-proof results writer: /output/results.json is ALWAYS valid JSON.

Malformed output scores zero, so this is the one component that must never fail:
answers default to empty strings, writes are atomic (tmp + rename), incremental after
every task, and I/O errors are logged but never raised.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("verdict.results")


class ResultsWriter:
    def __init__(self, path: Path, task_ids: list[str]) -> None:
        self._path = path
        self._answers: dict[str, str] = {tid: "" for tid in task_ids}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error("cannot create output dir %s: %r", path.parent, e)

    def set(self, task_id: str, answer: str) -> None:
        self._answers[task_id] = str(answer) if answer is not None else ""

    def flush(self) -> None:
        payload = [
            {"task_id": tid, "answer": ans} for tid, ans in self._answers.items()
        ]
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as e:
            log.error("results flush failed: %r", e)
