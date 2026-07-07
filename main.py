"""Container entrypoint: read /input/tasks.json, run the pipeline, write /output/results.json.

Exit code is 0 whenever a valid results file was produced — a partially-degraded run
still scores; a crashed one scores zero. All hard failure modes funnel into fallbacks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.core import settings
from src.core.schemas import Task
from src.pipeline.orchestrator import run
from src.pipeline.results_writer import ResultsWriter

log = logging.getLogger("verdict.main")


def load_tasks(path: Path) -> list[Task]:
    """Defensive parse: skip malformed entries rather than dying on them."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.error("cannot read tasks from %s: %r", path, e)
        return []
    if not isinstance(raw, list):
        log.error("tasks file is not a JSON array")
        return []
    tasks: list[Task] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        tid, prompt = item.get("task_id"), item.get("prompt")
        if tid is None or prompt is None:
            continue
        tid = str(tid)
        if tid in seen:
            continue
        seen.add(tid)
        tasks.append(Task(task_id=tid, prompt=str(prompt)))
    return tasks


def main() -> int:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="verdict routing agent")
    parser.add_argument("--input", type=Path, default=settings.INPUT_TASKS_PATH)
    parser.add_argument("--output", type=Path, default=settings.OUTPUT_RESULTS_PATH)
    args = parser.parse_args()

    tasks = load_tasks(args.input)
    log.info("loaded %d tasks from %s", len(tasks), args.input)
    writer = ResultsWriter(args.output, [t.task_id for t in tasks])
    writer.flush()  # valid (empty-answer) results exist from second zero

    try:
        asyncio.run(run(tasks, writer))
    except BaseException:  # noqa: BLE001 — flush what we have, exit clean
        log.exception("pipeline aborted; flushing partial results")
    finally:
        writer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
