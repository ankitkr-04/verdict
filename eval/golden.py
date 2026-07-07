"""Golden regression run — execute before every push.

Runs the full pipeline on the mock task set (mock backends by default) and asserts the
container contract: valid results.json, every task answered, ledger sane, and the
at-least-one-remote-call guarantee held.

Usage: python eval/golden.py [--tasks eval/mock_tasks/tasks.sample.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = REPO_ROOT / "eval" / "mock_tasks" / "tasks.sample.json"
OUT_DIR = REPO_ROOT / "outputs" / "golden"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--real-local", action="store_true",
                        help="use the llama backend instead of mock (GPU box)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"
    ledger_path = OUT_DIR / "run.ledger.jsonl"
    metrics_path = OUT_DIR / "run.metrics.json"

    env = os.environ.copy()
    env.setdefault("VERDICT_LOCAL_BACKEND", "llama" if args.real_local else "mock")
    env.setdefault("VERDICT_REMOTE_BACKEND", env.get("VERDICT_REMOTE_BACKEND", "mock"))
    env["VERDICT_INPUT"] = str(args.tasks)
    env["VERDICT_OUTPUT"] = str(results_path)
    env["VERDICT_LEDGER"] = str(ledger_path)
    env["VERDICT_METRICS"] = str(metrics_path)

    proc = subprocess.run([sys.executable, "main.py"], cwd=REPO_ROOT, env=env)

    failures: list[str] = []
    if proc.returncode != 0:
        failures.append(f"exit code {proc.returncode} != 0")

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"FAIL: results.json unreadable/invalid: {e!r}")
        return 1

    answered = {r["task_id"]: r["answer"] for r in results}
    for t in tasks:
        if t["task_id"] not in answered:
            failures.append(f"missing answer for {t['task_id']}")
        elif not str(answered[t["task_id"]]).strip():
            failures.append(f"empty answer for {t['task_id']}")

    remote_calls = 0
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                remote_calls += json.loads(line).get("remote_calls", 0)
    if remote_calls < 1:
        failures.append("no Fireworks/remote call was recorded (AMD-usage rule)")

    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        if m.get("status") != "complete":
            failures.append(f"metrics status is {m.get('status')!r}, not 'complete'")
    except (OSError, json.JSONDecodeError) as e:
        failures.append(f"run.metrics.json unreadable/invalid: {e!r}")

    if failures:
        print("GOLDEN RUN FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"GOLDEN RUN OK: {len(tasks)} tasks answered, {remote_calls} remote call(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
