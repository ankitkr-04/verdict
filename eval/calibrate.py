"""Offline calibration: fit Platt (A, B) per category and pick the cheapest theta whose
Wilson lower bound clears the accuracy floor.

Input: a labeled JSONL where each row has {"category", "mean_logprob", "correct": 0|1}
(produced by running the pipeline on mock tasks and judging answers). Output: rewrites
config/calibration.json in place.

Usage: python eval/calibrate.py labeled.jsonl --floor 0.85 --margin 0.03
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import settings  # noqa: E402
from src.routing.calibration import calibrated_confidence, fit_platt, wilson_lower_bound  # noqa: E402

THETA_GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
MIN_SAMPLES = 10


def pick_theta(scores: list[float], labels: list[int], a: float, b: float,
               floor: float, margin: float) -> float:
    """Cheapest theta (fewest escalations) whose accepted-set accuracy LB clears floor+margin."""
    best = max(THETA_GRID)  # most conservative fallback
    for theta in sorted(THETA_GRID):
        accepted = [(s, y) for s, y in zip(scores, labels)
                    if calibrated_confidence(s, a, b) >= theta]
        if not accepted:
            continue
        lb = wilson_lower_bound(sum(y for _, y in accepted), len(accepted))
        if lb >= floor + margin:
            return theta
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labeled", type=Path, help="JSONL: category, mean_logprob, correct")
    parser.add_argument("--floor", type=float, default=0.85, help="accuracy floor")
    parser.add_argument("--margin", type=float, default=0.03, help="safety margin above floor")
    args = parser.parse_args()

    by_cat: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for line in args.labeled.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mean_logprob") is None:
            continue
        by_cat[row["category"]].append((float(row["mean_logprob"]), int(row["correct"])))

    calibration = json.loads(settings.CALIBRATION_JSON.read_text(encoding="utf-8"))
    for cat, pairs in sorted(by_cat.items()):
        if len(pairs) < MIN_SAMPLES:
            print(f"{cat}: only {len(pairs)} samples, skipping (need {MIN_SAMPLES})")
            continue
        scores = [s for s, _ in pairs]
        labels = [y for _, y in pairs]
        a, b = fit_platt(scores, labels)
        theta = pick_theta(scores, labels, a, b, args.floor, args.margin)
        calibration[cat] = {"A": round(a, 4), "B": round(b, 4), "theta": theta}
        print(f"{cat}: A={a:.3f} B={b:.3f} theta={theta} (n={len(pairs)})")

    settings.CALIBRATION_JSON.write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {settings.CALIBRATION_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
