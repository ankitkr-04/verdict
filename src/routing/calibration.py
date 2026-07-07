"""Confidence calibration: Platt scaling over mean token logprobs + Wilson bound.

Pure Python on purpose — no numpy/scipy in the image. fit_platt() is used offline by
eval/calibrate.py; the runtime only evaluates the fitted sigmoid.
"""

from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def calibrated_confidence(mean_logprob: float | None, a: float, b: float) -> float:
    """p(correct) = sigmoid(-(A*s + B)) with s = mean token logprob (Platt form)."""
    if mean_logprob is None:
        return 0.0  # no signal -> no confidence; gate decides conservatively
    return sigmoid(-(a * mean_logprob + b))


def fit_platt(
    scores: list[float], labels: list[int], *, iters: int = 500, lr: float = 0.5,
) -> tuple[float, float]:
    """Fit (A, B) by gradient descent on log-loss. labels: 1 = local answer was correct."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be equal-length, non-empty")
    a, b = -1.0, 0.0
    n = len(scores)
    for _ in range(iters):
        ga = gb = 0.0
        for s, y in zip(scores, labels):
            p = sigmoid(-(a * s + b))
            err = p - y
            # d(loss)/dA = err * (-s), d(loss)/dB = err * (-1)
            ga += -err * s
            gb += -err
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)
