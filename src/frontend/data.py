"""Cached loaders for the run artifacts the UI reads."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "outputs" / "golden" / "run.metrics.json"
DEFAULT_LEDGER = REPO_ROOT / "outputs" / "golden" / "run.ledger.jsonl"

FREE_ROUTES = {"deterministic", "local", "local_repair", "local_unverified"}


@st.cache_data(ttl=2, max_entries=8, show_spinner=False)
def load_metrics(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=2, max_entries=8, show_spinner=False)
def load_ledger(path: str) -> list[dict]:
    rows: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # partial last line while a run is writing
    return rows
