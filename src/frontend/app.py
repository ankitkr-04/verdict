"""verdict — chat-first UI. Run from the repo root:

    streamlit run src/frontend/app.py

The chat talks to the real pipeline (dispatcher -> deterministic -> local+verify ->
escalate) through Playground; backends follow the same VERDICT_* env as any run.
Telemetry (optional) reads the run artifacts named in the sidebar settings.
Dev/demo tool only — never part of the container image.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.frontend.playground import Playground, PlaygroundError

FREE_ROUTES = {"deterministic", "local", "local_repair", "local_unverified"}
EXAMPLES = [
    "What is 847 * 36 + 129?",
    "What day of the week was 2024-03-15?",
    "A train travels 240 km in 3 hours. How far does it go in 5 hours?",
]
DEFAULT_METRICS = REPO_ROOT / "outputs" / "golden" / "run.metrics.json"

st.set_page_config(page_title="verdict", page_icon="⚖️", layout="centered")


@st.cache_resource
def playground() -> Playground:
    return Playground()


def meta_line(r: dict) -> str:
    free = r["route"] in FREE_ROUTES
    dot = "🔵" if free else "🟠"
    cost = "free · 0 tokens" if free else f"{r['remote_tokens']} tokens"
    return f"{dot} {r['route'].replace('_', ' ')} · {r['category'].replace('_', ' ')} · {cost} · {r['wall_ms']} ms"


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("⚖️ verdict")
    st.caption(
        "A routing agent that answers as much as it can **for free** — plain code "
        "first, then a small local model with verification — and only pays for a "
        "remote model when a task truly needs one."
    )

    st.divider()
    st.subheader("Run telemetry")
    with st.expander("settings"):
        metrics_path = Path(st.text_input("metrics file", str(DEFAULT_METRICS)))
        live = st.toggle("auto-refresh (2s)", value=False)

    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        m = None
        st.caption("No run artifacts yet — run the pipeline (or `eval/golden.py`) "
                   "and numbers appear here.")

    if m:
        tok = m.get("tokens", {})
        run = m.get("run", {})
        saved = tok.get("est_saved_vs_all_remote", 0)
        remote = tok.get("remote_total", 0)
        share = round(100 * saved / (saved + remote)) if (saved + remote) else 0
        a, b = st.columns(2)
        a.metric("Tokens saved", f"{saved:,}")
        b.metric("Paid tokens", f"{remote:,}")
        st.metric("Answered free", f"{share}%",
                  f"{run.get('tasks_done', 0)} tasks · {m.get('status', '')}")
        routes = m.get("routes", {})
        if routes:
            st.caption("How answers were produced")
            st.bar_chart(routes, horizontal=True, height=30 + 36 * len(routes),
                         color="#2a78d6")

# ---------------------------------------------------------------- chat

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("#### Ask anything")
    st.caption("Watch the answer arrive with the route that produced it and what it cost.")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("meta"):
            st.caption(msg["meta"])

prompt = st.chat_input("Type a task…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("routing…"):
                result = playground().solve(prompt)
            answer, meta = result["answer"], meta_line(result)
            st.markdown(answer)
            st.caption(meta)
        except Exception as e:  # noqa: BLE001 — show the failure, don't crash the UI
            answer, meta = f"⚠️ {e}", None
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "text": answer, "meta": meta})

if live:
    time.sleep(2)
    st.rerun()
