"""Chat — the default page. Every answer is captioned with route, cost, latency."""

from __future__ import annotations

import streamlit as st

from src.frontend.data import FREE_ROUTES
from src.frontend.playground import Playground

st.set_page_config(layout="centered")

SUGGESTIONS = [
    "What is 847 * 36 + 129?",
    "What day of the week was 2024-03-15?",
    "A train travels 240 km in 3 hours. How far does it go in 5 hours?",
    "Summarize in one sentence: Solar power grew rapidly as panel costs fell and storage matured.",
]


@st.cache_resource
def playground() -> Playground:
    return Playground()


def meta_line(r: dict) -> str:
    free = r["route"] in FREE_ROUTES
    dot = ":blue[●]" if free else ":orange[●]"
    cost = "free · 0 tokens" if free else f"{r['remote_tokens']} tokens"
    route = r["route"].replace("_", " ")
    category = r["category"].replace("_", " ")
    return f"{dot} {route} · {category} · {cost} · {r['wall_ms']} ms"


if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("#### Ask anything")
    st.caption("Watch the answer arrive with the route that produced it and what it cost.")
    with st.container(horizontal=True):
        for i, suggestion in enumerate(SUGGESTIONS):
            if st.button(suggestion, key=f"suggest_{i}"):
                st.session_state.messages.append({"role": "user", "text": suggestion})
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if prompt := st.chat_input("Type a task…", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "text": prompt})
    st.rerun()

# An unanswered user message at the tail means it's our turn: generate, render,
# store. Rendering always happens from state first, so the exchange survives
# any rerun (this is what fixed answers not showing up).
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    task = st.session_state.messages[-1]["text"]
    with st.chat_message("assistant"):
        try:
            with st.spinner("routing…"):
                result = playground().solve(task)
            answer, meta = result["answer"], meta_line(result)
        except Exception as e:  # noqa: BLE001 — show the failure, don't crash the UI
            answer, meta = f"⚠️ {e}", None
        st.markdown(answer)
        if meta:
            st.caption(meta)
    st.session_state.messages.append({"role": "assistant", "text": answer, "meta": meta})

with st.sidebar:
    st.caption("Backends follow `VERDICT_*` env — see Insights for the full run picture.")
