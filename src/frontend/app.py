"""verdict — chat-first UI with a detailed insights page. Run from the repo root:

    streamlit run src/frontend/app.py

Chat talks to the real pipeline (dispatcher -> deterministic -> local+verify ->
escalate) in-process via Playground; backends follow the same VERDICT_* env as any
run. Insights reads the run artifacts (metrics + ledger). Dev/demo tool only —
never part of the container image.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="verdict", page_icon="⚖️")

with st.sidebar:
    st.title("⚖️ verdict")
    st.caption(
        "Answers as much as it can **for free** — plain code first, then a small "
        "local model with verification — and only pays for a remote model when a "
        "task truly needs one."
    )

page = st.navigation([
    st.Page("app_pages/chat.py", title="Chat", icon=":material/forum:", default=True),
    st.Page("app_pages/insights.py", title="Insights", icon=":material/monitoring:"),
])
page.run()
