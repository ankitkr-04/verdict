"""Insights — the full run picture: savings, routes, timeline, latency, per-task detail."""

from __future__ import annotations

import time

import altair as alt
import pandas as pd
import streamlit as st

from src.frontend.data import (
    DEFAULT_LEDGER,
    DEFAULT_METRICS,
    FREE_ROUTES,
    load_ledger,
    load_metrics,
)

st.set_page_config(layout="wide")

FREE_COLOR, PAID_COLOR = "#2a78d6", "#eb6834"

with st.sidebar:
    with st.expander(":material/settings: data source"):
        metrics_path = st.text_input("metrics file", str(DEFAULT_METRICS))
        ledger_path = st.text_input("ledger file", str(DEFAULT_LEDGER))
    live = st.toggle("Auto-refresh (2s)", value=False,
                     help="Follow a run while it writes the artifacts.")

m = load_metrics(metrics_path)
ledger = load_ledger(ledger_path)

if not m:
    st.info("No run artifacts yet — run the pipeline (or `python eval/golden.py`), "
            "then numbers appear here. Point *data source* at any run's files.")
    st.stop()

tok, run = m.get("tokens", {}), m.get("run", {})
saved = tok.get("est_saved_vs_all_remote", 0)
remote = tok.get("remote_total", 0)
share = round(100 * saved / (saved + remote)) if (saved + remote) else 0
status = m.get("status", "unknown")

st.caption(f"run started {run.get('started_at', '?')} · status **{status}**")

# ---------------------------------------------------------------- KPI row
with st.container(horizontal=True):
    st.metric("Tokens saved", f"{saved:,}", f"{share}% answered free",
              delta_color="normal", border=True)
    st.metric("Paid tokens", f"{remote:,}",
              f"{tok.get('remote_prompt', 0):,} in · {tok.get('remote_completion', 0):,} out",
              delta_color="off", delta_arrow="off", border=True)
    st.metric("Free local tokens", f"{tok.get('local_total', 0):,}",
              "cost nothing, saved the rest", delta_color="off", delta_arrow="off",
              border=True)
    st.metric("Tasks", f"{run.get('tasks_done', 0)}",
              f"{run.get('tasks_remaining', 0)} remaining", delta_color="off",
              delta_arrow="off", border=True)
    st.metric("Wall clock", f"{run.get('wall_s', 0):.0f}s",
              f"{run.get('headroom_s', 0):.0f}s headroom", border=True)

# ---------------------------------------------------------------- routes + latency
left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**How answers were produced**")
    st.caption(":blue[●] free — deterministic code or the local model · "
               ":orange[●] paid — a remote call")
    routes = m.get("routes", {})
    if routes:
        df = pd.DataFrame(
            {"route": [r.replace("_", " ") for r in routes],
             "tasks": list(routes.values()),
             "kind": ["free" if r in FREE_ROUTES else "paid" for r in routes]}
        )
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=4, height=20)
            .encode(
                x=alt.X("tasks:Q", axis=alt.Axis(tickMinStep=1, title=None)),
                y=alt.Y("route:N", sort="-x", title=None),
                color=alt.Color("kind:N", legend=None,
                                scale=alt.Scale(domain=["free", "paid"],
                                                range=[FREE_COLOR, PAID_COLOR])),
                tooltip=["route", "tasks", "kind"],
            )
        )
        st.altair_chart(chart, width="stretch")

with right, st.container(border=True):
    st.markdown("**Latency by phase (ms)**")
    st.caption("Across finished tasks — p50 / p95 / max.")
    lat = m.get("latency_ms", {})
    nice = {"task_wall": "task wall", "local_gen_per_task": "local generation",
            "remote_call": "remote call", "dispatch": "dispatch", "queue_wait": "queue wait"}
    if lat:
        st.dataframe(
            pd.DataFrame(
                [{"phase": nice.get(k, k), **v} for k, v in lat.items()]
            ),
            hide_index=True, height="auto",
        )

# ---------------------------------------------------------------- token timeline
if ledger:
    with st.container(border=True):
        st.markdown("**Paid tokens over the run**")
        st.caption("Cumulative remote tokens against the run clock — flat is winning. "
                   "Dots mark paid calls.")
        rows = sorted((r for r in ledger if isinstance(r.get("t_s"), (int, float))),
                      key=lambda r: r["t_s"])
        cum, pts = 0, []
        for r in rows:
            rt = (r.get("remote_prompt_tokens", 0) or 0) + (r.get("remote_completion_tokens", 0) or 0)
            cum += rt
            pts.append({"t_s": r["t_s"], "cumulative": cum, "paid": rt > 0,
                        "task": r.get("task_id", ""), "tokens": rt})
        tl = pd.DataFrame(pts)
        line = (
            alt.Chart(tl)
            .mark_line(interpolate="step-after", color=FREE_COLOR, strokeWidth=2)
            .encode(x=alt.X("t_s:Q", title="run clock (s)"),
                    y=alt.Y("cumulative:Q", title="remote tokens"),
                    tooltip=["task", "tokens", "cumulative", "t_s"])
        )
        dots = (
            alt.Chart(tl[tl["paid"]])
            .mark_circle(size=90, color=PAID_COLOR)
            .encode(x="t_s:Q", y="cumulative:Q", tooltip=["task", "tokens", "cumulative"])
        )
        st.altair_chart(line + dots, width="stretch")

# ---------------------------------------------------------------- per category
by_cat = m.get("by_category", {})
if by_cat:
    with st.container(border=True):
        st.markdown("**Per category**")
        st.dataframe(
            pd.DataFrame([
                {
                    "category": name.replace("_", " "),
                    "tasks": c.get("tasks", 0),
                    "routes": ", ".join(f"{r.replace('_', ' ')} ×{n}"
                                        for r, n in (c.get("routes") or {}).items()),
                    "paid tokens": c.get("remote_tokens", 0),
                    "free tokens": c.get("local_tokens", 0),
                    "wall p50 ms": (c.get("wall_ms") or {}).get("p50"),
                    "wall max ms": (c.get("wall_ms") or {}).get("max"),
                }
                for name, c in sorted(by_cat.items(), key=lambda kv: -kv[1].get("tasks", 0))
            ]),
            hide_index=True, height="auto",
        )

# ---------------------------------------------------------------- every task
if ledger:
    with st.container(border=True):
        st.markdown("**Every task**")
        st.caption("One row per finished task, from the ledger — newest last.")
        st.dataframe(
            pd.DataFrame([
                {
                    "task": r.get("task_id"),
                    "category": (r.get("category") or "").replace("_", " "),
                    "route": (r.get("route") or "").replace("_", " "),
                    "free": "🔵" if r.get("route") in FREE_ROUTES else "🟠",
                    "confidence": r.get("confidence"),
                    "paid tokens": (r.get("remote_prompt_tokens", 0) or 0)
                                   + (r.get("remote_completion_tokens", 0) or 0),
                    "free tokens": (r.get("local_prompt_tokens", 0) or 0)
                                   + (r.get("local_completion_tokens", 0) or 0),
                    "wall ms": r.get("wall_ms"),
                    "t (s)": r.get("t_s"),
                    "detail": r.get("detail", ""),
                }
                for r in sorted(ledger, key=lambda r: r.get("t_s") or 0)
            ]),
            hide_index=True,
        )

# ---------------------------------------------------------------- environment
hw, cfg = m.get("hardware", {}), m.get("config", {})
if hw or cfg:
    with st.expander(":material/dns: environment & config snapshot"):
        a, b = st.columns(2)
        with a:
            st.caption("hardware")
            st.dataframe(pd.DataFrame(
                [{"key": k, "value": ", ".join(map(str, v)) if isinstance(v, list) else str(v)}
                 for k, v in hw.items()]), hide_index=True, height="auto")
        with b:
            st.caption("config")
            st.dataframe(pd.DataFrame(
                [{"key": k, "value": str(v)} for k, v in cfg.items()]),
                hide_index=True, height="auto")

if live:
    time.sleep(2)
    st.rerun()
