"""
Streamlit demo — Neural Beings: Met Museum Cultural Pattern Discovery.

Run from the project root:
    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from demo.predictor import get_sample_inputs, predict

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Beings — Met Museum ML",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<style>.stApp { background-color: #0e1117; }</style>",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_DATA_PATH = Path(__file__).parent / "data" / "umap_embeddings.csv"

_AGE_BUCKETS = [
    ("Ancient (< 500)",          -9999,  500),
    ("Medieval (500–1400)",         500, 1400),
    ("Early Modern (1400–1800)",   1400, 1800),
    ("Modern (1800–1950)",         1800, 1950),
    ("Contemporary (1950+)",       1950, 9999),
]
_AGE_COLORS = ["#2d6a4f", "#52b788", "#b7e4c7", "#f4a261", "#e76f51"]

# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_data
def load_umap_data() -> pd.DataFrame:
    return pd.read_csv(_DATA_PATH)


def _age_bucket(date_val) -> str:
    try:
        d = float(date_val)
    except (ValueError, TypeError):
        return "Unknown"
    for label, lo, hi in _AGE_BUCKETS:
        if lo <= d < hi:
            return label
    return _AGE_BUCKETS[-1][0]


def _embed(fig: go.Figure, height: int) -> str:
    """Wrap a Plotly figure in a minimal dark-background page for st.components.v1.html."""
    snippet = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return (
        "<!DOCTYPE html><html><head></head>"
        f'<body style="margin:0;padding:0;background:#0e1117;">{snippet}</body>'
        "</html>"
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏛️ Neural Beings — Met Museum Cultural Pattern Discovery")

tab1, tab2 = st.tabs(["🌌 UMAP Explorer", "🔮 Guess the Department"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UMAP EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    df_full = load_umap_data()
    all_departments = sorted(df_full["department"].unique().tolist())

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 2, 1])

    with ctrl1:
        color_by = st.selectbox(
            "Color by",
            ["Department", "Culture (top 15)", "Object Age Bucket"],
        )

    with ctrl2:
        department_filter = st.multiselect(
            "Filter departments",
            all_departments,
            default=all_departments,
        )

    with ctrl3:
        sample_size = st.slider(
            "Sample size",
            min_value=1_000,
            max_value=30_000,
            value=15_000,
            step=1_000,
        )

    # ── Filter & sample ───────────────────────────────────────────────────────
    df = df_full[df_full["department"].isin(department_filter)].copy()
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # ── Color mapping ─────────────────────────────────────────────────────────
    if color_by == "Department":
        color_col = "department"
        # Fixed palette keyed to all departments so colors never shift on filter
        color_map = {
            d: px.colors.qualitative.Dark24[i % 24]
            for i, d in enumerate(sorted(all_departments))
        }
        unique_labels = sorted(df["department"].unique().tolist())

    elif color_by == "Culture (top 15)":
        top14 = df_full["culture"].value_counts().head(14).index.tolist()
        df["_color_col"] = df["culture"].apply(
            lambda x: x if x in set(top14) else "Other"
        )
        color_col = "_color_col"
        ordered = top14 + ["Other"]
        color_map = {
            c: px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)]
            for i, c in enumerate(ordered)
        }
        present = set(df["_color_col"].unique())
        unique_labels = [c for c in ordered if c in present]

    else:  # Object Age Bucket
        df["_color_col"] = df["object_begin_date"].apply(_age_bucket)
        color_col = "_color_col"
        color_map = {b[0]: _AGE_COLORS[i] for i, b in enumerate(_AGE_BUCKETS)}
        present = set(df["_color_col"].unique())
        unique_labels = [b[0] for b in _AGE_BUCKETS if b[0] in present]

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    for lbl in unique_labels:
        sub = df[df[color_col] == lbl]

        hover = (
            "<b>" + sub["object_name"].astype(str) + "</b><br>"
            + "Dept: "    + sub["department"].astype(str) + "<br>"
            + "Culture: " + sub["culture"].astype(str) + "<br>"
            + "Medium: "  + sub["medium"].astype(str).str[:55] + "<br>"
            + "Date: "    + sub["object_begin_date"].astype(str)
        )

        fig.add_trace(go.Scatter3d(
            x=sub["umap_x"],
            y=sub["umap_y"],
            z=sub["umap_z"],
            mode="markers",
            name=lbl,
            marker=dict(size=2.5, color=color_map.get(lbl, "#888888"), opacity=0.7),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        height=700,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            bgcolor="#161b22",
        ),
        legend=dict(x=1.0, y=0.5, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=180, t=20, b=0),
    )

    components.html(_embed(fig, 700), height=710)

    # ── Metrics ───────────────────────────────────────────────────────────────
    top_cult_series = (
        df["culture"]
        .replace("Unknown", pd.NA)
        .dropna()
        .value_counts()
    )
    top_cult_name = top_cult_series.index[0] if len(top_cult_series) else "Unknown"

    m1, m2, m3 = st.columns(3)
    m1.metric("Points shown",        f"{len(df):,}")
    m2.metric("Departments visible",  df["department"].nunique())
    m3.metric("Most common culture",  top_cult_name)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GUESS THE DEPARTMENT
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    # Initialise session state defaults (only on first run)
    _PRED_DEFAULTS: dict = {
        "pred_medium":  "",
        "pred_culture": "",
        "pred_tags":    "",
        "pred_begin":   1850,
        "pred_end":     1900,
        "pred_result":  None,
    }
    for _k, _v in _PRED_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Callbacks (must be defined before any widget renders) ────────────────
    def _do_predict() -> None:
        """on_click callback for the Predict button."""
        st.session_state["pred_result"] = predict(
            medium=st.session_state["pred_medium"],
            culture=st.session_state["pred_culture"],
            tags=st.session_state["pred_tags"],
            object_begin_date=int(st.session_state["pred_begin"]),
            object_end_date=int(st.session_state["pred_end"]),
        )

    def _load_example(samp: dict) -> None:
        """on_click callback for each example button."""
        st.session_state["pred_medium"]  = samp.get("Medium",  "")
        st.session_state["pred_culture"] = samp.get("Culture", "")
        st.session_state["pred_tags"]    = samp.get("Tags",    "")
        st.session_state["pred_begin"]   = int(samp.get("Object Begin Date", 1850))
        st.session_state["pred_end"]     = int(samp.get("Object End Date",   1900))
        st.session_state["pred_result"]  = None

    left_col, right_col = st.columns(2)

    # ── Left: inputs ──────────────────────────────────────────────────────────
    with left_col:
        st.subheader("Artwork Details")

        st.text_input("Medium",  placeholder="e.g. Oil on canvas",       key="pred_medium")
        st.text_input("Culture", placeholder="e.g. French",              key="pred_culture")
        st.text_input("Tags",    placeholder="e.g. portrait, landscape", key="pred_tags")

        dc1, dc2 = st.columns(2)
        with dc1:
            st.number_input(
                "Object Begin Date", step=1,
                min_value=-7000, max_value=2026,
                key="pred_begin",
            )
        with dc2:
            st.number_input(
                "Object End Date", step=1,
                min_value=-7000, max_value=2026,
                key="pred_end",
            )

        st.button("🎯 Predict Department", type="primary", on_click=_do_predict)

        st.markdown("---")
        st.markdown("**Try an example:**")

        _samples = get_sample_inputs()
        _ex_cols = st.columns(len(_samples))
        for _i, (_col, _samp) in enumerate(zip(_ex_cols, _samples)):
            _lbl   = _samp.get("label", f"#{_i+1}")
            _short = (_lbl[:17] + "…") if len(_lbl) > 17 else _lbl
            _col.button(
                _short, key=f"ex_{_i}",
                on_click=_load_example, args=(_samp,),
            )

    # ── Right: results ────────────────────────────────────────────────────────
    with right_col:
        _result = st.session_state.get("pred_result")

        if _result:
            st.subheader("Prediction")

            st.metric(
                label="Predicted Department",
                value=_result["department"],
                delta=f"{_result['confidence'] * 100:.1f}% confidence",
            )

            st.markdown("#### Top 5 Departments")

            _probs = _result["probabilities"]
            _top5  = sorted(_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            _depts = [d for d, _ in _top5]
            _vals  = [v for _, v in _top5]

            # Warm gold → orange-red gradient (best → 5th best)
            _bar_colors = ["#f4a261", "#f19050", "#ee7e40", "#eb6c30", "#e76f51"]

            _bar_fig = go.Figure(go.Bar(
                x=_vals[::-1],
                y=_depts[::-1],
                orientation="h",
                marker_color=_bar_colors[::-1],
                text=[f"{v * 100:.1f}%" for v in _vals][::-1],
                textposition="outside",
                textfont=dict(color="#ffffff"),
            ))
            _bar_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                height=280,
                margin=dict(l=10, r=75, t=10, b=10),
                xaxis=dict(
                    range=[0, max(_vals) * 1.4],
                    showgrid=False,
                    showticklabels=False,
                ),
                yaxis=dict(showgrid=False),
            )
            components.html(_embed(_bar_fig, 280), height=290)

            st.info(
                "ℹ️ This model was trained on 388k artworks using XGBoost with "
                "214 engineered features including TF-IDF on Medium, Tags, and Period."
            )

        else:
            st.markdown(
                "<div style='color:#555;margin-top:80px;text-align:center;'>"
                "<h3>← Fill in details and click Predict</h3>"
                "<p>Or try one of the example artworks below the form</p>"
                "</div>",
                unsafe_allow_html=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "CS 6140 · Northeastern University "
)
