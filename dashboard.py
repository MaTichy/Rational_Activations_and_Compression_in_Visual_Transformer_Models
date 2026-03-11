"""Streamlit + Plotly dashboard for SimpleViT experiment results.

Launch with: streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────
RESULTS_PATH = Path("experiments/results.json")
COLOR_RELU = "#636EFA"
COLOR_RATIONAL = "#EF553B"
COLORS = {"relu": COLOR_RELU, "rational": COLOR_RATIONAL}
PLOTLY_TEMPLATE = "plotly_dark"

st.set_page_config(layout="wide", page_title="SimpleViT Experiment Dashboard")


@st.cache_data
def load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _has_section(data: dict, activation: str, section: str) -> bool:
    return section in data.get("models", {}).get(activation, {})


# ── Page: Training Overview ────────────────────────────────────────────
def page_training(data: dict):
    st.header("Training Overview")

    models = data.get("models", {})
    activations = [a for a in ["relu", "rational"] if _has_section(data, a, "training")]

    if not activations:
        st.info("No training data available yet.")
        return

    # Metric cards
    cols = st.columns(len(activations) * 3)
    for idx, act in enumerate(activations):
        t = models[act]["training"]
        cols[idx * 3 + 0].metric(f"{act.upper()} Test Acc", f"{t['test_acc']:.4f}" if t["test_acc"] else "N/A")
        cols[idx * 3 + 1].metric(f"{act.upper()} Params", f"{t['param_count']:,}")
        cols[idx * 3 + 2].metric(f"{act.upper()} Time", f"{t['training_time_seconds']}s")

    # 2x2 subplot grid
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Train Loss", "Val Loss", "Train Acc", "Val Acc"))

    chart_keys = [
        (1, 1, "train_loss"),
        (1, 2, "val_loss"),
        (2, 1, "train_acc"),
        (2, 2, "val_acc"),
    ]

    for act in activations:
        epochs_data = models[act]["training"].get("epochs", [])
        if not epochs_data:
            continue
        ep = [d["epoch"] for d in epochs_data]
        for row, col, key in chart_keys:
            vals = [d.get(key) for d in epochs_data]
            if any(v is not None for v in vals):
                fig.add_trace(
                    go.Scatter(x=ep, y=vals, name=act.upper(), legendgroup=act,
                               showlegend=(row == 1 and col == 1),
                               line=dict(color=COLORS[act])),
                    row=row, col=col,
                )

    fig.update_layout(template=PLOTLY_TEMPLATE, height=600, title_text="Training Curves")
    st.plotly_chart(fig, use_container_width=True)

    # Raw data expander
    with st.expander("Raw Epoch Data"):
        for act in activations:
            epochs_data = models[act]["training"].get("epochs", [])
            if epochs_data:
                st.subheader(act.upper())
                st.dataframe(pd.DataFrame(epochs_data), use_container_width=True)


# ── Page: Pruning Results ──────────────────────────────────────────────
def page_pruning(data: dict):
    st.header("Pruning Results (Lottery Ticket Hypothesis)")

    models = data.get("models", {})
    activations = [a for a in ["relu", "rational"] if _has_section(data, a, "pruned")]

    if not activations:
        st.info("No pruning data available yet.")
        return

    # Metric cards
    cols = st.columns(len(activations) * 3)
    for idx, act in enumerate(activations):
        p = models[act]["pruned"]
        t = models[act].get("training", {})
        final_acc = p.get("final_test_acc")
        baseline_acc = t.get("test_acc")
        delta = None
        if final_acc is not None and baseline_acc is not None:
            delta = f"{final_acc - baseline_acc:+.4f}"
        cols[idx * 3 + 0].metric(f"{act.upper()} Pruned Acc", f"{final_acc:.4f}" if final_acc else "N/A", delta=delta)
        cols[idx * 3 + 1].metric(f"{act.upper()} FF Sparsity", f"{p.get('total_sparsity_ff', 0)*100:.1f}%")
        cols[idx * 3 + 2].metric(f"{act.upper()} Attn Sparsity", f"{p.get('total_sparsity_attn', 0)*100:.1f}%")

    # Accuracy vs pruning round
    fig_acc = go.Figure()
    for act in activations:
        rounds = models[act]["pruned"].get("pruning_rounds", [])
        baseline = models[act].get("training", {}).get("test_acc")
        x = [0] + [r["round"] for r in rounds]
        y = [baseline] + [r.get("test_acc_after_round") for r in rounds]
        fig_acc.add_trace(go.Scatter(x=x, y=y, name=act.upper(), mode="lines+markers",
                                     line=dict(color=COLORS[act])))

    fig_acc.update_layout(template=PLOTLY_TEMPLATE, title="Test Accuracy vs Pruning Round",
                          xaxis_title="Pruning Round (0 = unpruned)", yaxis_title="Test Accuracy")
    st.plotly_chart(fig_acc, use_container_width=True)

    # Sparsity bar chart
    fig_sp = go.Figure()
    for act in activations:
        rounds = models[act]["pruned"].get("pruning_rounds", [])
        x = [f"Round {r['round']}" for r in rounds]
        fig_sp.add_trace(go.Bar(x=x, y=[r["sparsity_ff"] * 100 for r in rounds],
                                name=f"{act.upper()} FF", marker_color=COLORS[act]))
        fig_sp.add_trace(go.Bar(x=x, y=[r["sparsity_attn"] * 100 for r in rounds],
                                name=f"{act.upper()} Attn",
                                marker_color=COLORS[act], opacity=0.6))

    fig_sp.update_layout(template=PLOTLY_TEMPLATE, barmode="group",
                         title="Sparsity per Pruning Round", yaxis_title="Sparsity %")
    st.plotly_chart(fig_sp, use_container_width=True)


# ── Page: Quantisation ─────────────────────────────────────────────────
def page_quantisation(data: dict):
    st.header("Quantisation Results")

    models = data.get("models", {})
    activations = [a for a in ["relu", "rational"]
                   if _has_section(data, a, "quantised") or _has_section(data, a, "compressed")]

    if not activations:
        st.info("No quantisation data available yet.")
        return

    # Build comparison data
    variants = ["Original", "Quantised", "Pruned", "Compressed", "Compressed+Quantised"]
    section_keys = ["training", "quantised", "pruned", "compressed", "compressed_quantised"]
    size_key_map = {"training": "model_size_mb", "quantised": "model_size_mb",
                    "pruned": "model_size_mb", "compressed": "model_size_mb",
                    "compressed_quantised": "model_size_mb"}
    acc_key_map = {"training": "test_acc", "quantised": "test_acc",
                   "pruned": "final_test_acc", "compressed": "test_acc",
                   "compressed_quantised": "test_acc"}

    # Grouped bar chart: model sizes
    fig_size = go.Figure()
    for act in activations:
        sizes = []
        for sk in section_keys:
            sec = models[act].get(sk, {})
            sizes.append(sec.get(size_key_map[sk]))
        fig_size.add_trace(go.Bar(x=variants, y=sizes, name=act.upper(),
                                  marker_color=COLORS[act]))

    fig_size.update_layout(template=PLOTLY_TEMPLATE, barmode="group",
                           title="Model Size Across Variants", yaxis_title="Size (MB)")
    st.plotly_chart(fig_size, use_container_width=True)

    # Accuracy comparison table
    rows = []
    for act in activations:
        for variant, sk in zip(variants, section_keys):
            sec = models[act].get(sk, {})
            acc = sec.get(acc_key_map[sk])
            size = sec.get(size_key_map[sk])
            orig_size = models[act].get("training", {}).get("model_size_mb", 1)
            compression = round(orig_size / size, 2) if size and orig_size else None
            rows.append({
                "Activation": act.upper(),
                "Variant": variant,
                "Accuracy": f"{acc:.4f}" if acc is not None else "N/A",
                "Size (MB)": size,
                "Compression Ratio": f"{compression:.2f}x" if compression else "N/A",
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Metric cards
    for act in activations:
        orig_size = models[act].get("training", {}).get("model_size_mb", 0)
        cq_size = models[act].get("compressed_quantised", {}).get("model_size_mb")
        if cq_size and orig_size:
            st.metric(f"{act.upper()} Max Compression", f"{orig_size / cq_size:.2f}x",
                      delta=f"-{orig_size - cq_size:.1f} MB")


# ── Page: Summary ──────────────────────────────────────────────────────
def page_summary(data: dict):
    st.header("Summary")

    models = data.get("models", {})
    activations = [a for a in ["relu", "rational"] if a in models]

    if not activations:
        st.info("No results available yet.")
        return

    # Collect all variants
    variants = ["Original", "Quantised", "Pruned", "Compressed", "Compressed+Quantised"]
    section_keys = ["training", "quantised", "pruned", "compressed", "compressed_quantised"]
    acc_keys = {"training": "test_acc", "quantised": "test_acc",
                "pruned": "final_test_acc", "compressed": "test_acc",
                "compressed_quantised": "test_acc"}

    all_rows = []
    best_acc = (None, None, 0)
    smallest = (None, None, float("inf"))

    for act in activations:
        for variant, sk in zip(variants, section_keys):
            sec = models[act].get(sk, {})
            acc = sec.get(acc_keys[sk])
            size = sec.get("model_size_mb")
            if acc is not None and acc > best_acc[2]:
                best_acc = (act, variant, acc)
            if size is not None and size < smallest[2]:
                smallest = (act, variant, size)
            all_rows.append({
                "Activation": act.upper(),
                "Variant": variant,
                "Test Accuracy": acc,
                "Size (MB)": size,
            })

    # Hero metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Accuracy",
                f"{best_acc[2]:.4f}" if best_acc[0] else "N/A",
                delta=f"{best_acc[0].upper()} {best_acc[1]}" if best_acc[0] else None)
    col2.metric("Smallest Model",
                f"{smallest[2]:.2f} MB" if smallest[0] else "N/A",
                delta=f"{smallest[0].upper()} {smallest[1]}" if smallest[0] else None)

    # Best acc-to-size ratio
    best_ratio = (None, None, 0)
    for row in all_rows:
        acc = row["Test Accuracy"]
        size = row["Size (MB)"]
        if acc and size:
            ratio = acc / size
            if ratio > best_ratio[2]:
                best_ratio = (row["Activation"], row["Variant"], ratio)
    col3.metric("Best Acc/Size Ratio",
                f"{best_ratio[2]:.4f}" if best_ratio[0] else "N/A",
                delta=f"{best_ratio[0]} {best_ratio[1]}" if best_ratio[0] else None)

    # Full comparison table
    st.subheader("All Model Variants")
    df = pd.DataFrame(all_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart of all accuracies
    fig = go.Figure()
    for act in activations:
        act_rows = [r for r in all_rows if r["Activation"] == act.upper()]
        fig.add_trace(go.Bar(
            x=[r["Variant"] for r in act_rows],
            y=[r["Test Accuracy"] for r in act_rows],
            name=act.upper(),
            marker_color=COLORS[act],
        ))
    fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group",
                      title="Test Accuracy Across All Variants",
                      yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    st.title("SimpleViT Experiment Dashboard")
    st.caption("ReLU vs Rational Activations — Train, Prune, Quantise")

    data = load_results()
    if data is None:
        st.error(f"Results file not found at `{RESULTS_PATH}`. Run `python run_experiment.py` first.")
        return

    # Sidebar navigation
    page = st.sidebar.radio("Navigate", [
        "Training Overview",
        "Pruning Results",
        "Quantisation",
        "Summary",
    ])

    # Show metadata in sidebar
    meta = data.get("metadata", {})
    if meta:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Experiment Config**")
        st.sidebar.json(meta)

    if page == "Training Overview":
        page_training(data)
    elif page == "Pruning Results":
        page_pruning(data)
    elif page == "Quantisation":
        page_quantisation(data)
    elif page == "Summary":
        page_summary(data)


if __name__ == "__main__":
    main()
