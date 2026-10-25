"""
Streamlit Web Application: Network Load Analysis
Clean, streamlined dashboard for statistical network traffic & capacity analysis.
"""

import os
import sys

# Ensure local module directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from simulator import NetworkLoadSimulator, TrafficSimulationConfig
from statistical_model import NormalDistributionAnalyzer
from load_balancer import LoadBalancingSimulator

# Page configuration
st.set_page_config(
    page_title="Network Load Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined styling: clean typography, elegant cards, reduced visual noise
st.markdown(
    """
    <style>
    /* Global tweaks */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Elegant metric card */
    .metric-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .metric-num {
        font-size: 1.65rem;
        font-weight: 600;
        color: #f8fafc;
        margin-top: 2px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Subdued section headers */
    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📡 Network Load Analysis")
st.caption("Statistical traffic modeling, capacity planning, and routing simulation.")

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Traffic Settings")
    duration_hours = st.slider("Duration (Hours)", min_value=12, max_value=72, value=24, step=6)
    base_load = st.slider("Base Load (Gbps)", min_value=10.0, max_value=100.0, value=45.0, step=5.0)
    
    with st.expander("Advanced Parameters", expanded=False):
        diurnal_amp = st.slider("Diurnal Amplitude (Gbps)", min_value=0.0, max_value=40.0, value=20.0, step=2.0)
        noise_std = st.slider("Gaussian Noise σ (Gbps)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        burst_prob = st.slider("Burst Probability (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100.0
        burst_magnitude = st.slider("Burst Magnitude (Gbps)", min_value=5.0, max_value=50.0, value=20.0, step=5.0)
        seed = st.number_input("Random Seed", value=42, step=1)

# Generate Traffic Trace
config = TrafficSimulationConfig(
    duration_hours=duration_hours,
    sampling_rate_minutes=2,
    base_load_gbps=base_load,
    diurnal_amplitude_gbps=diurnal_amp,
    random_noise_std_gbps=noise_std,
    burst_probability=burst_prob,
    burst_magnitude_gbps=burst_magnitude,
    random_seed=int(seed),
)

simulator = NetworkLoadSimulator(config)
df_trace = simulator.generate_trace()
analyzer = NormalDistributionAnalyzer(df_trace["total_load_gbps"])
norm_metrics = analyzer.test_normality()

# High-Level Summary KPI Cards
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Mean Load</div>
            <div class="metric-num">{norm_metrics['mean']:.1f} <span style="font-size:0.95rem; font-weight:400; color:#94a3b8;">Gbps</span></div>
            <div class="metric-sub">Average traffic load</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Std Deviation</div>
            <div class="metric-num">±{norm_metrics['std']:.1f} <span style="font-size:0.95rem; font-weight:400; color:#94a3b8;">Gbps</span></div>
            <div class="metric-sub">Traffic volatility (σ)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    peak_load = float(df_trace["total_load_gbps"].max())
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Observed Peak</div>
            <div class="metric-num">{peak_load:.1f} <span style="font-size:0.95rem; font-weight:400; color:#94a3b8;">Gbps</span></div>
            <div class="metric-sub">Maximum load reached</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    is_normal = norm_metrics["is_approximately_normal"]
    status_label = "Gaussian" if is_normal else "Non-Gaussian"
    status_color = "#34d399" if is_normal else "#f87171"
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Distribution</div>
            <div class="metric-num" style="color: {status_color};">{status_label}</div>
            <div class="metric-sub">Skew: {norm_metrics['skewness']:.2f} | Kurt: {norm_metrics['kurtosis']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Clean Navigation Tabs
tab_overview, tab_stats, tab_capacity, tab_lb = st.tabs([
    "📈 Traffic Overview",
    "📊 Distribution Fitting",
    "🛡️ Capacity & SLA Risk",
    "⚖️ Load Balancer",
])

# ----------------- TAB 1: TRAFFIC OVERVIEW -----------------
with tab_overview:
    st.subheader("Network Load Over Time")

    fig_trace = go.Figure()
    fig_trace.add_trace(
        go.Scatter(
            x=df_trace["timestamp"],
            y=df_trace["total_load_gbps"],
            mode="lines",
            name="Total Load",
            line=dict(color="#38bdf8", width=1.6),
        )
    )
    fig_trace.add_trace(
        go.Scatter(
            x=df_trace["timestamp"],
            y=df_trace["deterministic_trend"],
            mode="lines",
            name="Diurnal Baseline",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
        )
    )

    bursts_df = df_trace[df_trace["is_burst"]]
    if not bursts_df.empty:
        fig_trace.add_trace(
            go.Scatter(
                x=bursts_df["timestamp"],
                y=bursts_df["total_load_gbps"],
                mode="markers",
                name="Bursts",
                marker=dict(color="#f87171", size=5, symbol="circle"),
            )
        )

    fig_trace.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time",
        yaxis_title="Load (Gbps)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trace, width="stretch")

    with st.expander("Show Data Preview"):
        st.dataframe(
            df_trace[["timestamp", "total_load_gbps", "deterministic_trend", "is_burst"]].head(25),
            width="stretch",
        )

# ----------------- TAB 2: DISTRIBUTION FITTING -----------------
with tab_stats:
    st.subheader("Normal Distribution Assessment")

    c_left, c_right = st.columns(2)

    with c_left:
        # PDF Histogram + Theoretical Normal Curve
        x_min = max(0, df_trace["total_load_gbps"].min() - 5)
        x_max = df_trace["total_load_gbps"].max() + 5
        x_vals = np.linspace(x_min, x_max, 150)
        fitted_pdf = stats.norm.pdf(x_vals, loc=norm_metrics["mean"], scale=norm_metrics["std"])

        fig_pdf = go.Figure()
        fig_pdf.add_trace(
            go.Histogram(
                x=df_trace["total_load_gbps"],
                histnorm="probability density",
                name="Observed Load",
                opacity=0.6,
                marker_color="#0284c7",
                nbinsx=35,
            )
        )
        fig_pdf.add_trace(
            go.Scatter(
                x=x_vals,
                y=fitted_pdf,
                mode="lines",
                name=f"Fitted N(μ={norm_metrics['mean']:.1f}, σ={norm_metrics['std']:.1f})",
                line=dict(color="#f43f5e", width=2),
            )
        )
        fig_pdf.update_layout(
            title="Density vs Fitted Normal Curve",
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Load (Gbps)",
            yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pdf, width="stretch")

    with c_right:
        # Q-Q Plot
        sorted_load = np.sort(df_trace["total_load_gbps"])
        n = len(sorted_load)
        theoretical_quantiles = stats.norm.ppf(
            (np.arange(1, n + 1) - 0.5) / n, loc=norm_metrics["mean"], scale=norm_metrics["std"]
        )

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_load,
                mode="markers",
                name="Quantiles",
                marker=dict(color="#a855f7", size=3, opacity=0.7),
            )
        )
        min_q = min(theoretical_quantiles.min(), sorted_load.min())
        max_q = max(theoretical_quantiles.max(), sorted_load.max())
        fig_qq.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                name="Ideal Normal",
                line=dict(color="#94a3b8", dash="dash"),
            )
        )
        fig_qq.update_layout(
            title="Q-Q Plot (Normal Fit)",
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Theoretical Quantiles (Gbps)",
            yaxis_title="Observed Quantiles (Gbps)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_qq, width="stretch")

    with st.expander("Goodness-of-Fit Tests"):
        test_df = pd.DataFrame([
            {
                "Test": "D'Agostino-Pearson",
                "Statistic": f"{norm_metrics['dagostino_stat']:.3f}",
                "p-value": f"{norm_metrics['dagostino_p_value']:.4e}",
                "Result": "Gaussian" if norm_metrics['dagostino_p_value'] > 0.05 else "Non-Gaussian tail/skew",
            },
            {
                "Test": "Kolmogorov-Smirnov",
                "Statistic": f"{norm_metrics['ks_stat']:.3f}",
                "p-value": f"{norm_metrics['ks_p_value']:.4e}",
                "Result": "Good Fit" if norm_metrics['ks_p_value'] > 0.01 else "Tail deviation",
            },
        ])
        st.dataframe(test_df, width="stretch", hide_index=True)

# ----------------- TAB 3: CAPACITY & SLA RISK -----------------
with tab_capacity:
    st.subheader("Capacity Planning & Risk Assessment")

    cap_left, cap_right = st.columns(2)

    with cap_left:
        target_sla = st.select_slider(
            "Target Reliability / SLA (%)",
            options=[95.0, 99.0, 99.9, 99.99],
            value=99.9,
        )
        provision_res = analyzer.recommend_provisioning_capacity(target_sla)

        st.metric(
            label="Recommended Capacity",
            value=f"{provision_res['recommended_capacity_gbps']:.1f} Gbps",
            delta=f"+{provision_res['safety_buffer_gbps']:.1f} Gbps buffer above mean",
        )
        st.caption(
            f"Formula: $C = \\mu + Z \\cdot \\sigma$ &nbsp;|&nbsp; Z = `{provision_res['z_multiplier']:.2f}` &nbsp;|&nbsp; "
            f"Safety buffer: `{provision_res['safety_buffer_gbps']:.1f} Gbps`"
        )

    with cap_right:
        default_cap = float(round(norm_metrics["mean"] + 2.5 * norm_metrics["std"], 1))
        fixed_cap = st.number_input(
            "Test Fixed Capacity (Gbps)",
            min_value=10.0,
            max_value=250.0,
            value=default_cap,
            step=5.0,
        )
        sla_eval = analyzer.calculate_sla_violation_probability(fixed_cap)

        st.metric(
            label="Breach Probability P(X > C)",
            value=f"{sla_eval['theoretical_violation_prob']*100:.3f}%",
            delta=f"Z-Score: {sla_eval['z_score']:.2f}",
        )
        st.caption(
            f"Empirical breaches in trace: `{sla_eval['empirical_violation_prob']*100:.2f}%` &nbsp;|&nbsp; "
            f"Reliability: `{sla_eval['theoretical_sla_reliability_percent']:.2f}%`"
        )

    # Clean CDF Reliability Curve
    x_range = np.linspace(
        max(0, norm_metrics["mean"] - 3 * norm_metrics["std"]),
        norm_metrics["mean"] + 4 * norm_metrics["std"],
        200,
    )
    cdf_vals = stats.norm.cdf(x_range, loc=norm_metrics["mean"], scale=norm_metrics["std"]) * 100.0

    fig_cdf = go.Figure()
    fig_cdf.add_trace(
        go.Scatter(
            x=x_range,
            y=cdf_vals,
            mode="lines",
            name="Reliability Curve",
            line=dict(color="#10b981", width=2),
        )
    )
    fig_cdf.add_vline(
        x=provision_res["recommended_capacity_gbps"],
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"{target_sla}%: {provision_res['recommended_capacity_gbps']:.1f} Gbps",
    )
    fig_cdf.update_layout(
        title="Reliability Curve: Provisioned Capacity vs SLA Coverage",
        template="plotly_dark",
        xaxis_title="Allocated Capacity (Gbps)",
        yaxis_title="SLA Coverage (%)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_cdf, width="stretch")

# ----------------- TAB 4: LOAD BALANCER -----------------
with tab_lb:
    st.subheader("Load Balancer Routing Comparison")

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        num_nodes = st.slider("Worker Gateway Nodes", min_value=2, max_value=8, value=4)
    with ctrl2:
        node_cap = st.slider("Node Capacity (Gbps)", min_value=10.0, max_value=50.0, value=25.0, step=2.5)

    lb_sim = LoadBalancingSimulator(num_nodes=num_nodes, node_capacity_gbps=node_cap)

    strategies = ["round_robin", "least_loaded", "statistical_aware"]
    results = {}
    for strat in strategies:
        results[strat] = lb_sim.run_simulation(df_trace["total_load_gbps"].values, strategy=strat)

    comparison_data = []
    for s in strategies:
        r = results[s]
        comparison_data.append({
            "Strategy": s.replace("_", " ").title(),
            "Served (Gbps)": f"{r['total_served_gbps']:.1f}",
            "Dropped (Gbps)": f"{r['total_dropped_gbps']:.1f}",
            "Drop Rate": f"{r['packet_drop_rate_pct']:.2f}%",
            "Imbalance (σ)": f"{r['load_imbalance_std']:.2f}",
        })
    st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)

    # Node load comparison chart
    fig_nodes = go.Figure()
    for s in strategies:
        node_metrics = results[s]["node_metrics"]
        node_names = [n["node_id"] for n in node_metrics]
        avg_loads = [n["avg_load_gbps"] for n in node_metrics]
        fig_nodes.add_trace(go.Bar(x=node_names, y=avg_loads, name=s.replace("_", " ").title()))

    fig_nodes.update_layout(
        barmode="group",
        title="Average Load Per Node by Strategy",
        template="plotly_dark",
        yaxis_title="Average Load (Gbps)",
        xaxis_title="Worker Node",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_nodes, width="stretch")
