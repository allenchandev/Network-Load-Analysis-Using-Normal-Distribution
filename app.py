"""
Streamlit Web Application: Network Load Analysis Using Normal Distribution
Interactive Data Science & Capacity Planning Suite for Distributed Systems.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from simulator import NetworkLoadSimulator, TrafficSimulationConfig
from statistical_model import NormalDistributionAnalyzer
from load_balancer import LoadBalancingSimulator

# Page configuration
st.set_page_config(
    page_title="Network Load Analysis & Capacity Planning",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern enterprise look
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .header-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<span class="header-badge">Distributed Systems & Capacity Planning</span>', unsafe_allow_html=True)
st.title("🌐 Network Load Analysis Using Normal Distribution")
st.caption(
    "Probabilistic Modeling, SLA Risk Estimation, and Capacity Provisioning for Hyperscale Data Centers (Meta / Google / Microsoft)"
)

# Sidebar: Simulation Controls
st.sidebar.header("⚙️ Simulation Parameters")

duration_hours = st.sidebar.slider("Simulation Duration (Hours)", min_value=12, max_value=72, value=24, step=6)
base_load = st.sidebar.slider("Base Traffic Load (Gbps)", min_value=10.0, max_value=100.0, value=45.0, step=5.0)
diurnal_amp = st.sidebar.slider("Diurnal Oscillation Amplitude (Gbps)", min_value=5.0, max_value=40.0, value=22.0, step=1.0)
noise_std = st.sidebar.slider("Gaussian Std Dev σ (Noise) (Gbps)", min_value=1.0, max_value=15.0, value=5.5, step=0.5)
burst_prob = st.sidebar.slider("Burst Anomaly Probability (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100.0
burst_magnitude = st.sidebar.slider("Average Burst Magnitude (Gbps)", min_value=5.0, max_value=50.0, value=20.0, step=5.0)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Run generator
config = TrafficSimulationConfig(
    duration_hours=duration_hours,
    sampling_rate_minutes=2,
    base_load_gbps=base_load,
    diurnal_amplitude_gbps=diurnal_amp,
    random_noise_std_gbps=noise_std,
    burst_probability=burst_prob,
    burst_magnitude_gbps=burst_magnitude,
    random_seed=seed,
)

simulator = NetworkLoadSimulator(config)
df_trace = simulator.generate_trace()
analyzer = NormalDistributionAnalyzer(df_trace["total_load_gbps"])
norm_metrics = analyzer.test_normality()

# Key Metric Summary Row
col1, col2, col4, col5 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Mean Load (μ)</div>
            <div class="metric-value">{norm_metrics['mean']:.2f} <span style="font-size:16px;color:#94a3b8">Gbps</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Standard Deviation (σ)</div>
            <div class="metric-value">{norm_metrics['std']:.2f} <span style="font-size:16px;color:#94a3b8">Gbps</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Skewness / Kurtosis</div>
            <div class="metric-value">{norm_metrics['skewness']:.2f} / {norm_metrics['kurtosis']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col5:
    is_normal = norm_metrics["is_approximately_normal"]
    badge_color = "#4ade80" if is_normal else "#f87171"
    status_text = "Approx. Gaussian" if is_normal else "Heavy-Tailed / Skewed"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Normality Status</div>
            <div class="metric-value" style="color:{badge_color}; font-size:20px;">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Tabs for Structured Exploration
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Traffic Traces & Diurnal Trends",
    "📊 Normal Distribution Fitting & Tests",
    "🛡️ Capacity Planning & SLA Risk (P(X > C))",
    "⚡ Intelligent Load Balancer Simulation",
])

# ----------------- TAB 1: TRAFFIC TRACES -----------------
with tab1:
    st.subheader("Time Series Traffic Trace Analysis")
    st.caption("Visualizing deterministic baseline diurnal flow combined with Gaussian variance and burst events.")

    fig_trace = go.Figure()
    fig_trace.add_trace(
        go.Scatter(
            x=df_trace["timestamp"],
            y=df_trace["total_load_gbps"],
            mode="lines",
            name="Observed Total Load",
            line=dict(color="#38bdf8", width=1.8),
        )
    )
    fig_trace.add_trace(
        go.Scatter(
            x=df_trace["timestamp"],
            y=df_trace["deterministic_trend"],
            mode="lines",
            name="Diurnal Mean Trend",
            line=dict(color="#fbbf24", width=2, dash="dash"),
        )
    )
    # Highlights bursts
    bursts_df = df_trace[df_trace["is_burst"]]
    if not bursts_df.empty:
        fig_trace.add_trace(
            go.Scatter(
                x=bursts_df["timestamp"],
                y=bursts_df["total_load_gbps"],
                mode="markers",
                name="Traffic Burst Anomalies",
                marker=dict(color="#ef4444", size=6, symbol="x"),
            )
        )

    fig_trace.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Timeline",
        yaxis_title="Bandwidth Load (Gbps)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trace, use_container_width=True)

    with st.expander("🔍 View Raw Generated Traffic Trace (First 50 Samples)"):
        st.dataframe(df_trace.head(50), use_container_width=True)

# ----------------- TAB 2: NORMAL DISTRIBUTION FITTING -----------------
with tab2:
    st.subheader("Statistical Fitting: PDF, CDF, and Q-Q Plot")
    st.caption("Validating Gaussian assumptions against synthetic network workloads.")

    col_fit1, col_fit2 = st.columns(2)

    with col_fit1:
        # PDF Histogram + Theoretical Normal Curve
        x_vals = np.linspace(df_trace["total_load_gbps"].min() - 5, df_trace["total_load_gbps"].max() + 5, 200)
        fitted_pdf = stats.norm.pdf(x_vals, loc=norm_metrics["mean"], scale=norm_metrics["std"])

        fig_pdf = go.Figure()
        fig_pdf.add_trace(
            go.Histogram(
                x=df_trace["total_load_gbps"],
                histnorm="probability density",
                name="Empirical Density",
                opacity=0.6,
                marker_color="#0284c7",
                nbinsx=40,
            )
        )
        fig_pdf.add_trace(
            go.Scatter(
                x=x_vals,
                y=fitted_pdf,
                mode="lines",
                name=f"Fitted Normal N({norm_metrics['mean']:.1f}, {norm_metrics['std']:.1f}²)",
                line=dict(color="#f43f5e", width=2.5),
            )
        )
        fig_pdf.update_layout(
            title="Probability Density Function (PDF) vs Empirical Histogram",
            template="plotly_dark",
            xaxis_title="Load (Gbps)",
            yaxis_title="Probability Density",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_pdf, use_container_width=True)

    with col_fit2:
        # Q-Q Plot
        sorted_load = np.sort(df_trace["total_load_gbps"])
        n = len(sorted_load)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n, loc=norm_metrics["mean"], scale=norm_metrics["std"])

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_load,
                mode="markers",
                name="Quantiles",
                marker=dict(color="#a855f7", size=4, opacity=0.7),
            )
        )
        # Reference 45-degree line
        min_q = min(theoretical_quantiles.min(), sorted_load.min())
        max_q = max(theoretical_quantiles.max(), sorted_load.max())
        fig_qq.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                name="Standard Gaussian 1:1 Line",
                line=dict(color="#e2e8f0", dash="dash"),
            )
        )
        fig_qq.update_layout(
            title="Quantile-Quantile (Q-Q) Plot for Normality Assessment",
            template="plotly_dark",
            xaxis_title="Theoretical Normal Quantiles (Gbps)",
            yaxis_title="Empirical Data Quantiles (Gbps)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # Statistical Tests Table
    st.markdown("#### Goodness-of-Fit & Hypothesis Test Outcomes")
    test_df = pd.DataFrame([
        {
            "Statistical Test": "D'Agostino-Pearson Omnibus Test",
            "Test Statistic": f"{norm_metrics['dagostino_stat']:.4f}",
            "p-value": f"{norm_metrics['dagostino_p_value']:.4e}",
            "Null Hypothesis (H0)": "Sample originates from Normal distribution",
            "Interpretation": "Normal" if norm_metrics['dagostino_p_value'] > 0.05 else "Non-Gaussian skew/kurtosis (due to diurnal/burst)",
        },
        {
            "Statistical Test": "Kolmogorov-Smirnov Test",
            "Test Statistic": f"{norm_metrics['ks_stat']:.4f}",
            "p-value": f"{norm_metrics['ks_p_value']:.4e}",
            "Null Hypothesis (H0)": "Cumulative empirical distribution matches Gaussian CDF",
            "Interpretation": "Good Fit" if norm_metrics['ks_p_value'] > 0.01 else "Deviates at extreme tails",
        },
    ])
    st.table(test_df)

# ----------------- TAB 3: CAPACITY PLANNING & SLA RISK -----------------
with tab3:
    st.subheader("Data Center Capacity Planning & SLA Risk Modeling")
    st.markdown(
        """
        Industry leaders like **Meta, Google, and Microsoft** use probabilistic bounds:
        $$\\text{Capacity } C = \\mu + Z \\cdot \\sigma$$
        The probability of an SLA violation is evaluated via Gaussian Survival Function:
        $$P(X > C) = 1 - \\Phi\\left(\\frac{C - \\mu}{\\sigma}\\right)$$
        """
    )

    cap_col1, cap_col2 = st.columns(2)

    with cap_col1:
        st.markdown("##### Provisioning Calculator")
        target_sla = st.select_slider(
            "Target SLA Uptime / Reliability Level (%)",
            options=[90.0, 95.0, 99.0, 99.9, 99.99, 99.999],
            value=99.9,
        )
        provision_res = analyzer.recommend_provisioning_capacity(target_sla)

        st.metric(
            label="Recommended Provisioned Capacity (C)",
            value=f"{provision_res['recommended_capacity_gbps']:.2f} Gbps",
            delta=f"+{provision_res['safety_buffer_gbps']:.2f} Gbps buffer above mean",
        )
        st.write(
            f"- **Z-score Multiplier**: `{provision_res['z_multiplier']:.3f}`"
            f"\n- **Statistical Headroom**: `{provision_res['safety_buffer_gbps']:.2f} Gbps` ({provision_res['safety_buffer_gbps']/provision_res['mean_load_gbps']*100:.1f}% over mean)"
        )

    with cap_col2:
        st.markdown("##### Evaluate SLA Breach Probability for Fixed Hardware")
        user_capacity = st.number_input(
            "Hypothetical Data Center Link Capacity (Gbps)",
            min_value=10.0,
            max_value=200.0,
            value=float(round(norm_metrics["mean"] + 2.5 * norm_metrics["std"], 1)),
            step=5.0,
        )
        sla_eval = analyzer.calculate_sla_violation_probability(user_capacity)

        st.metric(
            label="Predicted SLA Breach Probability P(X > C)",
            value=f"{sla_eval['theoretical_violation_prob']*100:.4f}%",
            delta=f"Z-Score: {sla_eval['z_score']:.2f}",
        )
        st.write(
            f"- **Empirical SLA Breach Rate in Simulation**: `{sla_eval['empirical_violation_prob']*100:.3f}%`"
            f"\n- **Theoretical System Reliability**: `{sla_eval['theoretical_sla_reliability_percent']:.4f}%`"
        )

    # CDF Reliability Curve
    x_range = np.linspace(
        max(0, norm_metrics["mean"] - 3.5 * norm_metrics["std"]),
        norm_metrics["mean"] + 4.5 * norm_metrics["std"],
        250,
    )
    cdf_vals = stats.norm.cdf(x_range, loc=norm_metrics["mean"], scale=norm_metrics["std"]) * 100.0

    fig_cdf = go.Figure()
    fig_cdf.add_trace(
        go.Scatter(
            x=x_range,
            y=cdf_vals,
            mode="lines",
            name="SLA Reliability (%)",
            line=dict(color="#10b981", width=2.5),
        )
    )
    fig_cdf.add_vline(
        x=provision_res["recommended_capacity_gbps"],
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"Target {target_sla}%: {provision_res['recommended_capacity_gbps']:.1f} Gbps",
    )
    fig_cdf.update_layout(
        title="Reliability Curve: Capacity vs Probability of Non-Saturating Load P(X ≤ C)",
        template="plotly_dark",
        xaxis_title="Allocated Network Bandwidth (Gbps)",
        yaxis_title="Probability of Sufficient Capacity (%)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_cdf, use_container_width=True)

# ----------------- TAB 4: LOAD BALANCER SIMULATION -----------------
with tab4:
    st.subheader("Intelligent Load Balancing Simulation")
    st.caption("Comparing standard routing algorithms with Normal Distribution / Headroom-Aware load balancers.")

    lb_col1, lb_col2 = st.columns([1, 2])
    with lb_col1:
        num_nodes = st.slider("Number of Worker Gateway Nodes", min_value=2, max_value=8, value=4)
        node_cap = st.slider("Node Bandwidth Capacity (Gbps)", min_value=10.0, max_value=50.0, value=25.0, step=2.5)

    lb_sim = LoadBalancingSimulator(num_nodes=num_nodes, node_capacity_gbps=node_cap)

    strategies = ["round_robin", "least_loaded", "statistical_aware"]
    results = {}
    for strat in strategies:
        results[strat] = lb_sim.run_simulation(df_trace["total_load_gbps"].values, strategy=strat)

    comparison_data = []
    for s in strategies:
        r = results[s]
        comparison_data.append({
            "Routing Strategy": s.replace("_", " ").title(),
            "Total Served (Gbps)": f"{r['total_served_gbps']:.1f}",
            "Dropped Traffic (Gbps)": f"{r['total_dropped_gbps']:.1f}",
            "Drop Rate (%)": f"{r['packet_drop_rate_pct']:.3f}%",
            "Load Imbalance (σ)": f"{r['load_imbalance_std']:.3f}",
        })
    st.table(pd.DataFrame(comparison_data))

    # Node load comparison chart
    fig_nodes = go.Figure()
    for s in strategies:
        node_metrics = results[s]["node_metrics"]
        node_names = [n["node_id"] for n in node_metrics]
        avg_loads = [n["avg_load_gbps"] for n in node_metrics]
        fig_nodes.add_trace(go.Bar(x=node_names, y=avg_loads, name=s.replace("_", " ").title()))

    fig_nodes.update_layout(
        barmode="group",
        title="Average Load Distribution Per Node Across Strategies",
        template="plotly_dark",
        yaxis_title="Average Load (Gbps)",
        xaxis_title="Worker Node",
        height=380,
    )
    st.plotly_chart(fig_nodes, use_container_width=True)

st.markdown("---")
st.markdown(
    "💡 **Academic & Industrial Context**: Integrated Course Outcomes (CO3: Mathematical Modeling, CO4: Probabilistic Risk Analysis, CO5: Distributed System Evaluation)."
)
