"""
Streamlit Web Application: Network Capacity & Traffic Analyzer
Ultra-clear, beginner-friendly dashboard explaining network load using the Normal Distribution.
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

try:
    from src.simulator import NetworkLoadSimulator, TrafficSimulationConfig
    from src.statistical_model import NormalDistributionAnalyzer
    from src.load_balancer import LoadBalancingSimulator
except ImportError:
    from simulator import NetworkLoadSimulator, TrafficSimulationConfig
    from statistical_model import NormalDistributionAnalyzer
    from load_balancer import LoadBalancingSimulator

st.set_page_config(
    page_title="Network Traffic & Capacity Planner",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom friendly styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }
    .plain-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .card-label {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .card-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0 2px 0;
    }
    .card-hint {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.3;
    }
    .guide-banner {
        background: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38bdf8;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .takeaway-box {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 14px 18px;
        border-radius: 10px;
        margin-top: 14px;
        font-size: 0.93rem;
        color: #dcfce7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Title & Simple Purpose
st.title("🌐 Network Traffic & Capacity Planner")
st.markdown(
    """
    <div class="guide-banner">
        💡 <b>What is this page?</b><br>
        Imagine you run a company like YouTube or Netflix. Millions of users stream videos.
        This tool uses <b>basic statistics (the bell curve)</b> to answer one vital question: 
        <b>"How much network bandwidth should we buy so our site never crashes?"</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Simple inputs
with st.sidebar:
    st.markdown("### 🎛️ Adjust Traffic Settings")
    st.write("Control how busy the network is:")
    
    base_load = st.slider(
        "Normal Everyday Traffic (Gbps)",
        min_value=10.0,
        max_value=100.0,
        value=45.0,
        step=5.0,
        help="The typical amount of data flowing through the network on average.",
    )
    
    duration_hours = st.slider(
        "Time to Simulate (Hours)",
        min_value=12,
        max_value=72,
        value=24,
        step=6,
        help="How many hours of internet traffic you want to view.",
    )

    with st.expander("More Controls (Spikes & Day/Night Swing)", expanded=False):
        diurnal_amp = st.slider(
            "Day vs Night Traffic Difference (Gbps)",
            min_value=0.0,
            max_value=40.0,
            value=20.0,
            step=2.0,
            help="Traffic is high during the afternoon and low at 4 AM.",
        )
        noise_std = st.slider("Random Jitter / Noise (Gbps)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        burst_prob = st.slider("Chance of Sudden Traffic Spikes (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100.0
        burst_magnitude = st.slider("Size of Traffic Spike (Gbps)", min_value=5.0, max_value=50.0, value=20.0, step=5.0)

# Generate Simulation Data
config = TrafficSimulationConfig(
    duration_hours=duration_hours,
    sampling_rate_minutes=2,
    base_load_gbps=base_load,
    diurnal_amplitude_gbps=diurnal_amp,
    random_noise_std_gbps=noise_std,
    burst_probability=burst_prob,
    burst_magnitude_gbps=burst_magnitude,
    random_seed=42,
)

simulator = NetworkLoadSimulator(config)
df_trace = simulator.generate_trace()
analyzer = NormalDistributionAnalyzer(df_trace["total_load_gbps"])
norm_metrics = analyzer.test_normality()
peak_load = float(df_trace["total_load_gbps"].max())
min_load = float(df_trace["total_load_gbps"].min())

# Step 1: Summary Cards with Plain English Explanations
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="plain-card">
            <div class="card-label">1. Average Traffic</div>
            <div class="card-val">{norm_metrics['mean']:.1f} <span style="font-size:1rem;color:#94a3b8;">Gbps</span></div>
            <div class="card-hint">The typical load flowing through your network.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="plain-card">
            <div class="card-label">2. Normal Fluctuations</div>
            <div class="card-val">±{norm_metrics['std']:.1f} <span style="font-size:1rem;color:#94a3b8;">Gbps</span></div>
            <div class="card-hint">How much traffic bounces up and down around the average.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="plain-card">
            <div class="card-label">3. Highest Peak</div>
            <div class="card-val">{peak_load:.1f} <span style="font-size:1rem;color:#94a3b8;">Gbps</span></div>
            <div class="card-hint">The single busiest moment observed during the simulation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    safe_rec = norm_metrics['mean'] + 3.0 * norm_metrics['std']
    st.markdown(
        f"""
        <div class="plain-card">
            <div class="card-label">4. Safe Bandwidth to Buy</div>
            <div class="card-val" style="color:#34d399;">{safe_rec:.1f} <span style="font-size:1rem;color:#94a3b8;">Gbps</span></div>
            <div class="card-hint">Enough capacity to safely handle 99.9% of all traffic surges.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Clean, Guided Tabs
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Live Traffic Timeline",
    "2️⃣ How Much Bandwidth to Buy? (SLA Calculator)",
    "3️⃣ The Bell Curve (Why Normal Distribution?)",
])

# ----------------- TAB 1: TIMELINE -----------------
with tab1:
    st.subheader("Step 1: Look at Network Traffic Over Time")
    st.write(
        "This graph shows how data flows through the network hour by hour. "
        "Notice the smooth rise and fall (day vs night) plus sudden red spikes when unexpected surges happen."
    )

    fig_trace = go.Figure()
    fig_trace.add_trace(
        go.Scatter(
            x=df_trace["timestamp"],
            y=df_trace["total_load_gbps"],
            mode="lines",
            name="Actual Traffic",
            line=dict(color="#38bdf8", width=1.8),
        )
    )
    fig_trace.add_trace(
        go.Scatter(
            x=df_trace["timestamp"],
            y=df_trace["deterministic_trend"],
            mode="lines",
            name="Day/Night Pattern",
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
                name="Sudden Spikes",
                marker=dict(color="#f87171", size=6, symbol="circle"),
            )
        )

    fig_trace.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=15, b=20),
        xaxis_title="Time of Day",
        yaxis_title="Bandwidth Load (Gigabits per second - Gbps)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trace, width="stretch")

    st.markdown(
        """
        <div class="takeaway-box">
            <b>Key takeaway from this chart:</b><br>
            If you only bought enough bandwidth for the <i>average</i> traffic (blue dashed line), your servers would crash every time traffic peaks during the afternoon or when sudden spikes hit. You need a safety buffer!
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- TAB 2: CAPACITY & SLA -----------------
with tab2:
    st.subheader("Step 2: Calculate the Exact Capacity to Buy")
    st.write(
        "Instead of guessing, network engineers use statistics to find the balance: "
        "**buy enough capacity so users never see errors, without wasting money on unused hardware.**"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🎯 Option A: Pick Your Desired Reliability (SLA)")
        sla_choice = st.select_slider(
            "How reliable must your service be?",
            options=[90.0, 95.0, 99.0, 99.9, 99.99],
            value=99.9,
            format_func=lambda x: f"{x}% uptime (Standard)" if x == 99.9 else f"{x}% uptime",
        )
        rec = analyzer.recommend_provisioning_capacity(sla_choice)

        st.metric(
            label="Recommended Capacity to Provision",
            value=f"{rec['recommended_capacity_gbps']:.1f} Gbps",
            delta=f"+{rec['safety_buffer_gbps']:.1f} Gbps safety buffer",
        )
        st.write(
            f"💡 **What this means:** If you buy **{rec['recommended_capacity_gbps']:.1f} Gbps**, "
            f"your network will successfully handle **{sla_choice}%** of all traffic peaks without dropping a single packet."
        )

    with c2:
        st.markdown("#### 🧪 Option B: Test a Fixed Bandwidth Limit")
        user_test_cap = st.slider(
            "What if our hardware is capped at:",
            min_value=float(int(min_load)),
            max_value=float(int(peak_load + 20)),
            value=float(int(norm_metrics['mean'] + norm_metrics['std'])),
            step=2.0,
            format="%.0f Gbps",
        )
        risk = analyzer.calculate_sla_violation_probability(user_test_cap)
        breach_pct = risk['theoretical_violation_prob'] * 100.0

        if breach_pct < 0.1:
            risk_status = "🟢 Safe (Less than 0.1% chance of overload)"
        elif breach_pct < 5.0:
            risk_status = "🟡 Moderate Risk (Occasional slowdowns)"
        else:
            risk_status = "🔴 Dangerous (Frequent overloads and crashes)"

        st.metric(
            label="Chance of Network Overload / Crash",
            value=f"{breach_pct:.2f}%",
            delta=risk_status,
            delta_color="off",
        )
        st.write(
            f"💡 **What this means:** At **{user_test_cap:.0f} Gbps**, "
            f"traffic will exceed capacity about **{breach_pct:.2f}% of the time**."
        )

# ----------------- TAB 3: THE BELL CURVE -----------------
with tab3:
    st.subheader("Step 3: Why Do We Use the Normal Distribution (Bell Curve)?")
    st.write(
        "When millions of independent users click, scroll, and stream, their combined activity naturally clusters around an average. "
        "This creates a **Bell Curve**."
    )

    x_min = max(0, df_trace["total_load_gbps"].min() - 5)
    x_max = df_trace["total_load_gbps"].max() + 5
    x_vals = np.linspace(x_min, x_max, 150)
    fitted_pdf = stats.norm.pdf(x_vals, loc=norm_metrics["mean"], scale=norm_metrics["std"])

    fig_bell = go.Figure()
    fig_bell.add_trace(
        go.Histogram(
            x=df_trace["total_load_gbps"],
            histnorm="probability density",
            name="Actual Traffic Frequency",
            opacity=0.6,
            marker_color="#0284c7",
            nbinsx=30,
        )
    )
    fig_bell.add_trace(
        go.Scatter(
            x=x_vals,
            y=fitted_pdf,
            mode="lines",
            name="Ideal Bell Curve (Normal Distribution)",
            line=dict(color="#f43f5e", width=2.5),
        )
    )

    fig_bell.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Traffic Load (Gbps)",
        yaxis_title="How Often This Load Occurs",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bell, width="stretch")

    st.markdown(
        """
        <div class="takeaway-box">
            <b>Why this helps:</b><br>
            Because the traffic matches a standard mathematical curve, we don't have to guess the future! 
            Using simple formulas ($\mu + 3\sigma$), we know that <b>99.7% of all traffic</b> will always fall within 3 standard deviations of the average.
        </div>
        """,
        unsafe_allow_html=True,
    )
