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

import src.simulator as sim_mod
import importlib
importlib.reload(sim_mod)
from src.simulator import NetworkLoadSimulator, TrafficSimulationConfig
from src.statistical_model import NormalDistributionAnalyzer
from src.load_balancer import LoadBalancingSimulator
from src.real_data import RealDataIngestionService

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
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }
    .plain-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .card-label {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .card-val {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 2px 0 6px 0;
    }
    .card-hint {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.4;
    }
    .intro-banner {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(99, 102, 241, 0.12));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 24px;
        color: #f1f5f9;
        font-size: 1rem;
        line-height: 1.55;
    }
    .takeaway-box {
        background: rgba(34, 197, 94, 0.08);
        border-left: 4px solid #22c55e;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 16px;
        font-size: 0.94rem;
        color: #dcfce7;
        line-height: 1.5;
    }
    .analogy-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
        font-size: 0.93rem;
        color: #fef3c7;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header & Intuitive Analogy
st.title("🌐 Network Traffic & Capacity Planner")
st.markdown(
    """
    <div class="intro-banner">
        👋 <b>Welcome! What does this tool do?</b><br>
        Imagine you operate a popular website or mobile app (like YouTube, Instagram, or Netflix). 
        At any moment, thousands of people are loading pages and watching videos.
        <br><br>
        • If you buy <b>too little capacity</b> ➔ Your site slows down, freezes, or crashes.<br>
        • If you buy <b>too much capacity</b> ➔ You waste thousands of dollars on empty servers.<br>
        <br>
        This dashboard shows how <b>statistics (the Bell Curve)</b> helps engineers find the 
        <b>exact sweet spot</b> to keep your service 100% fast without wasting money.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Simulator Mode & Settings
with st.sidebar:
    st.markdown("### 🎛️ Real-Time Simulator")
    
    live_streaming_on = st.toggle(
        "⚡ Live Real-Time Stream",
        value=True,
        help="When enabled, the simulator streams live traffic second-by-second using the actual clock time.",
    )

    if live_streaming_on:
        stream_speed = st.selectbox(
            "Live Refresh Speed:",
            options=[1, 5, 10, 20, 30],
            index=2,
            format_func=lambda s: f"Every {s} second{'s' if s > 1 else ''} (Live)",
        )
        if st.button("🔄 Reset Live Buffer"):
            st.session_state["live_history_df"] = None
    
    st.markdown("---")
    st.markdown("#### ⚙️ Traffic Parameters")
    base_load = st.slider(
        "Average Daily Traffic (Gbps)",
        min_value=10.0,
        max_value=100.0,
        value=45.0,
        step=5.0,
        help="The typical amount of data flowing into your servers on average.",
    )
    
    duration_hours = st.slider(
        "Historical Window (Hours)",
        min_value=12,
        max_value=72,
        value=24,
        step=6,
        help="Hours of baseline traffic history to analyze.",
    )

    with st.expander("More Controls (Spikes & Day/Night Swing)", expanded=False):
        diurnal_amp = st.slider("Day vs. Night Difference (Gbps)", 0.0, 40.0, 20.0, 2.0)
        noise_std = st.slider("Minute-to-Minute Wiggle / Jitter (Gbps)", 1.0, 15.0, 5.0, 0.5)
        burst_prob = st.slider("Chance of Sudden Surge (%)", 0.0, 10.0, 2.0, 0.5) / 100.0
        burst_magnitude = st.slider("Size of Sudden Surge (Gbps)", 5.0, 50.0, 20.0, 5.0)

# Build configuration
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

# Dynamic Dashboard Section with seamless auto-refresh (no full page reload/reset)
refresh_interval = stream_speed if live_streaming_on else None

@st.fragment(run_every=refresh_interval)
def render_live_dashboard():
    now_ts = pd.Timestamp.now()

    # Manage Real-Time Live Buffer
    if live_streaming_on:
        if "live_history_df" not in st.session_state or st.session_state["live_history_df"] is None:
            # Initialize with baseline ending right now
            init_steps = 180  # past 6 hours of 2-min points
            init_timestamps = pd.date_range(end=now_ts, periods=init_steps, freq="2min")
            init_df = simulator.generate_trace().tail(init_steps).copy()
            init_df["timestamp"] = init_timestamps
            st.session_state["live_history_df"] = init_df

        # Append current live second's point
        new_pt = simulator.next_live_point(now_ts)
        new_row = pd.DataFrame([new_pt])

        # Keep rolling buffer of recent points
        updated_df = pd.concat([st.session_state["live_history_df"], new_row], ignore_index=True)
        if len(updated_df) > 300:
            updated_df = updated_df.iloc[-300:]
        st.session_state["live_history_df"] = updated_df
        df_trace = updated_df.copy()
        data_origin_label = f"🟢 LIVE STREAMING ({now_ts.strftime('%H:%M:%S')})"
    else:
        df_trace = simulator.generate_trace()
        # Align timestamps to current date
        df_trace["timestamp"] = pd.date_range(end=now_ts, periods=len(df_trace), freq="2min")
        data_origin_label = "Real-Time Clock Aligned Simulation"

    # Compute exact statistical parameters from active data series
    analyzer = NormalDistributionAnalyzer(df_trace["total_load_gbps"])
    norm_metrics = analyzer.test_normality()
    peak_load = float(df_trace["total_load_gbps"].max())
    min_load = float(df_trace["total_load_gbps"].min())
    current_live_load = float(df_trace["total_load_gbps"].iloc[-1])

    # Industry standard 3-sigma capacity (99.73% confidence interval)
    safe_rec = norm_metrics['mean'] + 3.0 * norm_metrics['std']

    # 4 Key Scorecards (Clear & Jargon-Free)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0;">📊 At a Glance</h3>
            <span style="background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4); padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; color: #4ade80; font-weight: 600;">
                {data_origin_label} • Current: {current_live_load:.1f} Gbps
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="plain-card">
                <div>
                    <div class="card-label">1. Average Traffic</div>
                    <div class="card-val">{norm_metrics['mean']:.1f} <span style="font-size:1rem;color:#94a3b8;font-weight:400;">Gbps</span></div>
                </div>
                <div class="card-hint">The typical data flow throughout the day.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="plain-card">
                <div>
                    <div class="card-label">2. Natural Wiggle</div>
                    <div class="card-val">±{norm_metrics['std']:.1f} <span style="font-size:1rem;color:#94a3b8;font-weight:400;">Gbps</span></div>
                </div>
                <div class="card-hint">Standard deviation: how much load bounces around the average.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="plain-card">
                <div>
                    <div class="card-label">3. Highest Peak Seen</div>
                    <div class="card-val" style="color:#f87171;">{peak_load:.1f} <span style="font-size:1rem;color:#94a3b8;font-weight:400;">Gbps</span></div>
                </div>
                <div class="card-hint">The single busiest moment recorded across the day.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="plain-card" style="border-color: rgba(52, 211, 153, 0.4);">
                <div>
                    <div class="card-label" style="color:#34d399;">4. Safe Capacity (3-Sigma)</div>
                    <div class="card-val" style="color:#34d399;">{safe_rec:.1f} <span style="font-size:1rem;color:#94a3b8;font-weight:400;">Gbps</span></div>
                </div>
                <div class="card-hint">Covers 99.7% of all surges (Mean + 3×Std).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # 3 Guided Steps in Tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Step 1: See the Daily Traffic Flow",
        "🎯 Step 2: Test Different Capacity Limits",
        "🔔 Step 3: Understand the Bell Curve",
    ])

    # ----------------- TAB 1: TIMELINE -----------------
    with tab1:
        st.subheader("How Traffic Behaves Across 24 Hours")
        st.write(
            "Look at the interactive timeline below. Traffic isn't flat—it naturally swells during "
            "waking hours, drops late at night, and occasionally spikes when popular events happen."
        )

        fig_trace = go.Figure()
        fig_trace.add_trace(
            go.Scatter(
                x=df_trace["timestamp"],
                y=df_trace["total_load_gbps"],
                mode="lines",
                name="Actual Traffic (What servers see)",
                line=dict(color="#38bdf8", width=1.8),
            )
        )
        fig_trace.add_trace(
            go.Scatter(
                x=df_trace["timestamp"],
                y=df_trace["deterministic_trend"],
                mode="lines",
                name="Expected Day / Night Rhythm",
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
                    name="Surprise Spikes (Breaking news, flash crowd)",
                    marker=dict(color="#f87171", size=7, symbol="circle"),
                )
            )

        # Reference capacity line
        fig_trace.add_hline(
            y=safe_rec,
            line_dash="dot",
            line_color="#34d399",
            annotation_text=f"Safe Buffer ({safe_rec:.1f} Gbps)",
            annotation_position="top right",
        )

        fig_trace.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Time of Day",
            yaxis_title="Traffic Load (Gigabits per second - Gbps)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trace, width="stretch")

        st.markdown(
            """
            <div class="takeaway-box">
                <b>💡 The Main Lesson:</b><br>
                If you only bought enough server bandwidth for the <b>Average traffic</b> (around 45 Gbps), 
                your website would freeze or crash every single afternoon during peak hours! 
                To keep users happy, you always need a <b>Safety Buffer</b> above the average.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------- TAB 2: CAPACITY & SLA -----------------
    with tab2:
        st.subheader("Decide How Much Bandwidth to Buy")
        st.write(
            "Buying network capacity is a trade-off between **safety** and **cost**. "
            "Here you can see exactly what happens under different budgets."
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🎯 Strategy A: Target a Reliability Goal")
            st.write("Pick how reliable you want your service to be:")
            sla_choice = st.select_slider(
                "Uptime Target",
                options=[90.0, 95.0, 99.0, 99.7, 99.9, 99.99],
                value=99.7,
                format_func=lambda x: f"{x}% (Standard 3-Sigma)" if x == 99.7 else (f"{x}% (Ultra SLA)" if x == 99.9 else f"{x}% uptime"),
            )
            rec = analyzer.recommend_provisioning_capacity(sla_choice)
            z_val = stats.norm.ppf(sla_choice / 100.0)

            st.metric(
                label=f"Buy this much bandwidth for {sla_choice}% uptime:",
                value=f"{rec['recommended_capacity_gbps']:.1f} Gbps",
                delta=f"+{rec['safety_buffer_gbps']:.1f} Gbps safety cushion ({z_val:.2f}σ)",
            )
            st.markdown(
                f"""
                <div class="analogy-box">
                    <b>Mathematical Proof:</b><br>
                    <code>{norm_metrics['mean']:.1f} Gbps (Mean) + ({z_val:.2f} × {norm_metrics['std']:.1f} Gbps Std) = {rec['recommended_capacity_gbps']:.1f} Gbps</code><br>
                    By provisioning <b>{rec['recommended_capacity_gbps']:.1f} Gbps</b>, 
                    your servers will absorb <b>{sla_choice}%</b> of all traffic fluctuations with zero packet loss.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown("#### 🧪 Strategy B: Test Your Own Capacity Limit")
            st.write("Drag the slider to test what happens if you cap your bandwidth:")
            user_test_cap = st.slider(
                "Assume our maximum bandwidth limit is:",
                min_value=float(int(min_load)),
                max_value=float(int(peak_load + 20)),
                value=float(int(norm_metrics['mean'] + norm_metrics['std'])),
                step=2.0,
                format="%.0f Gbps",
            )
            risk = analyzer.calculate_sla_violation_probability(user_test_cap)
            breach_pct = risk['theoretical_violation_prob'] * 100.0

            if breach_pct <= 0.3:
                risk_status = "🟢 Safe (3-Sigma standard: covers 99.7%+ of all surges)"
                alert_type = "success"
            elif breach_pct < 5.0:
                risk_status = "🟡 Moderate Risk (Brief lag during peak afternoon)"
                alert_type = "warning"
            else:
                risk_status = "🔴 Dangerous (High chance of crashes & outages!)"
                alert_type = "error"

            st.metric(
                label="Chance of Overload / Service Crash:",
                value=f"{breach_pct:.2f}% of the time",
                delta=risk_status,
                delta_color="off",
            )

            if alert_type == "success":
                st.success(f"At {user_test_cap:.0f} Gbps, your capacity safely covers 99.7%+ of all traffic fluctuations!")
            elif alert_type == "warning":
                st.warning(f"At {user_test_cap:.0f} Gbps, you risk occasional buffering during unexpected spikes.")
            else:
                st.error(f"At {user_test_cap:.0f} Gbps, servers will be overwhelmed frequently during afternoon rush hours!")

    # ----------------- TAB 3: THE BELL CURVE -----------------
    with tab3:
        st.subheader("Why the Bell Curve Makes Sizing Easy")
        st.write(
            "When millions of independent users click links, stream videos, or download photos, "
            "their combined actions follow a predictable mathematical shape: the **Normal Distribution (Bell Curve)**."
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
                name="Actual Traffic Frequency (How often each load happened)",
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
                name="Fitted Bell Curve (Theoretical Model)",
                line=dict(color="#f43f5e", width=2.5),
            )
        )

        # Vertical line for recommended capacity
        fig_bell.add_vline(
            x=safe_rec,
            line_dash="dot",
            line_color="#34d399",
            annotation_text="99.9% Safe Line",
            annotation_position="top right",
        )

        fig_bell.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Traffic Load in Gbps (From quiet to very busy)",
            yaxis_title="How Often This Load Level Happens",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bell, width="stretch")

        st.markdown(
            """
            <div class="takeaway-box">
                <b>🪄 The 3-Step Rule That Every Engineer Uses:</b><br>
                Because internet traffic hugs this bell curve, we don't have to guess how high surges will go.<br>
                In statistics, <b>99.7% of all activity</b> always stays within <b>3 steps (3 standard deviations)</b> of the average.<br>
                So the simple formula to never crash is: <br>
                <center style="margin: 8px 0; font-size: 1.1rem; font-weight: 700; color: #38bdf8;">
                    Safe Bandwidth = Average Load + (3 × Normal Wiggle)
                </center>
                That's all it takes to guarantee smooth performance!
            </div>
            """,
            unsafe_allow_html=True,
        )

render_live_dashboard()

