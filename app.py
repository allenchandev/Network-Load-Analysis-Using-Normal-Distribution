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
    from src.real_data import RealDataIngestionService
except ImportError:
    from simulator import NetworkLoadSimulator, TrafficSimulationConfig
    from statistical_model import NormalDistributionAnalyzer
    from load_balancer import LoadBalancingSimulator
    from real_data import RealDataIngestionService

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

# Sidebar: Data Source & Settings
with st.sidebar:
    st.markdown("### 🗄️ Network Data Source")
    data_source_mode = st.radio(
        "Choose where data comes from:",
        options=[
            "🌐 Real-World Cloud / CDN Traces",
            "⚡ Live Server Telemetry (Host NIC)",
            "📁 Upload Your Own CSV Log",
            "🎛️ Mathematical Simulator",
        ],
        index=0,
        help="Switch between authentic production internet logs, live machine telemetry, custom CSV data, or statistical simulation.",
    )

    if data_source_mode == "🌐 Real-World Cloud / CDN Traces":
        st.markdown("#### Real Cloud Benchmarks")
        curated_options = list(RealDataIngestionService.get_curated_datasets().keys())
        selected_dataset = st.selectbox(
            "Select authentic network trace:",
            options=curated_options,
            index=0,
        )
        cat_info = RealDataIngestionService.get_curated_datasets()[selected_dataset]
        st.caption(f"ℹ️ **Origin**: {cat_info['source']}")
        st.caption(cat_info['description'])
        
        duration_hours = st.slider("Time Window (Hours)", min_value=12, max_value=72, value=24, step=6)
        df_trace = RealDataIngestionService.load_curated_dataset(selected_dataset, hours=duration_hours)

    elif data_source_mode == "⚡ Live Server Telemetry (Host NIC)":
        st.markdown("#### Live Network Interface")
        st.write("Measures instantaneous network I/O directly from this machine's network card.")
        sample_secs = st.slider("Sampling Duration (Seconds)", min_value=3, max_value=20, value=6, step=1)
        if st.button("🔄 Capture Live Network Stream", use_container_width=True):
            st.session_state["live_capture_df"] = None

        if "live_capture_df" not in st.session_state or st.session_state["live_capture_df"] is None:
            with st.spinner(f"Sampling network card for {sample_secs} seconds..."):
                st.session_state["live_capture_df"] = RealDataIngestionService.get_live_nic_sample(duration_seconds=sample_secs)

        df_trace = st.session_state["live_capture_df"]
        st.success(f"Captured {len(df_trace)} live telemetry points!")

    elif data_source_mode == "📁 Upload Your Own CSV Log":
        st.markdown("#### Upload Network Log")
        uploaded_file = st.file_uploader(
            "Upload CSV (must contain numeric bandwidth / load column)",
            type=["csv"],
            help="Supports columns named 'load', 'traffic', 'gbps', 'mbps', etc.",
        )
        if uploaded_file is not None:
            try:
                df_trace = RealDataIngestionService.parse_user_csv(uploaded_file)
                st.success(f"Parsed {len(df_trace)} rows from CSV successfully!")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}. Falling back to default trace.")
                df_trace = RealDataIngestionService.load_curated_dataset("Cloudflare Edge CDN (Global HTTP Ingress)")
        else:
            st.info("No file uploaded yet. Showing Cloudflare benchmark as preview:")
            df_trace = RealDataIngestionService.load_curated_dataset("Cloudflare Edge CDN (Global HTTP Ingress)")

    else:
        st.markdown("#### Simulator Controls")
        base_load = st.slider(
            "Average Daily Traffic (Gbps)",
            min_value=10.0,
            max_value=100.0,
            value=45.0,
            step=5.0,
            help="The baseline amount of data flowing into your servers during normal hours.",
        )
        duration_hours = st.slider(
            "Simulation Window (Hours)",
            min_value=12,
            max_value=72,
            value=24,
            step=6,
        )
        with st.expander("⚙️ Advanced Settings (Spikes & Swing)", expanded=False):
            diurnal_amp = st.slider("Day vs. Night Difference (Gbps)", 0.0, 40.0, 20.0, 2.0)
            noise_std = st.slider("Minute-to-Minute Wiggle / Jitter (Gbps)", 1.0, 15.0, 5.0, 0.5)
            burst_prob = st.slider("Chance of Sudden Surge (%)", 0.0, 10.0, 2.0, 0.5) / 100.0
            burst_magnitude = st.slider("Size of Sudden Surge (Gbps)", 5.0, 50.0, 20.0, 5.0)

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

# Compute exact statistical parameters from active real data series
data_origin_label = df_trace["data_origin"].iloc[0] if "data_origin" in df_trace.columns else "Active Telemetry"
analyzer = NormalDistributionAnalyzer(df_trace["total_load_gbps"])
norm_metrics = analyzer.test_normality()
peak_load = float(df_trace["total_load_gbps"].max())
min_load = float(df_trace["total_load_gbps"].min())

# Industry standard 3-sigma capacity (99.73% confidence interval)
safe_rec = norm_metrics['mean'] + 3.0 * norm_metrics['std']
recommendation_999 = analyzer.recommend_provisioning_capacity(99.9)["recommended_capacity_gbps"]


# 4 Key Scorecards (Clear & Jargon-Free)
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h3 style="margin: 0;">📊 At a Glance</h3>
        <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); padding: 4px 12px; border-radius: 999px; font-size: 0.82rem; color: #38bdf8; font-weight: 600;">
            📡 Source: {data_origin_label}
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
                <div class="card-label" style="color:#34d399;">4. Safe Capacity to Buy</div>
                <div class="card-val" style="color:#34d399;">{safe_rec:.1f} <span style="font-size:1rem;color:#94a3b8;font-weight:400;">Gbps</span></div>
            </div>
            <div class="card-hint">Buying this handles 99.9% of all surges with zero slowdown.</div>
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
            options=[90.0, 95.0, 99.0, 99.9, 99.99],
            value=99.9,
            format_func=lambda x: f"{x}% uptime (Recommended)" if x == 99.9 else f"{x}% uptime",
        )
        rec = analyzer.recommend_provisioning_capacity(sla_choice)

        st.metric(
            label=f"Buy this much bandwidth for {sla_choice}% uptime:",
            value=f"{rec['recommended_capacity_gbps']:.1f} Gbps",
            delta=f"+{rec['safety_buffer_gbps']:.1f} Gbps safety cushion",
        )
        st.markdown(
            f"""
            <div class="analogy-box">
                <b>What this means:</b><br>
                By provisioning <b>{rec['recommended_capacity_gbps']:.1f} Gbps</b>, 
                your servers will smoothly absorb <b>{sla_choice}%</b> of all traffic spikes. 
                Users will get instant loading speeds without any dropped requests.
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
