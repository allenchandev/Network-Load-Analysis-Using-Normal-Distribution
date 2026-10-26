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
    page_title="Network Load Analysis using Normal Distribution",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom friendly styling for White & Black Theme with High-Level Animations
st.markdown(
    """
    <style>
    /* ========================================================
       GLOBAL WHITE BACKGROUND & BLACK TEXT
       ======================================================== */
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Ensure all text elements are high-contrast black */
    h1, h2, h3, h4, h5, h6, p, span, label, div, li, strong, b, em, th, td {
        color: #000000 !important;
    }

    /* Sidebar Background and Typography */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
        background-color: #f8fafc !important;
        color: #000000 !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* Code styling */
    code {
        color: #000000 !important;
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* Layout Container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }

    /* ========================================================
       KEYFRAME ANIMATIONS
       ======================================================== */
    @keyframes slideDownFade {
        0% {
            opacity: 0;
            transform: translate3d(0, -25px, 0);
        }
        100% {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }

    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translate3d(0, 20px, 0);
        }
        100% {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }

    @keyframes floatLogo {
        0%, 100% {
            transform: translateY(0px) rotate(0deg);
        }
        50% {
            transform: translateY(-8px) rotate(1.5deg);
        }
    }

    @keyframes spinOrbit {
        0% {
            stroke-dashoffset: 0;
            transform: rotate(-25deg);
        }
        100% {
            stroke-dashoffset: 240;
            transform: rotate(335deg);
        }
    }

    @keyframes flowCurve {
        0% {
            stroke-dashoffset: 240;
        }
        100% {
            stroke-dashoffset: 0;
        }
    }

    @keyframes pulseRing {
        0% {
            r: 5;
            opacity: 1;
            stroke-width: 2.5;
        }
        60% {
            r: 16;
            opacity: 0.35;
            stroke-width: 1.5;
        }
        100% {
            r: 22;
            opacity: 0;
            stroke-width: 0.5;
        }
    }

    @keyframes nodeBreathe {
        0%, 100% {
            transform: scale(1);
            filter: drop-shadow(0 0 2px rgba(2, 132, 199, 0.5));
        }
        50% {
            transform: scale(1.2);
            filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.85));
        }
    }

    @keyframes radarPulse {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.85);
            transform: scale(0.95);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
            transform: scale(1.1);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            transform: scale(0.95);
        }
    }

    @keyframes gradientSheen {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ========================================================
       BRAND HEADER & INTERACTIVE ANIMATED LOGO
       ======================================================== */
    .brand-header-container {
        display: flex;
        align-items: center;
        gap: 24px;
        padding: 22px 26px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        box-shadow: 0 4px 20px -3px rgba(0, 0, 0, 0.06);
        margin-bottom: 24px;
        animation: slideDownFade 0.75s cubic-bezier(0.16, 1, 0.3, 1) both;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .brand-header-container:hover {
        border-color: #cbd5e1;
        box-shadow: 0 10px 32px -4px rgba(0, 0, 0, 0.1);
        transform: translateY(-3px);
    }
    .brand-header-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0284c7, #6366f1, #10b981, #f59e0b, #0284c7);
        background-size: 300% 100%;
        animation: gradientSheen 8s ease infinite;
    }

    /* Logo Wrapper & 3D Interactive Hover */
    .logo-wrapper {
        flex-shrink: 0;
        perspective: 800px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
    }
    .animated-brand-logo {
        animation: floatLogo 5s ease-in-out infinite;
        transition: transform 0.45s cubic-bezier(0.34, 1.56, 0.64, 1), filter 0.45s ease;
        filter: drop-shadow(0 6px 14px rgba(2, 132, 199, 0.25));
    }
    .logo-wrapper:hover .animated-brand-logo {
        transform: scale(1.14) rotate(7deg) translateY(-5px);
        filter: drop-shadow(0 12px 24px rgba(99, 102, 241, 0.4));
    }
    .orbit-ring {
        animation: spinOrbit 16s linear infinite;
        transform-origin: 50px 50px;
    }
    .bell-path {
        stroke-dasharray: 200;
        animation: flowCurve 4.5s ease-in-out infinite alternate;
    }
    .pulse-ring-core {
        animation: pulseRing 2.4s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        transform-origin: 50px 32px;
    }
    .node-peak {
        animation: nodeBreathe 3.2s ease-in-out infinite;
        transform-origin: 50px 32px;
    }

    /* Brand Header Typography */
    .brand-text-wrapper {
        flex-grow: 1;
    }
    .brand-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }
    .brand-badge {
        font-size: 0.75rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 11px;
        border-radius: 999px;
        border: 1px solid #cbd5e1;
        background: #f8fafc;
        color: #000000 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .brand-badge:hover {
        transform: translateY(-2px) scale(1.03);
        background: #000000;
        color: #ffffff !important;
        border-color: #000000;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .brand-title {
        font-size: 1.95rem !important;
        font-weight: 900 !important;
        margin: 0 0 6px 0 !important;
        color: #000000 !important;
        letter-spacing: -0.025em;
        line-height: 1.2 !important;
    }
    .brand-tagline {
        font-size: 0.96rem !important;
        color: #334155 !important;
        margin: 0 !important;
        font-weight: 550;
        line-height: 1.5 !important;
    }

    /* ========================================================
       LIVE STATUS INDICATOR
       ======================================================== */
    .live-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: #f0fdf4;
        border: 1px solid #a7f3d0;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.86rem;
        color: #047857 !important;
        font-weight: 750;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.12);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .live-status-pill:hover {
        transform: translateY(-3px) scale(1.02);
        background: #dcfce7;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.22);
    }
    .live-radar-dot {
        width: 10px;
        height: 10px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: radarPulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* ========================================================
       PLAIN CARDS: STAGGERED ENTRANCES & HOVER TRANSITIONS
       ======================================================== */
    .plain-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        transition: transform 0.38s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.38s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.35s ease;
        animation: fadeInUp 0.65s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .plain-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #e2e8f0;
        transition: height 0.35s ease, background 0.35s ease;
    }
    .card-entry-1 { animation-delay: 0.05s; }
    .card-entry-2 { animation-delay: 0.15s; }
    .card-entry-3 { animation-delay: 0.25s; }
    .card-entry-4 { animation-delay: 0.35s; }

    .plain-card:hover {
        transform: translateY(-8px) scale(1.018);
        box-shadow: 0 16px 36px -4px rgba(0, 0, 0, 0.11), 0 6px 12px -2px rgba(0, 0, 0, 0.05);
        border-color: #000000;
    }
    .plain-card:hover::before {
        height: 5px;
        background: linear-gradient(90deg, #0284c7, #6366f1);
    }
    .card-label {
        font-size: 0.85rem;
        color: #000000 !important;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        transition: color 0.3s ease;
    }
    .card-val {
        font-size: 2.15rem;
        font-weight: 850;
        color: #000000 !important;
        margin: 2px 0 6px 0;
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), color 0.3s ease;
        transform-origin: left center;
    }
    .plain-card:hover .card-val {
        transform: scale(1.06);
    }
    .card-hint {
        font-size: 0.86rem;
        color: #334155 !important;
        line-height: 1.45;
    }

    /* ========================================================
       INTERACTIVE NOTICE BOXES
       ======================================================== */
    .intro-banner {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        color: #000000 !important;
        font-size: 1rem;
        line-height: 1.6;
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.35s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.35s ease;
    }
    .intro-banner:hover {
        transform: translateY(-3px) translateX(4px);
        box-shadow: 0 10px 28px -3px rgba(0, 0, 0, 0.08);
        border-color: #000000;
    }

    .takeaway-box {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        border-top: 1px solid #bbf7d0;
        border-right: 1px solid #bbf7d0;
        border-bottom: 1px solid #bbf7d0;
        padding: 16px 22px;
        border-radius: 0 12px 12px 0;
        margin-top: 18px;
        font-size: 0.95rem;
        color: #000000 !important;
        line-height: 1.55;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 2px 6px rgba(22, 163, 74, 0.05);
    }
    .takeaway-box:hover {
        transform: translateX(8px);
        border-left-width: 9px;
        box-shadow: 0 8px 24px rgba(22, 163, 74, 0.15);
    }

    .analogy-box {
        background: #fffbeb;
        border-left: 5px solid #d97706;
        border-top: 1px solid #fde68a;
        border-right: 1px solid #fde68a;
        border-bottom: 1px solid #fde68a;
        padding: 16px 22px;
        border-radius: 0 12px 12px 0;
        margin-top: 16px;
        font-size: 0.94rem;
        color: #000000 !important;
        line-height: 1.55;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 2px 6px rgba(217, 119, 6, 0.05);
    }
    .analogy-box:hover {
        transform: translateX(8px);
        border-left-width: 9px;
        box-shadow: 0 8px 24px rgba(217, 119, 6, 0.15);
    }

    /* ========================================================
       STREAMLIT TABS, BUTTONS & WIDGETS TRANSITIONS
       ======================================================== */
    [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e2e8f0 !important;
        padding-bottom: 2px;
    }
    [data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 11px 20px !important;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    [data-baseweb="tab"]:hover {
        background-color: #f1f5f9 !important;
        transform: translateY(-2px);
        color: #000000 !important;
    }
    [data-baseweb="tab-highlight"] {
        background: #000000 !important;
        height: 3px !important;
        border-radius: 3px 3px 0 0;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stButton > button {
        transition: all 0.32s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border-radius: 10px !important;
        font-weight: 750 !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #000000 !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.025) !important;
        box-shadow: 0 8px 22px -3px rgba(0, 0, 0, 0.16) !important;
        border-color: #000000 !important;
        background: #f8fafc !important;
    }
    .stButton > button:active {
        transform: translateY(-1px) scale(0.99) !important;
    }

    [data-testid="stSlider"] {
        transition: transform 0.25s ease !important;
    }
    [data-testid="stSlider"]:hover {
        transform: translateY(-1px);
    }

    [data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        transition: all 0.3s ease !important;
        background: #ffffff !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #000000 !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.07) !important;
    }

    /* Plotly Chart Card Transition */
    [data-testid="stPlotlyChart"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.4s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.35s ease !important;
    }
    [data-testid="stPlotlyChart"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 14px 30px -4px rgba(0, 0, 0, 0.09) !important;
        border-color: #94a3b8 !important;
    }

    /* ========================================================
       NEW EXTENDED ANIMATIONS
       ======================================================== */

    /* Ticker tape scroll */
    @keyframes tickerScroll {
        0%   { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }

    /* Morphing border shimmer */
    @keyframes borderMorph {
        0%   { border-radius: 14px; box-shadow: 0 0 0 0 rgba(2,132,199,0.3); }
        33%  { border-radius: 22px 10px 20px 12px; box-shadow: 0 0 0 5px rgba(99,102,241,0.15); }
        66%  { border-radius: 10px 22px 12px 20px; box-shadow: 0 0 0 8px rgba(16,185,129,0.1); }
        100% { border-radius: 14px; box-shadow: 0 0 0 0 rgba(2,132,199,0.3); }
    }

    /* Gauge needle sweep */
    @keyframes gaugeSweep {
        0%   { transform: rotate(-90deg); }
        100% { transform: rotate(var(--needle-angle, 0deg)); }
    }

    /* Shimmer text highlight */
    @keyframes textShimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    /* Wave bar animation */
    @keyframes waveBar {
        0%, 100% { transform: scaleY(0.4); }
        50%       { transform: scaleY(1.0); }
    }

    /* Slide in from left */
    @keyframes slideInLeft {
        0%   { opacity: 0; transform: translateX(-30px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    /* Slide in from right */
    @keyframes slideInRight {
        0%   { opacity: 0; transform: translateX(30px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    /* Flip-in card */
    @keyframes flipIn {
        0%   { opacity: 0; transform: perspective(400px) rotateX(-90deg); }
        40%  { transform: perspective(400px) rotateX(20deg); }
        70%  { transform: perspective(400px) rotateX(-10deg); }
        100% { opacity: 1; transform: perspective(400px) rotateX(0deg); }
    }

    /* Bounce-in scale */
    @keyframes bounceScale {
        0%   { opacity: 0; transform: scale(0.3); }
        50%  { opacity: 1; transform: scale(1.08); }
        75%  { transform: scale(0.94); }
        100% { transform: scale(1.0); }
    }

    /* Progress bar fill */
    @keyframes fillBar {
        0%   { width: 0%; }
        100% { width: var(--bar-width, 50%); }
    }

    /* Pulsing glow ring on feature section headings */
    @keyframes sectionGlow {
        0%, 100% { text-shadow: 0 0 0 rgba(0,0,0,0); }
        50%       { text-shadow: 0 2px 20px rgba(2,132,199,0.2); }
    }

    /* ========================================================
       STATS TICKER
       ======================================================== */
    .ticker-outer {
        overflow: hidden;
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 10px 0;
        margin: 18px 0 24px 0;
        position: relative;
    }
    .ticker-outer::before, .ticker-outer::after {
        content: "";
        position: absolute;
        top: 0; bottom: 0;
        width: 60px;
        z-index: 2;
    }
    .ticker-outer::before { left:0;  background: linear-gradient(to right, #f8fafc, transparent); }
    .ticker-outer::after  { right:0; background: linear-gradient(to left,  #f8fafc, transparent); }
    .ticker-inner {
        display: flex;
        gap: 48px;
        white-space: nowrap;
        animation: tickerScroll 28s linear infinite;
        will-change: transform;
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #000000 !important;
        padding: 0 6px;
    }
    .ticker-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        animation: radarPulse 2s ease infinite;
    }

    /* ========================================================
       FEATURE SECTION HEADER
       ======================================================== */
    .feature-section-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 28px 0 18px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid #e2e8f0;
        animation: slideInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .feature-icon-circle {
        width: 48px; height: 48px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem;
        background: #f1f5f9;
        border: 2px solid #e2e8f0;
        flex-shrink: 0;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: borderMorph 6s ease-in-out infinite;
    }
    .feature-section-header:hover .feature-icon-circle {
        transform: rotate(15deg) scale(1.15);
        background: #000000;
        border-color: #000000;
        color: #ffffff;
    }
    .feature-heading {
        font-size: 1.25rem !important;
        font-weight: 850 !important;
        color: #000000 !important;
        margin: 0 !important;
        animation: sectionGlow 4s ease infinite;
    }
    .feature-subheading {
        font-size: 0.88rem !important;
        color: #475569 !important;
        margin: 2px 0 0 0 !important;
        font-weight: 500 !important;
    }

    /* ========================================================
       RISK GAUGE CARD
       ======================================================== */
    .gauge-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 24px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        animation: fadeInUp 0.7s cubic-bezier(0.16,1,0.3,1) both;
        position: relative;
        overflow: hidden;
    }
    .gauge-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 18px 40px -6px rgba(0,0,0,0.12);
        border-color: #000000;
    }
    .gauge-title {
        font-size: 0.8rem !important;
        font-weight: 750 !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #000000 !important;
        margin-bottom: 8px !important;
    }
    .gauge-value-large {
        font-size: 2.4rem !important;
        font-weight: 900 !important;
        line-height: 1.1 !important;
        color: #000000 !important;
        animation: bounceScale 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
    }
    .gauge-label {
        font-size: 0.82rem !important;
        color: #475569 !important;
        margin-top: 4px !important;
    }

    /* ========================================================
       PROGRESS / RISK BARS
       ======================================================== */
    .risk-bar-container {
        margin: 8px 0;
        animation: slideInRight 0.6s cubic-bezier(0.16,1,0.3,1) both;
    }
    .risk-bar-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        font-size: 0.84rem;
        font-weight: 700;
        color: #000000 !important;
    }
    .risk-bar-track {
        background: #f1f5f9;
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 999px;
        animation: fillBar 1.2s cubic-bezier(0.16,1,0.3,1) both;
        position: relative;
        overflow: hidden;
    }
    .risk-bar-fill::after {
        content: "";
        position: absolute;
        top: 0; left: -100%; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.5) 50%, transparent 100%);
        animation: textShimmer 2s linear infinite;
        background-size: 200% 100%;
    }

    /* ========================================================
       WAVE VISUALIZER
       ======================================================== */
    .wave-viz {
        display: flex;
        align-items: flex-end;
        gap: 4px;
        height: 40px;
        padding: 0 4px;
    }
    .wave-bar {
        flex: 1;
        background: linear-gradient(to top, #0284c7, #6366f1);
        border-radius: 3px 3px 0 0;
        animation: waveBar var(--dur, 1.2s) ease-in-out infinite;
    }

    /* ========================================================
       Z-SCORE CALCULATOR CARD
       ======================================================== */
    .zscore-result-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px 24px;
        margin-top: 16px;
        text-align: center;
        animation: flipIn 0.65s cubic-bezier(0.16,1,0.3,1) both;
        transition: all 0.35s cubic-bezier(0.16,1,0.3,1);
    }
    .zscore-result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px -4px rgba(0,0,0,0.1);
        border-color: #000000;
    }
    .zscore-big {
        font-size: 3rem !important;
        font-weight: 900 !important;
        color: #000000 !important;
        line-height: 1 !important;
        background: linear-gradient(135deg, #0284c7, #6366f1, #ec4899);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: textShimmer 3s linear infinite;
    }
    .zscore-interpretation {
        font-size: 0.9rem !important;
        color: #334155 !important;
        margin-top: 6px !important;
        font-weight: 600 !important;
    }

    /* ========================================================
       LB BATTLE STRATEGY CARD
       ======================================================== */
    .lb-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.38s cubic-bezier(0.16,1,0.3,1);
        animation: fadeInUp 0.65s cubic-bezier(0.16,1,0.3,1) both;
    }
    .lb-card:hover {
        transform: translateY(-6px) scale(1.012);
        box-shadow: 0 16px 36px -4px rgba(0,0,0,0.11);
        border-color: #000000;
    }
    .lb-card::before {
        content: "";
        position: absolute; top:0; left:0; right:0; height:4px;
        background: var(--lb-accent, #e2e8f0);
        transition: height 0.35s ease;
    }
    .lb-card:hover::before { height: 6px; }
    .lb-strategy-name {
        font-size: 1rem !important;
        font-weight: 800 !important;
        color: #000000 !important;
        margin-bottom: 6px !important;
    }
    .lb-stat-row {
        display: flex;
        justify-content: space-between;
        margin: 4px 0;
        font-size: 0.84rem;
        font-weight: 600;
        color: #000000 !important;
    }
    .lb-winner-badge {
        display: inline-block;
        background: #000000;
        color: #ffffff !important;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 3px 10px;
        border-radius: 999px;
        margin-top: 8px;
        animation: bounceScale 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
    }

    /* ========================================================
       FOOTER BAND
       ======================================================== */
    .footer-band {
        margin-top: 40px;
        padding: 18px 24px;
        background: #f8fafc;
        border-top: 2px solid #e2e8f0;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        animation: fadeInUp 0.8s cubic-bezier(0.16,1,0.3,1) both;
        transition: box-shadow 0.35s ease;
    }
    .footer-band:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.07);
    }
    .footer-text {
        font-size: 0.84rem !important;
        color: #475569 !important;
        font-weight: 600 !important;
    }
    .footer-formula {
        font-size: 0.9rem !important;
        font-weight: 800 !important;
        color: #000000 !important;
        background: linear-gradient(135deg, #0284c7, #6366f1);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: textShimmer 4s linear infinite;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Brand Header with Animated High-Level Vector Logo
st.markdown(
    """
    <div class="brand-header-container">
        <div class="logo-wrapper" title="Interactive Gaussian Network Core">
            <svg class="animated-brand-logo" width="84" height="84" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="bellGrad" x1="0%" y1="100%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#0284c7"/>
                        <stop offset="50%" stop-color="#6366f1"/>
                        <stop offset="100%" stop-color="#ec4899"/>
                    </linearGradient>
                    <linearGradient id="coreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#38bdf8"/>
                        <stop offset="100%" stop-color="#1d4ed8"/>
                    </linearGradient>
                    <filter id="logoGlow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="2.5" result="blur"/>
                        <feComposite in="SourceGraphic" in2="blur" operator="over"/>
                    </filter>
                </defs>
                <!-- Orbital Ring -->
                <ellipse class="orbit-ring" cx="50" cy="50" rx="42" ry="20" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="8 5" fill="none"/>
                <!-- Normal Distribution Bell Curve Foundation -->
                <path d="M 10 78 C 28 78, 38 72, 44 42 C 47 26, 53 26, 56 42 C 62 72, 72 78, 90 78" stroke="#e2e8f0" stroke-width="5" stroke-linecap="round" fill="none"/>
                <path class="bell-path" d="M 10 78 C 28 78, 38 72, 44 42 C 47 26, 53 26, 56 42 C 62 72, 72 78, 90 78" stroke="url(#bellGrad)" stroke-width="3.5" stroke-linecap="round" fill="none"/>
                <!-- Baseline Axis -->
                <line x1="8" y1="78" x2="92" y2="78" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round"/>
                <line x1="50" y1="20" x2="50" y2="78" stroke="#cbd5e1" stroke-width="1.4" stroke-dasharray="3 3"/>
                <!-- Network Nodes -->
                <circle cx="28" cy="74" r="3.5" fill="#0284c7" filter="url(#logoGlow)"/>
                <circle cx="72" cy="74" r="3.5" fill="#0284c7" filter="url(#logoGlow)"/>
                <!-- Central Gaussian Peak Node with Breathing Pulse Ring -->
                <circle class="pulse-ring-core" cx="50" cy="32" r="10" stroke="#6366f1" stroke-width="1.8" fill="none"/>
                <circle class="node-peak" cx="50" cy="32" r="6" fill="url(#coreGradient)" filter="url(#logoGlow)"/>
                <circle cx="50" cy="32" r="2.5" fill="#ffffff"/>
            </svg>
        </div>
        <div class="brand-text-wrapper">
            <div class="brand-badge-row">
                <span class="brand-badge">⚡ Enterprise Network Analytics</span>
                <span class="brand-badge">📈 Gaussian Distribution</span>
                <span class="brand-badge">🛡️ Live 3-Sigma Engine</span>
            </div>
            <h1 class="brand-title">Network Load Analysis using Normal Distribution</h1>
            <p class="brand-tagline">Probabilistic Cloud Workload Modeling • Stochastic Jitter Simulation • SLA Breach Prevention</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Header & Intuitive Analogy
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

@st.cache_data(show_spinner=False)
def get_cached_trace(duration_hours: int, base_load: float, diurnal_amp: float, noise_std: float, burst_prob: float, burst_mag: float) -> pd.DataFrame:
    cfg = TrafficSimulationConfig(
        duration_hours=duration_hours,
        sampling_rate_minutes=2,
        base_load_gbps=base_load,
        diurnal_amplitude_gbps=diurnal_amp,
        random_noise_std_gbps=noise_std,
        burst_probability=burst_prob,
        burst_magnitude_gbps=burst_mag,
        random_seed=42,
    )
    sim = NetworkLoadSimulator(cfg)
    return sim.generate_trace()


@st.cache_data(show_spinner=False)
def run_cached_lb_simulation(traffic_tuple: tuple, num_nodes: int, node_cap: float, strategy: str):
    sim = LoadBalancingSimulator(num_nodes=num_nodes, node_capacity_gbps=node_cap)
    return sim.run_simulation(np.array(traffic_tuple), strategy=strategy)


# Sidebar: Simulator Mode & Settings
with st.sidebar:
    st.markdown("### 🎛️ Real-Time Simulator")
    
    live_streaming_on = st.toggle(
        "⚡ Live Real-Time Stream",
        value=False,
        help="Toggle on for continuous second-by-second live streaming. Keep off for instant, zero-lag exploration.",
    )

    if live_streaming_on:
        stream_speed = st.selectbox(
            "Live Refresh Speed:",
            options=[2, 3, 5],
            index=0,
            format_func=lambda s: f"Every {s} seconds",
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
    raw_cached = get_cached_trace(
        duration_hours=duration_hours,
        base_load=base_load,
        diurnal_amp=diurnal_amp,
        noise_std=noise_std,
        burst_prob=burst_prob,
        burst_mag=burst_magnitude,
    )
    df_trace = raw_cached.copy()
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
recommendation_999 = analyzer.recommend_provisioning_capacity(99.9)["recommended_capacity_gbps"]

# 4 Key Scorecards (Clear & Jargon-Free with Staggered Entrance & Hover)
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
        <h3 style="margin: 0; color: #000000; font-size: 1.35rem; font-weight: 850; letter-spacing: -0.01em;">📊 Real-Time Network Telemetry</h3>
        <div class="live-status-pill" title="Live Clock-Synchronized Streaming Active">
            <span class="live-radar-dot"></span>
            <span>{data_origin_label} • Load: <b>{current_live_load:.1f} Gbps</b></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="plain-card card-entry-1">
            <div>
                <div class="card-label">1. Average Traffic</div>
                <div class="card-val">{norm_metrics['mean']:.1f} <span style="font-size:1rem;color:#000000;font-weight:600;">Gbps</span></div>
            </div>
            <div class="card-hint">The typical baseline data flow across all active users.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="plain-card card-entry-2">
            <div>
                <div class="card-label">2. Natural Wiggle</div>
                <div class="card-val">±{norm_metrics['std']:.1f} <span style="font-size:1rem;color:#000000;font-weight:600;">Gbps</span></div>
            </div>
            <div class="card-hint">Standard deviation: how much load bounces around the average.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="plain-card card-entry-3">
            <div>
                <div class="card-label">3. Highest Peak Seen</div>
                <div class="card-val" style="color:#b91c1c;">{peak_load:.1f} <span style="font-size:1rem;color:#000000;font-weight:600;">Gbps</span></div>
            </div>
            <div class="card-hint">The single busiest moment recorded across the day.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="plain-card card-entry-4" style="border: 2px solid #10b981;">
            <div>
                <div class="card-label" style="color:#047857;">4. Safe Capacity (3-Sigma)</div>
                <div class="card-val" style="color:#047857;">{safe_rec:.1f} <span style="font-size:1rem;color:#000000;font-weight:600;">Gbps</span></div>
            </div>
            <div class="card-hint">Covers 99.7% of all surges (Mean + 3×Std).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ── ANIMATED LIVE STATS TICKER ──────────────────────────────────────────────
z_live = (current_live_load - norm_metrics['mean']) / (norm_metrics['std'] + 1e-9)
capacity_util_pct = min(100, current_live_load / (safe_rec + 1e-9) * 100)
ticker_items = [
    ("🟢", "#10b981", f"Mean μ = {norm_metrics['mean']:.1f} Gbps"),
    ("🔵", "#0284c7", f"Std σ = {norm_metrics['std']:.1f} Gbps"),
    ("🟠", "#f59e0b", f"Peak Observed = {peak_load:.1f} Gbps"),
    ("🟣", "#6366f1", f"Safe Cap (3σ) = {safe_rec:.1f} Gbps"),
    ("🔴", "#dc2626", f"Live Z-Score = {z_live:+.2f}σ"),
    ("⚡", "#0284c7", f"Capacity Utilisation = {capacity_util_pct:.1f}%"),
    ("📊", "#10b981", f"99.9% SLA Capacity = {recommendation_999:.1f} Gbps"),
    ("🌐", "#6366f1", f"Min Recorded = {min_load:.1f} Gbps"),
]
items_html = "".join(
    f'<span class="ticker-item"><span class="ticker-dot" style="background:{c};"></span>{emoji}&nbsp;{txt}</span>'
    for emoji, c, txt in ticker_items
) * 2   # duplicate for seamless loop
st.markdown(
    f'<div class="ticker-outer"><div class="ticker-inner">{items_html}</div></div>',
    unsafe_allow_html=True,
)

# 5 Feature Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Traffic Flow",
    "🎯 Capacity Planner",
    "🔔 Bell Curve",
    "⚖️ Z-Score Lab",
    "🏆 Load Balancer Battle",
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
            line=dict(color="#0284c7", width=2),
        )
    )
    fig_trace.add_trace(
        go.Scatter(
            x=df_trace["timestamp"],
            y=df_trace["deterministic_trend"],
            mode="lines",
            name="Expected Day / Night Rhythm",
            line=dict(color="#475569", width=1.8, dash="dash"),
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
                marker=dict(color="#dc2626", size=8, symbol="circle"),
            )
        )

    # Reference capacity line
    fig_trace.add_hline(
        y=safe_rec,
        line_dash="dot",
        line_color="#059669",
        annotation_text=f"Safe Buffer ({safe_rec:.1f} Gbps)",
        annotation_position="top right",
        annotation_font_color="#000000",
    )

    fig_trace.update_layout(
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#000000"),
        uirevision="constant",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title=dict(text="Time of Day", font=dict(color="#000000")),
            tickfont=dict(color="#000000"),
            gridcolor="#e2e8f0",
            linecolor="#000000",
        ),
        yaxis=dict(
            title=dict(text="Traffic Load (Gigabits per second - Gbps)", font=dict(color="#000000")),
            tickfont=dict(color="#000000"),
            gridcolor="#e2e8f0",
            linecolor="#000000",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#000000"),
        ),
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
        line_color="#059669",
        annotation_text="99.9% Safe Line",
        annotation_position="top right",
        annotation_font_color="#000000",
    )

    fig_bell.update_layout(
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#000000"),
        uirevision="constant",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            title=dict(text="Traffic Load in Gbps (From quiet to very busy)", font=dict(color="#000000")),
            tickfont=dict(color="#000000"),
            gridcolor="#e2e8f0",
            linecolor="#000000",
        ),
        yaxis=dict(
            title=dict(text="How Often This Load Level Happens", font=dict(color="#000000")),
            tickfont=dict(color="#000000"),
            gridcolor="#e2e8f0",
            linecolor="#000000",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#000000"),
        ),
    )
    st.plotly_chart(fig_bell, width="stretch")

    st.markdown(
        """
        <div class="takeaway-box">
            <b>🪄 The 3-Step Rule That Every Engineer Uses:</b><br>
            Because internet traffic hugs this bell curve, we don't have to guess how high surges will go.<br>
            In statistics, <b>99.7% of all activity</b> always stays within <b>3 steps (3 standard deviations)</b> of the average.<br>
            So the simple formula to never crash is: <br>
            <center style="margin: 8px 0; font-size: 1.1rem; font-weight: 700; color: #000000;">
                Safe Bandwidth = Average Load + (3 × Normal Wiggle)
            </center>
            That's all it takes to guarantee smooth performance!
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── TAB 4: Z-SCORE INTERACTIVE LABORATORY ──────────────────────────────────
with tab4:
    st.markdown(
        """
        <div class="feature-section-header">
            <div class="feature-icon-circle">🔬</div>
            <div>
                <p class="feature-heading">Z-Score Interactive Laboratory</p>
                <p class="feature-subheading">Calculate exactly how many standard deviations any traffic load sits from the mean</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    zlab_col1, zlab_col2 = st.columns([1, 1])

    with zlab_col1:
        st.markdown("#### 🎛️ Input Your Traffic Scenario")
        z_query_load = st.slider(
            "Traffic load to analyse (Gbps)",
            min_value=float(int(min_load)),
            max_value=float(int(peak_load + 30)),
            value=float(int(norm_metrics['mean'] + norm_metrics['std'])),
            step=0.5,
            format="%.1f Gbps",
            key="zscore_slider",
        )
        z_direction = st.radio(
            "Which boundary are you testing?",
            options=["Upper tail — load EXCEEDS this value", "Lower tail — load FALLS BELOW this value"],
            index=0,
            key="z_direction_radio",
        )
        upper_tail = z_direction.startswith("Upper")

        z_computed = (z_query_load - norm_metrics['mean']) / (norm_metrics['std'] + 1e-9)
        from scipy.special import erfc
        if upper_tail:
            p_exceed = float(stats.norm.sf(z_query_load, loc=norm_metrics['mean'], scale=norm_metrics['std'])) * 100
            p_label = "Probability load EXCEEDS this"
        else:
            p_exceed = float(stats.norm.cdf(z_query_load, loc=norm_metrics['mean'], scale=norm_metrics['std'])) * 100
            p_label = "Probability load FALLS BELOW this"

        # Interpretation
        az = abs(z_computed)
        if az < 1:
            interp = "🟢 Within 1σ — highly typical operating range."
        elif az < 2:
            interp = "🟡 Between 1σ–2σ — elevated but manageable load."
        elif az < 3:
            interp = "🟠 Between 2σ–3σ — uncommon surge, activate buffer."
        else:
            interp = "🔴 Beyond 3σ — extreme rare event, SLA at risk!"

        st.markdown(
            f"""
            <div class="zscore-result-card">
                <div class="gauge-title">Computed Z-Score</div>
                <div class="zscore-big">{z_computed:+.3f}σ</div>
                <div class="zscore-interpretation">{interp}</div>
                <hr style="border:none;border-top:1px solid #e2e8f0;margin:12px 0;">
                <div style="font-size:0.88rem;font-weight:700;color:#000000;">
                    {p_label}: <span style="color:#0284c7;">{p_exceed:.4f}%</span>
                </div>
                <div style="font-size:0.78rem;color:#475569;margin-top:4px;">Formula: Z = (X − μ) / σ = ({z_query_load:.1f} − {norm_metrics['mean']:.1f}) / {norm_metrics['std']:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with zlab_col2:
        st.markdown("#### 📊 Z-Score Distribution Visualisation")
        x_z = np.linspace(norm_metrics['mean'] - 4*norm_metrics['std'], norm_metrics['mean'] + 4*norm_metrics['std'], 300)
        pdf_z = stats.norm.pdf(x_z, norm_metrics['mean'], norm_metrics['std'])

        # Build fill regions
        fill_x = x_z[x_z >= z_query_load] if upper_tail else x_z[x_z <= z_query_load]
        fill_y = stats.norm.pdf(fill_x, norm_metrics['mean'], norm_metrics['std'])

        fig_z = go.Figure()
        # Shaded tail
        fig_z.add_trace(go.Scatter(
            x=np.concatenate([[fill_x[0]], fill_x, [fill_x[-1]]]) if len(fill_x) else [],
            y=np.concatenate([[0], fill_y, [0]]) if len(fill_y) else [],
            fill='toself',
            fillcolor='rgba(239,68,68,0.18)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Tail Probability',
            hoverinfo='skip',
        ))
        # Bell curve
        fig_z.add_trace(go.Scatter(
            x=x_z, y=pdf_z, mode='lines',
            name='Normal Distribution',
            line=dict(color='#0284c7', width=2.5),
        ))
        # Vertical line at query
        fig_z.add_vline(
            x=z_query_load,
            line_dash='dash', line_color='#dc2626',
            annotation_text=f'{z_query_load:.1f} Gbps (Z={z_computed:+.2f}σ)',
            annotation_font_color='#000000',
            annotation_position='top left',
        )
        # Mean line
        fig_z.add_vline(
            x=norm_metrics['mean'],
            line_dash='dot', line_color='#10b981',
            annotation_text=f'μ={norm_metrics["mean"]:.1f}',
            annotation_font_color='#000000',
        )
        fig_z.update_layout(
            template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            font=dict(color='#000000'),
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title=dict(text='Traffic Load (Gbps)', font=dict(color='#000000')), tickfont=dict(color='#000000'), gridcolor='#e2e8f0'),
            yaxis=dict(title=dict(text='Probability Density', font=dict(color='#000000')), tickfont=dict(color='#000000'), gridcolor='#e2e8f0'),
            legend=dict(font=dict(color='#000000')),
            uirevision='zscore',
        )
        st.plotly_chart(fig_z, width='stretch')

        # Sigma level bars
        st.markdown("**Sigma Containment Reference:**")
        sigma_bars = [("1σ (68.27%)", 68.27, "#0284c7"), ("2σ (95.45%)", 95.45, "#6366f1"), ("3σ (99.73%)", 99.73, "#10b981"), ("4σ (99.994%)", 99.994, "#f59e0b")]
        for label, pct, color in sigma_bars:
            st.markdown(
                f"""
                <div class="risk-bar-container" style="animation-delay:{sigma_bars.index((label,pct,color))*0.1}s;">
                    <div class="risk-bar-label">
                        <span>{label}</span>
                        <span>{pct:.2f}%</span>
                    </div>
                    <div class="risk-bar-track">
                        <div class="risk-bar-fill" style="--bar-width:{pct:.1f}%;width:{pct:.1f}%;background:{color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="analogy-box">
            <b>📐 What does Z-Score mean for network engineers?</b><br>
            A Z-Score tells you how unusual a traffic reading is. Z = 0 means exactly average.
            Z = +2 means the load is 2 standard deviations above average — rare but not extreme.
            Z = +3.09 corresponds to a 99.9% SLA — beyond this, packet drops become statistically inevitable.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── TAB 5: LOAD BALANCER STRATEGY BATTLE ─────────────────────────────────────
with tab5:
    st.markdown(
        """
        <div class="feature-section-header">
            <div class="feature-icon-circle">⚔️</div>
            <div>
                <p class="feature-heading">Load Balancer Strategy Battle</p>
                <p class="feature-subheading">Head-to-head comparison: Round-Robin vs Least-Loaded vs Statistical Gaussian Routing</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lb_col_cfg, lb_col_result = st.columns([1, 2])

    with lb_col_cfg:
        st.markdown("#### ⚙️ Battle Configuration")
        lb_nodes = st.slider("Number of Server Nodes", 2, 8, 4, key="lb_nodes")
        lb_capacity = st.slider("Per-Node Capacity (Gbps)", 10.0, 60.0, 30.0, 2.0, key="lb_cap")

        st.markdown(
            """
            <div class="takeaway-box">
                <b>⚔️ Strategy Descriptions</b><br>
                • <b>Round-Robin:</b> Cycles traffic evenly — blind to actual load.<br>
                • <b>Least-Loaded:</b> Routes to least-busy node — reactive, not predictive.<br>
                • <b>Gaussian-Aware:</b> Uses statistical headroom probability — proactive, optimal.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lb_col_result:
        traffic_stream = df_trace["total_load_gbps"].values
        # Sample or limit to 200 points for ultra-fast, smooth calculation
        stream_sample = traffic_stream[:200] if len(traffic_stream) > 200 else traffic_stream
        traffic_tuple = tuple(np.round(stream_sample, 2))

        # Comparative line chart: utilisation over time per strategy
        st.markdown("**📈 Per-Node Load History (Gaussian-Aware Strategy)**")
        fig_lb = go.Figure()
        lb_result_detail = run_cached_lb_simulation(traffic_tuple, lb_nodes, lb_capacity, "statistical_aware")
        for ni, hist in enumerate(lb_result_detail.get('node_histories', [])):
            if hist:
                fig_lb.add_trace(go.Scatter(
                    y=hist[:120],
                    mode='lines',
                    name=f'Node {ni+1}',
                    line=dict(width=1.8),
                ))
        fig_lb.add_hline(y=lb_capacity, line_dash='dot', line_color='#dc2626',
                         annotation_text='Capacity Ceiling', annotation_font_color='#000000')
        fig_lb.update_layout(
            template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            font=dict(color='#000000'),
            height=300, margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title=dict(text='Time Step', font=dict(color='#000000')), tickfont=dict(color='#000000'), gridcolor='#e2e8f0'),
            yaxis=dict(title=dict(text='Load (Gbps)', font=dict(color='#000000')), tickfont=dict(color='#000000'), gridcolor='#e2e8f0'),
            legend=dict(font=dict(color='#000000')),
            uirevision='lb',
        )
        st.plotly_chart(fig_lb, width='stretch')


# ── ANIMATED RISK GAUGE PANEL ─────────────────────────────────────────────────
st.markdown(
    """
    <div class="feature-section-header">
        <div class="feature-icon-circle">📡</div>
        <div>
            <p class="feature-heading">Live Network Health Dashboard</p>
            <p class="feature-subheading">Real-time probabilistic risk metrics derived from the current traffic stream</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compute metrics
health_z = (current_live_load - norm_metrics['mean']) / (norm_metrics['std'] + 1e-9)
breach_prob_now = float(stats.norm.sf(current_live_load, loc=norm_metrics['mean'], scale=norm_metrics['std'])) * 100
headroom_gbps = max(0, safe_rec - current_live_load)
util_now_pct = min(100, current_live_load / (safe_rec + 1e-9) * 100)

g1, g2, g3, g4 = st.columns(4)
# Pre-compute bar widths per gauge metric (simple, readable)
gauge_data = [
    (g1, "Current Z-Score",    f"{health_z:+.2f}σ",        "Deviation from mean",    "#0284c7",  min(100, abs(health_z) / 4.0 * 100)),
    (g2, "Breach Probability", f"{breach_prob_now:.3f}%",   "P(Load > Current)",      "#dc2626",  min(100, breach_prob_now * 10)),
    (g3, "Safety Headroom",    f"{headroom_gbps:.1f} Gbps", "Until 3σ ceiling",       "#10b981",  min(100, headroom_gbps / (safe_rec + 1e-9) * 100)),
    (g4, "Capacity Util.",     f"{util_now_pct:.1f}%",      "of safe 3σ capacity",    "#f59e0b",  util_now_pct),
]
for col, title, value, label, bar_color, bar_pct in gauge_data:
    with col:
        st.markdown(
            f"""
            <div class="gauge-card">
                <div class="gauge-title">{title}</div>
                <div class="gauge-value-large">{value}</div>
                <div class="gauge-label">{label}</div>
                <div class="risk-bar-track" style="margin-top:10px;">
                    <div class="risk-bar-fill" style="--bar-width:{bar_pct:.1f}%;width:{bar_pct:.1f}%;background:{bar_color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Wave visualiser
wave_bars = "".join(
    f'<div class="wave-bar" style="--dur:{0.8+i*0.07:.2f}s; height:{int(15+abs(np.sin(i*0.8+health_z))*22)}px;"></div>'
    for i in range(20)
)
st.markdown(
    f"""
    <div style="margin:20px 0 8px 0;font-size:0.82rem;font-weight:700;color:#000;text-transform:uppercase;letter-spacing:0.06em;">⬇ Live Traffic Waveform</div>
    <div class="wave-viz" style="height:55px;">{wave_bars}</div>
    """,
    unsafe_allow_html=True,
)



# Real-time auto-streaming loop
if live_streaming_on:
    import time
    time.sleep(stream_speed)
    st.rerun()
