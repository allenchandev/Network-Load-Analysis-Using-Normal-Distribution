import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(16, 10.5), dpi=300)

DARK_TEXT = '#0f172a'
COLOR_SAFE = '#10b981'      # Green
COLOR_ELEVATED = '#f59e0b'  # Amber / Yellow
COLOR_WARNING = '#f97316'   # Orange
COLOR_DANGER = '#ef4444'    # Red
COLOR_BLUE = '#1d4ed8'      # Blue

mu = 50.0     # Mean load (Gbps)
sigma = 8.0   # Std dev (Gbps)

# -------------------------------------------------------------
# PANEL 1: Gaussian Curve Partitioned by Z-Score Risk Zones
# -------------------------------------------------------------
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

x = np.linspace(mu - 4.2*sigma, mu + 4.2*sigma, 1200)
y = norm.pdf(x, mu, sigma)

ax1.plot(x, y, color=DARK_TEXT, linewidth=2.5, label='Network Traffic Distribution')

# Shading Zones
# 1. Safe Zone: |Z| < 1
ax1.fill_between(x, 0, y, where=(x >= mu - sigma) & (x <= mu + sigma),
                 color=COLOR_SAFE, alpha=0.25, label=r'Safe Operating Zone (|Z| < 1$\sigma$, 68.27%)')

# 2. Elevated Zone: 1 <= |Z| < 2
ax1.fill_between(x, 0, y, where=((x >= mu - 2*sigma) & (x < mu - sigma)) | ((x > mu + sigma) & (x <= mu + 2*sigma)),
                 color=COLOR_ELEVATED, alpha=0.28, label=r'Elevated Busy Zone (1$\sigma$ $\leq$ |Z| < 2$\sigma$, 95.45% cum.)')

# 3. Warning Surge Zone: 2 <= Z < 3
ax1.fill_between(x, 0, y, where=(x > mu + 2*sigma) & (x <= mu + 3*sigma),
                 color=COLOR_WARNING, alpha=0.35, label=r'Warning Surge Zone (2$\sigma$ $\leq$ Z < 3$\sigma$, 99.73% cum.)')

# 4. Critical Danger Zone: Z >= 3
ax1.fill_between(x, 0, y, where=(x >= mu + 3*sigma),
                 color=COLOR_DANGER, alpha=0.55, label=r'Critical Anomaly Zone (Z $\geq$ 3$\sigma$, <0.27% tail risk)')

# Key Capacity Vertical Markers
cap_mean = mu                          # 50.0 Gbps (50% failure)
cap_95 = mu + 1.645 * sigma            # 63.2 Gbps (5% failure)
cap_999 = mu + 3.090 * sigma           # 74.7 Gbps (<0.1% failure)

ax1.axvline(cap_mean, color='#475569', linestyle=':', linewidth=2.2, label=rf'Mean Baseline Sizing: C = {cap_mean:.1f} Gbps (50.0% Crash Risk)')
ax1.axvline(cap_95, color=COLOR_WARNING, linestyle='-.', linewidth=2.2, label=rf'Moderate Sizing (95% SLA): C = {cap_95:.1f} Gbps (5.0% Breach)')
ax1.axvline(cap_999, color=COLOR_DANGER, linestyle='--', linewidth=2.6, label=rf'Safest Sizing (99.9% SLA): C = {cap_999:.1f} Gbps (<0.1% Breach)')

ax1.set_title('Gaussian Risk Spectrum: Safe Zones vs Critical Tail Boundaries (Z-Score Partitions)', fontsize=14, fontweight='bold', pad=10, color=DARK_TEXT)
ax1.set_xlabel('Network Throughput Load (Gbps)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Probability Density f(x)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=9, facecolor='#ffffff', edgecolor='#cbd5e1')
ax1.grid(True, linestyle='--', alpha=0.6)

# Annotations
ax1.annotate('Mean Baseline: 50 Gbps\nHIGHEST RISK (50% Crash)', xy=(cap_mean, norm.pdf(cap_mean, mu, sigma)),
             xytext=(cap_mean - 16, norm.pdf(cap_mean, mu, sigma)*0.85),
             arrowprops=dict(facecolor='#475569', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#475569')

ax1.annotate('SAFEST LEVEL\nC = 74.7 Gbps (99.9% SLA)', xy=(cap_999, norm.pdf(cap_999, mu, sigma)),
             xytext=(cap_999 + 1.5, 0.016),
             arrowprops=dict(facecolor=COLOR_DANGER, shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight='bold', color=COLOR_DANGER)


# -------------------------------------------------------------
# PANEL 2: Capacity Provisioning & Risk Comparison Matrix
# -------------------------------------------------------------
ax2 = plt.subplot2grid((2, 2), (1, 0))

strategies = [
    'Highest Risk\n(Mean Sizing)\nC = 50.0 Gbps',
    'Moderate Risk\n(95% SLA)\nC = 63.2 Gbps',
    'Safest Level\n(99.9% SLA)\nC = 74.7 Gbps'
]
breach_risks = [50.0, 5.0, 0.1]
bar_colors = [COLOR_DANGER, COLOR_ELEVATED, COLOR_SAFE]

bars = ax2.bar(strategies, breach_risks, color=bar_colors, width=0.55, edgecolor='#0f172a', linewidth=1.2)

for bar, risk in zip(bars, breach_risks):
    yval = bar.get_height()
    if risk >= 1.0:
        label_text = f'{risk:.1f}% Risk\nof Outage'
    else:
        label_text = f'< 0.1% Risk\n(Safe 3-Sigma)'
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, label_text,
             ha='center', va='bottom', fontsize=10, fontweight='bold', color=DARK_TEXT)

ax2.set_title('SLA Breach Probability Comparison by Sizing Strategy', fontsize=12, fontweight='bold', color=DARK_TEXT)
ax2.set_ylabel('Probability of SLA Violation / Crash (%)', fontsize=10, fontweight='bold')
ax2.set_ylim(0, 60)
ax2.grid(axis='y', linestyle='--', alpha=0.7)


# -------------------------------------------------------------
# PANEL 3: Real-Time Operating Threat Levels & Summary Card
# -------------------------------------------------------------
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.axis('off')

card_text = """CAPACITY SIZING & RISK SPECTRUM SUMMARY

1. SIZING BENCHMARKS & BREACH RISKS:
  * SAFEST LEVEL (99.9% SLA):  C = mu + 3.090*sigma = 74.7 Gbps
    -> SLA Breach Risk < 0.1%  (Near-zero packet drops)
  * MODERATE LEVEL (95.0% SLA): C = mu + 1.645*sigma = 63.2 Gbps
    -> SLA Breach Risk = 5.0%   (Occasional peak afternoon lag)
  * HIGHEST RISK (Mean Sizing): C = mu = 50.0 Gbps
    -> SLA Breach Risk = 50.0%  (CRITICAL: Crashes 50% of the day!)

2. REAL-TIME Z-SCORE OPERATING ZONES:
  * [Safe Zone] (|Z| < 1*sigma):
      Normal baseline traffic (68.27% of all operating time)
  * [Elevated Zone] (1*sigma <= |Z| < 2*sigma):
      Typical peak hour traffic (up to 95.45% coverage)
  * [Warning Zone] (2*sigma <= |Z| < 3*sigma):
      Uncommon surge (99.73% coverage; safety buffer engaged)
  * [Critical Danger Zone] (|Z| >= 3*sigma):
      Extreme burst anomaly (<0.27% of time; immediate outage risk!)

3. CORE MATHEMATICAL FORMULA:
  * Safe Bandwidth = mu + 3*sigma  (Guarantees 99.73%+ uptime)
  * Breach Probability: P(X > C) = 1 - Phi((C - mu) / sigma)"""

ax3.text(0.02, 0.98, card_text, transform=ax3.transAxes, fontsize=9.4, fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round,pad=1.0', facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1.5))

plt.suptitle('Network Load Analysis: Safest Level vs. Risk Level Evaluation', fontsize=16, fontweight='bold', y=0.98, color=DARK_TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = r'c:\Users\Arjun\OneDrive\Desktop\Maths microproject\Network-Load-Analysis-Using-Normal-Distribution\network_risk_and_safe_levels.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print('SUCCESS_RISK_IMAGE_SAVED:', output_path)
