import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(16, 10), dpi=300)

PRIMARY = '#1d4ed8'
ACCENT_GREEN = '#10b981'
ACCENT_RED = '#ef4444'
DARK_TEXT = '#0f172a'

# --- PANEL 1: Gaussian Bell Curve & SLA Capacity Sizing ---
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

mu = 50.0  # Mean load in Gbps
sigma = 8.0  # Std dev in Gbps
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = norm.pdf(x, mu, sigma)

ax1.plot(x, y, color=PRIMARY, linewidth=3, label=r'Gaussian Traffic PDF $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$')

ax1.fill_between(x, 0, y, where=(x >= mu - sigma) & (x <= mu + sigma), color='#3b82f6', alpha=0.18, label=r'1$\sigma$ Core Band (68.27%)')
ax1.fill_between(x, 0, y, where=(x >= mu - 2*sigma) & (x <= mu + 2*sigma), color='#60a5fa', alpha=0.10, label=r'2$\sigma$ Safe Band (95.45%)')

z_sla = 3.09
capacity = mu + z_sla * sigma
ax1.axvline(capacity, color=ACCENT_RED, linestyle='--', linewidth=2.5, label=rf'Capacity $C = \mu + Z\sigma = {capacity:.1f}$ Gbps (99.9% SLA)')
ax1.fill_between(x, 0, y, where=(x >= capacity), color=ACCENT_RED, alpha=0.45, label=r'SLA Breach Risk $P(X > C) = 1 - \Phi(Z) < 0.1\%$')
ax1.axvline(mu, color='#0f172a', linestyle=':', linewidth=2, label=rf'Mean Baseline $\mu = {mu:.1f}$ Gbps')

ax1.set_title(r'Gaussian Distribution & Capacity Sizing: $C = \mu + Z \cdot \sigma$', fontsize=14, fontweight='bold', pad=10, color=DARK_TEXT)
ax1.set_xlabel('Network Traffic Throughput (Gbps)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=9.5, facecolor='#ffffff', edgecolor='#cbd5e1')
ax1.grid(True, linestyle='--', alpha=0.6)

ax1.annotate(r'$\mu = 50$ Gbps', xy=(mu, norm.pdf(mu, mu, sigma)), xytext=(mu - 12, norm.pdf(mu, mu, sigma)*0.95),
             arrowprops=dict(facecolor='#0f172a', shrink=0.08, width=1.5, headwidth=6), fontsize=10, fontweight='bold')

ax1.annotate(r'Capacity $C = 74.7$ Gbps' + '\n' + r'($Z = 3.090$ for 99.9% SLA)', xy=(capacity, norm.pdf(capacity, mu, sigma)), xytext=(capacity + 1.2, 0.015),
             arrowprops=dict(facecolor=ACCENT_RED, shrink=0.08, width=1.5, headwidth=6), fontsize=10, fontweight='bold', color=ACCENT_RED)


# --- PANEL 2: Diurnal Time-Series with Stochastics & SLA Threshold ---
ax2 = plt.subplot2grid((2, 2), (1, 0))
np.random.seed(42)
t = np.linspace(0, 24, 720)
diurnal = 50 + 15 * np.sin(2 * np.pi * (t - 6) / 24)
noise = np.random.normal(0, 3.5, size=len(t))
traffic = diurnal + noise
traffic[420:435] += 12.0

ax2.plot(t, traffic, color='#0284c7', linewidth=1.2, alpha=0.85, label='Actual Stochastic Traffic $L(t)$')
ax2.plot(t, diurnal, color='#0f172a', linewidth=2.0, linestyle='--', label=r'Diurnal Wave $B + A\sin(\omega t)$')
ax2.axhline(capacity, color=ACCENT_RED, linewidth=2.2, linestyle='-', label=rf'Safe Capacity $C = {capacity:.1f}$ Gbps')
ax2.axhline(mu, color='#64748b', linewidth=1.5, linestyle=':', label=rf'Mean Baseline $\mu = {mu:.1f}$ Gbps')

ax2.set_title('24-Hour Network Load Trace vs Capacity Threshold', fontsize=12, fontweight='bold', color=DARK_TEXT)
ax2.set_xlabel('Time of Day (Hours)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Aggregate Load (Gbps)', fontsize=10, fontweight='bold')
ax2.set_xlim(0, 24)
ax2.set_xticks(np.arange(0, 25, 4))
ax2.legend(loc='lower center', frameon=True, fontsize=8.5, ncol=2)
ax2.grid(True, linestyle='--', alpha=0.6)


# --- PANEL 3: Mathematical Formula Summary Card ---
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.axis('off')

lines = [
    r"$\mathbf{MATHEMATICAL\ FORMULATION\ CHEAT\ SHEET}$",
    "",
    r"$\mathbf{1.\ Capacity\ Sizing\ Formula:}$",
    r"    $C = \mu + Z_{\alpha} \cdot \sigma$",
    r"    $\bullet\ 95.0\%\ \mathrm{SLA} \rightarrow Z = 1.645$",
    r"    $\bullet\ 99.0\%\ \mathrm{SLA} \rightarrow Z = 2.326$",
    r"    $\bullet\ 99.9\%\ \mathrm{SLA} \rightarrow Z = 3.090$",
    "",
    r"$\mathbf{2.\ Standard\ Z-Score:}$",
    r"    $Z = \frac{X - \mu}{\sigma}$",
    "",
    r"$\mathbf{3.\ SLA\ Breach\ Risk\ (Survival\ Function):}$",
    r"    $P(X > C) = 1 - \Phi\left(\frac{C - \mu}{\sigma}\right)$",
    "",
    r"$\mathbf{4.\ Traffic\ Simulation\ Model:}$",
    r"    $L(t) = B + A\sin\left(\frac{2\pi t}{1440} + \phi\right) + \epsilon(t) + \xi(t)$",
    r"    $\epsilon \sim \mathcal{N}(0, \sigma^2),\quad \xi \sim \mathrm{Bernoulli}(p)\times \mathrm{Exp}(\lambda)$",
    "",
    r"$\mathbf{5.\ Goodness-of-Fit\ Normality\ Test:}$",
    r"    $K^2 = Z^2(\sqrt{b_1}) + Z^2(b_2) \sim \chi^2(2),\quad p > 0.05 \rightarrow \mathrm{Accept}\ H_0$"
]
formula_text = "\n".join(lines)

ax3.text(0.04, 0.98, formula_text, transform=ax3.transAxes, fontsize=10.2,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.9', facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1.5))

plt.suptitle('Network Load Analysis & Capacity Planning using Normal Distribution', fontsize=16, fontweight='black', y=0.98, color='#0f172a')
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = r'c:\Users\Arjun\OneDrive\Desktop\Maths microproject\Network-Load-Analysis-Using-Normal-Distribution\network_load_math_infographic.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print('SUCCESS_IMAGE_SAVED:', output_path)
