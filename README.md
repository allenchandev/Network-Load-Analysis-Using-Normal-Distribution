# Network Load Analysis Using Normal Distribution

An enterprise capacity planning and load modeling platform designed for distributed systems and hyperscale data centers (Meta, Google, Microsoft).

## 📌 Project Overview
- **Industry Focus**: Distributed Systems & Cloud Infrastructure
- **Integrated Outcomes**: CO3 (Mathematical & Stochastic Modeling), CO4 (Probabilistic SLA Risk Analysis), CO5 (Distributed Architecture & Load Balancing Evaluation)
- **Problem Statement**: Cloud and data center network ingress/egress loads fluctuate following diurnal human cycles coupled with Gaussian stochastic variations and sudden bursts. This suite applies Normal Distribution theory ($X \sim \mathcal{N}(\mu, \sigma^2)$) to optimize bandwidth provisioning, guarantee 99.9%+ SLA compliance, and minimize packet drop rates.

---

## 🚀 Key Modules
1. **Traffic Simulation Engine (`src/simulator.py`)**:
   - Diurnal sinusoidal oscillation (24-hour business cycle).
   - Gaussian stochastic fluctuations $\mathcal{N}(0, \sigma^2)$.
   - Burst anomaly spikes modeled via exponential/Pareto distributions.
2. **Statistical Analyzer (`src/statistical_model.py`)**:
   - Skewness, Kurtosis, D'Agostino-Pearson omnibus normality test, and Kolmogorov-Smirnov test.
   - Gaussian Survival Function calculation: $P(X > C) = 1 - \Phi\left(\frac{C - \mu}{\sigma}\right)$.
   - Headroom capacity recommendation: $C = \mu + Z \cdot \sigma$.
3. **Load Balancer Simulator (`src/load_balancer.py`)**:
   - Comparative evaluation of Round-Robin, Least-Connections, and Gaussian Headroom-Aware routing algorithms.
4. **Interactive Dashboard (`app.py`)**:
   - Built with Streamlit and Plotly for real-time parameter tuning, Q-Q plots, PDF/CDF curves, and gateway node analytics.

---

## 🛠️ Installation & Setup

1. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

4. **Run Standalone Scripts**:
   ```bash
   python src/simulator.py
   python src/statistical_model.py
   python src/load_balancer.py
   ```
