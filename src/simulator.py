"""
Network Load Generator Module for Data Center Traffic Modeling.
Simulates diurnal cycles, peak business hours, and burst anomalies.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class TrafficSimulationConfig:
    duration_hours: int = 24
    sampling_rate_minutes: int = 1  # 1 sample per minute
    base_load_gbps: float = 40.0
    diurnal_amplitude_gbps: float = 25.0
    random_noise_std_gbps: float = 4.5
    peak_hours: Tuple[int, int] = (13, 17)  # 1 PM to 5 PM
    burst_probability: float = 0.02
    burst_magnitude_gbps: float = 18.0
    random_seed: int = 42


class NetworkLoadSimulator:
    """Simulates realistic data center ingress/egress network loads."""

    def __init__(self, config: TrafficSimulationConfig = None):
        self.config = config or TrafficSimulationConfig()

    def generate_trace(self) -> pd.DataFrame:
        """
        Generates time series network traffic trace with:
        - Base demand
        - Diurnal periodic trend (sine/cosine waves)
        - Stochastic Gaussian fluctuations
        - Burst anomalies (e.g. data syncs, backups, traffic spikes)
        """
        cfg = self.config
        np.random.seed(cfg.random_seed)

        total_steps = int((cfg.duration_hours * 60) / cfg.sampling_rate_minutes)
        timestamps = pd.date_range(
            start="2026-10-12 00:00:00",
            periods=total_steps,
            freq=f"{cfg.sampling_rate_minutes}min",
        )

        hours = np.linspace(0, cfg.duration_hours, total_steps)

        # Diurnal pattern modeled with primary cycle peaking during afternoon
        # Cosine shifted to peak at ~14:00 (2 PM)
        phase_shift = (14.0 / 24.0) * 2 * np.pi
        diurnal_pattern = cfg.diurnal_amplitude_gbps * np.sin(
            (2 * np.pi * (hours % 24) / 24.0) - (np.pi / 2) + (phase_shift / 2)
        )

        # Baseline + diurnal
        deterministic_load = cfg.base_load_gbps + diurnal_pattern

        # Gaussian / Normal noise N(0, sigma^2)
        normal_noise = np.random.normal(0, cfg.random_noise_std_gbps, total_steps)

        # Bursts (spikes) modeled as Bernoulli-triggered Pareto/Exponential bursts
        burst_mask = np.random.binomial(1, cfg.burst_probability, total_steps)
        burst_values = burst_mask * np.random.exponential(
            cfg.burst_magnitude_gbps, total_steps
        )

        total_load = np.maximum(5.0, deterministic_load + normal_noise + burst_values)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour_of_day": (timestamps.hour + timestamps.minute / 60.0),
                "total_load_gbps": total_load,
                "deterministic_trend": deterministic_load,
                "gaussian_fluctuation": normal_noise,
                "burst_anomaly": burst_values,
                "is_burst": burst_mask.astype(bool),
            }
        )

        return df

    def next_live_point(self, current_dt: pd.Timestamp) -> Dict[str, Any]:
        """
        Generates the next real-time streaming traffic data point anchored to actual wall clock time.
        """
        cfg = self.config
        hour_frac = current_dt.hour + (current_dt.minute / 60.0) + (current_dt.second / 3600.0)

        # Diurnal pattern peaking at 14:00 (2 PM)
        phase_shift = (14.0 / 24.0) * 2 * np.pi
        diurnal = cfg.diurnal_amplitude_gbps * np.sin(
            (2 * np.pi * hour_frac / 24.0) - (np.pi / 2) + (phase_shift / 2)
        )
        det_load = cfg.base_load_gbps + diurnal

        # Gaussian noise
        noise = np.random.normal(0, cfg.random_noise_std_gbps)

        # Burst anomaly check
        is_burst = bool(np.random.rand() < (cfg.burst_probability * 0.2))  # scaled for per-second sampling
        burst_val = float(np.random.exponential(cfg.burst_magnitude_gbps)) if is_burst else 0.0

        total_load = max(5.0, det_load + noise + burst_val)

        return {
            "timestamp": current_dt,
            "hour_of_day": hour_frac,
            "total_load_gbps": round(float(total_load), 2),
            "deterministic_trend": round(float(det_load), 2),
            "gaussian_fluctuation": round(float(noise), 2),
            "burst_anomaly": round(float(burst_val), 2),
            "is_burst": is_burst,
        }


if __name__ == "__main__":
    sim = NetworkLoadSimulator()
    data = sim.generate_trace()
    print("Generated data sample:")
    print(data.head())
    print(f"Summary Statistics:\n{data['total_load_gbps'].describe()}")
