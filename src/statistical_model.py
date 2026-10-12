"""
Statistical Modeling & Capacity Planning Module using Normal Distribution.
Addresses Course Outcomes (CO3 - Modeling, CO4 - Probabilistic Analysis, CO5 - System Evaluation).
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.stats as stats
import pandas as pd


class NormalDistributionAnalyzer:
    """
    Performs statistical analysis, hypothesis testing for normality,
    and capacity provisioning calculations.
    """

    def __init__(self, data_series: pd.Series):
        self.series = data_series.dropna()
        self.mean = float(np.mean(self.series))
        self.std = float(np.std(self.series, ddof=1))
        self.median = float(np.median(self.series))
        self.skewness = float(stats.skew(self.series))
        self.kurtosis = float(stats.kurtosis(self.series))

    def test_normality(self) -> Dict[str, Any]:
        """
        Conducts Shapiro-Wilk (or D'Agostino-Pearson) test and Kolmogorov-Smirnov test
        to evaluate how closely the load resembles a Gaussian distribution.
        """
        # For large samples (N > 5000), D'Agostino and Pearson's test is more standard
        dagostino_stat, dagostino_p = stats.normaltest(self.series)
        
        # Kolmogorov-Smirnov test against fitted normal distribution
        # Standardize series to test against standard normal (robust across scipy versions)
        z_series = (self.series - self.mean) / (self.std if self.std > 0 else 1.0)
        ks_res = stats.kstest(z_series, stats.norm.cdf)
        ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)

        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "dagostino_stat": float(dagostino_stat),
            "dagostino_p_value": float(dagostino_p),
            "ks_stat": float(ks_stat),
            "ks_p_value": float(ks_p),
            "is_approximately_normal": bool(ks_p > 0.01 and abs(self.skewness) < 1.0),
        }

    def compute_percentiles(self) -> Dict[str, float]:
        """Computes empirical percentiles vs theoretical normal distribution percentiles."""
        percentiles = [50, 75, 90, 95, 99, 99.9]
        empirical = {f"P{p}": float(np.percentile(self.series, p)) for p in percentiles}
        theoretical = {
            f"P{p}": float(stats.norm.ppf(p / 100.0, loc=self.mean, scale=self.std))
            for p in percentiles
        }
        return {"empirical": empirical, "theoretical": theoretical}

    def calculate_sla_violation_probability(self, capacity_limit_gbps: float) -> Dict[str, float]:
        """
        Calculates P(X > Capacity) using the Gaussian survival function (1 - CDF),
        representing the probability of an SLA breach or packet drop event.
        """
        if self.std <= 0:
            return {"z_score": 0.0, "violation_prob": 0.0, "empirical_violation_prob": 0.0}

        z_score = (capacity_limit_gbps - self.mean) / self.std
        theoretical_prob = float(stats.norm.sf(z_score))  # 1 - cdf(z)
        empirical_prob = float((self.series > capacity_limit_gbps).mean())

        return {
            "capacity_gbps": capacity_limit_gbps,
            "z_score": float(z_score),
            "theoretical_violation_prob": theoretical_prob,
            "empirical_violation_prob": empirical_prob,
            "theoretical_sla_reliability_percent": (1.0 - theoretical_prob) * 100.0,
            "empirical_sla_reliability_percent": (1.0 - empirical_prob) * 100.0,
        }

    def recommend_provisioning_capacity(
        self, target_reliability_percent: float = 99.9
    ) -> Dict[str, float]:
        """
        Computes the required network bandwidth capacity C such that:
        P(X <= C) >= target_reliability
        C = mu + Z * sigma
        """
        p = target_reliability_percent / 100.0
        z_multiplier = stats.norm.ppf(p)
        recommended_capacity = self.mean + (z_multiplier * self.std)

        # Baseline average capacity without statistical cushion
        cushion_gbps = recommended_capacity - self.mean

        return {
            "target_reliability_percent": target_reliability_percent,
            "z_multiplier": float(z_multiplier),
            "mean_load_gbps": self.mean,
            "std_gbps": self.std,
            "safety_buffer_gbps": float(cushion_gbps),
            "recommended_capacity_gbps": float(recommended_capacity),
        }
