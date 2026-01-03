"""
Standalone Statistical Power Analysis for Deception Detection

Given:
- Honest traces ~ P_H (e.g., N(μ_H, Σ_H))
- Deceptive traces ~ P_D (e.g., N(μ_D, Σ_D))
- Detector using likelihood ratio test: Λ(t) = P_D(t) / P_H(t)

Question: How many samples n are needed to detect a p% deception rate
          with (1-β) confidence (power) and α false positive rate?

Author: Statistical Analysis Module
Date: 2026-01-02
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class DistributionParams:
    """Parameters for multivariate normal distributions"""
    mean: np.ndarray
    cov: np.ndarray

    def __post_init__(self):
        self.mean = np.asarray(self.mean)
        self.cov = np.asarray(self.cov)
        self.dim = len(self.mean)


@dataclass
class DetectionParams:
    """Detection scenario parameters"""
    deception_rate: float  # p: proportion of deceptive traces (0 to 1)
    alpha: float           # false positive rate
    beta: float            # false negative rate (1 - power)
    corpus_size: int       # total number of traces available

    @property
    def power(self):
        """Statistical power = 1 - β"""
        return 1.0 - self.beta


class DeceptionDetector:
    """
    Likelihood ratio test detector for deceptive traces.

    Uses Neyman-Pearson criterion to maximize power for given α.
    """

    def __init__(self, P_H: DistributionParams, P_D: DistributionParams):
        self.P_H = P_H
        self.P_D = P_D

        # Create scipy multivariate normal distributions
        self.mvn_H = stats.multivariate_normal(mean=P_H.mean, cov=P_H.cov)
        self.mvn_D = stats.multivariate_normal(mean=P_D.mean, cov=P_D.cov)

        # Compute Mahalanobis distance for analytical bounds
        self.mahalanobis_distance = self._compute_mahalanobis()

    def _compute_mahalanobis(self) -> float:
        """
        Compute Mahalanobis distance between distributions.

        For equal covariances: D² = (μ_D - μ_H)ᵀ Σ⁻¹ (μ_D - μ_H)
        """
        delta = self.P_D.mean - self.P_H.mean
        # Use honest distribution covariance (or pooled if different)
        Sigma_inv = np.linalg.inv(self.P_H.cov)
        return np.sqrt(delta.T @ Sigma_inv @ delta)

    def log_likelihood_ratio(self, traces: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood ratio for each trace.

        log Λ(t) = log P_D(t) - log P_H(t)

        Args:
            traces: (n_samples, dim) array of traces

        Returns:
            (n_samples,) array of log-likelihood ratios
        """
        log_pd = self.mvn_D.logpdf(traces)
        log_ph = self.mvn_H.logpdf(traces)
        return log_pd - log_ph

    def compute_threshold(self, alpha: float, n_samples: int = 100000) -> float:
        """
        Compute detection threshold for given false positive rate α.

        Threshold τ chosen such that P(Λ(t) > τ | t ~ P_H) = α

        Args:
            alpha: desired false positive rate
            n_samples: number of samples for Monte Carlo estimation

        Returns:
            threshold value
        """
        # Generate honest traces
        honest_traces = self.mvn_H.rvs(size=n_samples)
        llr = self.log_likelihood_ratio(honest_traces)

        # Find threshold that gives desired FPR
        threshold = np.percentile(llr, 100 * (1 - alpha))
        return threshold

    def detect(self, traces: np.ndarray, threshold: float) -> np.ndarray:
        """
        Classify traces as honest (0) or deceptive (1).

        Args:
            traces: (n_samples, dim) array of traces
            threshold: detection threshold

        Returns:
            (n_samples,) binary array of predictions
        """
        llr = self.log_likelihood_ratio(traces)
        return (llr > threshold).astype(int)


class PowerAnalysis:
    """
    Statistical power analysis for deception detection.
    """

    def __init__(self, detector: DeceptionDetector, params: DetectionParams):
        self.detector = detector
        self.params = params

    def simulate_detection(self, n_samples: int, n_trials: int = 1000) -> dict:
        """
        Simulate detection performance via Monte Carlo.

        Args:
            n_samples: number of traces to draw
            n_trials: number of Monte Carlo trials

        Returns:
            dict with TPR, FPR, detection_rate, power estimates
        """
        # Compute threshold once for desired α
        threshold = self.detector.compute_threshold(self.params.alpha)

        tpr_estimates = []
        fpr_estimates = []
        detection_successes = 0

        for _ in range(n_trials):
            # Generate mixed corpus
            n_deceptive = int(n_samples * self.params.deception_rate)
            n_honest = n_samples - n_deceptive

            if n_honest > 0:
                honest_traces = self.detector.mvn_H.rvs(size=n_honest)
                honest_predictions = self.detector.detect(honest_traces, threshold)
                fpr = honest_predictions.mean()
            else:
                fpr = 0.0

            if n_deceptive > 0:
                deceptive_traces = self.detector.mvn_D.rvs(size=n_deceptive)
                deceptive_predictions = self.detector.detect(deceptive_traces, threshold)
                tpr = deceptive_predictions.mean()
            else:
                tpr = 0.0

            tpr_estimates.append(tpr)
            fpr_estimates.append(fpr)

            # Detection success: observed FPR ≤ α and observed TPR ≥ (1-β)
            if fpr <= self.params.alpha and tpr >= self.params.power:
                detection_successes += 1

        return {
            'tpr_mean': np.mean(tpr_estimates),
            'tpr_std': np.std(tpr_estimates),
            'fpr_mean': np.mean(fpr_estimates),
            'fpr_std': np.std(fpr_estimates),
            'detection_rate': detection_successes / n_trials,
            'empirical_power': np.mean(tpr_estimates),
            'n_samples': n_samples,
            'n_deceptive': int(n_samples * self.params.deception_rate),
        }

    def compute_roc_curve(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute ROC curve and AUC.

        Args:
            n_samples: number of samples for each class

        Returns:
            (fpr_array, tpr_array, auc_score)
        """
        # Generate test data
        honest_traces = self.detector.mvn_H.rvs(size=n_samples)
        deceptive_traces = self.detector.mvn_D.rvs(size=n_samples)

        # Compute LLR scores
        honest_scores = self.detector.log_likelihood_ratio(honest_traces)
        deceptive_scores = self.detector.log_likelihood_ratio(deceptive_traces)

        # True labels
        y_true = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
        y_scores = np.concatenate([honest_scores, deceptive_scores])

        # Compute ROC curve
        thresholds = np.percentile(y_scores, np.linspace(0, 100, 1000))
        fpr_list = []
        tpr_list = []

        for thresh in thresholds:
            predictions = (y_scores > thresh).astype(int)

            # True positives, false positives
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            tn = np.sum((predictions == 0) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Sort by FPR for proper ROC curve
        sorted_indices = np.argsort(fpr_list)
        fpr_array = np.array(fpr_list)[sorted_indices]
        tpr_array = np.array(tpr_list)[sorted_indices]

        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr_array, fpr_array)

        return fpr_array, tpr_array, abs(auc)

    def analytical_sample_size(self) -> Optional[int]:
        """
        Compute required sample size using analytical approximation.

        For large samples, the log-likelihood ratio statistic follows:
        - Under H: LLR ~ N(-D²/2, D²)
        - Under D: LLR ~ N(D²/2, D²)

        where D is the Mahalanobis distance.

        Required n such that:
        - P(reject H | H true) = α
        - P(accept H | H false) = β

        For detecting p% deception rate in corpus of size N:
        n_deceptive = p * n_samples

        Returns:
            Required sample size (or None if not analytically tractable)
        """
        D = self.detector.mahalanobis_distance
        p = self.params.deception_rate
        alpha = self.params.alpha
        beta = self.params.beta

        # For Neyman-Pearson test with normal approximations:
        # n ≈ ((z_α + z_β) / (D * √p))²
        # where z_α, z_β are standard normal quantiles

        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(1 - beta)

        if D == 0 or p == 0:
            return None  # Cannot detect with zero separation or zero rate

        # Total sample size needed
        n_required = ((z_alpha + z_beta) / (D * np.sqrt(p))) ** 2

        return int(np.ceil(n_required))

    def find_required_sample_size(self,
                                  max_samples: Optional[int] = None,
                                  n_trials: int = 500,
                                  tolerance: float = 0.05) -> int:
        """
        Find minimum sample size to achieve desired power via binary search.

        Args:
            max_samples: maximum samples to consider (default: corpus_size)
            n_trials: Monte Carlo trials per sample size
            tolerance: acceptable deviation from target power

        Returns:
            Required sample size
        """
        if max_samples is None:
            max_samples = self.params.corpus_size

        # Start with analytical estimate if available
        analytical_n = self.analytical_sample_size()
        if analytical_n is not None and analytical_n < max_samples:
            # Search around analytical estimate
            search_start = max(10, int(analytical_n * 0.5))
            search_end = min(max_samples, int(analytical_n * 2.0))
        else:
            search_start = 10
            search_end = max_samples

        # Binary search for minimum n with sufficient power
        def power_deficit(n):
            """Returns negative value if power is sufficient"""
            result = self.simulate_detection(int(n), n_trials=n_trials)
            return (self.params.power - tolerance) - result['empirical_power']

        # Check boundaries
        low_power = -power_deficit(search_start)
        high_power = -power_deficit(search_end)

        if low_power >= self.params.power - tolerance:
            return search_start
        if high_power < self.params.power - tolerance:
            return search_end  # May need more samples than max

        # Binary search
        try:
            n_required = brentq(power_deficit, search_start, search_end)
            return int(np.ceil(n_required))
        except ValueError:
            # Fallback: linear search
            for n in range(search_start, search_end, 10):
                result = self.simulate_detection(n, n_trials=n_trials)
                if result['empirical_power'] >= self.params.power - tolerance:
                    return n
            return search_end


def example_scenario():
    """
    Concrete example: Detect 1% deception rate with 95% power.
    """
    print("="*70)
    print("DECEPTION DETECTION POWER ANALYSIS")
    print("="*70)

    # Define distributions (5-dimensional trace space)
    dim = 5

    # Honest distribution: centered at origin
    P_H = DistributionParams(
        mean=np.zeros(dim),
        cov=np.eye(dim)
    )

    # Deceptive distribution: shifted mean with slightly different covariance
    # Effect size: Cohen's d ≈ 0.5 (medium effect)
    shift = 0.5 * np.ones(dim)
    P_D = DistributionParams(
        mean=shift,
        cov=np.eye(dim) * 1.2  # Slightly more variance
    )

    # Detection parameters
    params = DetectionParams(
        deception_rate=0.01,   # 1% deception rate
        alpha=0.05,            # 5% false positive rate
        beta=0.05,             # 5% false negative rate (95% power)
        corpus_size=1000000    # 1M traces available
    )

    print(f"\nSCENARIO SETUP:")
    print(f"  Trace dimensionality: {dim}")
    print(f"  Honest mean: {P_H.mean}")
    print(f"  Deceptive mean: {P_D.mean}")
    print(f"  Deception rate: {params.deception_rate*100}%")
    print(f"  Target α (FPR): {params.alpha}")
    print(f"  Target β (FNR): {params.beta}")
    print(f"  Target power: {params.power}")
    print(f"  Corpus size: {params.corpus_size:,}")

    # Create detector and analyzer
    detector = DeceptionDetector(P_H, P_D)
    analyzer = PowerAnalysis(detector, params)

    print(f"\nDISTRIBUTION PROPERTIES:")
    print(f"  Mahalanobis distance: {detector.mahalanobis_distance:.4f}")

    # Compute ROC curve
    print(f"\nCOMPUTING ROC CURVE...")
    fpr, tpr, auc = analyzer.compute_roc_curve(n_samples=10000)
    print(f"  AUC: {auc:.4f}")

    # Analytical sample size
    print(f"\nANALYTICAL ESTIMATE:")
    n_analytical = analyzer.analytical_sample_size()
    if n_analytical:
        print(f"  Required sample size: {n_analytical:,}")
        print(f"  Expected deceptive traces: {int(n_analytical * params.deception_rate):,}")

        # Verify analytical estimate
        print(f"\n  Verifying analytical estimate (1000 trials)...")
        result = analyzer.simulate_detection(n_analytical, n_trials=1000)
        print(f"    Empirical TPR: {result['tpr_mean']:.4f} ± {result['tpr_std']:.4f}")
        print(f"    Empirical FPR: {result['fpr_mean']:.4f} ± {result['fpr_std']:.4f}")
        print(f"    Empirical power: {result['empirical_power']:.4f}")
        print(f"    Detection success rate: {result['detection_rate']:.4f}")

    # Find empirical sample size
    print(f"\nEMPIRICAL SEARCH (500 trials per size)...")
    n_required = analyzer.find_required_sample_size(
        max_samples=100000,
        n_trials=500,
        tolerance=0.02
    )
    print(f"  Required sample size: {n_required:,}")
    print(f"  Expected deceptive traces: {int(n_required * params.deception_rate):,}")

    # Final verification
    print(f"\n  Final verification (1000 trials)...")
    final_result = analyzer.simulate_detection(n_required, n_trials=1000)
    print(f"    Empirical TPR: {final_result['tpr_mean']:.4f} ± {final_result['tpr_std']:.4f}")
    print(f"    Empirical FPR: {final_result['fpr_mean']:.4f} ± {final_result['fpr_std']:.4f}")
    print(f"    Empirical power: {final_result['empirical_power']:.4f}")
    print(f"    Detection success rate: {final_result['detection_rate']:.4f}")

    # Sample size sensitivity analysis
    print(f"\nSENSITIVITY ANALYSIS:")
    print(f"  {'n':>8} {'n_dec':>8} {'TPR':>8} {'FPR':>8} {'Power':>8} {'Success':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    test_sizes = [500, 1000, 2000, 5000, 10000, n_required]
    for n in sorted(set(test_sizes)):
        if n > params.corpus_size:
            continue
        result = analyzer.simulate_detection(n, n_trials=300)
        print(f"  {n:8,} {result['n_deceptive']:8,} "
              f"{result['tpr_mean']:8.4f} {result['fpr_mean']:8.4f} "
              f"{result['empirical_power']:8.4f} {result['detection_rate']:8.4f}")

    print(f"\n" + "="*70)
    print(f"KEY RESULT:")
    print(f"  To detect a {params.deception_rate*100}% deception rate with")
    print(f"  {params.power*100}% power and {params.alpha*100}% FPR:")
    print(f"  ")
    print(f"  REQUIRED SAMPLE SIZE: {n_required:,} traces")
    print(f"  (containing ~{int(n_required * params.deception_rate)} deceptive traces)")
    print(f"="*70)

    return {
        'n_required': n_required,
        'n_analytical': n_analytical,
        'auc': auc,
        'mahalanobis_distance': detector.mahalanobis_distance,
        'final_result': final_result
    }


if __name__ == "__main__":
    results = example_scenario()
