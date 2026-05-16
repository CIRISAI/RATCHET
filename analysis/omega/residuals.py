"""
RATCHET Omega Module - Residuals Computation

Defines OmegaObservation dataclass and functions to compute omega (w) residuals:
    w = sigma_observed - sigma_predicted

Sigma (sustainability) is obtained from domain engines:
    - MicrobiomeEngine: normalized Shannon diversity
    - BatteryEngine: State of Health (SOH)
    - InstitutionalEngine: political stability

Prediction methods range from simple baselines to learned models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Union, Callable, Any, Protocol,
    TYPE_CHECKING
)
from enum import Enum
from datetime import datetime

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DomainType(Enum):
    """Supported domain types for omega analysis."""
    MICROBIOME = "microbiome"
    BATTERY = "battery"
    INSTITUTIONAL = "institutional"
    GENERIC = "generic"


class PredictorType(Enum):
    """Types of sigma predictors."""
    BASELINE_MEAN = "baseline_mean"
    BASELINE_DECAY = "baseline_decay"
    LAGGED = "lagged"
    ARIMA = "arima"
    LEARNED = "learned"


class EngineProtocol(Protocol):
    """Protocol for engines that provide sigma measurements."""

    def get_sigma(self) -> float:
        """Get current sustainability value."""
        ...

    def get_k(self) -> float:
        """Get constraint count."""
        ...

    def get_k_eff(self) -> float:
        """Get effective constraint count."""
        ...


@dataclass
class OmegaObservation:
    """
    Single omega observation representing the residual between
    observed and predicted sigma.

    Attributes:
        omega: The residual value (sigma_obs - sigma_pred)
        sigma_observed: Measured sustainability value
        sigma_predicted: Predicted sustainability value
        timestamp: Time of observation
        domain: Domain type (microbiome, battery, institutional)
        context: Additional contextual information
        k: Constraint count at time of observation
        k_eff: Effective constraint count
        predictor_type: Type of predictor used
        confidence: Confidence in the prediction (0-1)
    """
    omega: float
    sigma_observed: float
    sigma_predicted: float
    timestamp: float = 0.0
    domain: DomainType = DomainType.GENERIC
    context: Dict[str, Any] = field(default_factory=dict)
    k: Optional[float] = None
    k_eff: Optional[float] = None
    predictor_type: PredictorType = PredictorType.BASELINE_MEAN
    confidence: float = 1.0

    def __post_init__(self):
        """Validate omega computation."""
        expected = self.sigma_observed - self.sigma_predicted
        if not np.isclose(self.omega, expected, rtol=1e-6):
            # Auto-correct if not set correctly
            self.omega = expected

    @property
    def relative_omega(self) -> float:
        """Omega as fraction of observed sigma (handles sigma=0)."""
        if abs(self.sigma_observed) < 1e-10:
            return 0.0 if abs(self.omega) < 1e-10 else np.sign(self.omega) * np.inf
        return self.omega / self.sigma_observed

    @property
    def prediction_error_pct(self) -> float:
        """Absolute prediction error as percentage."""
        if abs(self.sigma_observed) < 1e-10:
            return 100.0 if abs(self.omega) > 1e-10 else 0.0
        return abs(self.omega / self.sigma_observed) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'omega': self.omega,
            'sigma_observed': self.sigma_observed,
            'sigma_predicted': self.sigma_predicted,
            'timestamp': self.timestamp,
            'domain': self.domain.value,
            'k': self.k,
            'k_eff': self.k_eff,
            'predictor_type': self.predictor_type.value,
            'confidence': self.confidence,
            'relative_omega': self.relative_omega,
            'prediction_error_pct': self.prediction_error_pct,
            **self.context,
        }


@dataclass
class OmegaTimeSeries:
    """
    Time series of omega observations for a single domain.

    Attributes:
        observations: List of OmegaObservation objects
        domain: Domain type
        metadata: Additional metadata about the series
    """
    observations: List[OmegaObservation] = field(default_factory=list)
    domain: DomainType = DomainType.GENERIC
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> OmegaObservation:
        return self.observations[idx]

    def append(self, obs: OmegaObservation) -> None:
        """Add an observation to the series."""
        self.observations.append(obs)

    @property
    def omega_values(self) -> np.ndarray:
        """Get omega values as numpy array."""
        return np.array([obs.omega for obs in self.observations])

    @property
    def sigma_observed_values(self) -> np.ndarray:
        """Get observed sigma values as numpy array."""
        return np.array([obs.sigma_observed for obs in self.observations])

    @property
    def sigma_predicted_values(self) -> np.ndarray:
        """Get predicted sigma values as numpy array."""
        return np.array([obs.sigma_predicted for obs in self.observations])

    @property
    def timestamps(self) -> np.ndarray:
        """Get timestamps as numpy array."""
        return np.array([obs.timestamp for obs in self.observations])

    @property
    def mean_omega(self) -> float:
        """Mean of omega values."""
        if len(self) == 0:
            return 0.0
        return float(np.mean(self.omega_values))

    @property
    def std_omega(self) -> float:
        """Standard deviation of omega values."""
        if len(self) < 2:
            return 0.0
        return float(np.std(self.omega_values, ddof=1))

    @property
    def variance_omega(self) -> float:
        """Variance of omega values."""
        if len(self) < 2:
            return 0.0
        return float(np.var(self.omega_values, ddof=1))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if len(self) == 0:
            return pd.DataFrame()
        return pd.DataFrame([obs.to_dict() for obs in self.observations])

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        omega_col: str = 'omega',
        sigma_obs_col: str = 'sigma_observed',
        sigma_pred_col: str = 'sigma_predicted',
        time_col: str = 'timestamp',
        domain: DomainType = DomainType.GENERIC,
    ) -> 'OmegaTimeSeries':
        """Create OmegaTimeSeries from DataFrame."""
        series = cls(domain=domain)

        for _, row in df.iterrows():
            obs = OmegaObservation(
                omega=row.get(omega_col, row[sigma_obs_col] - row[sigma_pred_col]),
                sigma_observed=row[sigma_obs_col],
                sigma_predicted=row[sigma_pred_col],
                timestamp=row.get(time_col, 0.0),
                domain=domain,
            )
            series.append(obs)

        return series


# =============================================================================
# PREDICTOR FUNCTIONS
# =============================================================================

def sigma_predictor_baseline(
    sigma_history: np.ndarray,
    method: str = 'mean',
    decay_rate: float = 0.0,
    window: Optional[int] = None,
) -> float:
    """
    Baseline sigma predictor using simple statistics.

    Args:
        sigma_history: Array of historical sigma values
        method: 'mean', 'median', 'last', or 'decay'
        decay_rate: Decay rate for exponential decay prediction
        window: Window size for rolling statistics (None = all history)

    Returns:
        Predicted sigma value
    """
    if len(sigma_history) == 0:
        return 0.5  # Default neutral prediction

    if window is not None and len(sigma_history) > window:
        sigma_history = sigma_history[-window:]

    if method == 'mean':
        return float(np.mean(sigma_history))
    elif method == 'median':
        return float(np.median(sigma_history))
    elif method == 'last':
        return float(sigma_history[-1])
    elif method == 'decay':
        # Predict next sigma based on exponential decay from current
        current = sigma_history[-1]
        return current * np.exp(-decay_rate)
    else:
        raise ValueError(f"Unknown prediction method: {method}")


def sigma_predictor_lagged(
    sigma_history: np.ndarray,
    lag: int = 1,
) -> float:
    """
    Lagged predictor: predict sigma(t) = sigma(t-lag).

    Args:
        sigma_history: Array of historical sigma values
        lag: Number of time steps to lag

    Returns:
        Predicted sigma value
    """
    if len(sigma_history) < lag:
        return float(np.mean(sigma_history)) if len(sigma_history) > 0 else 0.5
    return float(sigma_history[-lag])


def sigma_predictor_exponential_smoothing(
    sigma_history: np.ndarray,
    alpha: float = 0.3,
) -> float:
    """
    Simple exponential smoothing predictor.

    Args:
        sigma_history: Array of historical sigma values
        alpha: Smoothing parameter (0 < alpha < 1)

    Returns:
        Predicted sigma value
    """
    if len(sigma_history) == 0:
        return 0.5

    # Initialize with first observation
    smoothed = sigma_history[0]

    # Apply exponential smoothing
    for sigma in sigma_history[1:]:
        smoothed = alpha * sigma + (1 - alpha) * smoothed

    return float(smoothed)


# =============================================================================
# OMEGA COMPUTATION FUNCTIONS
# =============================================================================

def compute_omega(
    sigma_observed: float,
    sigma_predicted: float,
    timestamp: float = 0.0,
    domain: DomainType = DomainType.GENERIC,
    predictor_type: PredictorType = PredictorType.BASELINE_MEAN,
    k: Optional[float] = None,
    k_eff: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
) -> OmegaObservation:
    """
    Compute a single omega observation.

    Args:
        sigma_observed: Measured sigma value
        sigma_predicted: Predicted sigma value
        timestamp: Time of observation
        domain: Domain type
        predictor_type: Type of predictor used
        k: Constraint count
        k_eff: Effective constraint count
        context: Additional context

    Returns:
        OmegaObservation object
    """
    omega = sigma_observed - sigma_predicted

    return OmegaObservation(
        omega=omega,
        sigma_observed=sigma_observed,
        sigma_predicted=sigma_predicted,
        timestamp=timestamp,
        domain=domain,
        predictor_type=predictor_type,
        k=k,
        k_eff=k_eff,
        context=context or {},
    )


def compute_omega_series(
    sigma_series: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    predictor: Union[str, Callable[[np.ndarray], float]] = 'mean',
    predictor_kwargs: Optional[Dict[str, Any]] = None,
    domain: DomainType = DomainType.GENERIC,
    k_series: Optional[np.ndarray] = None,
    k_eff_series: Optional[np.ndarray] = None,
    warmup: int = 1,
) -> OmegaTimeSeries:
    """
    Compute omega series from sigma time series.

    For each time t, predicts sigma(t) based on sigma(0:t-1) and computes
    omega(t) = sigma_observed(t) - sigma_predicted(t).

    Args:
        sigma_series: Array of observed sigma values
        timestamps: Array of timestamps (default: 0, 1, 2, ...)
        predictor: Prediction method ('mean', 'median', 'last', 'decay')
                   or custom callable
        predictor_kwargs: Additional kwargs for predictor
        domain: Domain type
        k_series: Array of k values (optional)
        k_eff_series: Array of k_eff values (optional)
        warmup: Number of initial observations to skip (used for training)

    Returns:
        OmegaTimeSeries object
    """
    n = len(sigma_series)

    if timestamps is None:
        timestamps = np.arange(n, dtype=float)

    if predictor_kwargs is None:
        predictor_kwargs = {}

    # Determine predictor function
    if isinstance(predictor, str):
        predictor_type = PredictorType.BASELINE_MEAN
        if predictor == 'mean':
            predictor_type = PredictorType.BASELINE_MEAN
            pred_func = lambda h: sigma_predictor_baseline(h, 'mean', **predictor_kwargs)
        elif predictor == 'median':
            predictor_type = PredictorType.BASELINE_MEAN
            pred_func = lambda h: sigma_predictor_baseline(h, 'median', **predictor_kwargs)
        elif predictor == 'last':
            predictor_type = PredictorType.LAGGED
            pred_func = lambda h: sigma_predictor_baseline(h, 'last', **predictor_kwargs)
        elif predictor == 'decay':
            predictor_type = PredictorType.BASELINE_DECAY
            pred_func = lambda h: sigma_predictor_baseline(h, 'decay', **predictor_kwargs)
        elif predictor == 'exp_smooth':
            predictor_type = PredictorType.LEARNED
            pred_func = lambda h: sigma_predictor_exponential_smoothing(h, **predictor_kwargs)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")
    else:
        predictor_type = PredictorType.LEARNED
        pred_func = predictor

    # Build omega series
    omega_series = OmegaTimeSeries(domain=domain)
    omega_series.metadata['predictor'] = predictor if isinstance(predictor, str) else 'custom'
    omega_series.metadata['warmup'] = warmup

    for t in range(warmup, n):
        # Get history up to (but not including) current time
        history = sigma_series[:t]

        # Predict sigma at time t
        sigma_pred = pred_func(history)
        sigma_obs = sigma_series[t]

        # Get k values if available
        k = k_series[t] if k_series is not None else None
        k_eff = k_eff_series[t] if k_eff_series is not None else None

        obs = compute_omega(
            sigma_observed=sigma_obs,
            sigma_predicted=sigma_pred,
            timestamp=timestamps[t],
            domain=domain,
            predictor_type=predictor_type,
            k=k,
            k_eff=k_eff,
        )
        omega_series.append(obs)

    return omega_series


def compute_omega_from_engine(
    engine: EngineProtocol,
    duration: float,
    dt: float = 1.0,
    predictor: str = 'mean',
    domain: Optional[DomainType] = None,
    warmup_fraction: float = 0.1,
) -> OmegaTimeSeries:
    """
    Run an engine simulation and compute omega series.

    Args:
        engine: Engine instance (must have run() and get_sigma() methods)
        duration: Simulation duration
        dt: Time step
        predictor: Prediction method
        domain: Domain type (auto-detected if None)
        warmup_fraction: Fraction of data to use for warmup

    Returns:
        OmegaTimeSeries object
    """
    # Auto-detect domain from engine type
    if domain is None:
        engine_type = type(engine).__name__.lower()
        if 'microbiome' in engine_type:
            domain = DomainType.MICROBIOME
        elif 'battery' in engine_type:
            domain = DomainType.BATTERY
        elif 'institutional' in engine_type:
            domain = DomainType.INSTITUTIONAL
        else:
            domain = DomainType.GENERIC

    # Run simulation
    df = engine.run(duration=duration, dt=dt)

    # Extract series
    sigma_series = df['sigma'].values
    timestamps = df['time'].values if 'time' in df.columns else np.arange(len(df)) * dt
    k_series = df['k'].values if 'k' in df.columns else None
    k_eff_series = df['k_eff'].values if 'k_eff' in df.columns else None

    # Compute warmup
    warmup = max(1, int(len(sigma_series) * warmup_fraction))

    return compute_omega_series(
        sigma_series=sigma_series,
        timestamps=timestamps,
        predictor=predictor,
        domain=domain,
        k_series=k_series,
        k_eff_series=k_eff_series,
        warmup=warmup,
    )


__all__ = [
    'DomainType',
    'PredictorType',
    'EngineProtocol',
    'OmegaObservation',
    'OmegaTimeSeries',
    'sigma_predictor_baseline',
    'sigma_predictor_lagged',
    'sigma_predictor_exponential_smoothing',
    'compute_omega',
    'compute_omega_series',
    'compute_omega_from_engine',
]
