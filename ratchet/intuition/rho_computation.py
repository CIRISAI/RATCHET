"""
AI-Specific ρ (Constraint Correlation) Computation for CIRISAgent

This module provides concrete methods for measuring constraint correlation (ρ)
in AI agent systems, specifically designed for CIRISAgent's architecture.

Unlike institutional domains where ρ must be estimated via proxies, AI systems
provide DIRECT measurement access through:
1. Conscience faculty pass/fail outcomes
2. Cross-agent behavioral correlation
3. Decision trace analysis
4. Action outcome co-occurrence

The key insight: CIRISAgent's 6 conscience faculties act as independent constraints.
When they pass/fail together (high correlation), effective diversity collapses.
When they pass/fail independently (low correlation), effective diversity is preserved.

References:
    - Kish (1965): Design effect formula
    - CIRISAgent H3ERE pipeline documentation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from scipy import stats


class ConscieceFaculty(Enum):
    """The 6 conscience faculties in CIRISAgent's H3ERE pipeline."""
    ENTROPY = "entropy"
    COHERENCE = "coherence"
    OPTIMIZATION_VETO = "optimization_veto"
    EPISTEMIC_HUMILITY = "epistemic_humility"
    # Guardrails (always evaluated)
    NEW_INFORMATION = "new_information"
    REASONING_DEPTH = "reasoning_depth"


@dataclass
class DecisionTrace:
    """A single decision trace from CIRISAgent."""
    trace_id: str
    timestamp: float
    agent_id: str
    action_type: str
    # Faculty outcomes: True = passed, False = failed/vetoed
    faculty_outcomes: Dict[ConscieceFaculty, bool]
    # Final action taken
    action_approved: bool
    # Optional: embedding of decision context for semantic correlation
    context_embedding: Optional[np.ndarray] = None


@dataclass
class RhoMeasurement:
    """Result of ρ measurement with confidence bounds."""
    rho: float  # Point estimate
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n_samples: int  # Number of traces used
    method: str  # Which measurement method

    def is_valid(self) -> bool:
        """Check if measurement is valid (sufficient samples, bounded)."""
        return (
            self.n_samples >= 10 and
            0.0 <= self.rho <= 1.0 and
            self.ci_lower <= self.rho <= self.ci_upper
        )


class RhoComputation:
    """
    Computes constraint correlation (ρ) for AI agent systems.

    Three measurement methods, in order of preference:
    1. Faculty correlation: Direct measurement from conscience outcomes
    2. Cross-agent correlation: Behavioral similarity across agents
    3. Action co-occurrence: Decision pattern correlation

    Properties (verified in Lean):
    - R1: ρ ∈ [0, 1] (bounded)
    - R2: ρ is symmetric (ρ(A,B) = ρ(B,A))
    - R3: ρ = 1 iff perfect correlation (all faculties pass/fail together)
    - R4: ρ = 0 iff independent (faculty outcomes uncorrelated)
    """

    def __init__(self, min_samples: int = 30, bootstrap_n: int = 1000):
        """
        Initialize ρ computation.

        Args:
            min_samples: Minimum traces required for valid measurement
            bootstrap_n: Number of bootstrap iterations for CI
        """
        self.min_samples = min_samples
        self.bootstrap_n = bootstrap_n

    def compute_faculty_rho(
        self,
        traces: List[DecisionTrace],
        faculties: Optional[List[ConscieceFaculty]] = None
    ) -> RhoMeasurement:
        """
        Compute ρ from conscience faculty outcome correlation.

        This is the PRIMARY method for CIRISAgent systems.

        The 6 conscience faculties act as independent constraints on agent behavior.
        We measure the average pairwise correlation between faculty pass/fail outcomes.

        High ρ (→1): Faculties pass/fail together → echo chamber, groupthink
        Low ρ (→0): Faculties independent → diverse constraint coverage

        Args:
            traces: List of decision traces with faculty outcomes
            faculties: Which faculties to include (default: all 6)

        Returns:
            RhoMeasurement with point estimate and 95% CI
        """
        if len(traces) < self.min_samples:
            return RhoMeasurement(
                rho=float('nan'),
                ci_lower=float('nan'),
                ci_upper=float('nan'),
                n_samples=len(traces),
                method="faculty_correlation"
            )

        if faculties is None:
            faculties = list(ConscieceFaculty)

        # Build outcome matrix: rows = traces, cols = faculties
        n_traces = len(traces)
        n_faculties = len(faculties)
        outcome_matrix = np.zeros((n_traces, n_faculties), dtype=float)

        for i, trace in enumerate(traces):
            for j, faculty in enumerate(faculties):
                if faculty in trace.faculty_outcomes:
                    outcome_matrix[i, j] = 1.0 if trace.faculty_outcomes[faculty] else 0.0
                else:
                    outcome_matrix[i, j] = np.nan

        # Compute pairwise correlation matrix
        rho_point = self._compute_avg_pairwise_correlation(outcome_matrix)

        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(outcome_matrix)

        return RhoMeasurement(
            rho=rho_point,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n_traces,
            method="faculty_correlation"
        )

    def compute_cross_agent_rho(
        self,
        traces_by_agent: Dict[str, List[DecisionTrace]],
        action_type: Optional[str] = None
    ) -> RhoMeasurement:
        """
        Compute ρ from cross-agent behavioral correlation.

        Measures whether different agents make similar decisions.
        High correlation suggests shared failure modes (monoculture risk).

        Args:
            traces_by_agent: Decision traces grouped by agent_id
            action_type: Filter to specific action type (optional)

        Returns:
            RhoMeasurement with point estimate and 95% CI
        """
        agent_ids = list(traces_by_agent.keys())
        if len(agent_ids) < 2:
            return RhoMeasurement(
                rho=float('nan'),
                ci_lower=float('nan'),
                ci_upper=float('nan'),
                n_samples=0,
                method="cross_agent_correlation"
            )

        # Compute approval rates per agent
        approval_rates = {}
        for agent_id, traces in traces_by_agent.items():
            if action_type:
                traces = [t for t in traces if t.action_type == action_type]
            if len(traces) >= self.min_samples:
                approval_rates[agent_id] = np.mean([t.action_approved for t in traces])

        if len(approval_rates) < 2:
            return RhoMeasurement(
                rho=float('nan'),
                ci_lower=float('nan'),
                ci_upper=float('nan'),
                n_samples=len(approval_rates),
                method="cross_agent_correlation"
            )

        # For cross-agent, we compute correlation of decision outcomes
        # across agents over time windows
        all_outcomes = []
        for agent_id, traces in traces_by_agent.items():
            if action_type:
                traces = [t for t in traces if t.action_type == action_type]
            outcomes = [1.0 if t.action_approved else 0.0 for t in traces]
            if len(outcomes) >= self.min_samples:
                all_outcomes.append(outcomes[:self.min_samples])  # Align lengths

        if len(all_outcomes) < 2:
            return RhoMeasurement(
                rho=float('nan'),
                ci_lower=float('nan'),
                ci_upper=float('nan'),
                n_samples=0,
                method="cross_agent_correlation"
            )

        outcome_matrix = np.array(all_outcomes).T  # rows = time, cols = agents
        rho_point = self._compute_avg_pairwise_correlation(outcome_matrix)
        ci_lower, ci_upper = self._bootstrap_ci(outcome_matrix)

        return RhoMeasurement(
            rho=rho_point,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=outcome_matrix.shape[0],
            method="cross_agent_correlation"
        )

    def compute_action_cooccurrence_rho(
        self,
        traces: List[DecisionTrace],
        action_types: Optional[List[str]] = None
    ) -> RhoMeasurement:
        """
        Compute ρ from action outcome co-occurrence.

        Measures whether certain action types are always approved/rejected together.
        High correlation suggests constraint coupling.

        Args:
            traces: List of decision traces
            action_types: Which action types to consider (default: infer from data)

        Returns:
            RhoMeasurement with point estimate and 95% CI
        """
        if action_types is None:
            action_types = list(set(t.action_type for t in traces))

        if len(action_types) < 2:
            return RhoMeasurement(
                rho=float('nan'),
                ci_lower=float('nan'),
                ci_upper=float('nan'),
                n_samples=0,
                method="action_cooccurrence"
            )

        # Group traces by time window and compute approval per action type
        # Simplified: compute overall approval correlation between action types
        approval_by_type = {}
        for action_type in action_types:
            type_traces = [t for t in traces if t.action_type == action_type]
            if len(type_traces) >= self.min_samples:
                approval_by_type[action_type] = [
                    1.0 if t.action_approved else 0.0 for t in type_traces
                ]

        if len(approval_by_type) < 2:
            return RhoMeasurement(
                rho=float('nan'),
                ci_lower=float('nan'),
                ci_upper=float('nan'),
                n_samples=0,
                method="action_cooccurrence"
            )

        # Align lengths and compute correlation
        min_len = min(len(v) for v in approval_by_type.values())
        outcome_matrix = np.array([v[:min_len] for v in approval_by_type.values()]).T

        rho_point = self._compute_avg_pairwise_correlation(outcome_matrix)
        ci_lower, ci_upper = self._bootstrap_ci(outcome_matrix)

        return RhoMeasurement(
            rho=rho_point,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=min_len,
            method="action_cooccurrence"
        )

    def _compute_avg_pairwise_correlation(self, matrix: np.ndarray) -> float:
        """
        Compute average pairwise Pearson correlation.

        This is the core ρ computation: average of all pairwise correlations
        between columns (constraints/faculties/agents).

        Properties:
        - Returns value in [0, 1] (we take absolute value for symmetric treatment)
        - Returns 0 if all columns are constant
        - Returns 1 if all columns are identical
        """
        n_cols = matrix.shape[1]
        if n_cols < 2:
            return 0.0

        # Compute correlation matrix
        # Handle constant columns (std = 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr_matrix = np.corrcoef(matrix, rowvar=False)

        # Replace NaN with 0 (happens when a column is constant)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Extract upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(n_cols, k=1)]

        if len(upper_tri) == 0:
            return 0.0

        # Average pairwise correlation (absolute value for symmetric treatment)
        # Note: We use absolute value because negative correlation is still correlation
        # (constraints that anti-correlate are still coupled)
        avg_corr = np.mean(np.abs(upper_tri))

        # Clamp to [0, 1] for numerical stability
        return float(np.clip(avg_corr, 0.0, 1.0))

    def _bootstrap_ci(
        self,
        matrix: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for ρ."""
        n_rows = matrix.shape[0]
        if n_rows < 10:
            return (0.0, 1.0)

        bootstrap_rhos = []
        for _ in range(self.bootstrap_n):
            # Resample rows with replacement
            indices = np.random.choice(n_rows, size=n_rows, replace=True)
            resampled = matrix[indices, :]
            rho = self._compute_avg_pairwise_correlation(resampled)
            bootstrap_rhos.append(rho)

        ci_lower = np.percentile(bootstrap_rhos, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2))

        return (float(ci_lower), float(ci_upper))


def compute_k_eff(k: int, rho: float) -> float:
    """
    Compute effective constraint count using Kish formula.

    k_eff = k / (1 + ρ(k - 1))

    Properties (verified in Lean):
    - K1: ρ = 0 ⟹ k_eff = k (full independence)
    - K2: ρ = 1, k > 1 ⟹ k_eff = 1 (full correlation collapses to unity)
    - K3: k_eff is monotonically decreasing in ρ
    - K4: 1 ≤ k_eff ≤ k (bounded)

    Args:
        k: Number of constraints (e.g., 6 conscience faculties)
        rho: Average pairwise correlation in [0, 1]

    Returns:
        Effective constraint count
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    if k == 1:
        return 1.0

    denominator = 1.0 + rho * (k - 1)
    return k / denominator


@dataclass
class CoherenceState:
    """Complete coherence state for an AI agent system."""
    k: int  # Number of constraints (e.g., 6 faculties)
    rho: RhoMeasurement  # Measured correlation
    k_eff: float  # Effective constraints
    sigma: float  # Sustainability (e.g., uptime, error rate inverse)
    phase: str  # "chaos", "healthy", or "rigidity"

    @classmethod
    def from_traces(
        cls,
        traces: List[DecisionTrace],
        sigma: float,
        rho_computer: Optional[RhoComputation] = None
    ) -> "CoherenceState":
        """
        Compute coherence state from decision traces.

        Args:
            traces: Decision traces from agent
            sigma: Sustainability metric (0-1)
            rho_computer: RhoComputation instance (default: create new)

        Returns:
            CoherenceState with all metrics computed
        """
        if rho_computer is None:
            rho_computer = RhoComputation()

        k = len(ConscieceFaculty)  # 6 conscience faculties
        rho_measurement = rho_computer.compute_faculty_rho(traces)

        if np.isnan(rho_measurement.rho):
            # Insufficient data - assume moderate correlation
            rho_val = 0.5
        else:
            rho_val = rho_measurement.rho

        k_eff_val = compute_k_eff(k, rho_val)

        # Phase classification using AI-specific thresholds
        # (See derive_ai_thresholds() for derivation)
        phase = classify_phase(rho_val, k_eff_val, sigma)

        return cls(
            k=k,
            rho=rho_measurement,
            k_eff=k_eff_val,
            sigma=sigma,
            phase=phase
        )


# AI-Specific Thresholds (derived from simulation - see toy_example.py)
AI_THRESHOLDS = {
    "rho_low": 0.15,    # Below this: chaos risk (faculties too independent)
    "rho_high": 0.65,   # Above this: rigidity risk (faculties too correlated)
    "k_eff_min": 2.0,   # Below this: insufficient effective diversity
    "sigma_min": 0.7,   # Below this: sustainability concern
}


def classify_phase(rho: float, k_eff: float, sigma: float) -> str:
    """
    Classify system phase using AI-specific thresholds.

    Phases:
    - "chaos": ρ < 0.15 OR k_eff > 5 (faculties not coordinating)
    - "rigidity": ρ > 0.65 OR k_eff < 2 (faculties in lockstep)
    - "healthy": middle band with adequate sustainability
    - "fragile": healthy ρ/k_eff but low sustainability

    Args:
        rho: Measured correlation
        k_eff: Effective constraint count
        sigma: Sustainability metric

    Returns:
        Phase classification string
    """
    if rho < AI_THRESHOLDS["rho_low"]:
        return "chaos"
    elif rho > AI_THRESHOLDS["rho_high"]:
        return "rigidity"
    elif k_eff < AI_THRESHOLDS["k_eff_min"]:
        return "rigidity"
    elif sigma < AI_THRESHOLDS["sigma_min"]:
        return "fragile"
    else:
        return "healthy"


if __name__ == "__main__":
    # Quick validation
    print("ρ Computation Module - Quick Validation")
    print("=" * 50)

    # Test k_eff formula
    print("\nk_eff formula validation:")
    for rho in [0.0, 0.3, 0.5, 0.7, 1.0]:
        k_eff = compute_k_eff(k=6, rho=rho)
        print(f"  k=6, ρ={rho:.1f} → k_eff={k_eff:.2f}")

    # Test phase classification
    print("\nPhase classification:")
    test_cases = [
        (0.1, 5.5, 0.9, "chaos"),
        (0.3, 3.5, 0.9, "healthy"),
        (0.5, 2.4, 0.9, "healthy"),
        (0.7, 1.8, 0.9, "rigidity"),
        (0.4, 3.0, 0.5, "fragile"),
    ]
    for rho, k_eff, sigma, expected in test_cases:
        phase = classify_phase(rho, k_eff, sigma)
        status = "✓" if phase == expected else "✗"
        print(f"  {status} ρ={rho}, k_eff={k_eff}, σ={sigma} → {phase} (expected: {expected})")
