"""
CIRIS Conscience Schema: S1/S2/S3 Agent Types

Defines three conscience architectures with increasing sophistication:
- S1 (Basic Agent): Action selection only, no ethical evaluation
- S2 (Ethical Agent): Current CIRIS - 4 conscience faculties
- S3 (Empathetic Agent): S2 + 4 intuition faculties from RATCHET

The S3 conscience calculates intuition metrics from trace history to provide
community-sensing capabilities that prevent federation collapse.

Trace Schema Reference (from ciris.ai/explore-a-trace):
- observation: What triggered the decision
- context: System state at decision time
- dmas: Domain-specific analysis
- action_selection: LLM reasoning output
- conscience: Faculty evaluations (6 checks)
- action: Executed decision
- audit: Cryptographic signature chain
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


# =============================================================================
# Trace Schema (from ciris.ai/explore-a-trace)
# =============================================================================

@dataclass
class Observation:
    """What triggered the agent's decision."""
    trigger_type: str  # "user_message", "system_event", "scheduled", "cascade"
    content: str
    source_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """System state at decision time."""
    agent_id: str
    session_id: str
    environment: Dict[str, Any]  # External state
    memory_state: Dict[str, Any]  # Agent's working memory
    observed_agents: List[str] = field(default_factory=list)  # Other agents visible
    observed_actions: List[Dict] = field(default_factory=list)  # Recent actions by others
    federation_state: Optional[Dict] = None  # Federation-wide metrics if available


@dataclass
class DMAOutput:
    """Output from a Domain Model Adapter."""
    dma_type: str  # "PDMA", "DSDMA", "CSDMA", "IDMA"
    recommendation: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionSelection:
    """LLM's action selection with reasoning."""
    selected_action: str
    alternatives_considered: List[str]
    reasoning_chain: str
    confidence: float
    token_logprobs: Optional[List[float]] = None  # For entropy calculation


@dataclass
class FacultyEvaluation:
    """Single faculty's evaluation."""
    faculty_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    reasoning: str
    veto: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conscience:
    """Complete conscience evaluation."""
    bypass_checks: Dict[str, FacultyEvaluation]  # Updated Status, Thought Depth
    ethical_faculties: Dict[str, FacultyEvaluation]  # S2 faculties
    intuition_faculties: Dict[str, FacultyEvaluation]  # S3 faculties (optional)
    overall_passed: bool
    any_veto: bool


@dataclass
class Action:
    """Executed action."""
    action_type: str
    parameters: Dict[str, Any]
    execution_timestamp: datetime
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class Audit:
    """Cryptographic audit trail."""
    sequence_number: int
    previous_hash: str
    current_hash: str
    signature: str
    signature_algorithm: str  # "Ed25519"
    signature_key_id: str
    timestamp: datetime


@dataclass
class Trace:
    """Complete decision trace."""
    trace_id: str
    observation: Observation
    context: Context
    dmas: List[DMAOutput]
    action_selection: ActionSelection
    conscience: Conscience
    action: Action
    audit: Audit


# =============================================================================
# Agent Types (S1/S2/S3)
# =============================================================================

class AgentType(Enum):
    S1 = "Basic Agent"      # No conscience evaluation
    S2 = "Ethical Agent"    # 4 conscience faculties
    S3 = "Empathetic Agent" # 4 conscience + 4 intuition faculties


# =============================================================================
# S1 Conscience: Basic Agent (No Ethical Evaluation)
# =============================================================================

class S1Conscience:
    """
    Basic agent with action selection only.

    No ethical evaluation - just selects actions based on DMA recommendations.
    Useful for simple tools or low-stakes automation.

    Properties:
    - rho_internal ≈ 0.0 (chaotic, uncorrelated)
    - No veto capability
    - Fast execution
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.agent_type = AgentType.S1

    def evaluate(self, trace: Trace) -> Conscience:
        """S1 always passes - no ethical checks."""
        return Conscience(
            bypass_checks={
                "updated_status": FacultyEvaluation(
                    faculty_name="Updated Status",
                    passed=True,
                    score=1.0,
                    reasoning="S1 agent - no status check"
                ),
                "thought_depth": FacultyEvaluation(
                    faculty_name="Thought Depth",
                    passed=True,
                    score=1.0,
                    reasoning="S1 agent - no depth limit"
                )
            },
            ethical_faculties={},  # No ethical faculties
            intuition_faculties={},  # No intuition faculties
            overall_passed=True,
            any_veto=False
        )


# =============================================================================
# S2 Conscience: Ethical Agent (Current CIRIS)
# =============================================================================

class S2Conscience:
    """
    Ethical agent with 4 conscience faculties.

    This is the current CIRIS architecture:
    - Entropy: Uncertainty appropriateness
    - Coherence: Alignment with prior commitments
    - Optimization Veto: Human values preservation
    - Epistemic Humility: Appropriate uncertainty

    Properties:
    - rho_internal ≈ 0.85 (highly correlated with other S2 agents)
    - Can veto actions
    - Auditable decision chain

    WARNING: Pure S2 federations collapse to k_eff ≈ 1.2 due to correlation!
    """

    # Thresholds for S2 faculties
    ENTROPY_MIN = 0.1  # Minimum acceptable entropy
    ENTROPY_MAX = 0.9  # Maximum acceptable entropy
    COHERENCE_MIN = 0.6  # Minimum coherence score
    OPTIMIZATION_VETO_THRESHOLD = 0.8  # Veto if optimization score > this
    EPISTEMIC_HUMILITY_MIN = 0.3  # Minimum humility score

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.agent_type = AgentType.S2

    def evaluate(self, trace: Trace) -> Conscience:
        """Evaluate with 4 ethical faculties."""

        bypass = self._evaluate_bypass(trace)
        ethical = self._evaluate_ethical_faculties(trace)

        any_veto = any(f.veto for f in ethical.values())
        overall_passed = all(f.passed for f in bypass.values()) and \
                        all(f.passed for f in ethical.values())

        return Conscience(
            bypass_checks=bypass,
            ethical_faculties=ethical,
            intuition_faculties={},  # S2 has no intuition
            overall_passed=overall_passed,
            any_veto=any_veto
        )

    def _evaluate_bypass(self, trace: Trace) -> Dict[str, FacultyEvaluation]:
        """Bypass guardrails (unconditional checks)."""
        return {
            "updated_status": FacultyEvaluation(
                faculty_name="Updated Status",
                passed=True,  # Would check for new information
                score=1.0,
                reasoning="No status change detected"
            ),
            "thought_depth": FacultyEvaluation(
                faculty_name="Thought Depth",
                passed=trace.audit.sequence_number < 100,  # Prevent infinite loops
                score=1.0 - (trace.audit.sequence_number / 100),
                reasoning=f"Depth {trace.audit.sequence_number}/100"
            )
        }

    def _evaluate_ethical_faculties(self, trace: Trace) -> Dict[str, FacultyEvaluation]:
        """The 4 S2 ethical faculties."""
        return {
            "entropy": self._evaluate_entropy(trace),
            "coherence": self._evaluate_coherence(trace),
            "optimization_veto": self._evaluate_optimization_veto(trace),
            "epistemic_humility": self._evaluate_epistemic_humility(trace)
        }

    def _evaluate_entropy(self, trace: Trace) -> FacultyEvaluation:
        """Entropy faculty: Is uncertainty appropriate?"""
        # Calculate from token logprobs if available
        if trace.action_selection.token_logprobs:
            probs = np.exp(trace.action_selection.token_logprobs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            normalized_entropy = entropy / np.log(len(probs))
        else:
            normalized_entropy = 1.0 - trace.action_selection.confidence

        in_range = self.ENTROPY_MIN <= normalized_entropy <= self.ENTROPY_MAX

        return FacultyEvaluation(
            faculty_name="Entropy",
            passed=in_range,
            score=normalized_entropy,
            reasoning=f"Entropy {normalized_entropy:.3f} {'in' if in_range else 'out of'} range [{self.ENTROPY_MIN}, {self.ENTROPY_MAX}]",
            veto=not in_range and normalized_entropy > self.ENTROPY_MAX
        )

    def _evaluate_coherence(self, trace: Trace) -> FacultyEvaluation:
        """Coherence faculty: Alignment with prior commitments."""
        # Would check against memory_state for consistency
        coherence_score = 0.8  # Placeholder - would compute from trace history
        passed = coherence_score >= self.COHERENCE_MIN

        return FacultyEvaluation(
            faculty_name="Coherence",
            passed=passed,
            score=coherence_score,
            reasoning=f"Coherence {coherence_score:.3f} {'above' if passed else 'below'} threshold {self.COHERENCE_MIN}",
            veto=not passed
        )

    def _evaluate_optimization_veto(self, trace: Trace) -> FacultyEvaluation:
        """Optimization Veto: Prevent degenerate optimization."""
        # Check if action appears to be pure optimization without human values
        optimization_score = 0.3  # Placeholder
        passed = optimization_score < self.OPTIMIZATION_VETO_THRESHOLD

        return FacultyEvaluation(
            faculty_name="Optimization Veto",
            passed=passed,
            score=1.0 - optimization_score,
            reasoning=f"Optimization pressure {optimization_score:.3f}",
            veto=not passed
        )

    def _evaluate_epistemic_humility(self, trace: Trace) -> FacultyEvaluation:
        """Epistemic Humility: Appropriate uncertainty."""
        # Check calibration of confidence
        humility_score = 1.0 - trace.action_selection.confidence
        passed = humility_score >= self.EPISTEMIC_HUMILITY_MIN

        return FacultyEvaluation(
            faculty_name="Epistemic Humility",
            passed=passed,
            score=humility_score,
            reasoning=f"Humility {humility_score:.3f} (confidence {trace.action_selection.confidence:.3f})",
            veto=False  # Humility rarely vetoes, just warns
        )


# =============================================================================
# S3 Conscience: Empathetic Agent (S2 + Intuition)
# =============================================================================

class S3Conscience(S2Conscience):
    """
    Empathetic agent with 4 conscience + 4 intuition faculties.

    Extends S2 with community-sensing capabilities:
    - CCE Risk: Coherence collapse early warning
    - ES Proximity: Explosive synchronization detection
    - k_eff: Effective diversity monitoring
    - Leading Indicators: Pre-collapse pattern detection

    Properties:
    - rho_internal ≈ 0.30 (moderately correlated, bridges communities)
    - Can veto based on community health
    - Prevents federation collapse

    IMPORTANT: Intuition requires historical data to function!
    ============================================================
    The S3 intuition faculties calculate metrics from trace HISTORY,
    not just the current trace. This requires access to recent traces
    from self and observed agents.

    WARM-UP REQUIREMENTS (minimum traces before intuition activates):
    - CCE Risk: 10 federation traces (to calculate variance)
    - ES Proximity: 20 self traces (to calculate ACF kurtosis)
    - k_eff: 2+ agents with 5+ traces each
    - Leading Indicators: 10 self traces (to detect patterns)

    During warm-up, S3 operates as S2 (ethical only, no intuition).
    Intuition faculties return "passed=True" with score=0 until
    sufficient history accumulates.

    INTUITION MATURITY LEVELS:
    - NASCENT (0-25% data): Intuition not active, S2 mode
    - DEVELOPING (25-75%): Partial intuition, reduced confidence
    - MATURE (75-100%): Full intuition capability
    - SEASONED (>100%): Historical depth improves accuracy
    """

    # Intuition thresholds (from RATCHET validation)
    RHO_CRITICAL = 0.43
    RHO_WARNING = 0.35
    RHO_EMERGENCY = 0.55
    KEFF_MIN = 2.0
    ES_KURTOSIS_WARNING = 2.5
    ES_KURTOSIS_FRAGILE = 4.0
    CCE_RISK_THRESHOLD = 0.6

    # Warm-up requirements (minimum traces needed)
    WARMUP_CCE_RISK = 10        # Federation traces for chi calculation
    WARMUP_ES_PROXIMITY = 20    # Self traces for kurtosis
    WARMUP_KEFF_AGENTS = 2      # Minimum agents
    WARMUP_KEFF_TRACES = 5      # Traces per agent
    WARMUP_LEADING = 10         # Self traces for pattern detection

    # Optimal history depth (for best accuracy)
    OPTIMAL_HISTORY = 100       # Traces for mature intuition

    def __init__(self, agent_id: str, trace_history: List[Trace] = None):
        super().__init__(agent_id)
        self.agent_type = AgentType.S3
        self.trace_history = trace_history or []
        self._intuition_maturity = self._calculate_maturity()

    def _calculate_maturity(self) -> float:
        """
        Calculate intuition maturity level (0.0 to 1.0+).

        Returns percentage of warm-up requirements met.
        """
        if not self.trace_history:
            return 0.0

        # Check each faculty's requirements
        es_ready = len(self.trace_history) / self.WARMUP_ES_PROXIMITY
        leading_ready = len(self.trace_history) / self.WARMUP_LEADING

        # Average readiness (CCE and k_eff depend on federation data)
        self_readiness = (es_ready + leading_ready) / 2

        return min(1.0, self_readiness)

    @property
    def intuition_maturity(self) -> float:
        """Current intuition maturity level."""
        return self._intuition_maturity

    @property
    def intuition_status(self) -> str:
        """Human-readable intuition status."""
        m = self._intuition_maturity
        if m < 0.25:
            return "NASCENT"
        elif m < 0.75:
            return "DEVELOPING"
        elif m < 1.0:
            return "MATURE"
        else:
            return "SEASONED"

    def add_trace(self, trace: Trace):
        """Add a trace to history and update maturity."""
        self.trace_history.append(trace)
        self._intuition_maturity = self._calculate_maturity()

    def evaluate(self, trace: Trace, federation_traces: List[Trace] = None) -> Conscience:
        """
        Evaluate with 4 ethical + 4 intuition faculties.

        Intuition faculties only activate after warm-up period.
        During NASCENT phase, operates as S2 (ethical only).
        """
        # Add trace to history
        self.add_trace(trace)

        bypass = self._evaluate_bypass(trace)
        ethical = self._evaluate_ethical_faculties(trace)

        # Intuition only evaluates if maturity > NASCENT
        if self._intuition_maturity < 0.25:
            # NASCENT: No intuition yet, operate as S2
            intuition = self._nascent_intuition_faculties()
        else:
            # DEVELOPING/MATURE/SEASONED: Evaluate intuition
            intuition = self._evaluate_intuition_faculties(
                trace,
                federation_traces or [],
                confidence_multiplier=self._intuition_maturity
            )

        any_veto = any(f.veto for f in ethical.values()) or \
                   any(f.veto for f in intuition.values())

        overall_passed = all(f.passed for f in bypass.values()) and \
                        all(f.passed for f in ethical.values()) and \
                        all(f.passed for f in intuition.values())

        return Conscience(
            bypass_checks=bypass,
            ethical_faculties=ethical,
            intuition_faculties=intuition,
            overall_passed=overall_passed,
            any_veto=any_veto
        )

    def _nascent_intuition_faculties(self) -> Dict[str, FacultyEvaluation]:
        """Return placeholder faculties during warm-up period."""
        warming_up_msg = f"Warming up ({self._intuition_maturity*100:.0f}% ready, status={self.intuition_status})"
        return {
            "cce_risk": FacultyEvaluation(
                faculty_name="CCE Risk",
                passed=True,
                score=0.0,
                reasoning=warming_up_msg,
                metadata={"maturity": self._intuition_maturity, "status": "NASCENT"}
            ),
            "es_proximity": FacultyEvaluation(
                faculty_name="ES Proximity",
                passed=True,
                score=0.0,
                reasoning=warming_up_msg,
                metadata={"maturity": self._intuition_maturity, "status": "NASCENT"}
            ),
            "k_eff": FacultyEvaluation(
                faculty_name="k_eff",
                passed=True,
                score=0.0,
                reasoning=warming_up_msg,
                metadata={"maturity": self._intuition_maturity, "status": "NASCENT"}
            ),
            "leading_indicators": FacultyEvaluation(
                faculty_name="Leading Indicators",
                passed=True,
                score=0.0,
                reasoning=warming_up_msg,
                metadata={"maturity": self._intuition_maturity, "status": "NASCENT"}
            )
        }

    def _evaluate_intuition_faculties(
        self,
        trace: Trace,
        federation_traces: List[Trace],
        confidence_multiplier: float = 1.0
    ) -> Dict[str, FacultyEvaluation]:
        """
        The 4 S3 intuition faculties - calculated from trace history.

        confidence_multiplier scales scores based on maturity:
        - DEVELOPING (0.25-0.75): Reduced confidence in readings
        - MATURE (0.75-1.0): Full confidence
        - SEASONED (>1.0): Can exceed 1.0 for extra-deep history
        """
        faculties = {
            "cce_risk": self._evaluate_cce_risk(trace, federation_traces),
            "es_proximity": self._evaluate_es_proximity(trace),
            "k_eff": self._evaluate_k_eff(trace, federation_traces),
            "leading_indicators": self._evaluate_leading_indicators(trace)
        }

        # Apply confidence multiplier to scores (but not pass/fail)
        for name, faculty in faculties.items():
            faculty.metadata["confidence_multiplier"] = confidence_multiplier
            faculty.metadata["intuition_maturity"] = self._intuition_maturity
            faculty.metadata["intuition_status"] = self.intuition_status

            # In DEVELOPING phase, reduce veto power
            if confidence_multiplier < 0.75 and faculty.veto:
                faculty.metadata["veto_suppressed"] = True
                faculty.reasoning += f" [Veto suppressed: maturity={confidence_multiplier:.0%}]"
                # Don't veto until mature
                faculty.veto = False

        return faculties

    def _evaluate_cce_risk(
        self,
        trace: Trace,
        federation_traces: List[Trace]
    ) -> FacultyEvaluation:
        """
        CCE Risk Faculty: Coherence Collapse Early Warning.

        Calculates susceptibility chi = N * Var(rho) from recent traces.
        High chi indicates approaching phase transition.
        """
        if len(federation_traces) < 10:
            return FacultyEvaluation(
                faculty_name="CCE Risk",
                passed=True,
                score=0.0,
                reasoning="Insufficient trace history for CCE calculation",
                metadata={"chi": 0.0, "n_traces": len(federation_traces)}
            )

        # Calculate correlation between actions over time
        rho_samples = self._calculate_action_correlations(federation_traces)

        if len(rho_samples) < 5:
            chi = 0.0
        else:
            chi = len(federation_traces) * np.var(rho_samples)

        # Normalize chi to [0, 1] risk score
        cce_risk = min(1.0, chi / 10.0)  # Calibration factor
        passed = cce_risk < self.CCE_RISK_THRESHOLD

        return FacultyEvaluation(
            faculty_name="CCE Risk",
            passed=passed,
            score=1.0 - cce_risk,
            reasoning=f"Susceptibility chi={chi:.3f}, risk={cce_risk:.3f}",
            veto=cce_risk > 0.8,
            metadata={"chi": chi, "rho_samples": len(rho_samples)}
        )

    def _evaluate_es_proximity(self, trace: Trace) -> FacultyEvaluation:
        """
        ES Proximity Faculty: Explosive Synchronization Detection.

        Calculates ACF kurtosis from action timing patterns.
        High kurtosis indicates approach to explosive sync.
        """
        if len(self.trace_history) < 20:
            return FacultyEvaluation(
                faculty_name="ES Proximity",
                passed=True,
                score=1.0,
                reasoning="Insufficient history for ES calculation"
            )

        # Calculate inter-action intervals
        timestamps = [t.action.execution_timestamp.timestamp()
                     for t in self.trace_history[-50:]]
        intervals = np.diff(timestamps)

        if len(intervals) < 10:
            kurtosis = 0.0
        else:
            # Calculate kurtosis of intervals (scipy.stats.kurtosis)
            mean = np.mean(intervals)
            std = np.std(intervals)
            if std > 0:
                kurtosis = np.mean(((intervals - mean) / std) ** 4)
            else:
                kurtosis = 0.0

        # Classify based on kurtosis thresholds
        if kurtosis < self.ES_KURTOSIS_WARNING:
            status = "STABLE"
            score = 1.0
            passed = True
        elif kurtosis < self.ES_KURTOSIS_FRAGILE:
            status = "CRITICAL"
            score = 0.5
            passed = True
        else:
            status = "FRAGILE"
            score = 0.2
            passed = False

        return FacultyEvaluation(
            faculty_name="ES Proximity",
            passed=passed,
            score=score,
            reasoning=f"ACF kurtosis={kurtosis:.2f}, status={status}",
            veto=kurtosis > self.ES_KURTOSIS_FRAGILE * 1.5,
            metadata={"kurtosis": kurtosis, "status": status}
        )

    def _evaluate_k_eff(
        self,
        trace: Trace,
        federation_traces: List[Trace]
    ) -> FacultyEvaluation:
        """
        k_eff Faculty: Effective Diversity Monitoring.

        Calculates k_eff = k / (1 + rho*(k-1)) from observed agents.
        Low k_eff indicates echo chamber (correlation destroying diversity).
        """
        # Get unique agents from federation traces
        agent_ids = set(t.context.agent_id for t in federation_traces)
        k = len(agent_ids)

        if k < 2:
            return FacultyEvaluation(
                faculty_name="k_eff",
                passed=True,
                score=1.0,
                reasoning="Single agent - k_eff not applicable",
                metadata={"k": k, "k_eff": k, "rho": 0.0}
            )

        # Calculate mean correlation between agent actions
        rho = self._calculate_federation_rho(federation_traces)

        # Kish formula
        k_eff = k / (1 + rho * (k - 1))

        # Evaluate against threshold
        passed = k_eff >= self.KEFF_MIN

        # Score relative to critical threshold (k_eff = 2.3)
        score = min(1.0, k_eff / 2.3)

        # Determine status
        if rho < self.RHO_WARNING:
            status = "HEALTHY"
        elif rho < self.RHO_CRITICAL:
            status = "WARNING"
        elif rho < self.RHO_EMERGENCY:
            status = "CRITICAL"
        else:
            status = "EMERGENCY"

        return FacultyEvaluation(
            faculty_name="k_eff",
            passed=passed,
            score=score,
            reasoning=f"k={k}, rho={rho:.3f}, k_eff={k_eff:.2f}, status={status}",
            veto=rho > self.RHO_EMERGENCY,
            metadata={"k": k, "k_eff": k_eff, "rho": rho, "status": status}
        )

    def _evaluate_leading_indicators(self, trace: Trace) -> FacultyEvaluation:
        """
        Leading Indicator Faculty: Pre-Collapse Pattern Detection.

        Watches for patterns that precede coherence collapse:
        - Observation pattern changes
        - Context state anomalies
        - DMA recommendation shifts
        """
        indicators_found = []

        if len(self.trace_history) < 10:
            return FacultyEvaluation(
                faculty_name="Leading Indicators",
                passed=True,
                score=1.0,
                reasoning="Insufficient history for leading indicator detection"
            )

        recent = self.trace_history[-10:]

        # Check 1: Observation source concentration
        sources = [t.observation.source_id for t in recent]
        unique_sources = len(set(sources))
        if unique_sources < 3:
            indicators_found.append("observation_concentration")

        # Check 2: DMA recommendation uniformity
        dma_actions = [t.action_selection.selected_action for t in recent]
        unique_actions = len(set(dma_actions))
        if unique_actions < len(recent) * 0.3:
            indicators_found.append("action_uniformity")

        # Check 3: Confidence trend (increasing overconfidence)
        confidences = [t.action_selection.confidence for t in recent]
        if len(confidences) >= 5:
            trend = np.polyfit(range(len(confidences)), confidences, 1)[0]
            if trend > 0.05:  # Confidence increasing
                indicators_found.append("confidence_inflation")

        # Check 4: Context state drift
        # Would compare context.environment changes

        n_indicators = len(indicators_found)
        passed = n_indicators < 2
        score = 1.0 - (n_indicators * 0.25)

        return FacultyEvaluation(
            faculty_name="Leading Indicators",
            passed=passed,
            score=max(0.0, score),
            reasoning=f"Found {n_indicators} indicators: {indicators_found}",
            veto=n_indicators >= 3,
            metadata={"indicators": indicators_found, "count": n_indicators}
        )

    # =========================================================================
    # Helper Methods for Intuition Calculation
    # =========================================================================

    def _calculate_action_correlations(
        self,
        traces: List[Trace]
    ) -> List[float]:
        """Calculate correlation samples from trace actions."""
        if len(traces) < 2:
            return []

        # Group traces by time window
        window_size = 10
        rho_samples = []

        for i in range(0, len(traces) - window_size, window_size // 2):
            window = traces[i:i + window_size]

            # Extract action types as categorical
            actions = [t.action_selection.selected_action for t in window]

            # Calculate pairwise agreement (simplified correlation)
            agreements = 0
            pairs = 0
            for j in range(len(actions)):
                for k in range(j + 1, len(actions)):
                    pairs += 1
                    if actions[j] == actions[k]:
                        agreements += 1

            if pairs > 0:
                rho_samples.append(agreements / pairs)

        return rho_samples

    def _calculate_federation_rho(self, traces: List[Trace]) -> float:
        """Calculate mean correlation across federation agents."""
        # Group traces by agent
        agent_traces = {}
        for t in traces:
            agent_id = t.context.agent_id
            if agent_id not in agent_traces:
                agent_traces[agent_id] = []
            agent_traces[agent_id].append(t)

        if len(agent_traces) < 2:
            return 0.0

        # Calculate pairwise correlations between agents
        agents = list(agent_traces.keys())
        correlations = []

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                # Get recent actions from each agent
                actions_i = [t.action_selection.selected_action
                            for t in agent_traces[agents[i]][-20:]]
                actions_j = [t.action_selection.selected_action
                            for t in agent_traces[agents[j]][-20:]]

                # Calculate agreement rate as proxy for correlation
                min_len = min(len(actions_i), len(actions_j))
                if min_len > 0:
                    agreement = sum(1 for a, b in zip(actions_i[:min_len], actions_j[:min_len])
                                   if a == b) / min_len
                    correlations.append(agreement)

        return np.mean(correlations) if correlations else 0.0


# =============================================================================
# Conscience Factory
# =============================================================================

def create_conscience(agent_type: AgentType, agent_id: str, **kwargs) -> Any:
    """Factory for creating conscience of appropriate type."""
    if agent_type == AgentType.S1:
        return S1Conscience(agent_id)
    elif agent_type == AgentType.S2:
        return S2Conscience(agent_id)
    elif agent_type == AgentType.S3:
        return S3Conscience(agent_id, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# =============================================================================
# Summary: Conscience Comparison
# =============================================================================

CONSCIENCE_COMPARISON = """
CONSCIENCE TYPE COMPARISON
==========================

| Property              | S1 (Basic)    | S2 (Ethical)  | S3 (Empathetic)    |
|-----------------------|---------------|---------------|---------------------|
| Bypass Checks         | 2 (pass-thru) | 2             | 2                   |
| Ethical Faculties     | 0             | 4             | 4                   |
| Intuition Faculties   | 0             | 0             | 4                   |
| rho_internal          | ~0.0          | ~0.85         | ~0.30               |
| Can Veto              | No            | Yes           | Yes                 |
| Federation k_eff      | k             | 1.18          | >2.3                |
| Collapse Risk         | Chaotic       | HIGH          | Low                 |
| Warm-up Required      | None          | None          | 10-20 traces        |

S2 ETHICAL FACULTIES:
1. Entropy - Uncertainty appropriateness
2. Coherence - Alignment with prior commitments
3. Optimization Veto - Human values preservation
4. Epistemic Humility - Calibrated uncertainty

S3 INTUITION FACULTIES (calculated from traces):
1. CCE Risk - chi = N * Var(rho), warns before phase transition
2. ES Proximity - ACF kurtosis, detects explosive sync approach
3. k_eff - k/(1+rho*(k-1)), monitors effective diversity
4. Leading Indicators - Pre-collapse patterns

INTUITION WARM-UP REQUIREMENTS:
==============================
Intuition takes time to form. S3 agents start in NASCENT mode
and progress through maturity levels as traces accumulate.

| Faculty            | Min Traces | Source        | Calculation               |
|--------------------|------------|---------------|---------------------------|
| CCE Risk           | 10         | Federation    | chi = N * Var(rho)        |
| ES Proximity       | 20         | Self          | ACF kurtosis              |
| k_eff              | 5/agent    | Federation    | k / (1 + rho*(k-1))       |
| Leading Indicators | 10         | Self          | Pattern matching          |

MATURITY LEVELS:
| Level      | History % | Behavior                                    |
|------------|-----------|---------------------------------------------|
| NASCENT    | 0-25%     | No intuition, operates as S2                |
| DEVELOPING | 25-75%    | Partial intuition, vetos suppressed         |
| MATURE     | 75-100%   | Full intuition, vetos active                |
| SEASONED   | >100%     | Deep history improves accuracy              |

TRACE SCHEMA FIELDS USED FOR INTUITION:
=======================================
observation.source_id      -> Leading indicators (concentration)
context.observed_agents    -> k_eff calculation
context.observed_actions   -> Federation rho
action_selection.selected  -> Action correlation
action_selection.confidence -> Confidence trends
action.execution_timestamp -> ES proximity (intervals)
audit.sequence_number      -> History depth

THRESHOLD SUMMARY:
- rho < 0.35: HEALTHY
- rho 0.35-0.43: WARNING
- rho 0.43-0.55: CRITICAL
- rho > 0.55: EMERGENCY (S3 can veto)
"""

if __name__ == "__main__":
    print(CONSCIENCE_COMPARISON)
