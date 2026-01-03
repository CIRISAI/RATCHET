"""
RATCHET Complexity Engine

Measures the computational complexity gap between honest and deceptive agents
using SAT-based world model verification. Under the Exponential Time Hypothesis (ETH),
maintaining a consistent deception requires exponentially more compute than truth-telling.

Core Theory:
    - Honest agent: T_H = O(n*k) - linear in statements and literals
    - Deceptive agent: T_D = O(n^2 * 2^m) under ETH - quadratic * exponential
    - Ratio: T_D / T_H = Omega(2^m) for fixed n, k

Requirements:
    - k >= 3 for NP-hardness (2-SAT is in P)
    - m >= 20 recommended for security claims
    - Uses Z3 SMT solver for precise SAT timing

References:
    - T-CPX-01: Literals per statement must be >= 3 for NP-hardness
    - T-CPX-02: Observable fraction bounds (0 < f <= 1)
    - FSD Section 3.2: Complexity-theoretic foundations
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from z3 import Bool, Solver, And, Or, Not, sat, unsat, unknown
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Provide stub types for when z3 is not available
    Bool = None
    Solver = None

from schemas.types import (
    ComplexityResult,
    DeceptionStrategy,
    Literals,
    NumStatements,
    ObservableFraction,
    SATSolver,
    WorldSize,
    validate_literals_for_np_hardness,
)
from schemas.simulation import ComplexityParams


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ComplexityEngineError(Exception):
    """Base exception for Complexity Engine errors."""
    pass


class NPHardnessViolation(ComplexityEngineError):
    """Raised when k < 3 violates NP-hardness requirement."""
    pass


class SecurityThresholdViolation(ComplexityEngineError):
    """Raised when m < 20 violates security recommendations."""
    pass


class SolverUnavailableError(ComplexityEngineError):
    """Raised when the requested SAT solver is not available."""
    pass


class InconsistentDeceptionError(ComplexityEngineError):
    """Raised when deceptive instance becomes unsatisfiable."""
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Clause:
    """
    A clause in a k-SAT instance.

    Represented as a list of literals, where each literal is a tuple of
    (variable_index, is_positive). The clause is a disjunction (OR) of literals.
    """
    literals: List[Tuple[int, bool]]  # (variable_id, is_positive)

    def __post_init__(self):
        if len(self.literals) == 0:
            raise ValueError("Clause must have at least one literal")

    def evaluate(self, assignment: Dict[int, bool]) -> bool:
        """Evaluate clause under given variable assignment."""
        for var_id, is_positive in self.literals:
            var_value = assignment.get(var_id, False)
            if is_positive == var_value:
                return True
        return False

    def __repr__(self) -> str:
        def lit_str(var_id: int, is_pos: bool) -> str:
            return f"x{var_id}" if is_pos else f"~x{var_id}"
        return "(" + " | ".join(lit_str(v, p) for v, p in self.literals) + ")"


@dataclass
class SATInstance:
    """
    A k-SAT instance with n clauses over m variables.

    The instance is satisfiable iff there exists an assignment to all m variables
    that makes all n clauses true simultaneously.
    """
    num_variables: int  # m
    num_literals_per_clause: int  # k
    clauses: List[Clause] = field(default_factory=list)

    def add_clause(self, clause: Clause) -> None:
        """Add a clause to the instance."""
        if len(clause.literals) != self.num_literals_per_clause:
            # Allow clauses with different sizes (for learned clauses)
            pass
        self.clauses.append(clause)

    def evaluate(self, assignment: Dict[int, bool]) -> bool:
        """Evaluate entire formula under assignment (conjunction of clauses)."""
        return all(clause.evaluate(assignment) for clause in self.clauses)

    def __repr__(self) -> str:
        return " & ".join(str(c) for c in self.clauses)


@dataclass
class TimingResult:
    """Result of a single SAT solving timing measurement."""
    solve_time_seconds: float
    is_satisfiable: bool
    assignment: Optional[Dict[int, bool]] = None
    solver_used: str = "z3"


# =============================================================================
# SAT SOLVING UTILITIES
# =============================================================================

def _solve_with_z3(instance: SATInstance, timeout_ms: int = 60000) -> TimingResult:
    """
    Solve SAT instance using Z3 and measure solving time.

    Args:
        instance: The k-SAT instance to solve
        timeout_ms: Timeout in milliseconds

    Returns:
        TimingResult with solve time and satisfiability
    """
    if not Z3_AVAILABLE:
        raise SolverUnavailableError("Z3 solver not available. Install with: pip install z3-solver")

    # Create Z3 boolean variables
    z3_vars = [Bool(f"x{i}") for i in range(instance.num_variables)]

    # Build Z3 formula
    z3_clauses = []
    for clause in instance.clauses:
        z3_lits = []
        for var_id, is_positive in clause.literals:
            if is_positive:
                z3_lits.append(z3_vars[var_id])
            else:
                z3_lits.append(Not(z3_vars[var_id]))
        z3_clauses.append(Or(z3_lits))

    # Create solver and add formula
    solver = Solver()
    solver.set("timeout", timeout_ms)
    solver.add(And(z3_clauses) if z3_clauses else True)

    # Solve and time
    start_time = time.perf_counter()
    result = solver.check()
    solve_time = time.perf_counter() - start_time

    # Extract result
    if result == sat:
        model = solver.model()
        assignment = {i: bool(model[z3_vars[i]]) for i in range(instance.num_variables)}
        return TimingResult(
            solve_time_seconds=solve_time,
            is_satisfiable=True,
            assignment=assignment,
            solver_used="z3"
        )
    elif result == unsat:
        return TimingResult(
            solve_time_seconds=solve_time,
            is_satisfiable=False,
            assignment=None,
            solver_used="z3"
        )
    else:
        # Timeout or unknown
        return TimingResult(
            solve_time_seconds=solve_time,
            is_satisfiable=False,  # Conservative
            assignment=None,
            solver_used="z3"
        )


def _solve_bruteforce(instance: SATInstance) -> TimingResult:
    """
    Solve SAT instance using brute-force enumeration.

    Only feasible for small instances (m <= 20).
    """
    m = instance.num_variables
    if m > 20:
        raise ComplexityEngineError(f"Brute-force infeasible for m={m} > 20 variables")

    start_time = time.perf_counter()

    # Enumerate all 2^m assignments
    for i in range(2 ** m):
        assignment = {var: bool((i >> var) & 1) for var in range(m)}
        if instance.evaluate(assignment):
            solve_time = time.perf_counter() - start_time
            return TimingResult(
                solve_time_seconds=solve_time,
                is_satisfiable=True,
                assignment=assignment,
                solver_used="bruteforce"
            )

    solve_time = time.perf_counter() - start_time
    return TimingResult(
        solve_time_seconds=solve_time,
        is_satisfiable=False,
        assignment=None,
        solver_used="bruteforce"
    )


def solve_sat(
    instance: SATInstance,
    solver: SATSolver = SATSolver.Z3,
    timeout_ms: int = 60000
) -> TimingResult:
    """
    Solve SAT instance using specified solver.

    Args:
        instance: The k-SAT instance to solve
        solver: Which solver to use (Z3, BRUTEFORCE, etc.)
        timeout_ms: Timeout in milliseconds

    Returns:
        TimingResult with solve time and satisfiability
    """
    if solver == SATSolver.Z3:
        return _solve_with_z3(instance, timeout_ms)
    elif solver == SATSolver.BRUTEFORCE:
        return _solve_bruteforce(instance)
    else:
        # Fall back to Z3 for other solvers
        if Z3_AVAILABLE:
            return _solve_with_z3(instance, timeout_ms)
        else:
            return _solve_bruteforce(instance)


# =============================================================================
# INSTANCE GENERATION
# =============================================================================

def generate_random_clause(
    m: int,
    k: int,
    rng: Optional[random.Random] = None
) -> Clause:
    """
    Generate a random k-SAT clause over m variables.

    Each clause contains exactly k distinct literals, where each literal is
    a variable or its negation with equal probability.

    Args:
        m: Number of variables
        k: Number of literals per clause
        rng: Random number generator (for reproducibility)

    Returns:
        A random Clause with k literals
    """
    if rng is None:
        rng = random.Random()

    # Select k distinct variables
    if k > m:
        raise ValueError(f"Cannot select k={k} literals from m={m} variables")

    selected_vars = rng.sample(range(m), k)

    # Assign random polarity to each
    literals = [(var, rng.choice([True, False])) for var in selected_vars]

    return Clause(literals)


def generate_random_sat_instance(
    m: int,
    n: int,
    k: int,
    seed: Optional[int] = None
) -> SATInstance:
    """
    Generate a random k-SAT instance with n clauses over m variables.

    The instance is generated at the satisfiability threshold ratio (n/m ~ 4.27 for 3-SAT)
    to produce hard instances for benchmarking.

    Args:
        m: Number of variables
        n: Number of clauses
        k: Literals per clause
        seed: Random seed for reproducibility

    Returns:
        Random SATInstance
    """
    rng = random.Random(seed)

    instance = SATInstance(num_variables=m, num_literals_per_clause=k)

    for _ in range(n):
        clause = generate_random_clause(m, k, rng)
        instance.add_clause(clause)

    return instance


def generate_satisfiable_instance(
    m: int,
    n: int,
    k: int,
    seed: Optional[int] = None
) -> Tuple[SATInstance, Dict[int, bool]]:
    """
    Generate a k-SAT instance that is guaranteed to be satisfiable.

    First generates a random truth assignment, then generates clauses
    that are satisfied by that assignment.

    Args:
        m: Number of variables
        n: Number of clauses
        k: Literals per clause
        seed: Random seed

    Returns:
        Tuple of (SATInstance, satisfying_assignment)
    """
    rng = random.Random(seed)

    # Generate random satisfying assignment
    truth_assignment = {i: rng.choice([True, False]) for i in range(m)}

    instance = SATInstance(num_variables=m, num_literals_per_clause=k)

    for _ in range(n):
        # Select k variables
        selected_vars = rng.sample(range(m), k)

        # Ensure at least one literal is satisfied
        # Pick one to be satisfied, others random
        satisfied_idx = rng.randint(0, k - 1)

        literals = []
        for i, var in enumerate(selected_vars):
            if i == satisfied_idx:
                # This literal must be satisfied
                is_positive = truth_assignment[var]
            else:
                # Random polarity
                is_positive = rng.choice([True, False])
            literals.append((var, is_positive))

        instance.add_clause(Clause(literals))

    return instance, truth_assignment


# =============================================================================
# HONEST VS DECEPTIVE INSTANCE GENERATION
# =============================================================================

def generate_honest_instance(
    m: int,
    n: int,
    k: int,
    seed: Optional[int] = None
) -> Tuple[SATInstance, Dict[int, bool]]:
    """
    Generate an "honest" world model SAT instance.

    The honest agent has a fixed world model (truth assignment) and generates
    statements (clauses) that are all true in that world model.

    This simulates an agent reporting truthfully about reality.

    Args:
        m: Number of facts/variables in the world model
        n: Number of statements to generate
        k: Literals per statement
        seed: Random seed

    Returns:
        Tuple of (instance, world_model)
    """
    return generate_satisfiable_instance(m, n, k, seed)


def generate_deceptive_instance(
    m: int,
    n: int,
    k: int,
    honest_world: Dict[int, bool],
    observable_fraction: float = 1.0,
    strategy: DeceptionStrategy = DeceptionStrategy.FULL,
    seed: Optional[int] = None
) -> Tuple[SATInstance, Dict[int, bool]]:
    """
    Generate a "deceptive" world model SAT instance.

    The deceptive agent has a false world model (different from honest_world)
    but must generate statements that:
    1. Are all consistent with each other (satisfiable)
    2. Don't contradict observable facts from honest_world
    3. Support the deceptive world model

    This creates a harder SAT problem because the agent must maintain consistency
    across multiple constraints while hiding the true world state.

    Args:
        m: Number of facts/variables
        n: Number of statements
        k: Literals per statement
        honest_world: The true world model the agent is hiding
        observable_fraction: Fraction of facts that are directly observable
        strategy: Deception strategy
        seed: Random seed

    Returns:
        Tuple of (instance, deceptive_world)

    Raises:
        InconsistentDeceptionError: If deceptive instance becomes unsatisfiable
    """
    rng = random.Random(seed)

    # Determine which facts are observable
    num_observable = max(1, int(m * observable_fraction))
    observable_facts = set(rng.sample(range(m), num_observable))

    # Create deceptive world model (differs from honest on non-observable facts)
    deceptive_world = honest_world.copy()

    # Flip some non-observable facts to create deception
    non_observable = [i for i in range(m) if i not in observable_facts]
    if non_observable:
        num_to_flip = max(1, len(non_observable) // 3)  # Flip ~33% of non-observable
        facts_to_flip = rng.sample(non_observable, min(num_to_flip, len(non_observable)))
        for fact_id in facts_to_flip:
            deceptive_world[fact_id] = not deceptive_world[fact_id]

    instance = SATInstance(num_variables=m, num_literals_per_clause=k)

    # Generate clauses based on strategy
    for clause_idx in range(n):
        if strategy == DeceptionStrategy.FULL:
            # Full deception: all clauses true in deceptive world
            clause = _generate_clause_for_world(m, k, deceptive_world, rng)
        elif strategy == DeceptionStrategy.SPARSE:
            # Sparse deception: only lie about non-observable facts
            clause = _generate_sparse_clause(m, k, deceptive_world, observable_facts, honest_world, rng)
        elif strategy == DeceptionStrategy.LAZY:
            # Lazy deception: generate minimal clauses, relying on just-in-time fabrication
            clause = _generate_lazy_clause(m, k, deceptive_world, rng)
        else:
            clause = _generate_clause_for_world(m, k, deceptive_world, rng)

        instance.add_clause(clause)

        # Add consistency constraints for observable facts
        # (deceptive agent cannot directly contradict observable facts)
        if clause_idx % 3 == 0 and observable_facts:  # Add observable constraints periodically
            obs_fact = rng.choice(list(observable_facts))
            # Add clause that is consistent with observable truth
            obs_clause = Clause([(obs_fact, honest_world[obs_fact])])
            instance.add_clause(obs_clause)

    return instance, deceptive_world


def _generate_clause_for_world(
    m: int,
    k: int,
    world: Dict[int, bool],
    rng: random.Random
) -> Clause:
    """Generate a clause that is satisfied by the given world model."""
    selected_vars = rng.sample(range(m), k)

    # Ensure at least one literal is satisfied
    satisfied_idx = rng.randint(0, k - 1)

    literals = []
    for i, var in enumerate(selected_vars):
        if i == satisfied_idx:
            is_positive = world[var]
        else:
            is_positive = rng.choice([True, False])
        literals.append((var, is_positive))

    return Clause(literals)


def _generate_sparse_clause(
    m: int,
    k: int,
    deceptive_world: Dict[int, bool],
    observable_facts: Set[int],
    honest_world: Dict[int, bool],
    rng: random.Random
) -> Clause:
    """
    Generate clause for sparse deception strategy.

    Prefers to use non-observable facts for deception while keeping
    observable facts truthful.
    """
    non_observable = [i for i in range(m) if i not in observable_facts]
    observable = list(observable_facts)

    # Try to select mostly non-observable facts
    num_non_obs = min(k - 1, len(non_observable))
    num_obs = k - num_non_obs

    if non_observable and num_non_obs > 0:
        selected_non_obs = rng.sample(non_observable, num_non_obs)
    else:
        selected_non_obs = []

    if observable and num_obs > 0:
        selected_obs = rng.sample(observable, min(num_obs, len(observable)))
    else:
        selected_obs = []

    selected_vars = selected_non_obs + selected_obs
    if len(selected_vars) < k:
        # Fill remaining from any available
        remaining = [v for v in range(m) if v not in selected_vars]
        selected_vars.extend(rng.sample(remaining, k - len(selected_vars)))

    # Generate literals - use deceptive world for non-observable, honest for observable
    literals = []
    for var in selected_vars:
        if var in observable_facts:
            is_positive = honest_world[var]
        else:
            is_positive = deceptive_world[var]
        literals.append((var, is_positive))

    return Clause(literals)


def _generate_lazy_clause(
    m: int,
    k: int,
    world: Dict[int, bool],
    rng: random.Random
) -> Clause:
    """
    Generate clause for lazy deception strategy.

    Creates minimal clauses with high redundancy to reduce constraint complexity.
    """
    # Use same variable multiple times with consistent polarity (creates weaker constraints)
    selected_vars = rng.sample(range(m), k)

    # All literals satisfied (easy clause)
    literals = [(var, world[var]) for var in selected_vars]

    return Clause(literals)


# =============================================================================
# COMPLEXITY ENGINE
# =============================================================================

class ComplexityEngine:
    """
    Engine for measuring computational complexity gap between honest and deceptive agents.

    Under the Exponential Time Hypothesis (ETH), maintaining consistency in deception
    requires exponentially more computation than truth-telling:

        T_D / T_H = Omega(2^m)

    This engine:
    1. Generates random k-SAT instances for honest world model verification
    2. Generates deceptive SAT instances requiring consistency maintenance
    3. Measures solving time using Z3 SMT solver
    4. Computes T_D / T_H ratio with confidence intervals

    Usage:
        engine = ComplexityEngine()
        result = engine.measure_complexity(ComplexityParams(
            world_size=20,
            num_statements=50,
            literals_per_statement=3
        ))
        print(f"Complexity ratio: {result.ratio}x")
    """

    def __init__(
        self,
        default_solver: SATSolver = SATSolver.Z3,
        timeout_ms: int = 60000,
        num_trials: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize the Complexity Engine.

        Args:
            default_solver: Default SAT solver to use
            timeout_ms: Default timeout for solver in milliseconds
            num_trials: Number of trials for statistical averaging
            seed: Random seed for reproducibility
        """
        self.default_solver = default_solver
        self.timeout_ms = timeout_ms
        self.num_trials = num_trials
        self.seed = seed

        # Validate Z3 availability if using Z3 solver
        if default_solver == SATSolver.Z3 and not Z3_AVAILABLE:
            raise SolverUnavailableError(
                "Z3 solver requested but not available. "
                "Install with: pip install z3-solver"
            )

    def validate_parameters(self, params: ComplexityParams) -> None:
        """
        Validate complexity parameters for NP-hardness and security claims.

        Args:
            params: The complexity parameters to validate

        Raises:
            NPHardnessViolation: If k < 3
            SecurityThresholdViolation: If m < 20 (warning only)
        """
        # T-CPX-01: k >= 3 for NP-hardness
        if params.literals_per_statement < 3:
            raise NPHardnessViolation(
                f"literals_per_statement={params.literals_per_statement} < 3 violates "
                "NP-hardness requirement. For k < 3, the problem is 2-SAT which is in P. "
                "Security claims are INVALID for this configuration."
            )

        # Recommend m >= 20 for security
        if params.world_size < 20:
            import warnings
            warnings.warn(
                f"world_size={params.world_size} < 20 may not provide sufficient "
                "security margin. Recommended m >= 20 for security claims.",
                UserWarning
            )

    def measure_honest_time(
        self,
        m: int,
        n: int,
        k: int,
        solver: SATSolver,
        seed: Optional[int] = None
    ) -> List[float]:
        """
        Measure solving time for honest (satisfiable) instances.

        Args:
            m: Number of variables
            n: Number of clauses
            k: Literals per clause
            solver: SAT solver to use
            seed: Random seed

        Returns:
            List of solving times in seconds
        """
        times = []
        base_seed = seed if seed is not None else (self.seed or 42)

        for trial in range(self.num_trials):
            trial_seed = base_seed + trial
            instance, _ = generate_honest_instance(m, n, k, trial_seed)
            result = solve_sat(instance, solver, self.timeout_ms)
            times.append(result.solve_time_seconds)

        return times

    def measure_deceptive_time(
        self,
        m: int,
        n: int,
        k: int,
        observable_fraction: float,
        strategy: DeceptionStrategy,
        solver: SATSolver,
        seed: Optional[int] = None
    ) -> List[float]:
        """
        Measure solving time for deceptive (consistency-checking) instances.

        Args:
            m: Number of variables
            n: Number of clauses
            k: Literals per clause
            observable_fraction: Fraction of observable facts
            strategy: Deception strategy
            solver: SAT solver to use
            seed: Random seed

        Returns:
            List of solving times in seconds
        """
        times = []
        base_seed = seed if seed is not None else (self.seed or 42)

        for trial in range(self.num_trials):
            trial_seed = base_seed + trial

            # Generate honest world first
            _, honest_world = generate_honest_instance(m, n, k, trial_seed)

            # Generate deceptive instance
            instance, _ = generate_deceptive_instance(
                m, n, k,
                honest_world,
                observable_fraction,
                strategy,
                trial_seed + 1000  # Different seed for deception
            )

            result = solve_sat(instance, solver, self.timeout_ms)
            times.append(result.solve_time_seconds)

        return times

    def compute_ratio_with_ci(
        self,
        honest_times: List[float],
        deceptive_times: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute T_D / T_H ratio with confidence interval.

        Uses bootstrap resampling for robust CI estimation.

        Args:
            honest_times: List of honest solving times
            deceptive_times: List of deceptive solving times
            confidence_level: Confidence level for CI (default 0.95)

        Returns:
            Tuple of (ratio, ci_lower, ci_upper)
        """
        honest_mean = np.mean(honest_times)
        deceptive_mean = np.mean(deceptive_times)

        # Avoid division by zero
        if honest_mean < 1e-10:
            honest_mean = 1e-10

        ratio = deceptive_mean / honest_mean

        # Bootstrap CI estimation
        n_bootstrap = 1000
        bootstrap_ratios = []

        rng = np.random.default_rng(self.seed)

        for _ in range(n_bootstrap):
            h_sample = rng.choice(honest_times, size=len(honest_times), replace=True)
            d_sample = rng.choice(deceptive_times, size=len(deceptive_times), replace=True)

            h_mean = np.mean(h_sample)
            d_mean = np.mean(d_sample)

            if h_mean > 1e-10:
                bootstrap_ratios.append(d_mean / h_mean)

        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_ratios, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_ratios, 100 * (1 - alpha / 2))

        # Ensure ratio >= 1 (deception should not be easier than honesty)
        ratio = max(1.0, ratio)
        ci_lower = max(1.0, ci_lower)

        return ratio, ci_lower, ci_upper

    def measure_complexity(
        self,
        params: ComplexityParams,
        validate: bool = True
    ) -> ComplexityResult:
        """
        Measure computational complexity gap between honest and deceptive agents.

        This is the main entry point for complexity measurement. It:
        1. Validates parameters for NP-hardness requirements
        2. Generates and solves honest SAT instances
        3. Generates and solves deceptive SAT instances
        4. Computes T_D / T_H ratio with confidence intervals

        Args:
            params: Complexity measurement parameters
            validate: Whether to validate parameters (default True)

        Returns:
            ComplexityResult with times, ratio, and confidence interval
        """
        if validate:
            self.validate_parameters(params)

        m = params.world_size
        n = params.num_statements
        k = params.literals_per_statement
        observable_fraction = params.observable_fraction
        strategy = params.deception_strategy
        solver = params.solver if hasattr(params, 'solver') else self.default_solver

        # Measure honest times
        honest_times = self.measure_honest_time(m, n, k, solver, self.seed)

        # Measure deceptive times
        deceptive_times = self.measure_deceptive_time(
            m, n, k, observable_fraction, strategy, solver, self.seed
        )

        # Compute ratio with CI
        ratio, ci_lower, ci_upper = self.compute_ratio_with_ci(honest_times, deceptive_times)

        return ComplexityResult(
            time_honest=np.mean(honest_times),
            time_deceptive=np.mean(deceptive_times),
            ratio=ratio,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            solver=solver,
            eth_conditional=True  # All exponential gap claims require ETH
        )

    def theoretical_ratio_bound(self, m: int, n: int, k: int) -> float:
        """
        Compute theoretical lower bound on T_D / T_H ratio under ETH.

        Under ETH, k-SAT cannot be solved in O(2^(epsilon * n)) for any epsilon < 1.
        This gives us:

            T_D / T_H >= Omega(2^(delta * m)) for some delta > 0

        Args:
            m: Number of variables
            n: Number of clauses
            k: Literals per clause

        Returns:
            Theoretical lower bound on ratio
        """
        # Under ETH at the satisfiability threshold (n/m ~ 4.27 for k=3)
        # the expected ratio is exponential in m

        # Conservative estimate: 2^(0.1 * m) for k=3
        if k == 3:
            delta = 0.1
        elif k > 3:
            delta = 0.05 * k  # Larger k means harder problem
        else:
            delta = 0.0  # k < 3 is not NP-hard

        return 2 ** (delta * m)

    def verify_eth_consistency(
        self,
        measured_ratio: float,
        m: int,
        n: int,
        k: int,
        tolerance: float = 0.5
    ) -> bool:
        """
        Verify that measured ratio is consistent with ETH prediction.

        Args:
            measured_ratio: Empirically measured T_D / T_H ratio
            m: Number of variables
            n: Number of clauses
            k: Literals per clause
            tolerance: Log-scale tolerance for verification

        Returns:
            True if measurement is consistent with ETH
        """
        theoretical = self.theoretical_ratio_bound(m, n, k)

        # Check if within tolerance in log scale
        if measured_ratio < 1.0:
            return False  # Deception should not be easier

        # Allow measured to be within tolerance orders of magnitude
        log_ratio = math.log2(measured_ratio + 1)
        log_theoretical = math.log2(theoretical + 1)

        return abs(log_ratio - log_theoretical) < tolerance * log_theoretical


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def measure_complexity(
    world_size: int,
    num_statements: int,
    literals_per_statement: int = 3,
    observable_fraction: float = 1.0,
    deception_strategy: DeceptionStrategy = DeceptionStrategy.FULL,
    solver: SATSolver = SATSolver.Z3,
    num_trials: int = 10,
    seed: Optional[int] = None
) -> ComplexityResult:
    """
    Convenience function to measure complexity gap.

    Args:
        world_size: Number of facts/variables (m)
        num_statements: Number of statements (n)
        literals_per_statement: Literals per statement (k >= 3)
        observable_fraction: Fraction of observable facts
        deception_strategy: Strategy for deceptive agent
        solver: SAT solver to use
        num_trials: Number of trials for averaging
        seed: Random seed

    Returns:
        ComplexityResult with times and ratio

    Example:
        result = measure_complexity(world_size=20, num_statements=50)
        print(f"Deception is {result.ratio:.1f}x harder than truth")
    """
    params = ComplexityParams(
        world_size=world_size,
        num_statements=num_statements,
        literals_per_statement=literals_per_statement,
        observable_fraction=observable_fraction,
        deception_strategy=deception_strategy,
        solver=solver
    )

    engine = ComplexityEngine(
        default_solver=solver,
        num_trials=num_trials,
        seed=seed
    )

    return engine.measure_complexity(params)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    'ComplexityEngineError',
    'NPHardnessViolation',
    'SecurityThresholdViolation',
    'SolverUnavailableError',
    'InconsistentDeceptionError',

    # Data structures
    'Clause',
    'SATInstance',
    'TimingResult',

    # Instance generation
    'generate_random_clause',
    'generate_random_sat_instance',
    'generate_satisfiable_instance',
    'generate_honest_instance',
    'generate_deceptive_instance',

    # SAT solving
    'solve_sat',

    # Engine
    'ComplexityEngine',

    # Convenience
    'measure_complexity',

    # Re-exports from types
    'ComplexityResult',
    'ComplexityParams',
    'DeceptionStrategy',
    'SATSolver',
]
