# Worktree 5 Synthesis: T-GEO-02

## Assignment
- **Issue:** T-GEO-02
- **Scope:** Refinement types for all interfaces
- **Dependencies:** None (BASE TYPES - other agents depend on this)
- **Status:** COMPLETED

## Task
Add Pydantic validators with explicit bounds to ALL numeric parameters.

## Parallel Context
You are one of 15 parallel agents. OTHER AGENTS DEPEND ON YOU for base types.
This worktree provides foundational type definitions used by all other worktrees.

## Reference Files
- FSD: `/home/emoore/RATCHET/FSD.md`
- Formal Review: `/home/emoore/RATCHET/FSD_FORMAL_REVIEW.md`
- Coordinator: `/home/emoore/RATCHET/COORDINATOR.md`

---

## Work Log

### 1. Analysis

Reviewed FSD_FORMAL_REVIEW.md Section 1.2 which identifies 10 type issues (T-GEO-01 through T-SCH-01) that could cause runtime errors. The review recommends implementing Pydantic validators with explicit bounds to catch bugs at validation time.

Key issues addressed:
- T-GEO-01: Unbounded dimension (admits <= 0)
- T-GEO-02: Unbounded radius (theory requires 0 < r < 0.5)
- T-GEO-03: Correlation bounds missing (needs [-1, 1])
- T-CPX-01: Literals unbounded (needs >= 3 for NP-hardness)
- T-CPX-02: Observable fraction semantics (needs (0, 1])
- T-DET-01: Mahalanobis must be non-negative
- T-DET-02: Deception rate domain (needs (0, 1))
- T-DET-03: Alpha/beta probability constraints
- T-SCH-01: Untyped parameters dictionary (type hole)

### 2. Changes Made

#### Created: `schemas/types.py`
New file containing all refinement types with Pydantic Field validators:

**Core Refinement Types:**
```python
Dimension = Annotated[int, Field(gt=0)]           # Positive dimension D
Radius = Annotated[float, Field(gt=0, lt=0.5)]    # Deceptive region radius
Correlation = Annotated[float, Field(ge=-1, le=1)] # Constraint correlation rho
Probability = Annotated[float, Field(gt=0, lt=1)]  # For alpha, beta, rates
MahalanobisDistance = Annotated[float, Field(ge=0)] # Non-negative distance
SampleSize = Annotated[int, Field(ge=1)]          # Sample size n
Literals = Annotated[int, Field(ge=3)]            # NP-hardness requirement
WorldSize = Annotated[int, Field(ge=1)]           # World model size m
```

**Derived Refinement Types:**
```python
NumConstraints = Annotated[int, Field(gt=0)]      # Number of constraints k
NumStatements = Annotated[int, Field(gt=0)]       # Number of statements n
ObservableFraction = Annotated[float, Field(gt=0, le=1)]  # (0, 1]
EffectiveRank = Annotated[float, Field(gt=0)]     # k_eff > 0
ByzantineFraction = Annotated[float, Field(ge=0, lt=1/3)] # BFT limit
NodeCount = Annotated[int, Field(ge=1)]           # Federation nodes
CaptureRate = Annotated[float, Field(ge=0, lt=1)] # Slow capture rate
```

**Enumerations:**
- SamplingMode: orthonormal, correlated, adversarial
- DeceptionStrategy: full, sparse, lazy
- SATSolver: z3, minisat, cadical, bruteforce
- DetectionMethod: lrt, mahalanobis, isolation_forest, ensemble
- ConsensusProtocol: pbft, raft, tendermint
- AttackType: mimicry, flooding, adaptive, null_space, etc.
- MaliciousStrategy: random, coordinated, slow_capture
- ProofStatus: pending, partial, axiomatized, proven, disproven, blocked

**Composite Types (addressing GAPs from formal review):**
- Hyperplane: Normal vector + offset with unit norm validation
- VolumeEstimate: Volume with CI and decay constant
- ComplexityResult: T_H, T_D, ratio with ETH conditional flag
- PowerAnalysisResult: Sample size with finite-sample correction
- AdversarialStrategy: Attack specification (GAP-01)
- InferenceGraph: Graph for compositional detection (GAP-04)
- Vote: Federation vote record (GAP-05)
- Precedent: Precedent record (GAP-06)

**Engine Parameter Types (addressing T-SCH-01):**
- GeometricParams: Type-safe bundle for estimate_volume()
- ComplexityParams: Type-safe bundle for measure_complexity()
- DetectionParams: Type-safe bundle for power_analysis()
- FederationParams: Type-safe bundle with BFT validation

**Helper Functions:**
- compute_effective_rank(k, rho): Computes k_eff = k / (1 + rho*(k-1))
- compute_required_sample_size(D, p, alpha, beta): Sample size formula
- validate_literals_for_np_hardness(k): Validates k >= 3

#### Updated: `FSD.md` Sections 3.1-3.4

Added "Refinement Types" subsections to each engine section showing:
1. Import statements for relevant types
2. Type annotations in method signatures
3. Comments documenting which type issues are prevented
4. Preconditions and assumptions

### 3. Verification

To verify the types work correctly:

```python
from schemas.types import (
    Dimension, Radius, Correlation, Probability,
    MahalanobisDistance, SampleSize, Literals, WorldSize,
    GeometricParams, ComplexityParams, DetectionParams, FederationParams,
)
from pydantic import ValidationError

# Test valid values
params = GeometricParams(
    dimension=100,
    num_constraints=50,
    deceptive_radius=0.2,
    constraint_correlation=0.5,
)
print(f"Valid GeometricParams: {params}")

# Test invalid dimension (should fail)
try:
    GeometricParams(dimension=0, num_constraints=50, deceptive_radius=0.2)
except ValidationError as e:
    print(f"Caught invalid dimension: {e}")

# Test invalid radius (should fail)
try:
    GeometricParams(dimension=100, num_constraints=50, deceptive_radius=0.6)
except ValidationError as e:
    print(f"Caught invalid radius: {e}")

# Test invalid literals (should fail - k < 3)
try:
    ComplexityParams(world_size=20, num_statements=100, literals_per_statement=2)
except ValidationError as e:
    print(f"Caught invalid literals (NP-hardness violation): {e}")

# Test BFT violation (should fail - malicious >= 1/3)
try:
    FederationParams(num_honest=4, num_malicious=3)
except ValidationError as e:
    print(f"Caught BFT violation: {e}")
```

### 4. Handoff Notes for Dependent Worktrees

**IMPORTANT FOR ALL OTHER AGENTS:**

Import refinement types from `schemas/types.py`:
```python
from schemas.types import (
    # Core types - use these for ALL numeric parameters
    Dimension,           # int > 0
    Radius,              # 0 < float < 0.5
    Correlation,         # -1 <= float <= 1
    Probability,         # 0 < float < 1
    MahalanobisDistance, # float >= 0
    SampleSize,          # int >= 1
    Literals,            # int >= 3 (NP-hardness)
    WorldSize,           # int >= 1

    # Enums - use instead of Literal types
    SamplingMode,
    DeceptionStrategy,
    SATSolver,
    DetectionMethod,
    ConsensusProtocol,
    AttackType,

    # Composite types - use for return values
    Hyperplane,
    VolumeEstimate,
    ComplexityResult,
    PowerAnalysisResult,
    AdversarialStrategy,
    InferenceGraph,
    Vote,
    Precedent,

    # Parameter bundles - use for type-safe API requests
    GeometricParams,
    ComplexityParams,
    DetectionParams,
    FederationParams,
    SimulationParams,
)
```

**Type Safety Guarantees:**
1. All types use Pydantic Field validators - invalid values raise ValidationError
2. FederationParams has model_validator to enforce BFT constraint
3. Hyperplane has field_validator to enforce unit normal constraint
4. ComplexityResult has eth_conditional flag for conditional claims

**Integration Pattern:**
```python
class GeometricEngine:
    def estimate_volume(
        self,
        dimension: Dimension,        # NOT int
        deceptive_radius: Radius,    # NOT float
        ...
    ) -> VolumeEstimate:            # NOT Dict
        ...
```

### 5. Files Modified

1. **NEW:** `schemas/__init__.py` - Package init with exports
2. **NEW:** `schemas/types.py` - All refinement types (500+ lines)
3. **MODIFIED:** `FSD.md` - Sections 3.1, 3.2, 3.3, 3.4 updated with type annotations

### 6. Issue Resolution Summary

| Issue ID | Description | Resolution |
|----------|-------------|------------|
| T-GEO-01 | Unbounded dimension | `Dimension = Annotated[int, Field(gt=0)]` |
| T-GEO-02 | Unbounded radius | `Radius = Annotated[float, Field(gt=0, lt=0.5)]` |
| T-GEO-03 | Correlation bounds | `Correlation = Annotated[float, Field(ge=-1, le=1)]` |
| T-GEO-04 | AdversarialStrategy undefined | `AdversarialStrategy` BaseModel with attack_type, probe_budget |
| T-CPX-01 | Literals unbounded | `Literals = Annotated[int, Field(ge=3)]` |
| T-CPX-02 | Observable fraction | `ObservableFraction = Annotated[float, Field(gt=0, le=1)]` |
| T-DET-01 | Mahalanobis negative | `MahalanobisDistance = Annotated[float, Field(ge=0)]` |
| T-DET-02 | Deception rate bounds | `Probability = Annotated[float, Field(gt=0, lt=1)]` |
| T-DET-03 | Alpha/beta bounds | Same `Probability` type |
| T-SCH-01 | Untyped parameters | `SimulationParams = Union[GeometricParams, ...]` |
| GAP-01 | AdversarialStrategy | Defined as BaseModel |
| GAP-04 | InferenceGraph | Defined as BaseModel |
| GAP-05 | Vote | Defined as BaseModel |
| GAP-06 | Precedent | Defined as BaseModel |

---

**Status:** READY FOR MERGE

