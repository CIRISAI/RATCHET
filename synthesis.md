# Worktree 6 Synthesis: TC-GAPS

## Assignment
- **Issue:** TC-GAPS
- **Scope:** Topological collapse proof gaps
- **Dependencies:** None

## Task
Add specifications for TC-2(Fubini), TC-3(volume scaling), TC-4(error bounds), TC-8(uniform convergence)

## Parallel Context
You are one of 15 parallel agents. If your work requires output from a dependent worktree, note the interface assumption and proceed. The coordinator will merge.

## Reference Files
- FSD: `/home/emoore/RATCHET/FSD.md`
- Formal Review: `/home/emoore/RATCHET/FSD_FORMAL_REVIEW.md`
- Coordinator: `/home/emoore/RATCHET/COORDINATOR.md`

## Output Format
1. **Analysis**: Your assessment of the issue
2. **Changes**: Specific edits/additions to FSD.md
3. **Code**: Any new Python/Lean code required
4. **Verification**: How to verify the fix is correct
5. **Handoff**: Notes for dependent worktrees

---

## Work Log

### Analysis

The FORMALIZATION_ROADMAP.md Section 6.1 identifies 8 proof obligations for the Topological Collapse theorem. Four of these were identified as gaps requiring explicit specification:

| ID | Obligation | Difficulty | Status Before |
|----|------------|------------|---------------|
| TC-2 | Independence/Fubini | CORE | Implicit |
| TC-3 | Volume scaling | MEDIUM | Implicit |
| TC-4 | Error bound O(r^2k) | HARD | Unquantified |
| TC-8 | Uniform convergence | HARD | Unstated |

These gaps were blocking formal verification because:
1. TC-2: Independence assumption was used but not stated as a verifiable property
2. TC-3: Volume scaling after manifold intersection lacked precise bounds
3. TC-4: The O(r^2k) error term was mentioned but not formalized
4. TC-8: No statement about uniformity over center positions

### Changes Made

#### 1. FSD.md Updates

Added new section **4.1.1 Topological Collapse Proof Obligations (TC-GAPS)** containing:

- **TC-2: Independence/Fubini Property**
  - Formal statement of product measure factorization
  - Lean 4 theorem sketch: `independence_fubini`
  - Verification protocol: Chi-squared test with 10^6 samples

- **TC-3: Volume Scaling After Manifold Intersection**
  - Precise statement: E[mu(intersection)] = Theta(r^D)
  - Lean 4 theorems: `volume_scaling_manifold`, `volume_fraction_recursion`
  - Verification protocol: Numerical integration for D in {10, 50, 100}

- **TC-4: Error Bound O(r^2k)**
  - Precise statement: |V(k) - V(0)e^{-2rk}| <= V(0)e^{-2rk} * C * r^2 * k
  - Lean 4 theorem: `exponential_error_bound` with helper lemmas
  - Verification protocol: Fit C empirically, verify |C - 1.5| < 0.5

- **TC-8: Uniform Convergence Over Centers**
  - Precise statement: Bound holds uniformly over [0.25, 0.75]^D
  - Lean 4 theorem: `uniform_convergence_centers` with supporting lemmas
  - Verification protocol: F-test across 100 random centers

#### 2. New Files Created

- `formal/proofs/TopologicalCollapseGaps.lean` (220+ lines)
  - Complete Lean 4 proof sketches for TC-2, TC-3, TC-4, TC-8
  - Type definitions for hyperplanes, balls, interior cube
  - Dependency graph showing proof structure

- `formal/proofs/TCGapsVerification.lean` (180+ lines)
  - Verification lemmas bridging formal proofs and simulation
  - Simulation interface types (parameters and results)
  - Expected outcome theorems

### Verification

To verify these specifications are correct:

1. **Syntactic Check**: Lean 4 files should parse without syntax errors
   ```bash
   cd formal/proofs && lake build
   ```

2. **Monte Carlo Validation**: Run simulations to confirm bounds
   - TC-2: Chi-squared p-value > 0.01
   - TC-3: Volume in range [C1*r^D, C2*r^D] with C1, C2 > 0
   - TC-4: Fitted C in [1.0, 2.0]
   - TC-8: F-test p-value > 0.05

3. **Cross-Reference**: Check consistency with FORMALIZATION_ROADMAP.md Section 6.1

---

## Handoff: Notes for Dependent Worktrees

### For wt-9 (Verification Implementation)

**Key Interfaces Provided:**

1. `TC2SimParams` / `TC2SimResult` - Use these types when implementing independence verification
2. `TC4SimParams` / `TC4SimResult` - Use for error bound fitting
3. `TC8SimParams` / `TC8SimResult` - Use for uniformity testing

**Critical Requirements:**
- TC-2 requires at least 10^6 samples for statistical power
- TC-4 fitting should use weighted least squares (errors are heteroscedastic)
- TC-8 centers must be sampled from [0.25, 0.75]^D to avoid boundary effects

**Theorems to Verify:**
```lean
theorem independence_fubini        -- TC-2
theorem exponential_error_bound    -- TC-4
theorem uniform_convergence_centers -- TC-8
```

### For wt-11 (Lean Formalization)

**Proof Dependencies:**

```
TC-2 (Independence)
  |
  +--> TC-4 (Error Bound)
  |      |
  |      +--> Dimension Independence (existing)
  |
  +--> TC-3 (Volume Scaling)
         |
         +--> Exponential Decay (existing)
               |
               +--> TC-8 (Uniform Convergence)
                      |
                      +--> Robustness Analysis
```

**Suggested Proof Order:**
1. TC-2 first (uses standard probability independence)
2. TC-3 (requires coarea formula, geometric probability)
3. TC-4 (Taylor series, error accumulation - uses TC-2, TC-3)
4. TC-8 (compactness, uniformity - uses TC-3, TC-4)

**Missing Mathlib Dependencies:**
- `Probability.Independence.Kernel` for product measure factorization
- Geometric probability lemmas (may need to develop)
- Coarea formula (exists in Mathlib, needs connection)

### For wt-12 (Integration Tests)

**Test Cases to Implement:**

| Test ID | Obligation | Parameters | Expected |
|---------|------------|------------|----------|
| `test_tc2_independence` | TC-2 | D=20, k=10, r=0.05 | chi_sq < 6.635 |
| `test_tc3_volume_bounds` | TC-3 | D in {10,50,100} | C1 <= V/r^D <= C2 |
| `test_tc4_error_fit` | TC-4 | D=50, k_max=50 | 1.0 <= C <= 2.0 |
| `test_tc8_uniformity` | TC-8 | n_centers=100 | p_value > 0.05 |

**Regression Tests:**
- Ensure existing volume shrinkage tests still pass
- Add edge cases: r=0.01 (small), r=0.1 (large), D=1000 (high dim)

---

## Files Modified/Created

| File | Action | Lines |
|------|--------|-------|
| `FSD.md` | Modified | +220 |
| `formal/proofs/TopologicalCollapseGaps.lean` | Created | 225 |
| `formal/proofs/TCGapsVerification.lean` | Created | 185 |
| `synthesis.md` | Updated | (this file) |

## Status: COMPLETE

All TC-GAPS specifications have been added. Ready for merge into main branch.
