# Worktree 12 Synthesis: M-05

## Assignment
- **Issue:** M-05
- **Scope:** Independence invariant
- **Dependencies:** wt-6 (TC gaps)

## Task
Add invariant for constraint independence, specify correlation impact on k_eff

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

Issue M-05 addresses a critical gap identified in the Formal Methods Review: the exponential decay of deceptive volume V(k) = V(0) * exp(-lambda*k) assumes constraints are independent. The Red Team Analysis (Section 1.2) demonstrates that correlated constraints dramatically reduce effective protection:

**Key Finding from Red Team Analysis 1.2:**
- For k=100 constraints with rho=0.7 correlation: k_eff = 100 / (1 + 0.7 * 99) = 1.42
- This means 100 correlated constraints provide protection equivalent to only ~1.4 independent constraints

**Root Cause:**
- Real-world federations suffer from shared training data, memetic propagation, and authority bias
- Current simulations use `ortho_group.rvs(D)` which generates maximally independent (orthonormal) constraints
- This is the best case, not the typical case

### Changes Made to FSD.md

#### Section 3.1 - Geometric Engine

1. **Added Critical Invariant M-05:**
   ```
   INVARIANT M-05: forall i, j: i != j => hyperplanes[i] independent_of hyperplanes[j]
   ```

2. **Added Formal Definition:** Two hyperplanes are independent iff their normal vectors are linearly independent.

3. **Added Correlation Impact Table:** Shows k_eff for various rho values (0.0 to 0.9) and k values (10, 50, 100).

4. **Added Adjusted Decay Rate Formula:**
   ```
   lambda_eff = lambda_0 / (1 + rho * (k - 1))
   ```

5. **Extended `compute_effective_rank` method:** Added detailed algorithm specification including:
   - Gram matrix computation
   - SVD-based alternative (participation ratio)
   - Return values including independence_violations list
   - Invariant check with warning threshold

6. **Added `verify_independence` method:** For checking invariant M-05 compliance with configurable thresholds.

7. **Added `enforce_independence` method:** Three remediation strategies:
   - `drop_correlated`: Remove highly correlated constraints
   - `orthogonalize`: Apply Gram-Schmidt orthogonalization
   - `diversify`: Add new orthogonal constraints from null space

#### Section 7.2 - Security Invariants

1. **Added three M-05 related invariants:**
   - Independence invariant (forall i,j: i != j => independent)
   - Effective rank minimum (k_eff >= MIN_EFFECTIVE_RANK)
   - Correlation bound (average_correlation <= 0.3)

2. **Added Effective Rank Calculation specification** with security implications:
   - k_eff < 10: WEAK protection
   - k_eff < 5: MINIMAL protection
   - k_eff < 2: FAILS (adversary can evade)

#### Section 7.3 - Independence Enforcement Protocol (NEW)

Added comprehensive operational protocol:
- Continuous monitoring with alert levels (INFO, WARNING, ALERT, CRITICAL)
- Three enforcement action types: Preventive, Corrective, Compensatory

### Code

No new Python/Lean code added. The specification additions provide sufficient detail for implementation. The algorithms (Gram matrix, SVD, Gram-Schmidt) are standard linear algebra operations.

### Verification

To verify the fix is correct:

1. **Check invariant coverage:** Grep for "M-05" in FSD.md - should appear in Sections 3.1, 7.2, and 7.3.

2. **Verify formula consistency:** The k_eff formula `k / (1 + rho * (k-1))` should match across all occurrences.

3. **Test correlation impact table:** Manually verify one entry:
   - k=50, rho=0.5: k_eff = 50 / (1 + 0.5 * 49) = 50 / 25.5 = 1.96 (matches table)

4. **Formal verification:** The invariant statement is suitable for encoding in Lean 4.

### Handoff

#### Dependency on wt-6 (TC gaps)

This worktree depends on wt-6 which addresses TC (Topological Collapse) gaps. Specifically:
- wt-6 should provide the proof obligation TC-2 (Independence via Fubini) specification
- M-05 provides the invariant that TC-2 depends on
- The effective rank calculation in this worktree (k_eff formula) should be referenced by TC-7

**Interface Assumption:** wt-6 will use the invariant M-05 as stated here:
```
forall i, j: i != j => hyperplanes[i] independent_of hyperplanes[j]
```

**Merge Order:** wt-6 should merge after wt-12 to ensure the independence invariant is in place.

#### Notes for Coordinator

1. The k_eff formula appears in multiple places - ensure consistency during merge.
2. Section 7.3 is new - verify it doesn't conflict with other worktrees modifying Section 7.
3. The `verify_independence` and `enforce_independence` methods extend the GeometricEngine interface.

---

## Summary

This worktree addresses M-05 by adding a comprehensive specification for constraint independence, including:
- Formal invariant statement
- Effective rank calculation formula with security implications
- Methods for verification and enforcement
- Operational monitoring protocol with alert levels

The changes ensure that geometric security claims are qualified by the independence assumption and provide practical tools for maintaining this invariant in production systems.
