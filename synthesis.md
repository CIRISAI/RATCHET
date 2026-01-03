# Worktree 1 Synthesis: U-01

## Assignment
- **Issue:** U-01
- **Scope:** Power formula preconditions
- **Dependencies:** None

## Task
Add preconditions to detection power formula: D≥0.5, p∈(0.001,1), document n≥100 validity regime

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

### 2026-01-02: U-01 Implementation Complete

#### Analysis

The Formal Review (Section 5.2) identified **U-01** as a critical unsoundness risk: the detection power formula in Section 3.3 lacked explicit preconditions, which could lead to:

1. **Division by zero** when `deception_rate = 0`
2. **Negative or invalid sample sizes** when `mahalanobis_distance` is too small
3. **Invalid asymptotic approximations** when sample size `n < 100`
4. **Formula breakdown** when `deception_rate = 1` (no honest baseline)

The specific issues from the formal review:
- T-DET-01: Mahalanobis distance must be non-negative (extended to D >= 0.5 for practical feasibility)
- T-DET-02: Deception rate domain must be (0, 1) exclusive
- GAP-08: Finite sample regime n < 100 undefined
- M-03: Missing finite sample validity invariant

#### Changes Made to FSD.md Section 3.3

1. **Added Section 3.3.1 "Power Formula Preconditions"** with:
   - Explicit statement of the sample complexity formula
   - Precondition table with constraints and rationale:
     - `mahalanobis_distance >= 0.5` (detection infeasible below)
     - `deception_rate in (0.001, 1)` exclusive (prevents division by zero, degenerate cases)
     - `n >= 100` validity regime (asymptotic normality requirement)
     - Gaussian distribution assumption documented

2. **Added Berry-Esseen finite-sample correction guidance**:
   - For `30 <= n < 100`: apply correction `power_corrected = power - 0.4748/sqrt(n)`
   - For `n < 30`: asymptotic formula unreliable, use exact methods

3. **Updated `DetectionEngine.power_analysis` docstring** with explicit PRECONDITIONS section:
   - Precondition 1: D >= 0.5 with guidance for smaller effect sizes
   - Precondition 2: p in (0.001, 1) with rationale for bounds
   - Precondition 3: n >= 100 with Berry-Esseen correction formula
   - Precondition 4: Gaussian assumption with guidance for heavy-tailed distributions

4. **Enhanced return type documentation**:
   - `n`: asymptotic sample size
   - `n_corrected`: finite-sample adjusted size
   - `power`: achieved power
   - `validity_regime`: indicates which approximation regime applies
   - `warnings`: precondition concerns

#### Verification

The fix addresses U-01 by:
1. Preventing unsound implementations that use the formula outside its valid regime
2. Documenting the finite-sample correction (Berry-Esseen) for n < 100
3. Providing guidance for edge cases (small D, rare deception, non-Gaussian)
4. Adding structured return type with validity indicators

To verify correctness:
1. Check that `D >= 0.5` yields practical sample sizes (n < 100K for typical alpha/beta)
2. Confirm `p in (0.001, 1)` avoids division by zero and maintains n < 10^7
3. Validate Berry-Esseen constant C_BE <= 0.4748 matches literature
4. Ensure docstring preconditions match the specification table

#### Handoff Notes

- **Dependencies:** None (this fix is self-contained)
- **Consumers:** Any worktree implementing DetectionEngine should enforce these preconditions in code
- **Related:** T-DET-01, T-DET-02, T-DET-03 from formal review Section 1.1.3 are partially addressed; full refinement types (REC-C1) may be handled by another worktree
- **Assumption:** Other worktrees handling type refinements (Pydantic validators) will use these documented bounds

