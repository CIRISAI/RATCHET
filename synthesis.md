# Worktree 10 Synthesis: M-03

## Assignment
- **Issue:** M-03
- **Scope:** Finite sample invariant
- **Dependencies:** wt-7 (DP gaps)

## Task
Add invariant for n≥100 validity with Berry-Esseen correction specification

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

### 2026-01-02: Finite Sample Invariant Implementation

**Analysis:**
The FORMALIZATION_ROADMAP.md Section 8.1 Gap 3 identifies that detection power formulas
use asymptotic normal approximation, but the code tests on finite n. For n < 100,
Berry-Esseen bounds are needed to quantify the approximation error.

The key insight is that the Berry-Esseen theorem provides an explicit bound on the
error of the normal approximation:
```
|F_n(x) - Phi(x)| <= C * rho / (sigma^3 * sqrt(n))
```
where C <= 0.4748 (Shevtsova 2011).

For the 0.05 tolerance to hold, we need:
```
0.4748 / sqrt(n) <= 0.05
=> sqrt(n) >= 9.496
=> n >= 90.2
```
Rounding up to n >= 100 provides a conservative threshold.

**Changes Made:**

1. **FSD.md Section 3.3 (Detection Engine):**
   - Added "Finite Sample Validity Invariant (M-03)" subsection
   - Documented n >= 100 as asymptotic validity threshold
   - Added Invariant FS-1 (asymptotic validity threshold)
   - Added Invariant FS-2 (power approximation accuracy: <= 0.05)
   - Added Berry-Esseen correction formula for 30 <= n < 100
   - Added small sample fallback specification for n < 30 (permutation, bootstrap, conservative)
   - Noted dependency on wt-7 (DP-4 asymptotic validity)

2. **FSD.md Section 7.2 (Security Invariants):**
   - Added FINITE_SAMPLE_INVARIANTS array with FS-1 through FS-4
   - Added validate_finite_sample_regime() function for runtime validation
   - Added academic references (Berry 1941, Esseen 1942, Shevtsova 2011)
   - Documented derivation of n >= 100 threshold from Berry-Esseen constant

**Dependency on wt-7:**
This work assumes that wt-7 (DP-4 asymptotic validity) has specified the asymptotic
regime correctly. The Berry-Esseen corrections here are validated against that regime.
Specifically:
- The asymptotic power formula from DP-4 is assumed valid for large n
- The finite sample corrections provide the bridge to small n
- The n >= 100 threshold marks the boundary between regimes

**Verification:**
1. Check that FSD.md Section 3.3 includes all four invariants (FS-1 through FS-4)
2. Check that FSD.md Section 7.2 includes FINITE_SAMPLE_INVARIANTS array
3. Verify Berry-Esseen constant 0.4748 matches literature
4. Confirm n >= 100 threshold derivation: 0.4748/sqrt(100) = 0.0475 < 0.05

**Handoff Notes:**
- Downstream worktrees should use the validate_finite_sample_regime() function
- Any worktree implementing detection power should check sample size regime
- The 0.05 tolerance in FS-2 may need adjustment based on domain requirements

