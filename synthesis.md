# Worktree 7 Synthesis: DP-GAPS

## Assignment
- **Issue:** DP-GAPS
- **Scope:** Detection power proof gaps
- **Dependencies:** wt-1 (preconditions)

## Task
Add specifications for DP-4(asymptotic validity), DP-5(plug-in estimation), DP-6(monotonicity)

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

### 2026-01-02: Detection Power Proof Obligations (DP-4, DP-5, DP-6)

#### Analysis

The FSD_FORMAL_REVIEW.md (Section 3.1.3) identified three missing detection power proof obligations:

| ID | Obligation | Status Before | Difficulty |
|----|------------|---------------|------------|
| DP-4 | Asymptotic validity | MISSING | HARD |
| DP-5 | Plug-in estimation error | MISSING | MEDIUM |
| DP-6 | Power monotonicity | IMPLICIT | EASY |

These gaps meant that only 57% (4/7) of detection power obligations were covered.

#### Changes Made

1. **FSD.md Section 4.1** - Added detailed specifications for:

   - **DP-4 (Asymptotic Validity)**: Specified O(1/sqrt(n)) error term for sample complexity formula using Berry-Esseen bound. Key bound: |beta_hat(n) - beta| <= C/sqrt(n) where C <= 0.56/D for D >= 0.5. Validity regime: n >= 100 for error <= 0.05.

   - **DP-5 (Plug-in Estimation Error)**: Specified error bounds for empirical Mahalanobis distance D_hat. Key bound: |D_hat - D| <= C_p * sqrt(p/n_train) + C_Sigma * sqrt(p^2/n_train) where C_Sigma depends on condition number kappa(Sigma).

   - **DP-6 (Power Monotonicity)**: Specified three monotonicity properties:
     - (a) Power increases with n (sample size)
     - (b) Power increases with D (Mahalanobis distance)
     - (c) Power decreases with p (deception rate) for fixed n

2. **formal/proofs/DetectionPower.lean** - Created Lean 4 theorem sketches:
   - `DetectionPreconditions` structure encoding constraints from wt-1
   - `asymptotic_validity` theorem with Berry-Esseen error bound
   - `plugin_estimation_error` theorem with high-probability bound
   - `power_monotone_in_n`, `power_monotone_in_D`, `power_monotone_in_p` theorems
   - `detection_power_guarantees` combined interface theorem for downstream use

#### Dependencies (Inputs)

**From wt-1 (Preconditions):**
- D >= 0.5 (Mahalanobis distance lower bound)
- p > 0.001 (deception rate lower bound)
- n >= 100 (sample size lower bound)

These preconditions are embedded in the `DetectionPreconditions` structure in DetectionPower.lean and referenced in all proof obligation specifications.

#### Handoff (Outputs for wt-10)

**Available for wt-10:**

1. **Theorem interfaces** in `formal/proofs/DetectionPower.lean`:
   - `DetectionPreconditions` - type encoding the wt-1 constraints
   - `asymptotic_validity` - O(1/sqrt(n)) error bound theorem
   - `plugin_estimation_error` - estimation error bound theorem
   - `power_monotone_in_n/D/p` - monotonicity theorems
   - `detection_power_guarantees` - combined verification interface

2. **Specification text** in FSD.md Section 4.1:
   - Complete mathematical statements for DP-4, DP-5, DP-6
   - Explicit precondition references to wt-1
   - Proof sketches for verification

**To complete proofs, wt-10 will need:**
- Mathlib.Probability.Distributions.Gaussian (when available in Mathlib)
- Formalization of standard normal CDF Phi and inverse z_alpha
- Berry-Esseen theorem for Gaussian LRT statistic
- Concentration bounds for sample covariance matrix estimation

#### Verification

The specifications can be verified by:

1. **Mathematical consistency**: Check that error bounds are dimensionally correct
2. **Precondition coverage**: Verify D >= 0.5, p > 0.001, n >= 100 are referenced
3. **Lean type-checking**: Run `lake build` to verify theorem sketches parse correctly
4. **Coverage improvement**: Detection power coverage should now be 7/7 (100%)

#### Files Modified

- `/home/emoore/RATCHET_WORKTREES/wt-7/FSD.md` (Section 4.1)
- `/home/emoore/RATCHET_WORKTREES/wt-7/formal/proofs/DetectionPower.lean` (new file)
- `/home/emoore/RATCHET_WORKTREES/wt-7/synthesis.md` (this file)

