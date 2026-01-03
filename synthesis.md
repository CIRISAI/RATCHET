# Worktree 9 Synthesis: M-02

## Assignment
- **Issue:** M-02
- **Scope:** Hyperplane distribution invariant
- **Dependencies:** wt-6 (TC gaps)

## Task
Add invariant specifying offset distribution and lambda adjustment formula

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

**Issue M-02** addresses a critical discrepancy between theory and implementation for hyperplane distributions, as identified in the Formal Review Section 4 Question 1.

**The Problem:**
- **Theory:** Hyperplane offsets are drawn from Uniform([0, 1])
- **Code:** Hyperplane offsets are drawn from Uniform([0.2, 0.8])

This discrepancy affects the cutting probability formula and, consequently, the exponential volume decay constant lambda.

**Root Cause Analysis:**
The implementation uses [0.2, 0.8] to avoid boundary effects at the edges of [0,1]^D. However, this introduces a systematic bias that changes the effective lambda from 2r to 2r/(b-a) = 2r/0.6 = 3.33r.

**Error Classification:**
- The distribution mismatch causes an O(r) error in cutting probability
- The theoretical higher-order terms cause an O(r^2) error
- The O(r) error DOMINATES the O(r^2) error, making this a significant issue

### Changes Made to FSD.md

#### 1. Section 3.1 - New Subsection 3.1.1: Hyperplane Distribution Specification (M-02)

Added comprehensive documentation including:
- **Discrepancy identification:** Explicit statement of the theory vs. code difference
- **Canonical distribution specification:** Defined the theoretical standard (Uniform([0,1]) offset)
- **Lambda adjustment formula:** `lambda_adjusted = 2r / (b - a)`
- **Error analysis:** Documented O(r) vs O(r^2) error distinction
- **Configuration class:** `HyperplaneDistributionConfig` with lambda_multiplier property
- **Dependency note:** Explicit reference to wt-6 for TC-4 error bound

#### 2. Section 7.2 - Added M-02 Invariant to Security Invariants

Added the distribution consistency invariant to the SECURITY_INVARIANTS list:
```python
# M-02: Hyperplane Distribution Consistency (NEW)
"""
geometric.hyperplane_distribution.offset_distribution == 'uniform_0_1' OR
(
    geometric.hyperplane_distribution.offset_distribution == 'uniform_a_b' AND
    geometric.lambda == 2*r / (b - a) AND
    |cutting_probability_error| <= C * r^2
)
"""
```

#### 3. Section 7.2.1 - Detailed Invariant Specification

Added new subsection with:
- **Formal statement:** Mathematical definition of the invariant
- **M-02-A (Canonical):** Invariant for Uniform([0,1]) case
- **M-02-B (Adjusted):** Invariant for Uniform([a,b]) case with adjustment
- **Error bound dependency:** How this invariant connects to TC-4 from wt-6
- **O(r) vs O(r^2) table:** Clear comparison of error severities
- **Security implications:** Why this matters for deployment
- **Verification protocol:** How to test the invariant

### Code

New configuration class added to Section 3.1:

```python
class HyperplaneDistributionConfig:
    """
    Hyperplane sampling distribution configuration.
    """
    offset_distribution: Literal["uniform_0_1", "uniform_a_b"] = "uniform_0_1"
    offset_range: Tuple[float, float] = (0.0, 1.0)  # [a, b] for uniform_a_b

    @property
    def lambda_multiplier(self) -> float:
        """Lambda adjustment factor for non-standard offset range."""
        a, b = self.offset_range
        return 1.0 / (b - a)  # For uniform_0_1, this is 1.0

    def cutting_probability(self, r: float) -> float:
        """Expected cutting probability for ball of radius r."""
        return 2 * r * self.lambda_multiplier
```

### Verification

1. **Unit test:** Compare Monte Carlo cutting probability against:
   - `2r` for Uniform([0,1])
   - `2r/(b-a)` for Uniform([a,b])

2. **Integration test:** Verify volume decay follows `exp(-lambda_adjusted * k)` for both distributions

3. **Regression test:** CI check that any modification to offset distribution triggers lambda recalculation

### Handoff

#### Dependency on wt-6 (TC gaps)

This work **depends on wt-6** for:
- **TC-4 Error Bound:** The exponential approximation error bound
- The combined error formula in Section 7.2.1 references:
  ```
  V(k) = V(0) * exp(-lambda_adjusted * k) * (1 + O(r^2 * k))
  ```
  where the O(r^2 * k) term is specified by TC-4.

**Interface Assumption:** Assuming TC-4 provides an error bound of the form O(r^2 * k) for the exponential approximation. If wt-6 specifies a different form, Section 7.2.1 may need adjustment.

#### Notes for Coordinator

1. **Merge Order:** This work can be merged independently, but the TC-4 reference in Section 7.2.1 should be verified against wt-6's output.

2. **Potential Conflicts:**
   - If wt-6 modifies Section 3.1 or 7.2, manual merge may be needed
   - The new Section 3.1.1 and 7.2.1 are additions, not modifications, minimizing conflict risk

3. **Open Decision:** The FSD now specifies that `uniform_0_1` is the canonical distribution. Implementers must choose whether to:
   - Change code to use Uniform([0,1]) and accept boundary effects
   - Keep Uniform([0.2, 0.8]) and apply lambda adjustment

   The invariant enforces consistency regardless of choice.

---

## Summary

| Item | Status |
|------|--------|
| Document discrepancy | COMPLETE |
| Lambda adjustment formula | COMPLETE |
| Distribution invariant | COMPLETE |
| Canonical distribution specified | COMPLETE (uniform_0_1) |
| O(r) vs O(r^2) distinction | COMPLETE |
| Dependency on wt-6 noted | COMPLETE |
| Ready for commit | YES |
