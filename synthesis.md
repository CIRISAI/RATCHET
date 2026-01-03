# Worktree 11 Synthesis: M-04

## Assignment
- **Issue:** M-04
- **Scope:** Convexity invariant
- **Dependencies:** wt-6 (TC gaps)

## Task
Add invariant: geometric.deceptive_region.is_convex==True, document non-convex as unsupported

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

### 2026-01-02: Convexity Invariant Implementation

**Analysis:**
The M-04 issue addresses the implicit convexity assumption in the Topological Collapse theorem
(Theorem 1.1). Per FORMALIZATION_ROADMAP.md Section 1.1 and 4.1.2, the exponential volume
shrinkage proof explicitly requires convex geometry for the cutting probability calculation.
Non-convex regions (torus, fractal, disconnected components) invalidate the independence
assumptions in the proof.

**Changes Made:**

1. **FSD.md Section 3.1 (GeometricEngine):**
   - Added `PRECONDITION-GE-01: geometric.deceptive_region.is_convex == True`
   - Added invariant documentation explaining why convexity is required
   - Added requirement to enforce convexity precondition on inputs
   - Updated Red Team requirements to flag non-convex as attack vector

2. **FSD.md Section 7.1 (Attack Mitigations):**
   - Added `RT-06 Non-Convex Evasion` attack to the table
   - Added full attack description with severity (HIGH) and type (FUNDAMENTAL)
   - Documented research directions for future work

3. **FSD.md Section 7.2 (Security Invariants):**
   - Added `geometric.deceptive_region.is_convex == True` as first invariant
   - Added detailed violation analysis table
   - Documented known limitation and required extensions

4. **FSD.md Section 10 (Open Questions):**
   - Added item 6: Non-Convex Deceptive Regions research question
   - Documented potential approaches (convex decomposition, topological methods, etc.)

**Dependency Notes:**
- **Depends on wt-6 (TC gaps):** This worktree documents the convexity assumption that wt-6
  may be filling in as a formal gap. The precondition added here should be consistent with
  any type constraints or invariants wt-6 establishes.
- **Interface assumption:** We assume wt-6 will establish the formal type structure for
  `DeceptiveRegion` that includes an `is_convex` property or similar constraint.

**Verification:**
1. Check that FSD.md Section 3.1 contains the precondition
2. Check that FSD.md Section 7.2 includes convexity in SECURITY_INVARIANTS
3. Check that RT-06 is documented as an attack vector
4. Grep for "is_convex" to verify all references are consistent

**Handoff:**
- wt-6 should ensure the formal type `DeceptiveRegion` in Lean includes convexity constraint
- Any implementation of GeometricEngine must validate convexity at input
- Non-convex testing should be restricted to attack surface analysis (RT-06 scenarios)

