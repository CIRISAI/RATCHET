# Worktree 2 Synthesis: U-02

## Assignment
- **Issue:** U-02
- **Scope:** k≥3 NP-hardness enforcement
- **Dependencies:** None

## Task
Enforce literals_per_statement≥3 or add prominent warning for P-time regime in 2-SAT

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

**Issue U-02: k >= 3 NP-Hardness Enforcement**

The Formal Review (Section 5.2, Issue T-CPX-01) identified a critical security vulnerability in the ComplexityEngine interface: the `literals_per_statement` parameter was unbounded, allowing values k < 3 that completely undermine the computational security model.

**The Problem:**
- For k >= 3: CONSISTENT-LIE reduces to k-SAT, which is NP-hard
- For k = 2: CONSISTENT-LIE reduces to 2-SAT, which is in P (solvable in O(n) time via Aspvall-Plass-Tarjan 1979 implication graph algorithm)
- For k = 1: Problem is trivially solvable

**Security Impact:**
The entire complexity-based security argument (T_D / T_H = Omega(2^m)) assumes NP-hardness. With k = 2, an adversary can maintain deceptive consistency with the SAME computational cost as an honest agent. The complexity gap vanishes entirely, defeating the core security property.

**Attack Scenario:**
An adversary could craft a world model using only 2-literal Horn clauses or implications, pass all validation checks, and then exploit the P-time consistency checking to maintain arbitrary deception indefinitely.

### Changes Made to FSD.md

1. **Section 3.2 - ComplexityEngine Interface (Lines 249-295)**
   - Added `LiteralsPerStatement` refinement type with `ge=3` constraint
   - Added security-critical documentation explaining why k >= 3 is required
   - Updated `measure_complexity` signature to use the constrained type
   - Added class-level docstring warning about k < 3 security implications

2. **Section 3.2 - Validation Protocol (Lines 326-338)**
   - Added explicit security boundary test requirement for k=2 vs k=3
   - Documented the "security cliff edge" at k in {2, 3}
   - Added CI pipeline enforcement requirement

3. **Section 7.2 - Security Invariants (Lines 791-831)**
   - Added new invariant: `complexity.literals_per_statement >= 3`
   - Added complexity regime documentation table showing k vs security status
   - Referenced the Aspvall-Plass-Tarjan algorithm for 2-SAT tractability

### Code (Type Constraint)

```python
from typing import Annotated
from pydantic import Field

LiteralsPerStatement = Annotated[
    int,
    Field(
        ge=3,
        description="Number of literals per statement. MUST be >= 3 for NP-hardness. "
                    "k=2 reduces to 2-SAT (P-time tractable). "
                    "k<3 voids all complexity gap security claims."
    )
]
```

### Verification

To verify this fix is correct:

1. **Type-Level Enforcement:** Any call to `measure_complexity` with `literals_per_statement < 3` should fail Pydantic validation at runtime.

2. **Security Invariant Check:** The CI pipeline should validate `complexity.literals_per_statement >= 3` before accepting any configuration.

3. **Empirical Validation:** Run complexity benchmarks with k=2 and k=3 on identical world models:
   - k=2: Expect T_D/T_H approx 1 (no gap)
   - k=3: Expect T_D/T_H >> 1 (exponential gap)

4. **Unit Test:** Add test that verifies rejection of k < 3 configurations.

### Handoff Notes for Dependent Worktrees

**For Worktrees Working on Complexity Engine Implementation:**
- The `literals_per_statement` parameter now has a minimum value of 3
- Any code that constructs ComplexityEngine queries MUST ensure k >= 3
- Legacy code allowing k=2 must be updated to either reject or warn

**For Worktrees Working on Proof Obligations:**
- The 3-SAT reduction proof (CA-2) now has an explicit precondition: k >= 3
- The complexity gap claim is conditional on this precondition

**For Worktrees Working on Security Invariants:**
- New invariant added: `complexity.literals_per_statement >= 3`
- This invariant gates ALL complexity-based security claims

**Interface Assumptions:**
- I assume other worktrees will update any code paths that dynamically set `literals_per_statement`
- The Pydantic validation will catch violations at runtime, but static analysis may need updates

### Summary

This fix closes the k < 3 security hole identified in Formal Review issue U-02 by:
1. Adding a type-level constraint (refinement type with ge=3)
2. Adding prominent warnings in documentation
3. Adding a security invariant for CI enforcement
4. Documenting the complexity/security tradeoff table

The fix follows the Formal Review recommendation REC-C4: "Enforce or Document k >= 3 for NP-Hardness"

