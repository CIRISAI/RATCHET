# Worktree 13 Synthesis: M-06

## Status: READY FOR MERGE

**Dependency Resolved:** wt-2 (k>=3 enforcement) has been merged to master.

## Assignment
- **Issue:** M-06
- **Scope:** World model size invariant
- **Dependencies:** wt-2 (k>=3) - **MERGED**

## Task
Add invariant: complexity.world_size grows with security parameter

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

### 2026-01-02: World Model Size Invariant Implementation

**Analysis:**
The M-06 issue identifies a critical gap in the computational asymmetry claims. Per FORMALIZATION_ROADMAP Section 4.2.2:
- For m <= 15: brute-force SAT is feasible, making security claims vacuous
- For m = O(log n): 2^m = O(poly(n)), eliminating exponential separation
- Asymptotic separation requires m to grow independently with threat model

The world model size `m` functions as the security parameter - analogous to key length in cryptography. Without sufficient m, the complexity gap T_D / T_H = Omega(2^m / poly(n)) provides no meaningful security.

**Changes Made:**

1. **FSD.md Section 3.2 (Complexity Engine):**
   - Added "Security Parameter Relationship" subsection
   - Documented invariant: `complexity.world_size >= security_parameter`
   - Added security level table (minimal: m>=20, moderate: m>=40, high: m>=64)
   - Referenced FORMALIZATION_ROADMAP Section 4.2.2 for small m regime
   - Added requirement: "MUST enforce minimum world size m >= 20 for security claims"
   - Added cross-reference to wt-2 for k>=3 fix

2. **FSD.md Section 7.2 (Security Invariants):**
   - Added world size invariants to SECURITY_INVARIANTS list
   - Added SECURITY_PARAMETER_SCALING specification with threat model mapping
   - Documented dependency on wt-2 for NP-hardness to apply

**Dependency on wt-2:**
This work depends on wt-2 (k>=3 fix) because NP-hardness only applies when statements have k>=3 literals per clause (3-SAT). With k=2 (2-SAT), the problem is in P and the complexity gap disappears regardless of m. The security parameter scaling assumes k>=3 is enforced.

**Verification:**
- The invariant `complexity.world_size >= 20` ensures minimum meaningful security
- The scaling specification provides clear guidance for threat model alignment
- Cross-references to FORMALIZATION_ROADMAP maintain traceability

**Handoff Notes:**
- wt-2 must merge before these invariants are fully effective (k>=3 requirement)
- Downstream implementations should validate world_size at configuration time
- Security claims should be conditional on both m>=20 AND k>=3

---

### 2026-01-02: Integration with Merged wt-2 Changes

**Status Update:** wt-2 has been merged to master. Updating wt-13 to incorporate the k>=3 enforcement and properly cross-reference the combined invariant.

**Changes Made (Post-Dependency Resolution):**

1. **FSD.md Section 3.2 - Critical Invariant M-06:**
   - Added explicit combined invariant statement:
     `INVARIANT M-06: Security claims require BOTH m >= 20 AND k >= 3 (per Section 3.2.1)`
   - Added "Why BOTH are Required" explanation
   - Added Combined Security Table showing all k/m combinations and their security status

2. **FSD.md Section 3.2.2 - k >= 3 Enforcement:**
   - Incorporated `LiteralsPerStatement` refinement type from merged wt-2
   - Added full type definition with Pydantic Field constraints
   - Updated ComplexityEngine class to use typed parameter

3. **FSD.md Section 7.2 - Security Invariants:**
   - Added M-06 combined invariant as first entry in SECURITY_INVARIANTS list
   - Added separate k >= 3 enforcement invariant
   - Added M06_COMBINED_INVARIANT specification with violation consequences
   - Updated SECURITY_PARAMETER_SCALING to include `literals_per_statement: 3` at all levels

**Verification:**
- M-06 invariant now explicitly states: "Security claims require BOTH m>=20 AND k>=3 (per Section 3.2.1)"
- Cross-references between m>=20 and k>=3 requirements are complete
- Combined security table provides clear guidance on all parameter combinations

**This worktree is now READY FOR MERGE.**

