# Worktree 13 Synthesis: M-06

## Assignment
- **Issue:** M-06
- **Scope:** World model size invariant
- **Dependencies:** wt-2 (k≥3)

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

The world model size `m` functions as the security parameter - analogous to key length in cryptography. Without sufficient m, the complexity gap T_D / T_H = Ω(2^m / poly(n)) provides no meaningful security.

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

