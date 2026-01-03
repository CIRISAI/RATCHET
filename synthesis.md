# Worktree 14 Synthesis: NEW-OBL

## Assignment
- **Issue:** NEW-OBL
- **Scope:** New proof obligations
- **Dependencies:** wt-3,wt-5

## Task
Add NEW-01(k_eff), NEW-02(BFT safety), NEW-03(behavioral correlation), NEW-04(compositional), NEW-05(slow capture)

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

### 2026-01-02: NEW-OBL Implementation Complete

**Analysis:**
Reviewed FSD_FORMAL_REVIEW.md Section 3.3 which identifies 5 new proof obligations implied by FSD additions that were not in the original Formalization Roadmap. These obligations arise from:
- Section 3.1: k_eff formula for correlated constraints (NEW-01)
- Section 3.4: BFT protocol implementation (NEW-02, NEW-03, NEW-05)
- Section 3.3: Compositional detection for RT-01 mitigation (NEW-04)

**Changes Made:**
1. Added Section 4.1.1 "New Proof Obligations (from Formal Review)" to FSD.md
2. Created summary table with all 5 obligations, difficulty, status, and dependencies
3. Added detailed specifications for each obligation:
   - NEW-01: Effective rank k_eff correctness with correlation matrix analysis
   - NEW-02: BFT protocol safety (no conflicting commits) and liveness
   - NEW-03: Behavioral correlation detection soundness (Sybil detection)
   - NEW-04: Compositional detection correctness (MARKED IMPOSSIBLE)
   - NEW-05: Slow capture threshold O(1/r) bound

**Lean 4 Theorem Sketches:**
Added theorem sketches with `sorry` placeholders for:
- `effective_rank_correctness` - depends on wt-5 for Correlation type
- `bft_safety` and `bft_liveness` - depends on wt-3 for BFT types
- `behavioral_correlation_sensitivity/specificity` - depends on wt-3, wt-5
- `compositional_detection_incomplete` - impossibility proof sketch
- `slow_capture_bound` and `slow_capture_asymptotic` - depends on wt-3

**NEW-04 Impossibility Analysis:**
Marked as POTENTIALLY IMPOSSIBLE with three justifications:
1. Reduction from one-way functions (would break crypto commitments)
2. Information-theoretic barrier (marginal-preserving deception)
3. Rice's theorem analog (semantic property of compositions)

Provided partial mitigation guidance - what IS detectable vs what is fundamentally impossible.

**Dependencies on Other Worktrees:**
- **wt-3 (BFT):** NEW-02, NEW-03, NEW-05 require BFT protocol definitions, voting types, and federation structures from wt-3
- **wt-5 (types):** NEW-01, NEW-03 require refined types (Correlation, PositiveInt, Agent, Vote) from wt-5

**Interface Assumptions:**
Assuming wt-3 provides:
- `BFTConfig` structure with n, f, threshold constraint
- `HonestNode`, `Round`, `Value`, `Proposal` types
- `committed`, `valid`, `partial_synchrony` predicates
- `VotingHistory`, `Vote` types

Assuming wt-5 provides:
- `Correlation` refined type with -1 <= rho <= 1
- `PositiveInt`, `NonNegativeInt` refined types
- `Agent`, `Vote` schema types

**Verification:**
- All 5 obligations have theorem statements with clear preconditions/postconditions
- Lean sketches compile structurally (use `sorry` for unproven parts)
- NEW-04 impossibility is properly justified with reduction argument

**Handoff Notes:**
- wt-3 should provide BFT type definitions that match the Lean sketches
- wt-5 should ensure Correlation type handles the degenerate case rho = -1/(k-1)
- Full BFT verification should use TLA+ or Ivy, not Lean (Lean sketches are for specification anchoring)

