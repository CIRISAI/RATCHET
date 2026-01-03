# Worktree 8 Synthesis: M-01

## Assignment
- **Issue:** M-01
- **Scope:** Non-adaptive invariant
- **Dependencies:** None

## Task
Add explicit invariant: adversary.strategy_fixed_before(detection.threshold)

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

The M-01 issue identifies a critical missing invariant: the non-adaptive adversary assumption. Per the Formal Review Section 4 Question 2, all detection power guarantees assume the adversary's strategy is fixed BEFORE the detection threshold is chosen. This is currently implicit throughout the FSD but should be explicit.

The formal review states:
- Non-adaptive means: P_D is fixed BEFORE threshold tau is chosen
- Adaptive adversary can evade with O(sqrt(n)) queries
- Detection power degrades by O(B / sqrt(n)) for B-adaptive adversary

### Changes Made to FSD.md

1. **Section 1 (Executive Summary):** Added SECURITY_ASSUMPTION block with:
   - Formal statement: `adversary.strategy_fixed_before(detection.threshold)`
   - Conditions when assumption is violated
   - Consequence: O(sqrt(n)) evasion, power degradation O(B/sqrt(n))
   - Partial mitigations listed

2. **Section 3.3 (Detection Engine):** Added:
   - CRITICAL ASSUMPTION banner at top of section
   - Note that power formula assumes P_D fixed before threshold
   - LRT optimality requires non-adaptive adversary
   - Updated power_analysis() docstring with ASSUMES/INVALID FOR clauses

3. **Section 7.2 (Security Invariants):** Added:
   - New invariant: `adversary.strategy_fixed_before(detection.threshold)`
   - Formal sigma-algebra statement
   - Reference to Section 11 for violation consequences
   - Note on detection.power claim being conditional

4. **Section 11 (Known Limitations):** NEW SECTION with:
   - Detailed M-01 analysis (non-adaptive assumption)
   - Theorem statement: O(sqrt(n)) adaptive evasion
   - Proof sketch (5 steps)
   - Power degradation formula
   - Failure table comparing non-adaptive vs adaptive
   - Four partial mitigation strategies
   - Verification status (NOT formally verifiable)
   - Recommendations
   - Table of other known limitations (M-02 through M-06)

### Code Changes

None required - this is a specification document update.

### Verification

The fix can be verified by:
1. Grep for "non-adaptive" in FSD.md - should appear in Sections 1, 3.3, 7.2, 11
2. Grep for "strategy_fixed_before" - should appear in all detection-related sections
3. Check Section 11 exists with adaptive adversary analysis
4. Verify O(sqrt(n)) evasion is documented

### Handoff Notes

- Worktrees addressing detection power claims should reference the non-adaptive assumption
- Any worktree modifying Section 3.3 should preserve the CRITICAL ASSUMPTION block
- The Section 11 Known Limitations is a new section that other worktrees may need to update
- The SECURITY_ASSUMPTION in Section 1 is now the canonical statement of this assumption

