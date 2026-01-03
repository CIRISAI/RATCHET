# RATCHET Parallel Development Coordination

## Worktree Assignments

| WT | Branch | Issue ID | Scope | Dependencies |
|----|--------|----------|-------|--------------|
| wt-1 | issue-1 | U-01 | Power formula preconditions | None |
| wt-2 | issue-2 | U-02 | k≥3 NP-hardness enforcement | None |
| wt-3 | issue-3 | U-03 | BFT protocol implementation | None |
| wt-4 | issue-4 | T-SCH-01 | Type hole fix (discriminated union) | wt-5 (types) |
| wt-5 | issue-5 | T-GEO-02 | Refinement types for all interfaces | None |
| wt-6 | issue-6 | TC-GAPS | Missing proof obligations (TC-2,3,4,8) | None |
| wt-7 | issue-7 | DP-GAPS | Missing proof obligations (DP-4,5,6) | wt-1 |
| wt-8 | issue-8 | M-01 | Invariant: Non-adaptive assumption | None |
| wt-9 | issue-9 | M-02 | Invariant: Hyperplane distribution | wt-6 |
| wt-10 | issue-10 | M-03 | Invariant: Finite sample validity | wt-7 |
| wt-11 | issue-11 | M-04 | Invariant: Convexity assumption | wt-6 |
| wt-12 | issue-12 | M-05 | Invariant: Independence of constraints | wt-6 |
| wt-13 | issue-13 | M-06 | Invariant: World model size | wt-2 |
| wt-14 | issue-14 | NEW-OBL | New proof obligations (NEW-01 to NEW-05) | wt-3,wt-5 |
| wt-15 | issue-15 | Q-RESOLVE | Open question resolutions (Q1-Q5) | All |

## Coordination Protocol

1. Each agent writes to `synthesis.md` in their worktree
2. If your work depends on another worktree, note it and proceed with interface assumptions
3. wt-15 synthesizes all outputs into final recommendations
4. All agents commit to their branch when complete
5. Coordinator merges all branches to master

## File Locations

- Main FSD: `/home/emoore/RATCHET/FSD.md`
- Formal Review: `/home/emoore/RATCHET/FSD_FORMAL_REVIEW.md`
- Formalization Roadmap: `/home/emoore/RATCHET/FORMALIZATION_ROADMAP.md`
- Red Team Analysis: `/home/emoore/RATCHET/proposals/red_team_analysis.md`

## Merge Order

1. wt-5 (base types) → 2. wt-4 (schema types) → 3. wt-1,2,3 (critical fixes)
4. wt-6,7 (proof obligations) → 5. wt-8-13 (invariants) → 6. wt-14 (new obligations)
7. wt-15 (synthesis)
