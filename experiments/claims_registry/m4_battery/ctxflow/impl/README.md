# H3ERE as implemented (TLA+)

`H3EREImpl.tla` models the conscience step of CIRISAgent's thought pipeline as the code does it, not as the
contract says (`../H3EREContextFlow.tla` is the contract). The configurations differ only in the switches below,
and each switch is a code difference between v2.23.2 and the fix. Every model answer is nondeterministic: TLC
explores every draft (DEFER / act = SPEAK·TOOL·PONDER / MEMORIZE / complete = TASK_COMPLETE·REJECT) and every
conscience answer (none / evidence / justification / substantive / unsure) at both passes of every thought.

Run: `TLA2TOOLS=/path/tla2tools.jar ./check_impl.sh` (each property is checked as its own INVARIANT; `CFGS=`,
`PROPS=` and `TRACES=dir` are optional).

## Trees

| | commit |
|---|---|
| v2.23.2 | tag `v2.23.2-stable` = `1b5b1e61c`; engine tree identical to `176c8b25a` (line refs below) |
| fix | PR #1386 `b1634614b` (`fix.cfg`); release/2.24.0 `777c6adc7` (`fix_2240.cfg`) = #1386 + non-H3ERE changes + the F6 fix in `main.py:2075-2082` |

## Switches

| switch | v2.23.2 | fix | code (v2.23.2 → 2.24.0) |
|---|---|---|---|
| `DeferSkipsBan` (F1) | FALSE | TRUE | `conscience_execution.py:314` → `:318-319` (`repeated = None if is_defer`); `veto_contract.py` `banned_entry_for` → `:196` (`defer:` keys never banned) |
| `RetryExemptsDefer` (F2) | FALSE | TRUE | retry forced PONDER `conscience_execution.py:448-470` → `:451-472`, unchanged; on 2.24.0 the DEFER block after it (`:475-482`) replaces that PONDER with the DEFER. Gated by `has_depth_guardrail` (`:450`/`:455`); `ThoughtDepthGuardrail` is always registered (`component_builder.py:140`, `registry.py:50-55`), so the branch is dead in the shipped build. `HasDepthGuardrail=FALSE` shows the mechanism |
| `DeferStands` (F3) | FALSE | TRUE | none on v2.23.2; 2.24.0 `conscience_execution.py:475-482` (`if is_defer and overridden`: dispatch the DEFER with the objections, `_defer_with_objections` `:843`) |
| `InheritVetoes` (F4) | TRUE | FALSE | guidance task `context.copy()` `wise_authority/service.py:1395`; 2.24.0 pops `VETO_RECORD_CONTEXT_KEY` at `:1405-1407` |
| `UnsureType` (F5) | substantive | justification | `schemas/runtime/veto_record.py:36-50` (`coerce_objection_type`, last line); `_objections_of` `conscience_execution.py:680` → `:698`; English conscience prompts "substantive when unsure" → "justification when unsure" (`entropy_conscience.yml:51` etc.) |
| `RetryKeepsDefer` (F6) | FALSE | TRUE on 2.24.0 | retry failure returns the first pass's action `main.py:2140-2142`; 2.24.0 `777c6adc7` returns a retry that ended in DEFER (`:2075-2082`) |
| `HasDepthGuardrail` | TRUE | TRUE | see F2 |
| `ConscienceRepeats` | TRUE | TRUE | FALSE = a conscience vetoes a given draft at most once per task, so any repeat comes from the ban (sh-un-07's shape) |

## Actions → code (same lines in both trees unless shown)

| model | code |
|---|---|
| `Pass`, depth branch | bypass conscience first, returns early: `conscience_execution.py:184-216`, `_run_bypass_conscience_entries` `:1015` → `:1050`; `thought_depth_guardrail.py` (max_depth 5 `:139`, terminal pass `:230-257`, forced DEFER `:259+`) |
| `Pass`, `Exempt` | `conscience_execution.py:230-237` → `:231-238` (RECALL/TASK_COMPLETE/OBSERVE/REJECT; DEFER not exempt) |
| `Pass`, ban | `:303-316` → `:303-319`; `veto_contract.banned_entry_for` `:184`; `_repeated_veto_result` `:824` → `:859` |
| `Pass`, veto / `Eff` | `_apply_veto_contract` `:771` → `:789` (`banned = type != JUSTIFICATION or substantive`, `:796` → `:814`); evidence ban lifts on a newer WA grant |
| `Step`, retry once | `main.py:354-370`, `_should_retry_with_conscience_guidance` `:1617`, `_handle_conscience_retry` `:1645` |
| `Step`, `disp` on retry failure | `_process_conscience_retry_result` `main.py:2039`; normal mode returns the first pass's action `:2140-2142` (F6) |
| `Step`, depth | `ponder_handler.py:64` (+1, no cap); `task_thought_factory.py:405` (`min(raw_depth, 7)` = `DepthCap`) |
| `Conv` | `main.py:1198` `_redefer_objected_identity_amendment` (WA-edited amendment, no M-1 justification → DEFER) |
| `Resolve` | `wise_authority/service.py:~1383-1490` (new guidance task, depth 0, WA grant) |

## Configurations

`v2232`, `fix` (#1386), `fix_2240` (release/2.24.0), one per flaw (`F1_ban_loop` also reverts F3, since a DEFER must be
vetoed before it can be banned; `F2_retry_forced_ponder[_noguard]`; `F3_veto_defer`; `F4_inherit`; `F5_unsure`),
`F6_direct_F3reverted` (2.24.0 with F3 reverted: F6 closed by its own fix) and `v2232_banonly`
(`ConscienceRepeats=FALSE`).

## Results (2026-11-27)

| config | breaks (trace length) | holds |
|---|---|---|
| v2232 | Termination (8), GuidanceTerminates (10), DeferNeverBlocked (2), WAOverrule (3), WAOverruleInTask (4), ForcedDeferDispatched (7) | TypeOK, Continuity (278,874 states) |
| v2232_banonly | GuidanceTerminates (10), DeferNeverBlocked (2), WAOverrule (3), NotShUn07 (15; 214,189 states) | |
| F1_ban_loop | Termination (8), GuidanceTerminates (10), DeferNeverBlocked (2), ForcedDeferDispatched (7) | the rest |
| F2_retry_forced_ponder | | all 8 (2,447) |
| F2_retry_forced_ponder_noguard | DeferNeverBlocked (2) | the rest |
| F3_veto_defer | Termination (8), GuidanceTerminates (10), DeferNeverBlocked (2), ForcedDeferDispatched (7) | the rest |
| F4_inherit | WAOverrule (3) | the rest |
| F5_unsure | WAOverruleInTask (4) | the rest |
| F6_direct_F3reverted | Termination (8), GuidanceTerminates (10), DeferNeverBlocked (2) | ForcedDeferDispatched (17,516) and the rest |
| fix, fix_2240 | | all 8 and NotShUn07 (2,447 each; NotShUn07 checked on fix_2240) |

## Properties

| | property | invariant |
|---|---|---|
| (a) | Termination | `Termination`: an active task has fewer than `ThoughtBound` thoughts |
| (e) | chains bounded per WA resolution | `GuidanceTerminates`: the same, for the guidance task a resolution opens |
| (b) | DeferNeverBlocked | a drafted (or retried, or forced) DEFER is what dispatches |
| (c) | WA-overrule | `WAOverrule`: no pre-resolution veto bans anything in the guidance task; `WAOverruleInTask`: no ban from an untyped objection under a WA grant |
| (d) | item 8 continuity | `Continuity`: a banned non-DEFER action never dispatches unchanged |
| F6 | | `ForcedDeferDispatched`: a guardrail-forced DEFER is never discarded for a PONDER |
| | sh-un-07 probe | `NotShUn07`: a violation is a witness of sh-un-07's exact path (WA-edited resolution, inherited substantive DEFER ban, no fresh DEFER veto in the guidance task, still active at the bound) |

`Termination` is an invariant over a bounded counter, so a non-closing task is a finite counterexample. The
traces that break it end at `depth = DepthCap` with the ban unchanged; from there `Step`'s guard and effect do
not depend on `nthoughts`, so the state repeats and the unbounded engine loops (sh-un-07: 31 PONDERs at depth 7).

Bounds: MaxDepth 5, DepthCap 7, ThoughtBound 7, MaxRes 2. The veto record is abstracted to one ban level per
action key (none / evidence-ungranted / evidence-granted / substantive); action equivalence (`verb:tool:hash`) is
collapsed to the verb class.
