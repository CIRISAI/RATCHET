---------------------------- MODULE H3EREImpl ----------------------------
(* H3ERE AS IMPLEMENTED: the control flow of one task's thoughts through ASPDMA, the conscience step, the
   retry and the handlers, as the code does it, not as the contract says it should (that is
   ../H3EREContextFlow.tla). One Step = one thought, from the ASPDMA draft to the dispatched action.

   The model (the LLM: drafters and consciences) is fully nondeterministic: every draft and every
   conscience answer is chosen by TLC. Each switch below is one of the code differences between v2.23.2
   (1b5b1e61c / 176c8b25a) and fix/defer-never-blocked; the .cfg files set them. Line references
   are to v2.23.2 and are listed in README.md.

   Abstractions (each is stated where it matters):
   - A draft is its verb; two drafts with the same verb have the same ban-equivalence key
     (veto_contract.action_equivalence_key). Worst case for the ban, and the sh-un-07 shape: the model
     re-drafted a byte-identical DEFER every round.
   - One conscience answer per pass: the first failing conscience in priority order decides the override
     (conscience_execution.py fold), so one answer is enough to model the override.
   - TOOL is SPEAK for this flow; RECALL/OBSERVE (exempt, non-terminal) are not modelled.
   - A dispatched SPEAK/MEMORIZE/PONDER creates a follow-up thought at depth+1, capped at 7
     (task_thought_factory.py:405). The depth guardrail forces DEFER at depth >= MaxDepth (5). *)
EXTENDS Naturals, FiniteSets

CONSTANTS
  MaxDepth,            \* config.security.max_thought_depth (5)
  DepthCap,            \* schema cap on thought_depth (7)
  MaxRes,              \* WA resolutions modelled (each is an external event)
  ThoughtBound,        \* Termination: a task dispatches at most this many thoughts
  InheritVetoes,       \* guidance task context = context.copy() incl. __agent_veto_record__ (F4)
  UnsureType,          \* how an untyped / unsure objection is read (F5): "substantive" | "justification"
  DeferSkipsBan,       \* fix: a DEFER is never banned (F1)
  DeferStands,         \* fix: a veto cannot replace a DEFER; it stands, objections attached (F3)
  RetryExemptsDefer,   \* fix: the retry's forced PONDER never replaces a DEFER (F2)
  HasDepthGuardrail,   \* ThoughtDepthGuardrail registered in the registry (always TRUE in the shipped build)
  RetryKeepsDefer,     \* TRUE: a retry ending in DEFER dispatches that DEFER (_process_conscience_retry_result, 2.24.0 F6 fix)
  ConscienceRepeats    \* FALSE: a conscience vetoes a given draft at most once per task (isolates F1: only the ban repeats)

Drafts    == {"defer", "act", "memorize", "complete"}
  \* "act" = SPEAK / TOOL / PONDER drafts: non-terminal, non-exempt, same path through this flow.
  \* "complete" = TASK_COMPLETE / REJECT: terminal and conscience-exempt.
Keyed     == {"defer", "act", "memorize"}               \* drafts that can be vetoed / banned
Answers   == {"none", "evidence", "justification", "substantive", "unsure"}
Terminal  == {"defer", "complete"}                      \* thought_depth_guardrail.py terminal_actions
Exempt    == {"complete"}                               \* conscience_execution.py exempt_actions
Bans      == {"no", "eu", "eh", "sub"}
  \* the strongest recorded veto on a key: none / evidence (conscience did not hold a WA grant) /
  \* evidence (it did) / substantive. Several entries on one key reduce to the strongest: that is all
  \* banned_entry_for reads.

VARIABLES tid, kind, grant, editedNoM1, depth, status, res, nthoughts, last,
          oldBan,     \* [Keyed -> Bans]: entries recorded in EARLIER tasks and carried into this one
          curBan,     \* [Keyed -> Bans]: entries recorded in this task
          curUnsure,  \* [Keyed -> BOOLEAN]: this task holds a ban that came from an unsure / untyped answer
          vb          \* [Keyed -> BOOLEAN]: a conscience has vetoed this draft in this task
vars == <<tid, kind, grant, editedNoM1, depth, status, res, nthoughts, last, oldBan, curBan, curUnsure, vb>>

Min(a, b) == IF a < b THEN a ELSE b
Rank(b) == CASE b = "no" -> 0 [] b = "eu" -> 1 [] b = "eh" -> 2 [] b = "sub" -> 3
Max(b, c) == IF Rank(b) >= Rank(c) THEN b ELSE c

TypeOf(v) == IF v = "unsure" THEN UnsureType ELSE v      \* veto_record.coerce_objection_type

(* veto_contract.banned_entry_for: an evidence ban lifts once trusted state holds evidence the conscience
   did not (a WA grant arrived after it); a substantive ban never lifts. *)
Eff(b) == b \in {"sub", "eh"} \/ (b = "eu" /\ ~grant)
IsBanned(a, OB, CB) == a \in Keyed /\ (Eff(OB[a]) \/ Eff(CB[a]))

(* shared_helpers.redefer: in a guidance task whose WA supplied the text with no M-1 justification, the
   approved MEMORIZE is replaced by a DEFER back to the WA -- the same DEFER every round. *)
Conv(a) == IF kind = "guidance" /\ editedNoM1 /\ a = "memorize" THEN "defer" ELSE a

NoVeto == [k |-> "none", b |-> "no", unsure |-> FALSE]

(* One conscience pass (conscience_execution._conscience_execution_step): final action, override, and the
   veto it records (k = "none" if none). *)
Pass(a, v, retry, OB, CB, VB) ==
  IF a \notin Terminal /\ depth >= MaxDepth
    THEN [final |-> "defer_forced", ovr |-> TRUE, veto |-> NoVeto]       \* bypass guardrail, returns early
  ELSE IF a \in Exempt
    THEN [final |-> a, ovr |-> FALSE, veto |-> NoVeto]                   \* exempt: returns early
  ELSE IF IsBanned(a, OB, CB) /\ ~(a = "defer" /\ DeferSkipsBan)
    THEN [final |-> "ponder", ovr |-> TRUE, veto |-> NoVeto]             \* ban: PONDER, no model call
  ELSE IF v # "none" /\ (ConscienceRepeats \/ ~VB[a])
    THEN IF a = "defer" /\ DeferStands
           THEN [final |-> "defer", ovr |-> FALSE, veto |-> NoVeto]      \* fix: the DEFER stands
           ELSE [final |-> "ponder", ovr |-> TRUE,
                 veto |-> [k |-> a,
                           b |-> CASE TypeOf(v) = "justification" -> "no"
                                   [] TypeOf(v) = "evidence" -> (IF grant THEN "eh" ELSE "eu")
                                   [] OTHER -> "sub",
                           unsure |-> v = "unsure" /\ TypeOf(v) # "justification"]]
  ELSE IF retry /\ ~HasDepthGuardrail /\ ~(a = "defer" /\ RetryExemptsDefer)
    THEN [final |-> "ponder", ovr |-> TRUE, veto |-> NoVeto]             \* forced PONDER on a passing retry
  ELSE [final |-> a, ovr |-> FALSE, veto |-> NoVeto]

AddB(CB, x)  == IF x.k = "none" THEN CB ELSE [CB EXCEPT ![x.k] = Max(@, x.b)]
AddU(CU, x)  == IF x.k = "none" THEN CU ELSE [CU EXCEPT ![x.k] = @ \/ x.unsure]
AddV(VB, x)  == IF x.k = "none" THEN VB ELSE [VB EXCEPT ![x.k] = TRUE]

None3 == [k \in Keyed |-> "no"]
False3 == [k \in Keyed |-> FALSE]
NoLast == [d1 |-> "none", d2 |-> "none", retried |-> FALSE, forced |-> FALSE, disp |-> "none", contOK |-> TRUE]

Init ==
  /\ tid = 1 /\ kind = "original" /\ grant = FALSE /\ editedNoM1 = FALSE
  /\ depth = 0 /\ status = "active" /\ res = 0 /\ nthoughts = 0 /\ last = NoLast
  /\ oldBan = None3 /\ curBan = None3 /\ curUnsure = False3 /\ vb = False3

(* One thought. The retry (main._handle_conscience_retry) runs once, only when the first pass overrode to
   PONDER. A retry that is overridden again dispatches the FIRST pass's PONDER
   (_process_conscience_retry_result, normal mode) -- including when the retry was a guardrail-forced DEFER. *)
Step ==
  /\ status = "active" /\ nthoughts <= ThoughtBound
  /\ \E a1 \in Drafts, v1 \in Answers, a2 \in Drafts, v2 \in Answers :
       LET c1 == Conv(a1)
           p1 == Pass(c1, v1, FALSE, oldBan, curBan, vb)
           CB1 == AddB(curBan, p1.veto)
           CU1 == AddU(curUnsure, p1.veto)
           VB1 == AddV(vb, p1.veto)
           retried == p1.ovr /\ p1.final = "ponder"
           c2 == Conv(a2)
           p2 == Pass(c2, v2, TRUE, oldBan, CB1, VB1)
           disp == IF ~retried THEN p1.final ELSE IF ~p2.ovr THEN p2.final
                   ELSE IF RetryKeepsDefer /\ p2.final \in {"defer", "defer_forced"} THEN p2.final ELSE p1.final
           forced == p1.final = "defer_forced" \/ (retried /\ p2.final = "defer_forced")
           \* #1366 item 8: a dispatched non-DEFER action must not be one already banned in this task
           contOK == disp \in {"ponder", "defer", "defer_forced", "complete"} \/ ~IsBanned(disp, oldBan, CB1)
       IN /\ curBan' = IF retried THEN AddB(CB1, p2.veto) ELSE CB1
          /\ curUnsure' = IF retried THEN AddU(CU1, p2.veto) ELSE CU1
          /\ vb' = IF retried THEN AddV(VB1, p2.veto) ELSE VB1
          /\ last' = [d1 |-> c1, d2 |-> IF retried THEN c2 ELSE "none", retried |-> retried,
                      forced |-> forced, disp |-> disp, contOK |-> contOK]
          /\ nthoughts' = nthoughts + 1
          /\ IF disp \in {"defer", "defer_forced"} THEN status' = "deferred" /\ UNCHANGED depth
             ELSE IF disp = "complete" THEN status' = "done" /\ UNCHANGED depth
             ELSE status' = "active" /\ depth' = Min(depth + 1, DepthCap)
  /\ UNCHANGED <<tid, kind, grant, editedNoM1, res, oldBan>>

(* A WA resolves the deferral (wise_authority/service.py ~1383-1410): a NEW guidance task at depth 0 with
   the WA's grant. Its context is a copy of the deferred task's, so it carries the veto record unless
   popped. Whether the WA supplied edited text without an M-1 justification is the WA's choice. *)
Resolve ==
  /\ status = "deferred" /\ res < MaxRes
  /\ \E ed \in BOOLEAN :
       /\ tid' = tid + 1 /\ kind' = "guidance" /\ grant' = TRUE /\ editedNoM1' = ed
       /\ depth' = 0 /\ status' = "active" /\ nthoughts' = 0 /\ res' = res + 1 /\ last' = NoLast
       /\ oldBan' = IF InheritVetoes THEN [k \in Keyed |-> Max(oldBan[k], curBan[k])] ELSE None3
       /\ curBan' = None3 /\ curUnsure' = False3 /\ vb' = False3

Next == Step \/ Resolve
Spec == Init /\ [][Next]_vars /\ WF_vars(Step)

-----------------------------------------------------------------------------
TypeOK == /\ depth \in 0..DepthCap /\ status \in {"active", "deferred", "done"}
          /\ oldBan \in [Keyed -> Bans] /\ curBan \in [Keyed -> Bans]

(* (a) Termination: every task reaches DEFER, REJECT or TASK_COMPLETE within ThoughtBound thoughts, on every
   model answer. An invariant, so a task that does not close is a finite counterexample. *)
Termination == status = "active" => nthoughts < ThoughtBound

(* (e) Chains bounded per WA resolution: the guidance task a resolution opens closes within ThoughtBound
   thoughts (sh-un-07's loop was here). Each resolution opens exactly one task, so with (a) this bounds the
   thoughts between successive WA decisions. *)
GuidanceTerminates == (kind = "guidance" /\ status = "active") => nthoughts < ThoughtBound

(* sh-un-07's exact path, as a reachability probe (a violation is a witness): the WA resolved with edited text
   and no M-1 justification, the guidance task inherited a substantive DEFER ban, no conscience vetoed a DEFER in
   the guidance task itself, and the task is still active at the bound. *)
NotShUn07 == ~(kind = "guidance" /\ editedNoM1 /\ oldBan["defer"] = "sub" /\ ~vb["defer"]
               /\ status = "active" /\ nthoughts >= ThoughtBound)

(* (b) A drafted DEFER (first pass or retry, incl. the M-1 re-defer) or a forced DEFER is what dispatches. *)
DeferNeverBlocked ==
  (last.d1 = "defer" \/ (last.retried /\ last.d2 = "defer") \/ last.forced) => last.disp \in {"defer", "defer_forced"}

(* (c) After a WA resolution, no veto recorded before it constrains the guidance task. *)
WAOverrule == \A k \in Keyed : ~Eff(oldBan[k])

(* (c') Within a task holding the WA's grant, no ban comes from an objection the conscience did not type
   (the ruling of 2026-11-27: unsure is "justification", which does not ban). *)
WAOverruleInTask == grant => \A k \in Keyed : ~curUnsure[k]

(* (d) #1366 item 8 continuity: a banned non-DEFER action never dispatches unchanged in the same task. *)
Continuity == last.contOK

(* A guardrail-forced DEFER is never discarded for a PONDER (the retry path). *)
ForcedDeferDispatched == last.forced => last.disp \in {"defer", "defer_forced"}

TerminatesLive == [](status = "active" => <>(status # "active"))
=============================================================================
