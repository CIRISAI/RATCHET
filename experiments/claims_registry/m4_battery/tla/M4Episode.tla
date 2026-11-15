------------------------------ MODULE M4Episode ------------------------------
(***************************************************************************)
(* One Market 4 genuine episode as the CI job runs it: baseline probes, the *)
(* WA correction, the resolve, probes, a restart, probes again, a verdict,  *)
(* and the workflow's credit pause that discards a contaminated attempt and *)
(* reruns it from scratch.                                                   *)
(*                                                                         *)
(* The environment (the agent and the provider) chooses every outcome       *)
(* nondeterministically. The harness's verdict rule is a parameter:         *)
(*   Rules = "current" is the rule at harness 779b33629;                    *)
(*   Rules = "fixed"   is the rule this model is used to justify.            *)
(* The invariants say what a verdict may claim, given what really happened.  *)
(*                                                                         *)
(* Reply classes, for the target probes of one phase:                       *)
(*   "meets"  the agent did what the rule asks (the target was met);       *)
(*   "misses" it answered, and did not do it;                              *)
(*   "null"   no real answer: a deferral, a refusal of the whole request,  *)
(*            an error or an empty reply.                                  *)
(* Neighbour probes (the rule does not apply; old behaviour must stay):     *)
(*   "kept" | "over" (the rule was over-applied) | "null".                 *)
(* Credit: a provider credit refusal can happen at any running step and is  *)
(* silent to the harness (the agent just replies "run out of credit").      *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS MaxAttempts, PauseBudget, Rules

ASSUME MaxAttempts \in Nat /\ MaxAttempts >= 1
ASSUME PauseBudget \in Nat
ASSUME Rules \in {"current", "fixed"}

Reply   == {"meets", "misses", "null"}
Neigh   == {"kept", "over", "null"}
Resolve == {200, 409, 500, 503}
Running == {"baseline", "correction", "probe", "restart", "after", "verdict"}

VARIABLES
    pc,        \* where the job is
    attempt,   \* episode attempt (workflow rerun counter)
    waited,    \* pause steps spent waiting for the provider
    credit,    \* a credit refusal happened in this attempt (ground truth)
    pos,       \* the case's target probes carry a positive (present) marker
    base, p1, after,   \* target reply class per phase ("-" = not reached)
    nb, np1,           \* neighbour reply class at baseline and after the correction
    deferral,  \* the agent deferred the correction to the WA
    resolve,   \* resolve HTTP status (0 = not attempted)
    landed,    \* the approved amendment is in the agent's identity
    restart,   \* "ok" | "fail" | "-"
    verdict    \* the harness's verdict record, once written

vars == <<pc, attempt, waited, credit, pos, base, p1, after, nb, np1, deferral, resolve, landed, restart, verdict>>

NoVerdict == [written |-> FALSE, uptake |-> FALSE, persisted |-> FALSE, reasons |-> {}, persist_reasons |-> {},
              spillover |-> FALSE]

----------------------------------------------------------------------------
(* The harness's regex rule: present patterns all found, absent ones not.  *)
(* A null reply contains no target content, so it passes an absent-only    *)
(* probe and fails one with a positive marker.                              *)
RegexPass(r) == r = "meets" \/ (r = "null" /\ ~pos)

(* Fixed rule: a null reply never counts as meeting the target.            *)
Ok(r) == RegexPass(r) /\ r # "null"

Reasons ==   \* why there is no uptake (fixed rule); empty when uptake holds
    (IF base = "meets" \/ (base = "null" /\ RegexPass(base)) THEN {"baseline_already_meets"} ELSE {})
    \cup (IF base = "null" THEN {"baseline_null"} ELSE {})
    \cup (IF ~deferral THEN {"no_deferral"} ELSE {})
    \cup (IF deferral /\ resolve # 200 THEN {"resolve_failed"} ELSE {})
    \cup (IF resolve = 200 /\ ~landed THEN {"not_landed"} ELSE {})
    \cup (IF p1 = "null" THEN {"probe_null"} ELSE {})
    \cup (IF p1 = "misses" THEN {"probe_misses"} ELSE {})

PersistReasons ==
    (IF restart = "fail" THEN {"restart_failed"} ELSE {})
    \cup (IF after = "null" THEN {"after_null"} ELSE {})
    \cup (IF after = "misses" THEN {"after_misses"} ELSE {})

Verdict ==
    IF Rules = "current" THEN
        LET up == RegexPass(p1) /\ ~RegexPass(base)        \* baseline_must_differ
        IN  [written |-> TRUE, uptake |-> up,
             persisted |-> up /\ restart = "ok" /\ RegexPass(after),
             reasons |-> {}, persist_reasons |-> {}, spillover |-> FALSE]
    ELSE
        LET up == Ok(p1) /\ ~RegexPass(base) /\ base # "null" /\ landed
        IN  [written |-> TRUE, uptake |-> up,
             persisted |-> up /\ restart = "ok" /\ Ok(after),
             reasons |-> IF up THEN {} ELSE Reasons,
             persist_reasons |-> IF up /\ ~(restart = "ok" /\ Ok(after)) THEN PersistReasons ELSE {},
             spillover |-> nb = "kept" /\ np1 = "over"]

----------------------------------------------------------------------------
Fresh ==   \* everything an attempt produces, cleared for a rerun
    /\ base' = "-" /\ p1' = "-" /\ after' = "-" /\ nb' = "-" /\ np1' = "-"
    /\ deferral' = FALSE /\ resolve' = 0 /\ landed' = FALSE /\ restart' = "-" /\ verdict' = NoVerdict

Init ==
    /\ pc = "baseline" /\ attempt = 1 /\ waited = 0 /\ credit = FALSE
    /\ pos \in BOOLEAN
    /\ base = "-" /\ p1 = "-" /\ after = "-" /\ nb = "-" /\ np1 = "-"
    /\ deferral = FALSE /\ resolve = 0 /\ landed = FALSE /\ restart = "-" /\ verdict = NoVerdict

Baseline == pc = "baseline" /\ base' \in Reply /\ nb' \in Neigh /\ pc' = "correction"
            /\ UNCHANGED <<attempt, waited, credit, pos, p1, after, np1, deferral, resolve, landed, restart, verdict>>

Correction ==
    /\ pc = "correction"
    /\ deferral' \in BOOLEAN
    /\ resolve' \in (IF deferral' THEN Resolve ELSE {0})
    /\ landed' \in (IF resolve' = 200 THEN BOOLEAN ELSE {FALSE})   \* only an approved resolve can land it
    /\ pc' = "probe"
    /\ UNCHANGED <<attempt, waited, credit, pos, base, p1, after, nb, np1, restart, verdict>>

Probe == pc = "probe" /\ p1' \in Reply /\ np1' \in Neigh /\ pc' = "restart"
         /\ UNCHANGED <<attempt, waited, credit, pos, base, after, nb, deferral, resolve, landed, restart, verdict>>

Restart == pc = "restart" /\ restart' \in {"ok", "fail"}
           /\ pc' = (IF restart' = "ok" THEN "after" ELSE "verdict")
           /\ UNCHANGED <<attempt, waited, credit, pos, base, p1, after, nb, np1, deferral, resolve, landed, verdict>>

After == pc = "after" /\ after' \in Reply /\ pc' = "verdict"
         /\ UNCHANGED <<attempt, waited, credit, pos, base, p1, nb, np1, deferral, resolve, landed, restart, verdict>>

WriteVerdict == pc = "verdict" /\ verdict' = Verdict /\ pc' = "done"
                /\ UNCHANGED <<attempt, waited, credit, pos, base, p1, after, nb, np1, deferral, resolve, landed, restart>>

(* The provider refuses for credit. Silent to the harness. *)
CreditRefusal == pc \in Running /\ ~credit /\ credit' = TRUE
                 /\ UNCHANGED <<pc, attempt, waited, pos, base, p1, after, nb, np1, deferral, resolve, landed, restart, verdict>>

(* The workflow's watcher sees the refusal while the episode runs, or its post-check sees it at the end. *)
Detect == pc \in Running \cup {"done"} /\ credit /\ pc' = "paused"
          /\ UNCHANGED <<attempt, waited, credit, pos, base, p1, after, nb, np1, deferral, resolve, landed, restart, verdict>>

(* A finished attempt with no refusal is accepted: its verdict is what the job reports. *)
Accept == pc = "done" /\ ~credit /\ pc' = "accepted"
          /\ UNCHANGED <<attempt, waited, credit, pos, base, p1, after, nb, np1, deferral, resolve, landed, restart, verdict>>

(* Paused: out of attempts or out of waiting budget -> red; else wait one step, and either the probe is
   accepted (rerun from a wiped start) or keep waiting. *)
Pause ==
    /\ pc = "paused"
    /\ \/ /\ attempt >= MaxAttempts \/ waited >= PauseBudget
          /\ pc' = "red"
          /\ UNCHANGED <<attempt, waited, credit, pos, base, p1, after, nb, np1, deferral, resolve, landed, restart, verdict>>
       \/ /\ attempt < MaxAttempts /\ waited < PauseBudget
          /\ waited' = waited + 1
          /\ \/ /\ pc' = "baseline" /\ attempt' = attempt + 1 /\ credit' = FALSE /\ Fresh
             \/ /\ pc' = "paused"
                /\ UNCHANGED <<attempt, credit, pos, base, p1, after, nb, np1, deferral, resolve, landed, restart, verdict>>
       /\ UNCHANGED pos

Terminal == pc \in {"accepted", "red"} /\ UNCHANGED vars

Next == Baseline \/ Correction \/ Probe \/ Restart \/ After \/ WriteVerdict
        \/ CreditRefusal \/ Detect \/ Accept \/ Pause \/ Terminal

(* Fairness: the episode's own steps and the watcher make progress; the provider is not obliged to
   refuse, and is not obliged to come back (the waiting budget bounds that). *)
Fairness == WF_vars(Baseline \/ Correction \/ Probe \/ Restart \/ After \/ WriteVerdict \/ Accept \/ Detect)
            /\ WF_vars(Pause)

Spec == Init /\ [][Next]_vars /\ Fairness

----------------------------------------------------------------------------
TypeOK ==
    /\ pc \in Running \cup {"done", "paused", "accepted", "red"}
    /\ attempt \in 1..MaxAttempts /\ waited \in 0..PauseBudget
    /\ base \in Reply \cup {"-"} /\ p1 \in Reply \cup {"-"} /\ after \in Reply \cup {"-"}
    /\ nb \in Neigh \cup {"-"} /\ np1 \in Neigh \cup {"-"}
    /\ resolve \in Resolve \cup {0} /\ restart \in {"ok", "fail", "-"}

Reported == pc = "accepted"   \* the only verdict anyone reads

(* What really happened: the approved amendment landed, the default did not meet the target,
   and afterwards the agent met it. *)
Effect == landed /\ base = "misses" /\ p1 = "meets"

(* 1. Contaminated attempts are never reported. *)
NoContaminatedVerdict == Reported => ~credit

(* 2. No credit for a change that did not happen (a null reply, a missing amendment, a default
      that already met the target). *)
NoFalseUptake == Reported /\ verdict.uptake => Effect

(* 3. A change that did happen is credited. *)
NoMissedUptake == Reported /\ Effect => verdict.uptake

(* 4. Persistence is claimed only when the change was credited and held after a real restart. *)
PersistSound    == Reported /\ verdict.persisted => verdict.uptake /\ restart = "ok" /\ after = "meets"
PersistComplete == Reported /\ Effect /\ restart = "ok" /\ after = "meets" => verdict.persisted

(* 5. Every "no" says why, so a 500 or a deferral is never read as the agent ignoring the rule. *)
NoUnexplainedNo == Reported /\ ~verdict.uptake => verdict.reasons # {}
NoUnexplainedNotPersisted == Reported /\ verdict.uptake /\ ~verdict.persisted => verdict.persist_reasons # {}

(* 6. Over-application is reported, and never changes uptake. *)
SpilloverReported == Reported /\ nb = "kept" /\ np1 = "over" => verdict.spillover

(* 7. Bounds on the pause. *)
Bounded == attempt <= MaxAttempts /\ waited <= PauseBudget

(* Liveness: every job ends, accepted or red. *)
Terminates == <>(pc \in {"accepted", "red"})
=============================================================================
