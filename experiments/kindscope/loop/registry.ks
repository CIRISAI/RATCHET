# Kindscope misfit registry — the source of truth (MISFITS.md is rendered from it).
# `applies` binds a stake; `requires` names engine obligations the bound stake's verb must meet; the auditor refuses a
# stake whose verb lacks one. `signature` classifies a missed prediction as an instance of the misfit.
# Status follows CIRISConstitution CC 2.4 row acts: candidate (a statement) -> registered (authorised by its
# test) | recanted (never right; the row stays) | replaced (superseded by a refined id; the row stays).
# Holon ids are ADOPTED from CIRISHolon conformance/gravity/MISFITS.md at 8477c00 (rule text there).

misfit KS-DIRECT-ONLY
  line: a write read straight through ln_f and the head explains only the last block (mlp0 real 4.11 nats vs direct 0.08)
  rule: every causal or attribution claim about a non-final block is measured through the full downstream network
  applies: verb in {attribute, amnesic, patch, steer}
  requires: downstream
  born: CAL-1 2026-11-11
  status: registered

misfit M-PLANT-OBS
  line: the pipeline must fire on a planted, known-carried label on THIS carrier before it may report absence
  rule: every stake runs a plant on its own state and site first; a plant that does not fire VOIDs the stake
  applies: verb in {probe, amnesic, patch, steer, attribute}
  requires: plant
  born: holon 2026-08-30
  status: adopted

misfit M-JOINT-PASS-REGION
  line: a conjunction of bars must be shown satisfiable on the carrier before it is staked
  rule: the plant must pass the WHOLE conjunction the stake reads (V1 and V2 and V3), not one bar of it
  applies: verb in {probe}
  requires: plant_joint
  born: holon 2026-08-30
  status: adopted

misfit M-BASE-RATE-OMITTED
  line: an accuracy without its population's base rate is an enrichment claim with its denominator deleted
  rule: every accuracy record carries the majority-class rate and a permutation null beside it
  applies: verb in {probe}
  requires: base_rate, null
  born: holon 2026-08-30
  status: adopted

misfit M-TAG-AS-PROPERTY
  line: a verdict a relabelling reproduces is a lookup wearing a measurement's clothes
  rule: every probe carries the type-consistent random-label control task
  applies: verb in {probe}
  requires: control_task
  born: holon 2026-08-30
  status: adopted

misfit M-EXIT-DISCRIMINATOR
  line: a fit record keeping only its residual cannot tell an iteration cap from convergence
  rule: every fit records its iteration count and why it stopped, and the auditor reads it
  applies: verb in {probe, amnesic}
  requires: exit_reason
  born: holon 2026-08-30
  status: adopted

misfit M-DEVICE-CLASS
  line: the device class is part of the artifact; determinism holds only within one class
  rule: every record carries device, GPU model and dtype
  applies: verb in {probe, amnesic, patch, steer, attribute}
  requires: device_class
  born: holon 2026-08-30
  status: adopted

misfit M-FLOOR-UNSTAKED
  line: a clause reading a difference fires on arithmetic when the difference has no measured floor
  rule: every difference a clause reads has an identity-intervention floor on the same instrument
  applies: verb in {amnesic, patch, steer}
  requires: floor
  born: holon 2026-09-04
  status: adopted

misfit M-ONE-MODEL-DELTA
  line: beating one chosen baseline earns "beats that baseline", nothing wider
  rule: a V3 pass is reported as "beyond the declared surface baseline", naming it
  applies: verb in {probe}
  requires: cite
  born: holon 2026-08-30
  status: adopted

# --- adopted from CIRISHolon (cirisholon-d7, 2026-11-11 message): obligations a generator can check mechanically
misfit H-POWER
  line: a stake that cannot produce both outcomes at the real sample size is foregone, whatever it reads
  rule: a null at the real sample size (the plant label on uninjected features) must return the OTHER outcome
  applies: verb in {probe}
  requires: power
  born: holon P-NULL-YEAR 2026-11-11
  status: adopted

misfit H-PLANT-BY-CONSTRUCTION
  line: a plant injected where the stake reads it recovers its label by construction
  rule: the plant's carrier is generated one state below the read and passes through the model to it
  applies: verb in {probe}
  requires: plant_not_by_construction
  born: holon 2026-11-11
  status: adopted

# --- adopted from the Gemma hand-over (cirisagent-fd, 2026-11-11 message); evidence and numbers are theirs
misfit G-LENS-IS-NOT-THE-PATH
  line: an early-layer readout through the final head is a different reader, not a view of what the head will say (Priorities 0.84 at L28, 0.02 at L29; skipping the block restored nothing)
  rule: a lens flip between two states is not evidence of a removable mechanism; test it by intervention through the network
  applies: verb in {lens}
  requires: downstream
  born: Gemma hand-over 2026-11-11
  status: adopted

misfit G-REBUILT-AFTER-REMOVAL
  line: one-shot removal of a subspace under-reports anything later layers recompute (more than half back within 4 layers)
  rule: a removal verb measures re-emergence downstream and admits removals in order, not marginally
  applies: verb in {amnesic}
  requires: reemergence
  born: Gemma hand-over 2026-11-11
  status: adopted

misfit G-SMALLEST-PASSING-STEP
  line: the smallest change passing a float gate did not survive 4-bit quantisation (Premises 0/12 twice)
  rule: an edit is gated on the value as it will be written (dtype and quantisation of the artifact)
  applies: verb in {fork-delta, edit}
  requires: written_value_gate
  born: Gemma hand-over 2026-11-11
  status: adopted

misfit G-DEFAULT-NAME
  line: before the head can read anything, one name is picked for 66-100% of items, so a high recall was partly a leftover default
  rule: every recall or accuracy carries precision and the per-class prediction rate beside it
  applies: verb in {lens, readout}
  requires: precision
  born: Gemma hand-over 2026-11-11
  status: adopted

misfit G-STYLE-BOUND-READER
  line: readers fit on curated items agreed only 29-58% on real revisions; held-out from the same authoring process is not transfer
  rule: a transfer claim needs items from a different authoring process than the fit
  applies: verb in {probe} & noun in {kind}
  requires: cross_process
  born: Gemma hand-over 2026-11-11
  status: adopted

misfit G-SECOND-READ
  line: a test set read twice is not a test; a per-step hold against the previous file lets a kind lose one item per step
  rule: proposer stakes read EXPLORE only; diagnosis tests read CONFIRM only; reads per split are counted
  applies: verb in {probe}
  requires: split
  born: Gemma hand-over 2026-11-11
  status: adopted

misfit G-RECALIBRATION-CONTROL
  line: an output-row edit on a tied embedding was matched by a 12x12 readout map on the same split (+0.024 [-0.028, +0.077])
  rule: a fork-delta carries the cheapest readout-only fix on the same split, that fix refitted on the fork, a paired interval on fork minus fix, and the output-only prediction
  applies: verb in {fork-delta}
  requires: readout_control
  born: Gemma hand-over 2026-11-11
  status: adopted

# --- born 2026-11-11 in this series
misfit KS-EXACT-ZERO-ACROSS-BATCH
  line: an "exactly 0" bar on a comparison whose two arms ran in different batch shapes reads float summation-order noise (MLP0-2: self-patch deltas ~1e-9, not 0, so the plant failed by its letter)
  rule: arms compared against zero run in the identical batch layout, or the bar is "within the self-donor floor", never "exactly 0"
  applies: verb in {patch, steer, edit}
  requires: same_batch_or_floor
  born: MLP0-2 2026-11-11
  status: registered

misfit KS-READ-NOT-WRITE
  line: the direction where a category is readable is not the direction that writes it (EDIT-1 exploration: reading vs pulled-back writing directions 84-89 degrees apart at every MLP; probe-direction rank-one edits UNREACHABLE at 200x write norm)
  rule: an edit direction is pulled back from the destination through the network (gradient or Jacobian), never taken from a probe; report the parallax angle beside every edit
  applies: verb in {steer, edit}
  requires: pulled_back_direction
  born: EDIT-1 2026-11-11
  status: candidate

misfit KS-READOUT-GRAIN
  line: a readout-only fix cannot reproduce a change whose category is coarser than the output unit (EDIT-1: byte-level readout map vs full fine-tune on adjectives-after-determiners, A2-A1 +0.29 [0.22, 0.37]; Gemma's single-token answer let a readout match)
  rule: before staking a readout control as sufficient, state the output unit and the category's grain; a category spanning several output units needs an inside-the-network arm
  applies: verb in {fork-delta, edit}
  requires: grain_declared
  born: EDIT-1 P1 2026-11-11
  status: candidate

misfit KS-FORCE-AT-WRONG-POINT
  line: a restoring force read as the base model's loss gradient AT the base weights is the pull of a converged model — near zero, mostly batch noise — so any change sits near 90 degrees to it (EDIT-2: 91-93 degrees in every tensor, read by the author as "slightly against the grain"; reviewer cirisagent-fd, 2026-11-12)
  rule: the pull opposing a change delta is the ordinary-loss gradient at theta + delta (about H delta); report its angle to delta and delta^T H delta per tensor, and call it restoring force; the agent's TORQUE is output-space deflection (share of the base-to-target log-odds gap closed per item) and keeps that name
  applies: verb in {fork-delta, edit, tune}
  requires: force_at_displaced_point
  born: EDIT-2 2026-11-12 (author ratchet-b9; found by reviewer cirisagent-fd)
  status: registered

misfit KS-READ-NOT-WRITE-EARLY
  line: refines the recanted KS-READ-NOT-WRITE — at mlp0-mlp2 a pulled-back writing direction moves behaviour where the probe (reading) direction does not (EDIT-2 at half write norm: +0.08, +0.07, +0.10 over reading, CIs exclude 0); in stiff late layers the same first-order direction overshoots and harms (mlp5 -0.56)
  rule: an edit direction is pulled back from the destination, never taken from a probe, AND its size is kept inside the range where the first-order direction holds (measured per layer); report the parallax angle against the random-direction band beside every edit
  applies: verb in {steer, edit}
  requires: pulled_back_direction, linear_range
  replaces: KS-READ-NOT-WRITE
  born: EDIT-2 2026-11-12
  status: candidate

misfit KS-CASEFOLD-DONOR
  line: a byte-exact plant assumed identical raw bytes from donors matched on lower-cased word forms (MLP0-3: 2 of 71,195 suffix donors had a differently cased last byte, so the "exactly within floor" plant failed by its letter)
  rule: donors for a byte-level plant are matched on raw bytes, and the plant's exactness assumption is checked over every job before the run reads anything
  applies: verb in {patch}
  requires: raw_byte_match
  born: MLP0-3 2026-11-12
  status: registered

misfit KS-CONTROL-BASE-RATE
  line: V2 compares the probe's raw accuracy with the control task's raw accuracy, but type-level random labels at the class prior give a token-level positive rate that depends on which frequent types drew a 1; for a rare one-vs-rest class the control's own base rate can exceed the real one, so V2 fails without any evidence about the category (S0013: acc 0.942, base 0.926, control max 0.967)
  rule: compare excess over each label set's own majority rate, (acc - base) / (1 - base), for the real and the control fits, and record the control's token-level base rate beside it
  applies: verb in {probe}
  requires: control_base_rate
  signature: verb=probe & V1=true & V2=false & noun!=pos10
  born: D0001 2026-11-12
  status: candidate
  rests_on: S0012, S0013, S0014, S0015, S0016, S0017, S0018; engine 13274181efbdb8f2
