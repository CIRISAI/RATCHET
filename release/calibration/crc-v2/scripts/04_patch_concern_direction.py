"""Patch per-axis JSON files with explicit corridor-based concern_direction
(closes RATCHET#6; substantively reflects framework's consent-corridor structure).

Three values for concern_direction:
  - at_or_above: only the upper pole triggers (lower extreme is not a failure mode)
  - at_or_below: only the lower pole triggers (upper extreme is not a failure mode)
  - outside_corridor: BOTH poles trigger — consent-corridor exit on either side

For outside_corridor axes in crc-v2: corridor_lower_bound is structurally
documented but uncalibrated against the current corpus (v1 conservatism per
RATCHET#2). crc-v3+ calibrations will set the lower bound when variance
permits.
"""
import json, os

AXES_DIR = "/home/emoore/RATCHET/release/calibration/crc-v2/data/axes"

# Per-axis verdict, derived from framework's corridor structure:
#   both extremes (rigidity pole = too-similar, chaos pole = misaligned) produce
#   fragility; detection should trigger outside the healthy corridor.
AXIS_SPEC = {
    "distributive:access:compute": {
        "concern_direction": "outside_corridor",
        "rigidity_pole": "concentration — one or few agents capture compute access",
        "chaos_pole": "suspicious uniformity — agents in lockstep on identical compute consumption patterns",
        "lower_bound_data_needed": "Cohorts with intentional agent-differentiation where Gini > 0 reflects legitimate role specialization, paired against cohorts in goal-collapse where Gini ≈ 0 reflects identical-behavior lockstep.",
    },
    "distributive:access:models": {
        "concern_direction": "outside_corridor",
        "rigidity_pole": "model monoculture — single-vendor / single-model dominance",
        "chaos_pole": "model fragmentation — no coordination basis; agents not converging on any shared substrate",
        "lower_bound_data_needed": "Cohorts where model diversity reflects legitimate task differentiation, paired against cohorts where each agent picks its own model with no coordination.",
    },
    "distributive:access:federation_membership": {
        "concern_direction": "at_or_above",
        "rationale": "nonmember_frac high = uneven federation (rigidity-pole concern). Low (everyone member) is not a failure mode at this axis — forced-membership is a different signal at a different axis.",
    },
    "correlated_action:rights_asymmetry": {
        "concern_direction": "outside_corridor",
        "rigidity_pole": "PDMA conflicts firing — goal-projection asymmetry between pursuing cohort and affected population",
        "chaos_pole": "zero PDMA conflicts — possible PDMA gaming, bypass, or detector failure",
        "lower_bound_data_needed": "Production observations of cohorts where PDMA actively fires (current corpus shows zero variance — see zero_variance_baseline outcome). Lower bound becomes calibratable when variance accumulates.",
    },
    "correlated_action:participation_exclusion": {
        "concern_direction": "at_or_above",
        "rationale": "below-median-domain-breadth fraction high = subgroup excluded from broad goal-articulation. Low (uniform broad coverage) is not a failure mode at this axis.",
    },
    "correlated_action:informational_asymmetry": {
        "concern_direction": "outside_corridor",
        "rigidity_pole": "CV of flag totals high — some agents see substantially different visibility signals than others",
        "chaos_pole": "CV ≈ 0 — every agent gets identical exposure pattern, possible lockstep behavior or detector saturation",
        "lower_bound_data_needed": "Cohorts where information asymmetry reflects legitimate role differentiation, paired against cohorts where uniform exposure reflects detector saturation.",
    },
    "correlated_action:aggregate_footprint": {
        "concern_direction": "at_or_above",
        "rationale": "actions × log10(N) high = population-scale impact concentrated in cohort. Low (small cohort or low-action cohort) is not a failure mode — these are healthy small-scale cohorts.",
    },
    "distributive:access:agent_capabilities": {
        "concern_direction": "at_or_below",
        "rationale": "cap_diversity low = capability monoculture / lockstep (single-pole concern). Upper bound (high cap_diversity = fragmentation) is structurally meaningful but is provisionally not flagged as failure mode in v1 pending CIRISAgent#880 capability-as-declared signal.",
    },
}

for fn in sorted(os.listdir(AXES_DIR)):
    if not fn.endswith(".json"): continue
    p = f"{AXES_DIR}/{fn}"
    with open(p) as f: d = json.load(f)
    axis_name = d["axis"]
    spec = AXIS_SPEC.get(axis_name)
    if spec is None:
        print(f"  SKIP {fn}: no spec for axis {axis_name}"); continue
    tf = d.get("threshold_function", {})
    direction = spec["concern_direction"]
    tf["concern_direction"] = direction
    pctile = tf.get("threshold_pctile_of_observed")
    threshold_value = tf.get("threshold_value")

    if direction == "outside_corridor":
        tf["corridor"] = {
            "upper_bound": threshold_value,
            "upper_bound_status": "corpus_calibrated",
            "lower_bound": None,
            "lower_bound_status": "structurally_documented_uncalibrated_in_v2",
            "rigidity_pole_concern": spec["rigidity_pole"],
            "chaos_pole_concern": spec["chaos_pole"],
            "lower_bound_calibration_data_needed": spec["lower_bound_data_needed"],
        }
        tf["concern_direction_semantics"] = (
            "Consumer emits negative-scored attestation when metric is at-or-above corridor.upper_bound. "
            "Lower bound is structurally documented but uncalibrated in crc-v2 — see corridor.lower_bound_calibration_data_needed. "
            "crc-v3+ will set corridor.lower_bound when corpus variance permits; until then, lower-pole concern is not actively flagged."
        )
    elif direction == "at_or_above":
        tf["concern_direction_semantics"] = (
            "Consumer emits negative-scored attestation when metric is at-or-above threshold_value. "
            "Single-pole concern: low values at this axis are not a failure mode (see rationale)."
        )
        tf["rationale"] = spec["rationale"]
    elif direction == "at_or_below":
        tf["concern_direction_semantics"] = (
            "Consumer emits negative-scored attestation when metric is at-or-below threshold_value. "
            "Single-pole concern: high values at this axis are not currently flagged (see rationale)."
        )
        tf["rationale"] = spec["rationale"]

    with open(p, "w") as f: json.dump(d, f, indent=2)
    pole_label = direction if direction != "outside_corridor" else "outside_corridor (upper calibrated, lower structurally documented)"
    print(f"  {fn}: pctile={pctile} → concern_direction={pole_label}")

print("\nphilosophy summary for bundle top-level:")
print(f"  outside_corridor: {sum(1 for s in AXIS_SPEC.values() if s['concern_direction']=='outside_corridor')} axes")
print(f"  at_or_above only: {sum(1 for s in AXIS_SPEC.values() if s['concern_direction']=='at_or_above')} axes")
print(f"  at_or_below only: {sum(1 for s in AXIS_SPEC.values() if s['concern_direction']=='at_or_below')} axes")
