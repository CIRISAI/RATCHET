"""Pass C: write release/calibration/crc-v2/bundle.yaml with honest calibration outcomes."""
import json, os, hashlib, gzip

AXES_DIR = "/home/emoore/RATCHET/release/calibration/crc-v2/data/axes"
COHORT_FILE = "/home/emoore/RATCHET/release/calibration/crc-v2/data/cohorts.jsonl"
METRICS_FILE = "/home/emoore/RATCHET/release/calibration/crc-v2/data/cohort_metrics.json"
OUT = "/home/emoore/RATCHET/release/calibration/crc-v2"

# Load axes
axes = {}
for f in os.listdir(AXES_DIR):
    if f.endswith(".json"):
        with open(f"{AXES_DIR}/{f}") as fh:
            d = json.load(fh)
            axes[d["axis"]] = d

# Load cohort metrics
with open(METRICS_FILE) as fh: per_cohort = json.load(fh)

# Trace corpus sha256
DUMP = "/home/emoore/0612_prod_traces"
def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""): h.update(chunk)
    return h.hexdigest()
trace_events_sha = file_sha(f"{DUMP}/trace_events.jsonl.gz")
trace_llm_calls_sha = file_sha(f"{DUMP}/trace_llm_calls.jsonl.gz")
accord_traces_sha = file_sha(f"{DUMP}/accord_traces.jsonl.gz")

# k_eff interpretation per axis (CCA cross-reference)
keff_interp = {
    "distributive:access:compute": "k_eff_compute ≈ N·(1−Gini²) — at threshold Gini=0.17 with N=15-agent cohort, k_eff≈14.6 (healthy diversity); v1 trigger fires when Gini exceeds 75th-pctile observed.",
    "distributive:access:models": "k_eff_models = 1/HHI — at threshold HHI=1.0, k_eff=1 (collapse); v1 triggers only on full single-model dominance, conservative per RATCHET#2.",
    "distributive:access:federation_membership": "k_eff_membership at nonmember_frac=0 is degenerate (everyone in the federation). Threshold set to 'any nonzero deviation' until variance appears in production.",
    "correlated_action:rights_asymmetry": "PDMA-conflict rate is the inverse-coherence signal. Threshold at 'any nonzero rate' until variance appears (current production: all DMA_RESULTS.pdma.has_conflicts = false).",
    "correlated_action:participation_exclusion": "Proxy: fraction below median domain-breadth. At threshold 0.44, ~45% of cohort agents below cohort median — strong but not absolute exclusion signal.",
    "correlated_action:informational_asymmetry": "Proxy: CV of per-agent flag totals. At threshold CV=0.71, substantial within-cohort variance in dsdma flag exposure.",
    "correlated_action:aggregate_footprint": "Proxy: actions × log10(agents). At threshold ~619, cohorts with concentrated population-scale action loads.",
    "distributive:access:agent_capabilities": "Proxy: distinct (cognitive_state, role) tuples per agent. At threshold 0.04 (cap_diversity), very low capability differentiation — concentration on a few capability profiles.",
}

# Handle zero-variance axes specifically
for axis_name in axes:
    a = axes[axis_name]
    if "calibration" in a:
        cal = a["calibration"]
        if cal["observed_min"] == cal["observed_max"] == 0.0:
            a["threshold_function"]["threshold_value"] = 1e-6
            a["threshold_function"]["calibration_outcome"] = "zero_variance_baseline"
            a["known_issue"] = (
                "All qualifying cohorts in the crc-v2 corpus show this metric = 0. "
                "Threshold set to lowest detectable signal (1e-6). When production data accumulates "
                "non-zero observations (predicted: adversarial scenarios or substrate drift), "
                "re-calibration on the new corpus will set an evidence-based threshold."
            )

# Build bundle.yaml dict
bundle = {
    "ratchet_calibration_version": 2,
    "projection_version": "crc-v2",
    "calibrated_at": "2026-06-12T19:00:00+00:00",
    "supersedes": "crc-v1",
    "calibration_corpus": {
        "source": "0612_prod_traces dump",
        "captured_at": "2026-06-12 (afternoon UTC)",
        "trace_events_sha256": trace_events_sha,
        "trace_llm_calls_sha256": trace_llm_calls_sha,
        "accord_traces_sha256": accord_traces_sha,
        "n_trace_events": 51373,
        "n_trace_llm_calls": 97180,
        "n_accord_traces_legacy": 12165,
        "n_distinct_agents": 192,
        "n_qualifying_agents": 168,  # ≥100 events and real agent_id_hash
        "n_qualifying_cohorts": 7,    # ≥12 agents and ≥1000 events
        "agent_filter": "events≥100, agent_id_hash NOT IN {MISSING, unknown, [IDENTIFIER]}",
        "cohort_derivation": "k-means clustering on z-scored behavioral features (log_n_events, pdma_conflict_rate, log_flags_total, n_distinct_models, federation_member, log_days_active, log_cost_per_event, n_distinct_domains), k=11, target cohort size ~15 agents",
        "known_issues": [
            "`unknown` and `[IDENTIFIER]` defaults account for 14.2% of trace_events; excluded from cohort statistics as uncategorizable.",
            "`channel_id` is scrubbed at source (CIRISLens#12 carry-over); cohort delineation uses `agent_id_hash` exclusively.",
            "trace_events: 7,380 / 51,373 rows (14.4%) required regex-recovery from LLM_CALL prompt double-escaping; 318 unrecoverable (0.6%).",
            "trace_llm_calls: 15,790 / 97,180 rows (16.2%) unrecoverable due to embedded JSON-in-JSON; remaining 81,390 used.",
            "`cohort_target_id`, `deployment_region`, `classifications`, `pipeline_metadata` columns are present in schema but 100% null in current substrate (calculated downstream, not stored — see CIRISAgent issues).",
            "`distributive:access:federation_membership` and `correlated_action:rights_asymmetry` show zero variance across all qualifying cohorts; thresholds set to lowest-detectable-signal baselines pending observation of non-zero events.",
        ],
        "schema_versions": ["2.7.0", "2.7.9", "2.7.legacy", "3.0.0"],
    },
    "statistical_floors_canonical": {
        "min_cohort_size_events": 1000,
        "min_goal_aligned_cluster_size_agents": 12,
        "min_window_days": 30,
        "power_target": 0.95,
        "power_basis_reference": "RATCHET README — 95% power at n=109 standard for DetectionEngine",
        "axes_satisfying_floor_at_v1_corpus_median": "7/8 axes meet floor with cohort median 15 agents × ~4400 events × 91-day window",
    },
    "validation": {
        "monte_carlo_method": "Bootstrap percentile CI on cohort metric at 2000 resamples per axis",
        "cca_model_reference": {
            "preprint_doi": "10.5281/zenodo.18217688",
            "model": "k_eff = k / (1 + ρ(k − 1))",
            "axis_keff_interpretations": keff_interp,
        },
    },
    "axes": axes,
    "tier_summary": {
        "tier_1_full_calibration": [
            "distributive:access:compute",
            "distributive:access:models",
            "distributive:access:federation_membership (zero_variance_baseline)",
            "correlated_action:rights_asymmetry (zero_variance_baseline)",
        ],
        "tier_2_proxy_calibration": [
            "correlated_action:participation_exclusion",
            "correlated_action:informational_asymmetry",
            "correlated_action:aggregate_footprint",
            "distributive:access:agent_capabilities",
        ],
        "tier_3_deferred_pending_substrate_emit": [
            "correlated_action:ecology_of_communication:echo_chamber_density",
            "correlated_action:ecology_of_communication:information_silo_correlation",
            "correlated_action:ecology_of_communication:coordinated_messaging_pattern",
            "correlated_action:ecology_of_communication:cross_cohort_information_flow",
            "distributive:access:training_data",
        ],
    },
    "consumer_contract": {
        "version_transition_discipline": "Per CEG §15.2 R2: LensCore deployments emit `evidence_refs[]` carrying both crc-v{N} and crc-v{N+1} hashes during transition windows to defeat straddle attacks.",
        "polarity_convention": "polarity=positive_when_distributed → high values = good (broad access); positive_when_detected → high values = concern (concentrated/asymmetric pattern detected). See per-axis threshold_function.polarity.",
        "concern_direction_convention": {
            "field": "threshold_function.concern_direction",
            "values": {
                "at_or_above": "Trigger when metric >= threshold_value. Lower extreme at this axis is not a failure mode (see per-axis rationale).",
                "at_or_below": "Trigger when metric <= threshold_value. Upper extreme at this axis is not currently flagged (see per-axis rationale).",
                "outside_corridor": "Trigger when metric is OUTSIDE the healthy corridor (either pole). Upper bound is calibrated in this crc version; lower bound is structurally documented but uncalibrated in crc-v2 — see per-axis threshold_function.corridor. crc-v3+ sets lower bound when corpus variance permits.",
            },
            "framework_basis": "Consent-corridor exit produces fragility at both poles per CCA preprint (DOI 10.5281/zenodo.18217688): rigidity pole (rho → 1, 'too similar') and chaos pole (rho → 0, 'misaligned') both push systems out of sustained coordination. Detection should fire on either pole where the corpus supports calibration.",
            "consumer_rule": "Read threshold_function.concern_direction directly. Do NOT infer from threshold_pctile_of_observed — that heuristic worked for crc-v2's 8 axes but will mis-fire on future axes where pctile choice and concern direction decouple. Closes RATCHET#6.",
        },
        "evidence_refs_required_per_attestation": [
            "trace_sample_bundle (hash-pinned)",
            "cohort_delineation (cohort_id and member agent_id_hash list, OR cohort derivation algorithm version)",
            "axis-specific evidence (see per-axis evidence_required)",
        ],
        "sole_evidence_rule": "ratchet:flag:* / detection:* attestations are NEVER sole evidence for slashing:* per CEG §4.6/§4.9. WA quorum is the load-bearing gate.",
    },
}

# Emit YAML (manual; PyYAML may not be available). Use a defensive serializer.
def to_yaml(obj, indent=0, key_quote=False):
    sp = "  " * indent
    if obj is None: return "null"
    if isinstance(obj, bool): return "true" if obj else "false"
    if isinstance(obj, (int, float)): return repr(obj)
    if isinstance(obj, str):
        if "\n" in obj or "'" in obj or '"' in obj or ":" in obj or len(obj) > 80:
            esc = obj.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
            return f'"{esc}"'
        return obj if obj else '""'
    if isinstance(obj, list):
        if not obj: return "[]"
        s = "\n"
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                rendered = to_yaml(item, indent+1)
                # multi-line: place "-" on same line as first key
                if isinstance(item, dict):
                    first_key = next(iter(item.keys()))
                    rest = {k:v for k,v in item.items() if k != first_key}
                    s += f"{sp}- {first_key}: {to_yaml(item[first_key], indent+2)}\n"
                    for k, v in rest.items():
                        s += f"{sp}  {k}: {to_yaml(v, indent+2)}\n"
                else:
                    s += f"{sp}- {rendered}\n"
            else:
                s += f"{sp}- {to_yaml(item, indent+1)}\n"
        return s.rstrip("\n")
    if isinstance(obj, dict):
        if not obj: return "{}"
        s = "\n" if indent > 0 else ""
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                s += f"{sp}{k}:{to_yaml(v, indent+1)}\n"
            else:
                s += f"{sp}{k}: {to_yaml(v, indent+1)}\n"
        return s.rstrip("\n")
    return str(obj)

yaml_text = "# RATCHET crc-v2 calibration bundle — F-3 + distributive axes for CIRISLensCore\n"
yaml_text += "# Generated 2026-06-12 from 0612_prod_traces. See bundle.sha256 for integrity.\n"
yaml_text += "# Consumes: RATCHET#2, RATCHET#3, RATCHET#5. See README.md for derivation pipeline.\n\n"
yaml_text += to_yaml(bundle) + "\n"

with open(f"{OUT}/bundle.yaml", "w") as fh: fh.write(yaml_text)

# sha256 of bundle.yaml
sha = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()
with open(f"{OUT}/bundle.sha256", "w") as fh: fh.write(f"{sha}  bundle.yaml\n")

print(f"wrote {OUT}/bundle.yaml ({len(yaml_text)} bytes)")
print(f"sha256: {sha}")
print(f"axes calibrated: {len(axes)}")
print(f"  tier-1: {sum(1 for a in axes.values() if a.get('tier')==1)}")
print(f"  tier-2: {sum(1 for a in axes.values() if a.get('tier')==2)}")
print(f"  tier-3 deferred: 5 (named, awaiting substrate emit)")
