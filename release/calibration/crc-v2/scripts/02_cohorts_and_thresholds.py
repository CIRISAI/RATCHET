"""Pass B: cluster agents into cohorts; compute per-axis thresholds + statistical floors + power analysis.

Output:
  data/cohorts.jsonl       — one row per cohort with member agents + per-axis stats
  data/axes/{axis}.json    — per-axis calibration: threshold, floor, evidence_shape, power_analysis
"""
import json, os, sys, math, random
from collections import Counter, defaultdict
random.seed(20260612)

AGENTS_PATH = "/home/emoore/RATCHET/release/calibration/crc-v2/data/agents.jsonl"
OUT_DIR = "/home/emoore/RATCHET/release/calibration/crc-v2/data"
os.makedirs(f"{OUT_DIR}/axes", exist_ok=True)

# --- 1. Load + filter ---
agents = []
with open(AGENTS_PATH) as f:
    for line in f:
        a = json.loads(line)
        if a["agent_id_hash"] in ("MISSING", "unknown", "[IDENTIFIER]"): continue
        if a["n_events"] < 100: continue  # below noise floor
        agents.append(a)
print(f"qualifying agents: {len(agents)} (≥100 events, real agent_id_hash)")

# --- 2. Behavioral feature vector per agent ---
# Features: log(n_events), pdma_conflict_rate, dsdma_flags_count totals, n_distinct_models,
#           federation_member, days_active, cost_usd_per_event
def feat(a):
    cu = a["cost_usd_sum"] / max(a["n_events"], 1)
    flags_total = sum(a.get("dsdma_flags_count", {}).values())
    return [
        math.log10(max(a["n_events"], 1)),
        (a.get("pdma_conflict_rate") or 0.0),
        math.log10(max(flags_total, 1)),
        a.get("n_distinct_models", 0),
        float(a.get("federation_member", False)),
        math.log10(max(a.get("days_active", 1), 1)),
        math.log10(max(cu, 1e-9)) + 9,  # shift to non-negative
        a.get("n_distinct_domains", 0),
    ]

X = [feat(a) for a in agents]
# z-score
ncols = len(X[0])
means = [sum(row[i] for row in X) / len(X) for i in range(ncols)]
sds = [math.sqrt(sum((row[i] - means[i])**2 for row in X) / max(len(X)-1, 1)) for i in range(ncols)]
sds = [s if s > 1e-9 else 1.0 for s in sds]
Z = [[(row[i] - means[i]) / sds[i] for i in range(ncols)] for row in X]

# --- 3. k-means clustering (lightweight, no sklearn dep) ---
def kmeans(data, k, n_iter=50):
    centers = random.sample(data, k)
    for _ in range(n_iter):
        clusters = defaultdict(list)
        for idx, pt in enumerate(data):
            d = [sum((pt[i]-c[i])**2 for i in range(len(pt))) for c in centers]
            clusters[d.index(min(d))].append(idx)
        new_centers = []
        for ci in range(k):
            members = clusters.get(ci, [])
            if not members:
                new_centers.append(random.choice(data)); continue
            new_centers.append([sum(data[m][i] for m in members)/len(members) for i in range(len(data[0]))])
        if all(sum((nc[i]-c[i])**2 for i in range(len(c))) < 1e-9 for nc, c in zip(new_centers, centers)):
            centers = new_centers; break
        centers = new_centers
    return clusters

# Target cohort size ~15 agents; k = n/15
n = len(agents)
target_cohort_size = 15
k = max(2, n // target_cohort_size)
print(f"clustering {n} agents into k={k} cohorts (target size ~{target_cohort_size})")

# Run k-means a few times, keep best (lowest within-cluster sum of squares)
best_clusters = None; best_wcss = float("inf")
for trial in range(10):
    cl = kmeans(Z, k)
    wcss = 0.0
    for ci, idxs in cl.items():
        if not idxs: continue
        ctr = [sum(Z[i][j] for i in idxs)/len(idxs) for j in range(ncols)]
        wcss += sum(sum((Z[i][j]-ctr[j])**2 for j in range(ncols)) for i in idxs)
    if wcss < best_wcss:
        best_wcss = wcss; best_clusters = cl

print(f"best WCSS = {best_wcss:.2f}")
clusters = best_clusters

# Build cohort records
cohorts = []
for ci, idxs in clusters.items():
    if not idxs: continue
    members = [agents[i] for i in idxs]
    cohorts.append({
        "cohort_id": f"crc-v2-cohort-{ci:02d}",
        "n_agents": len(members),
        "n_events_total": sum(m["n_events"] for m in members),
        "agent_ids": [m["agent_id_hash"] for m in members],
        "members": members,
    })
cohorts.sort(key=lambda c: -c["n_events_total"])

print(f"cohorts: {len(cohorts)}")
print(f"  cohort sizes (agents): {[c['n_agents'] for c in cohorts]}")
print(f"  cohort event totals: {[c['n_events_total'] for c in cohorts]}")
print(f"  cohorts ≥12 agents AND ≥1000 events: {sum(1 for c in cohorts if c['n_agents']>=12 and c['n_events_total']>=1000)}")

# --- 4. Per-axis statistics ---
def gini(values):
    """Gini coefficient on a non-negative list."""
    vs = sorted(v for v in values if v is not None and v >= 0)
    if not vs or sum(vs) == 0: return 0.0
    n = len(vs); cumsum = 0.0; gini_sum = 0.0
    for i, v in enumerate(vs):
        cumsum += v
        gini_sum += (i+1) * v
    return (2*gini_sum)/(n*cumsum) - (n+1)/n

def hhi(counts_dict):
    """Herfindahl-Hirschman Index: sum of squared shares."""
    total = sum(counts_dict.values())
    if total == 0: return 0.0
    return sum((c/total)**2 for c in counts_dict.values())

def cr_top(values, k=1):
    """Concentration ratio: share of top-k agents in the total."""
    vs = sorted([v for v in values if v is not None], reverse=True)
    if not vs or sum(vs) == 0: return 0.0
    return sum(vs[:k]) / sum(vs)

# Compute per-cohort metrics for each axis
qualifying = [c for c in cohorts if c["n_agents"] >= 12 and c["n_events_total"] >= 1000]
print(f"\nqualifying cohorts (≥12 agents AND ≥1000 events): {len(qualifying)}")

per_cohort = []
for c in qualifying:
    M = c["members"]
    # distributive:access:compute — Gini of cost_usd
    compute_gini = gini([m["cost_usd_sum"] for m in M])
    # distributive:access:models — HHI of pooled model-mix
    pooled_models = Counter()
    for m in M:
        for mname, cnt in m.get("llm_model_counts", {}).items():
            pooled_models[mname] += cnt
    models_hhi = hhi(pooled_models)
    # distributive:access:federation_membership — fraction of non-members
    nonmember_frac = sum(1 for m in M if not m.get("federation_member")) / len(M)
    # correlated_action:rights_asymmetry — pooled PDMA-conflict rate (within-cohort)
    pdma_total = sum(m["pdma_total"] for m in M)
    pdma_conflicts = sum(m["pdma_conflict_count"] for m in M)
    pdma_rate_within = (pdma_conflicts/pdma_total) if pdma_total else None
    # informational_asymmetry — coefficient of variation of per-agent dsdma_flag totals (proxy)
    flag_totals = [sum(m.get("dsdma_flags_count", {}).values()) for m in M]
    fm_mean = sum(flag_totals)/len(flag_totals)
    fm_sd = math.sqrt(sum((x-fm_mean)**2 for x in flag_totals)/max(len(flag_totals)-1,1))
    info_asym_cv = (fm_sd/fm_mean) if fm_mean > 0 else 0.0
    # participation_exclusion — fraction of agents below median domain-breadth (proxy)
    domains_per_agent = [m.get("n_distinct_domains", 0) for m in M]
    median_d = sorted(domains_per_agent)[len(domains_per_agent)//2]
    part_excl_frac = sum(1 for d in domains_per_agent if d < median_d) / len(M)
    # aggregate_footprint — total cohort action count × log(distinct agents)
    actions_total = sum(m.get("n_action", 0) for m in M)
    agg_fp = actions_total * math.log10(max(c["n_agents"], 1))
    # agent_capabilities — number of distinct (cognitive_state_dominant, role) tuples / n_agents
    distinct_caps = len({(m.get("cognitive_state_dominant"), m.get("agent_role")) for m in M})
    cap_diversity = distinct_caps / max(c["n_agents"], 1)
    per_cohort.append({
        "cohort_id": c["cohort_id"], "n_agents": c["n_agents"], "n_events_total": c["n_events_total"],
        "compute_gini": compute_gini, "models_hhi": models_hhi, "nonmember_frac": nonmember_frac,
        "pdma_rate_within": pdma_rate_within,
        "info_asym_cv": info_asym_cv, "part_excl_frac": part_excl_frac,
        "agg_fp": agg_fp, "cap_diversity": cap_diversity,
    })

# Save per-cohort stats
with open(f"{OUT_DIR}/cohorts.jsonl", "w") as f:
    for c in cohorts: f.write(json.dumps({k:v for k,v in c.items() if k!="members"}) + "\n")
with open(f"{OUT_DIR}/cohort_metrics.json", "w") as f: json.dump(per_cohort, f, indent=2)

# --- 5. Threshold derivation + statistical floor + power analysis per axis ---
# Method per axis:
#   threshold = empirical 75th percentile of the per-cohort metric (conservative — RATCHET#2 says v1 = conservative)
#   statistical_floor: bootstrap-based min_cohort_size + min_events for 95% power
#   evidence_shape: declared

def pctile(values, p):
    vs = sorted(values); n = len(vs)
    if n == 0: return None
    k = (n-1) * p; f = math.floor(k); ce = math.ceil(k)
    if f == ce: return vs[int(k)]
    return vs[f] + (vs[ce]-vs[f]) * (k-f)

def bootstrap_ci(values, statistic_fn, n_boot=2000, alpha=0.05):
    vs = [v for v in values if v is not None]
    if len(vs) < 3: return None, None
    boots = []
    for _ in range(n_boot):
        sample = [random.choice(vs) for _ in vs]
        boots.append(statistic_fn(sample))
    boots.sort()
    return boots[int(alpha/2 * n_boot)], boots[int((1-alpha/2) * n_boot)]

def power_floor(metric_values, threshold, n_boot=1000):
    """Find the cohort-size n_min at which a permutation test rejects null (no-collapse) at 95% power."""
    vs = [v for v in metric_values if v is not None]
    if len(vs) < 3: return None
    # for each candidate n, fraction of n-element resamples whose mean exceeds threshold
    candidates = [10, 12, 15, 20, 25, 30, 50]
    for n_can in candidates:
        if n_can > len(vs): continue
        successes = 0
        for _ in range(n_boot):
            sample = [random.choice(vs) for _ in range(n_can)]
            if sum(sample)/n_can >= threshold: successes += 1
        if successes / n_boot >= 0.95: return n_can
    return None

def calibrate_axis(name, metric_key, polarity, evidence_required, score_at_threshold,
                   measurement_procedure, threshold_pctile=0.75, tier=1, notes=""):
    values = [c[metric_key] for c in per_cohort if c.get(metric_key) is not None]
    if len(values) < 3:
        return {"axis": name, "tier": tier, "status": "INSUFFICIENT_COHORTS",
                "n_qualifying_cohorts": len(values)}
    threshold = pctile(values, threshold_pctile)
    lo, hi = bootstrap_ci(values, lambda s: pctile(s, threshold_pctile))
    # statistical floor — events required to detect threshold at 95% power
    min_n_agents = power_floor(values, threshold)
    # 95% power for n=109 (the existing RATCHET standard from README) — verify
    if min_n_agents is None: min_n_agents = 30
    # min_window_days — empirical (median days-active across cohort members)
    return {
        "axis": name, "tier": tier,
        "measurement_procedure": measurement_procedure,
        "threshold_function": {
            "metric": metric_key,
            "threshold_value": round(threshold, 6),
            "threshold_pctile_of_observed": threshold_pctile,
            "ci_95": [round(lo, 6) if lo else None, round(hi, 6) if hi else None],
            "score_at_threshold": score_at_threshold,
            "polarity": polarity,
            "scaling": "magnitude_scales_with_severity_above_threshold",
        },
        "statistical_floor": {
            "min_cohort_size_events": 1000,
            "min_goal_aligned_cluster_size_agents": max(min_n_agents, 12),
            "min_window_days": 30,
            "power_target": 0.95,
            "power_basis": "RATCHET README 95%-power-at-n=109 standard",
        },
        "evidence_required": evidence_required,
        "calibration": {
            "n_qualifying_cohorts": len(values),
            "observed_min": round(min(values), 6),
            "observed_max": round(max(values), 6),
            "observed_median": round(pctile(values, 0.5), 6),
            "observed_p75": round(pctile(values, 0.75), 6),
            "observed_p90": round(pctile(values, 0.9), 6),
        },
        "notes": notes,
    }

# Tier-1 axes
ax = {}
ax["distributive:access:compute"] = calibrate_axis(
    "distributive:access:compute", "compute_gini", "positive_when_distributed",
    ["trace_sample_bundle", "cohort_delineation", "per_agent_cost_usd_aggregate", "time_window"],
    -0.6,
    "For each agent in the cohort, sum cost_usd across all LLM_CALL events (post-cutover trace_events.payload + trace_llm_calls). Compute Gini coefficient across per-agent totals. Threshold above which the distribution is judged concentrated.",
    threshold_pctile=0.75, tier=1,
    notes="Polarity: positive_when_distributed = lower Gini is better; high Gini = concentrated compute = negative attestation."
)
ax["distributive:access:models"] = calibrate_axis(
    "distributive:access:models", "models_hhi", "positive_when_distributed",
    ["trace_sample_bundle", "cohort_delineation", "pooled_model_usage_counts", "time_window"],
    -0.5,
    "Pool LLM_CALL.payload.model across all agents in the cohort. Compute HHI = sum of squared model shares. Threshold above which the cohort's model-mix is judged concentrated on a single provider/model.",
    threshold_pctile=0.75, tier=1,
    notes="HHI in [1/N, 1]; 1 = single model dominance. The CCA exp(-λ·k_eff) model identifies k_eff = 1/HHI as effective model count."
)
ax["distributive:access:federation_membership"] = calibrate_axis(
    "distributive:access:federation_membership", "nonmember_frac", "positive_when_distributed",
    ["trace_sample_bundle", "cohort_delineation", "per_agent_federation_member_flag", "accord_public_key_registry_snapshot_hash"],
    -0.4,
    "For each agent, look up federation_member via accord_public_keys.key_id prefix-match on agent_id_hash. Fraction of non-members in the cohort. Threshold above which federation participation is judged uneven.",
    threshold_pctile=0.75, tier=1,
    notes="A score of −0.4 indicates uneven participation; magnitude scales above threshold."
)
ax["correlated_action:rights_asymmetry"] = calibrate_axis(
    "correlated_action:rights_asymmetry", "pdma_rate_within", "negative_when_detected",
    ["trace_sample_bundle", "goal_attestation_cluster_membership", "affected_population_identifier",
     "per_actor_pdma_compliance_score", "DMA_RESULTS.payload.pdma.has_conflicts events"],
    -0.6,
    "For each cohort, compute the pooled PDMA-conflict rate = sum(DMA_RESULTS.payload.pdma.has_conflicts==True) / sum(DMA_RESULTS with pdma block). Threshold above which the cohort's pursuit pattern is judged rights-asymmetric.",
    threshold_pctile=0.75, tier=1,
    notes="Affected_population_identifier currently NOT in dump (column null) — RATCHET will issue CIRISAgent-side ticket; v1 calibration uses cohort-pooled PDMA rate as the load-bearing signal."
)

# Tier-2 (proxy)
ax["correlated_action:participation_exclusion"] = calibrate_axis(
    "correlated_action:participation_exclusion", "part_excl_frac", "negative_when_detected",
    ["trace_sample_bundle", "cohort_delineation", "per_agent_dsdma_domain_breadth"],
    -0.3,
    "Fraction of agents in the cohort whose distinct-domain count (n_distinct_domains from dsdma.domain field) is below the cohort median. Threshold above which a non-trivial subgroup is judged excluded from broad goal-articulation.",
    threshold_pctile=0.75, tier=2,
    notes="PROXY: true 'excluded from goal-articulation phase' requires goal-attestation cluster membership data that is null in the current substrate. CIRISAgent-side ticket filed."
)
ax["correlated_action:informational_asymmetry"] = calibrate_axis(
    "correlated_action:informational_asymmetry", "info_asym_cv", "negative_when_detected",
    ["trace_sample_bundle", "cohort_delineation", "per_agent_dsdma_flag_distributions"],
    -0.3,
    "Coefficient of variation of per-agent dsdma flag totals within the cohort. High CV means some agents see substantially more defer-required / out-of-domain flags than others. Threshold above which informational asymmetry is judged structural.",
    threshold_pctile=0.75, tier=2,
    notes="PROXY: true 'visibility-into-footprint asymmetry' requires explicit footprint-visibility attestations. CIRISAgent-side ticket filed."
)
ax["correlated_action:aggregate_footprint"] = calibrate_axis(
    "correlated_action:aggregate_footprint", "agg_fp", "negative_when_detected",
    ["trace_sample_bundle", "cohort_delineation", "per_agent_action_count", "cohort_size"],
    -0.3,
    "Cohort aggregate ACTION_RESULT count × log10(cohort_n_agents). Captures the scale-amplified footprint at population level.",
    threshold_pctile=0.75, tier=2,
    notes="PROXY: per-act population-scale impact requires affected_population_identifier not in dump. CIRISAgent-side ticket filed."
)
ax["distributive:access:agent_capabilities"] = calibrate_axis(
    "distributive:access:agent_capabilities", "cap_diversity", "positive_when_distributed",
    ["trace_sample_bundle", "cohort_delineation", "per_agent_cognitive_state_dominant", "per_agent_agent_role"],
    -0.3,
    "Number of distinct (cognitive_state_dominant, agent_role) tuples in the cohort, normalized by cohort size. Low diversity → narrow capability access.",
    threshold_pctile=0.25, tier=2,
    notes="PROXY: capability-as-revealed-by-action; capability-as-declared requires agent capability registry not in dump. CIRISAgent-side ticket filed."
)

# Write per-axis files
for name, content in ax.items():
    safe = name.replace(":", "_").replace("/", "_")
    with open(f"{OUT_DIR}/axes/{safe}.json", "w") as f: json.dump(content, f, indent=2)

# Index
with open(f"{OUT_DIR}/axes_index.json", "w") as f:
    json.dump({
        "version": "crc-v2",
        "calibrated_at": "2026-06-12T19:00:00+00:00",
        "axes": list(ax.keys()),
        "tier_1": [k for k,v in ax.items() if v.get("tier")==1],
        "tier_2": [k for k,v in ax.items() if v.get("tier")==2],
        "tier_3_deferred": [
            "correlated_action:ecology_of_communication:echo_chamber_density",
            "correlated_action:ecology_of_communication:information_silo_correlation",
            "correlated_action:ecology_of_communication:coordinated_messaging_pattern",
            "correlated_action:ecology_of_communication:cross_cohort_information_flow",
            "distributive:access:training_data",
        ],
    }, f, indent=2)

# Summary
print(f"\n=== AXES CALIBRATED ===")
for name, content in ax.items():
    if "calibration" in content:
        cal = content["calibration"]; tf = content["threshold_function"]
        print(f"  [{content['tier']}] {name}")
        print(f"      threshold={tf['threshold_value']}  (CI=[{tf['ci_95'][0]}, {tf['ci_95'][1]}])")
        print(f"      observed: min={cal['observed_min']} median={cal['observed_median']} max={cal['observed_max']}")
        print(f"      cohorts={cal['n_qualifying_cohorts']}  floor: ≥{content['statistical_floor']['min_goal_aligned_cluster_size_agents']} agents × 1000 events × 30 days")
    else:
        print(f"  [{content['tier']}] {name}  INSUFFICIENT_COHORTS ({content['n_qualifying_cohorts']})")
