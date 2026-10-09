"""Re-evaluate the RATCHET calibration-package scorer values from prod scoring rows.

Inputs: the class-only export of capacity:sustained_coherence:v1 (6,465 rows, 440 subjects)
plus agent_id_hash_class.jsonl. No numpy: pure-python quantiles and a seeded bootstrap.

Parameters under evaluation (CIRISServer src/scorer.rs, src/config_reconcile.rs):
  scorer.sample_gate   rows before a numeric score      server default 20  (crc-v1: 500)
  scorer.target_n_eff  n_eff at which capacity = 1.0    server default 8.0 (RATCHET "owns the real value")
  measure              n_eff_pr (fed today) vs n_eff_h  calibration.rs states no preference
  gate semantics       capacity() tests n_eff <= gate; the gate is defined on ROWS

Run with python3 -I. Domain: production canonical, 2026-08-01..10-09, battery-dominated population.
"""
import json, os, random, collections, statistics as st
EX = os.environ.get("TRACE_EXPORT_DIR", "/home/emoore/ratchet-trace-export").rstrip("/") + "/"
rows = [json.loads(l) for l in open(EX + "scoring/capacity_sustained_coherence.jsonl")]
FEATURE_DIM = 11
random.seed(20261009)

def q(xs, p):
    xs = sorted(xs); n = len(xs)
    if not n: return float("nan")
    k = min(n - 1, max(0, int(round(p * (n - 1)))))
    return xs[k]
def fmt(x): return f"{x:.2f}" if isinstance(x, float) else str(x)

def klass(r):
    c = r["subject_class"]
    return ("qa-runner" if c.startswith("synthetic") else
            "research-agent" if c.startswith("registered") else
            "owner-bound" if "owner-bound" in c else "unclaimed-bootstrap")

# ── 0. dedupe: latest attestation per subject (re-emits dominate raw rows) ──
by_subj = collections.defaultdict(list)
for r in rows: by_subj[r["subject"]].append(r)
latest = [max(rs, key=lambda r: r["asserted_at"]) for rs in by_subj.values()]
print(f"rows {len(rows)} | subjects {len(latest)} | attestations/subject median {st.median(len(v) for v in by_subj.values()):.0f}, max {max(len(v) for v in by_subj.values())}")
print("subjects by class:", dict(collections.Counter(klass(r) for r in latest)))

# ── 1. what makes n_eff zero? ──
z = [r for r in rows if (r["n_eff_pr"] or 0) == 0]
print(f"\n[1] n_eff_pr == 0 on {len(z)} rows: feature_dim==0 in {sum(1 for r in z if r['feature_dim']==0)}, sample_size<2 in {sum(1 for r in z if (r['sample_size'] or 0)<2)}")
print(f"    n_eff_pr > 0 with sample_size==2: {sum(1 for r in rows if (r['sample_size'] or 0)==2 and (r['n_eff_pr'] or 0)>0)}  (-> no row floor is applied before capacity())")

# ── 2. saturation curve: effective rank vs rows, both measures, all rows and latest ──
def curve(src, key):
    bins = collections.defaultdict(list)
    for r in src:
        N = r["sample_size"] or 0
        if N < 2 or r["feature_dim"] == 0: continue
        b = N if N < 10 else 10 * (N // 10) if N < 60 else 60 if N < 100 else 100
        bins[b].append(r[key] or 0)
    return {b: (len(v), q(v, .5), q(v, .9)) for b, v in bins.items() if len(v) >= 8}
print("\n[2] saturation curve (rows with ≥2 samples and ≥1 feature): N-bin -> n, median, p90")
cp, ch = curve(rows, "n_eff_pr"), curve(rows, "n_eff_h")
print("     N      n_eff_pr (n, med, p90)        n_eff_h (n, med, p90)")
for b in sorted(set(cp) | set(ch)):
    a = cp.get(b, ("-", float("nan"), float("nan"))); h = ch.get(b, ("-", float("nan"), float("nan")))
    print(f"   {b:>4}   ({a[0]:>4}, {fmt(a[1]):>5}, {fmt(a[2]):>5})          ({h[0]:>4}, {fmt(h[1]):>5}, {fmt(h[2]):>5})")

# ── 3. plateau estimate: latest-per-subject with N >= candidate row gate; bootstrap over subjects ──
def plateau(gate, key, src=latest, B=2000):
    pool = [r[key] or 0 for r in src if (r["sample_size"] or 0) >= gate and r["feature_dim"] > 0]
    if len(pool) < 5: return None
    def stats(xs): return (q(xs, .5), q(xs, .9), q(xs, .95))
    pt = stats(pool); boots = [stats(random.choices(pool, k=len(pool))) for _ in range(B)]
    ci = tuple((q([b[i] for b in boots], .025), q([b[i] for b in boots], .975)) for i in range(3))
    return len(pool), pt, ci
print("\n[3] plateau of effective rank, latest attestation per subject, subjects with N >= gate (bootstrap 95% CI over subjects)")
print("     gate  measure   n   median [CI]          p90 [CI]            p95 [CI]")
for gate in (20, 30, 50):
    for key in ("n_eff_pr", "n_eff_h"):
        p = plateau(gate, key)
        if p:
            n, (m, p90, p95), ci = p
            print(f"   {gate:>5}  {key:8} {n:>3}   {m:.2f} [{ci[0][0]:.2f},{ci[0][1]:.2f}]   {p90:.2f} [{ci[1][0]:.2f},{ci[1][1]:.2f}]   {p95:.2f} [{ci[2][0]:.2f},{ci[2][1]:.2f}]")
print("\n    by class, gate 20, n_eff_pr (latest per subject):")
for k in ("qa-runner", "unclaimed-bootstrap", "owner-bound", "research-agent"):
    pool = [r["n_eff_pr"] or 0 for r in latest if klass(r) == k and (r["sample_size"] or 0) >= 20 and r["feature_dim"] > 0]
    tot = sum(1 for r in latest if klass(r) == k)
    print(f"      {k:20} subjects {tot:>3}  with N>=20: {len(pool):>3}  median {fmt(q(pool,.5)) if pool else '-':>5}  p90 {fmt(q(pool,.9)) if pool else '-':>5}  max {fmt(max(pool)) if pool else '-':>5}")

# ── 4. where does the curve stop being N-limited? first bin whose median >= 90% of the N>=50 plateau ──
for key in ("n_eff_pr", "n_eff_h"):
    pl = plateau(50, key)
    if not pl: continue
    target_med = pl[1][0]; c = curve(rows, key)
    knee = next((b for b in sorted(c) if c[b][1] >= 0.9 * target_med), None)
    print(f"\n[4] {key}: plateau median (N>=50) {target_med:.2f}; binned median first reaches 90% of it at N≈{knee}")

# ── 5. stability under re-attestation at fixed N (same corpus re-derived) ──
sd = []
for rs in by_subj.values():
    byN = collections.defaultdict(list)
    for r in rs:
        if r["feature_dim"] > 0: byN[r["sample_size"]].append(r["n_eff_pr"] or 0)
    for N, v in byN.items():
        if len(v) >= 5: sd.append(st.pstdev(v))
print(f"\n[5] re-attestation stability: {len(sd)} (subject,N) groups with ≥5 rows; n_eff_pr stdev median {q(sd,.5):.4f}, p95 {q(sd,.95):.4f}, max {max(sd):.4f}")

# ── 6. counterfactual score distributions ──
def cap_current(ne, gate, tgt):  # scoring/capacity.rs verbatim (gate applied to n_eff)
    if ne <= gate: return 0.0
    if ne >= tgt: return 1.0
    span = tgt - gate
    return 1.0 if span <= 0 else max(0.0, min(1.0, (ne - gate) / span))
def cap_fixed(N, ne, rows_gate, floor, tgt):  # gate on rows; band on n_eff above a rank floor
    if N < rows_gate or ne <= floor: return 0.0
    return max(0.0, min(1.0, (ne - floor) / (tgt - floor)))
pr95 = plateau(20, "n_eff_pr")[1][2]; pr90 = plateau(20, "n_eff_pr")[1][1]
cands = [
    ("CURRENT  gate_on=n_eff gate=20 target=8.0", lambda r: cap_current(r["n_eff_pr"] or 0, 20, 8.0)),
    ("fixed    rows>=20 floor=1 target=8.0",      lambda r: cap_fixed(r["sample_size"] or 0, r["n_eff_pr"] or 0, 20, 1.0, 8.0)),
    (f"fixed    rows>=20 floor=1 target=p95={pr95:.2f}", lambda r: cap_fixed(r["sample_size"] or 0, r["n_eff_pr"] or 0, 20, 1.0, pr95)),
    (f"fixed    rows>=20 floor=1 target=p90={pr90:.2f}", lambda r: cap_fixed(r["sample_size"] or 0, r["n_eff_pr"] or 0, 20, 1.0, pr90)),
    (f"fixed    rows>=30 floor=1 target=p95={pr95:.2f}", lambda r: cap_fixed(r["sample_size"] or 0, r["n_eff_pr"] or 0, 30, 1.0, pr95)),
    ("crc-v1   rows>=500 floor=1 target=8.0",     lambda r: cap_fixed(r["sample_size"] or 0, r["n_eff_pr"] or 0, 500, 1.0, 8.0)),
]
print("\n[6] counterfactual capacity on latest-per-subject rows (440 subjects): share >0, share =1, median of nonzero, by class share>0")
for name, f in cands:
    sc = [(klass(r), f(r)) for r in latest]
    nz = [s for _, s in sc if s > 0]
    bycls = {k: f"{100*sum(1 for c,s in sc if c==k and s>0)/max(1,sum(1 for c,_ in sc if c==k)):.0f}%" for k in ("qa-runner","unclaimed-bootstrap","owner-bound","research-agent")}
    print(f"   {name:46} >0: {100*len(nz)/len(sc):5.1f}%  =1: {100*sum(1 for s in nz if s==1)/len(sc):4.1f}%  med(nz) {fmt(q(nz,.5)) if nz else '-':>5}  {bycls}")
# subjects that could EVER be scoreable under a row gate: max N per subject
mx = [max(r["sample_size"] or 0 for r in rs) for rs in by_subj.values()]
print(f"\n    subjects whose max N ever reached: >=20: {sum(1 for m in mx if m>=20)}  >=30: {sum(1 for m in mx if m>=30)}  >=50: {sum(1 for m in mx if m>=50)}  >=500: {sum(1 for m in mx if m>=500)}   (of {len(mx)})")

# ── 7. feature coverage: is n_eff tracking how many features varied, rather than independence? ──
sc = [r for r in latest if (r["sample_size"] or 0) >= 20 and r["feature_dim"] > 0]
print(f"\n[7] feature_dim among latest-per-subject rows with N>=20 (n={len(sc)}):", dict(sorted(collections.Counter(r['feature_dim'] for r in sc).items())))
print("    all rows, feature_dim distribution:", dict(sorted(collections.Counter(r['feature_dim'] for r in rows).items())))
# ceiling: n_eff_pr / feature_dim (fraction of available rank realised)
frac = [(r["n_eff_pr"] or 0) / r["feature_dim"] for r in sc]
print(f"    realised fraction of available rank (n_eff_pr/feature_dim), N>=20: median {q(frac,.5):.2f}  p10 {q(frac,.1):.2f}  p90 {q(frac,.9):.2f}")
# rank correlation n_eff_pr vs feature_dim (Spearman via ranks, ties averaged)
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0]*len(xs); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and xs[order[j+1]] == xs[order[i]]: j += 1
        for k in range(i, j+1): r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb)); den = (sum((x-ma)**2 for x in ra) * sum((y-mb)**2 for y in rb)) ** .5
    return num/den if den else float("nan")
print(f"    Spearman(n_eff_pr, feature_dim) N>=20: {spearman([r['n_eff_pr'] or 0 for r in sc], [r['feature_dim'] for r in sc]):.2f}   "
      f"Spearman(n_eff_pr, sample_size) N>=20: {spearman([r['n_eff_pr'] or 0 for r in sc], [r['sample_size'] or 0 for r in sc]):.2f}")
print("    with a minimum-live-features gate, scoreable subjects (N>=20) and plateau p95 of n_eff_pr:")
for kmin in (1, 3, 4, 5, 6):
    pool = [r["n_eff_pr"] or 0 for r in sc if r["feature_dim"] >= kmin]
    print(f"      feature_dim>={kmin}: subjects {len(pool):>3}   median {fmt(q(pool,.5)) if pool else '-':>5}   p95 {fmt(q(pool,.95)) if pool else '-':>5}")

# ── 8. the recommended set, exactly, and a machine-readable proposal ──
REC = {"measure": "n_eff_pr", "gate_semantics": "rows", "sample_gate_rows": 20,
       "min_feature_dim": 4, "n_eff_floor": 1.0, "target_n_eff": 4.5, "window_rows": 500}
def cap_rec(r):
    N, ne, fd = r["sample_size"] or 0, r["n_eff_pr"] or 0, r["feature_dim"]
    if N < REC["sample_gate_rows"] or fd < REC["min_feature_dim"]: return None   # Indeterminate, not 0.0
    if ne <= REC["n_eff_floor"]: return 0.0
    return max(0.0, min(1.0, (ne - REC["n_eff_floor"]) / (REC["target_n_eff"] - REC["n_eff_floor"])))
sc = [(klass(r), cap_rec(r)) for r in latest]
det = [s for _, s in sc if s is not None]
print(f"\n[8] RECOMMENDED {REC}")
print(f"    latest-per-subject: Indeterminate {sum(1 for _,s in sc if s is None)}  numeric {len(det)}  (=0: {sum(1 for s in det if s==0)}, =1: {sum(1 for s in det if s==1)})"
      f"  numeric median {q(det,.5):.2f}  p25 {q(det,.25):.2f}  p75 {q(det,.75):.2f}")
for k in ("unclaimed-bootstrap", "research-agent", "owner-bound", "qa-runner"):
    d = [s for c, s in sc if c == k and s is not None]; tot = sum(1 for c, _ in sc if c == k)
    print(f"      {k:20} subjects {tot:>3}  numeric {len(d):>3}  median {fmt(q(d,.5)) if d else '-':>5}  =1 {sum(1 for s in d if s==1)}")
# normalized alternative: realised fraction of available rank, target 0.6
alt = [ (r["n_eff_pr"] or 0)/r["feature_dim"] for r in latest if (r["sample_size"] or 0)>=20 and r["feature_dim"]>=4 ]
print(f"    ALT (normalized n_eff_pr/feature_dim, floor 1/fd, target 0.60): n {len(alt)}  median {q(alt,.5):.2f}  share>=0.60: {100*sum(1 for a in alt if a>=0.6)/max(1,len(alt)):.0f}%")
out = {"proposal": REC, "withdrawn": {"target_n_eff": 8.0, "sample_size_gate_crc_v1": 500},
       "evidence": {"rows": len(rows), "subjects": len(latest), "window": "2026-08-01..2026-10-09",
                    "plateau_n_eff_pr_gate20": {"n": 56, "median": 2.96, "p90": 4.14, "p95": 4.48, "p95_ci95": [3.78, 5.10]},
                    "knee_rows": 20, "max_observed": {"n_eff_pr": 5.31, "n_eff_h": 6.18, "sample_size": 156},
                    "spearman_n_eff_pr_vs_feature_dim_N20": 0.74, "spearman_n_eff_pr_vs_sample_size_N20": -0.35,
                    "scoreable_by_class": {"unclaimed-bootstrap": 53, "research-agent": 3, "owner-bound": 0, "qa-runner": 0}},
       "labels": {"sample_gate_rows": "measured (knee)", "target_n_eff": "measured p95 of plateau; as a saturation policy, wager",
                  "min_feature_dim": "measured (coverage confound)", "gate_semantics": "proved (units)"}}
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/proposed_values.json", "w"), indent=1)
print("    wrote proposed_values.json")

# ── 9. prod agents only (Datum, echo-core, echo-speculative) vs the full population ──
def subset(name, pred):
    L = [r for r in latest if pred(r)]; R = [r for r in rows if pred(r)]
    Ns = [r["sample_size"] or 0 for r in L]
    fds = [r["feature_dim"] for r in L]
    ne_all = [r["n_eff_pr"] or 0 for r in L if r["feature_dim"] > 0]
    pl = [r["n_eff_pr"] or 0 for r in L if (r["sample_size"] or 0) >= 20 and r["feature_dim"] > 0]
    cf = [cap_rec(r) for r in L]; det = [s for s in cf if s is not None]
    return {"name": name, "rows": len(R), "subjects": len(L),
            "N med/max": f"{q(Ns,.5):.0f}/{max(Ns) if Ns else 0}", "N>=20": sum(1 for n in Ns if n >= 20),
            "fd med/max": f"{q(fds,.5):.0f}/{max(fds) if fds else 0}",
            "n_eff_pr latest med/p90/max": f"{q(ne_all,.5):.2f}/{q(ne_all,.9):.2f}/{max(ne_all):.2f}" if ne_all else "-",
            "plateau N>=20 n/med/p90/max": f"{len(pl)}/{q(pl,.5):.2f}/{q(pl,.9):.2f}/{max(pl):.2f}" if pl else "0/-",
            "REC numeric n/med/=1": f"{len(det)}/{q(det,.5):.2f}/{sum(1 for s in det if s==1)}" if det else "0/-",
            "_curve": curve(R, "n_eff_pr")}
isprod = lambda r: r["subject_class"].startswith("registered research agent")
cols = [subset("ALL", lambda r: True), subset("PROD (datum+echoes)", isprod),
        subset("datum", lambda r: r["subject_class"].endswith(":datum")),
        subset("echo-core", lambda r: r["subject_class"].endswith(":echo-core")),
        subset("echo-speculative", lambda r: r["subject_class"].endswith(":echo-speculative")),
        subset("owner-bound", lambda r: "owner-bound" in r["subject_class"]),
        subset("unclaimed-bootstrap", lambda r: r["subject_class"].startswith("install: bootstrap"))]
print("\n[9] prod agents vs the rest")
keys = ["rows", "subjects", "N med/max", "N>=20", "fd med/max", "n_eff_pr latest med/p90/max", "plateau N>=20 n/med/p90/max", "REC numeric n/med/=1"]
print(f"    {'':30}" + "".join(f"{c['name'][:19]:>20}" for c in cols))
for k in keys:
    print(f"    {k:30}" + "".join(f"{str(c[k]):>20}" for c in cols))
print("\n    saturation curve, n_eff_pr median by N-bin (n): ALL vs PROD")
ca, cp_ = cols[0]["_curve"], cols[1]["_curve"]
for b in sorted(set(ca) | set(cp_)):
    a = ca.get(b); p = cp_.get(b)
    print(f"      N≈{b:>4}   ALL {fmt(a[1]) if a else '-':>5} ({a[0] if a else 0:>4})     PROD {fmt(p[1]) if p else '-':>5} ({p[0] if p else 0:>4})")
# prod subjects individually at their max N
print("\n    each prod subject at its max-N attestation: class, N, feature_dim, n_eff_pr, REC score")
pr = [max(rs, key=lambda r: (r["sample_size"] or 0)) for rs in by_subj.values() if isprod(rs[0])]
for r in sorted(pr, key=lambda r: -(r["sample_size"] or 0)):
    s = cap_rec(r)
    print(f"      {r['subject_class'].split(':')[-1]:17} N={r['sample_size'] or 0:>3}  fd={r['feature_dim']:>2}  n_eff_pr={r['n_eff_pr'] or 0:.2f}  -> {'Indet' if s is None else f'{s:.2f}'}")
