#!/usr/bin/env python3
"""Honest re-scoring of CCA sec.11 institutional validation.
Pre-registration: PREREGISTRATION.md sha256 1d3b0c390e43835c1b6185307ae7db3b611c45420d35712e9dd1695e9bfb4b1e
frozen 2026-07-31T23:04:27Z, before any outcome was computed.
"""
import sys, json, itertools
import numpy as np, pandas as pd

sys.path.insert(0, '/home/emoore/RATCHET')
from ratchet.engines.institutional import InstitutionalCollapseEngine, InstitutionalParams

VDEM = '/home/emoore/RATCHET/data/institutional/vdem/v-dem-v15.parquet'
POLITY = '/home/emoore/RATCHET/data/institutional/polity5.xls'
WGI = '/home/emoore/RATCHET/data/institutional/wgi_processed.csv'

COUNTRIES = ['Venezuela', 'Türkiye', 'Hungary', 'Poland', 'Zimbabwe', 'Syria', 'Libya',
             'Yemen', 'Tunisia', 'Egypt', 'Ukraine', 'Germany', 'Canada', 'Australia']

cols = ['country_name', 'country_text_id', 'year', 'v2x_regime', 'v2x_polyarchy',
        'v2x_libdem', 'v2x_corr', 'e_p_polity']
v = pd.read_parquet(VDEM, columns=cols)
v = v[v.country_name.isin(COUNTRIES) & v.year.between(2000, 2024)].copy()
v['year'] = v.year.astype(int)


def series(c, col):
    s = v[v.country_name == c].set_index('year')[col].sort_index()
    return s.reindex(range(2000, 2025))


def sustained_drop(s, baseline, drop, persist, hi=2024):
    """First year t where s <= baseline-drop for `persist` consecutive fully-observed years."""
    for t in range(2001, hi + 1):
        yrs = list(range(t, t + persist))
        if yrs[-1] > hi:
            return None
        vals = [s.get(y, np.nan) for y in yrs]
        if any(pd.isna(x) for x in vals):
            continue
        if all(x <= baseline - drop + 1e-12 for x in vals):
            return t
    return None


# ---------------- outcomes ----------------
rows = []
for c in COUNTRIES:
    row = {'country': c}
    R = series(c, 'v2x_regime')
    L = series(c, 'v2x_libdem')
    P = series(c, 'e_p_polity')
    row['row_2000'] = R[2000]
    row['row_2024'] = R[2024]
    row['libdem_2000'] = L[2000]
    row['libdem_2024'] = L[2024]
    row['polity_2000'] = P[2000]
    row['polity_last'] = P.dropna().iloc[-1] if P.notna().any() else np.nan
    row['polity_last_yr'] = int(P.dropna().index[-1]) if P.notna().any() else None
    # PO-1 primary: RoW >=1 category decline sustained 3y
    t = sustained_drop(R, R[2000], 1, 3)
    row['PO1'] = int(t is not None); row['PO1_year'] = t
    # PO-4 variants
    row['PO1_p1'] = int(sustained_drop(R, R[2000], 1, 1) is not None)
    row['PO1_p1_year'] = sustained_drop(R, R[2000], 1, 1)
    row['PO1_p5'] = int(sustained_drop(R, R[2000], 1, 5) is not None)
    # PO-2 libdem absolute
    t2 = sustained_drop(L, L[2000], 0.10, 3)
    row['PO2'] = int(t2 is not None); row['PO2_year'] = t2
    # PO-5 libdem relative 25%
    t5 = sustained_drop(L, L[2000], 0.25 * L[2000], 3)
    row['PO5'] = int(t5 is not None); row['PO5_year'] = t5
    # PO-3 polity >=3 points
    t3 = sustained_drop(P.dropna(), P[2000], 3, 3, hi=int(P.dropna().index[-1]))
    row['PO3'] = int(t3 is not None); row['PO3_year'] = t3
    rows.append(row)
out = pd.DataFrame(rows).set_index('country')

# ---------------- predictor: the engine as configured in scripts/test_institutional_collapse.py ----------
pol = pd.read_excel(POLITY)
wgi = pd.read_csv(WGI)
WGI_NAME = {'Türkiye': 'Turkiye', 'Venezuela': 'Venezuela, RB', 'Syria': 'Syrian Arab Republic',
            'Egypt': 'Egypt, Arab Rep.', 'Yemen': 'Yemen, Rep.'}
POL_NAME = {'Türkiye': 'Turkey'}

preds = []
for c in COUNTRIES:
    sig0 = series(c, 'v2x_polyarchy')[2000]
    f0 = series(c, 'v2x_corr')[2000]
    pc = POL_NAME.get(c, c)
    xc = pol[(pol.country == pc) & (pol.year == 2000)]['xconst']
    xconst = float(xc.iloc[0]) if len(xc) else np.nan
    k0 = np.clip((xconst - 1) / 6.0, 0, 1) if not np.isnan(xconst) else 0.5
    rho0 = float(np.clip(f0 * (1 - k0), 0, 1))
    wc = WGI_NAME.get(c, c)
    wr = wgi[(wgi.country == wc) & (wgi.year == 2000)]
    lam0 = float(wr['lambda_'].iloc[0]) if len(wr) else 0.5

    eng = InstitutionalCollapseEngine(
        params=InstitutionalParams(alpha=0.01, d=0.02, noise_sigma=0.005), seed=42)
    eng.initialize_manual(k=k0, rho=rho0, sigma=float(sig0), f=float(f0),
                          lambda_=lam0, country_code=c, year=2000)
    eng.run(duration=20)
    pred_year = 2000 + int(eng.get_collapse_time()) if eng.is_collapsed() else None
    preds.append({'country': c, 'sigma0': sig0, 'f0': f0, 'k0': k0, 'rho0': rho0,
                  'k_eff0': k0 * 10 / max(1 + rho0 * (k0 * 10 - 1), 0.01),
                  'xconst2000': xconst, 'lambda0': lam0,
                  'pred': int(eng.is_collapsed()), 'pred_year': pred_year})
pred = pd.DataFrame(preds).set_index('country')
df = pred.join(out)

# ---------------- confusion matrices ----------------
def cm(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    bal = np.nanmean([sens, spec])
    num = tp * tn - fp * fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = num / den if den > 0 else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, n=n, acc=acc, sens=sens, spec=spec,
                bal_acc=bal, mcc=mcc)


def perm_test(y, p, iters=100000, seed=0):
    rng = np.random.default_rng(seed)
    obs = cm(y, p)
    y = np.asarray(y); p = np.asarray(p)
    ge_bal = ge_mcc = 0
    for _ in range(iters):
        q = rng.permutation(p)
        s = cm(y, q)
        if s['bal_acc'] >= obs['bal_acc'] - 1e-12: ge_bal += 1
        if s['mcc'] >= obs['mcc'] - 1e-12: ge_mcc += 1
    return (ge_bal + 1) / (iters + 1), (ge_mcc + 1) / (iters + 1)


print("=" * 100)
print("PER-COUNTRY TABLE (n=14 candidate set)")
print("=" * 100)
show = df[['sigma0', 'f0', 'k0', 'rho0', 'k_eff0', 'pred', 'pred_year',
           'row_2000', 'row_2024', 'PO1', 'PO1_year', 'PO2', 'PO2_year', 'PO3', 'PO3_year',
           'PO5', 'PO1_p1', 'PO1_p5']]
print(show.round(3).to_string())

results = {}
for po in ['PO1', 'PO2', 'PO3', 'PO5', 'PO1_p1', 'PO1_p5']:
    m = cm(df[po], df['pred'])
    results[po] = m
    print(f"\n--- outcome {po}: base rate {df[po].mean():.3f} ({int(df[po].sum())}/{len(df)}) ---")
    print("   ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()})
    pb, pm = perm_test(df[po].values, df['pred'].values, iters=20000, seed=1)
    print(f"    permutation p (bal_acc) = {pb:.4f}   p (MCC) = {pm:.4f}")
    # floors
    n = len(df); pos = int(df[po].sum())
    always = cm(df[po], np.ones(n)); never = cm(df[po], np.zeros(n))
    maj = always if pos > n - pos else never
    print(f"    floors: always-flag acc={always['acc']:.3f} bal={always['bal_acc']:.3f} | "
          f"never-flag acc={never['acc']:.3f} bal={never['bal_acc']:.3f} | "
          f"majority acc={maj['acc']:.3f}")
    results[po]['perm_p_bal'] = pb; results[po]['perm_p_mcc'] = pm

# Fisher exact on primary
from scipy.stats import fisher_exact
m = results['PO1']
odds, p = fisher_exact([[m['TP'], m['FP']], [m['FN'], m['TN']]])
print(f"\nFisher exact (PO-1): odds={odds}, p={p:.4f}")

# ---------------- leave-one-out over 13-country subsets ----------------
print("\n" + "=" * 100)
print("LEAVE-ONE-OUT: all 14 possible 13-country subsets, primary outcome PO-1")
print("=" * 100)
loo = []
for drop in COUNTRIES:
    sub = df.drop(index=drop)
    m = cm(sub['PO1'], sub['pred'])
    loo.append({'dropped': drop, **{k: m[k] for k in ['TP', 'FP', 'TN', 'FN']},
                'acc': round(m['acc'], 3), 'bal_acc': round(m['bal_acc'], 3),
                'mcc': round(m['mcc'], 3)})
print(pd.DataFrame(loo).to_string(index=False))

# ---------------- timing ----------------
print("\n" + "=" * 100)
print("TIMING (PO-1 observed event year vs engine predicted year)")
print("=" * 100)
tt = df[(df.pred == 1) & (df.PO1 == 1)][['pred_year', 'PO1_year']].copy()
tt['offset'] = tt.pred_year - tt.PO1_year
print(tt.to_string())
print(f"mean offset = {tt.offset.mean():+.2f} y ; median = {tt.offset.median():+.1f} ; "
      f"n = {len(tt)}  (negative = early)")
# all countries with a prediction, incl. those with no observed event
print("\nAll flagged countries and predicted years:")
print(df[df.pred == 1][['pred_year', 'PO1', 'PO1_year', 'sigma0', 'f0']].to_string())

# does k_eff enter the decision at all?
print("\n" + "=" * 100)
print("DOES k_eff ENTER THE PREDICTION? sweep k,rho over full range for a fixed (sigma0,f0)")
print("=" * 100)
base = df.loc['Hungary']
outs = set()
for kk in np.linspace(0.05, 1.0, 20):
    for rr in np.linspace(0.0, 0.99, 20):
        e = InstitutionalCollapseEngine(
            params=InstitutionalParams(alpha=0.01, d=0.02, noise_sigma=0.005), seed=42)
        e.initialize_manual(k=kk, rho=rr, sigma=float(base.sigma0), f=float(base.f0),
                            lambda_=0.5, year=2000)
        e.run(duration=20)
        outs.add((e.is_collapsed(), e.get_collapse_time()))
print(f"distinct (collapsed, time) outcomes across 400 (k,rho) combinations: {outs}")

df.to_csv('/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad/rescore_table.csv')
