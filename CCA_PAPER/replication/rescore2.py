#!/usr/bin/env python3
"""Part 2: score the paper's own stated verdicts; power floor at n=13/14; k_eff signal check."""
import numpy as np, pandas as pd, itertools
from scipy.stats import fisher_exact

df = pd.read_csv('/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad/rescore_table.csv',
                 index_col='country')

def cm(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((p==1)&(y==1)).sum()); fp = int(((p==1)&(y==0)).sum())
    tn = int(((p==0)&(y==0)).sum()); fn = int(((p==0)&(y==1)).sum())
    n = tp+fp+tn+fn
    sens = tp/(tp+fn) if tp+fn else np.nan; spec = tn/(tn+fp) if tn+fp else np.nan
    den = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return dict(TP=tp,FP=fp,TN=tn,FN=fn,n=n,acc=(tp+tn)/n,sens=sens,spec=spec,
                bal_acc=np.nanmean([sens,spec]), mcc=(tp*tn-fp*fn)/den if den>0 else 0.0)

# ---- 1. The paper's own stated verdicts (sec.11), scored against PO-1 ----
paper_pred = {'Germany':0,'Canada':0,'Australia':0,'Poland':0,'Hungary':0,
              'Türkiye':1,'Venezuela':1,'Tunisia':1,'Egypt':1,'Zimbabwe':1}
sub = df.loc[list(paper_pred)]
pp = np.array([paper_pred[c] for c in sub.index])
print("=== Paper's 10 named countries, its own stated verdicts vs pre-registered PO-1 ===")
tab = pd.DataFrame({'paper_verdict': ['flagged' if x else 'healthy/stable' for x in pp],
                    'PO1_outcome': sub.PO1.values, 'PO1_year': sub.PO1_year.values,
                    'cell': ['TP' if a==1 and b==1 else 'FP' if a==1 else 'FN' if b==1 else 'TN'
                             for a, b in zip(pp, sub.PO1.values)]}, index=sub.index)
print(tab.to_string())
m = cm(sub.PO1.values, pp); print(' ', {k:(round(x,3) if isinstance(x,float) else x) for k,x in m.items()})
odds,p = fisher_exact([[m['TP'],m['FP']],[m['FN'],m['TN']]]); print(f'  Fisher p={p:.3f}')
print(f"  Paper's claimed 'healthy phase 5/5' -> actual "
      f"{int((sub.loc[['Germany','Canada','Australia','Poland','Hungary']].PO1==0).sum())}/5 correct")

# ---- 2. What would it take to be significant at n=13/14? ----
print("\n=== Minimum detectable skill, n=14, 7 positives, 9 positive predictions (one-sided Fisher) ===")
for tp in range(0, 8):
    fp = 9-tp
    if fp < 0 or fp > 7: continue
    fn = 7-tp; tn = 7-fp
    if fn < 0 or tn < 0: continue
    o,p = fisher_exact([[tp,fp],[fn,tn]], alternative='greater')
    mm = cm([1]*7+[0]*7, [1]*tp+[0]*fn+[1]*fp+[0]*tn)
    print(f'  TP={tp} FP={fp} TN={tn} FN={fn}  acc={mm["acc"]:.3f} bal={mm["bal_acc"]:.3f} '
          f'mcc={mm["mcc"]:+.3f}  Fisher one-sided p={p:.4f}{"  <-- first significant" if p<0.05 else ""}')
print("\n=== Best possible p at n=13, balanced base rate, PERFECT classifier ===")
for n, pos in [(13,6),(13,7),(14,7),(20,10),(30,15)]:
    o,p = fisher_exact([[pos,0],[0,n-pos]], alternative='greater')
    print(f'  n={n}, {pos} positives, perfect classification: Fisher one-sided p={p:.5f}')

# ---- 3. Does k_eff (or rho, or k) carry any signal for PO-1? ----
print("\n=== Continuous predictors vs PO-1 (rank AUC, n=14) ===")
def auc(score, y):
    y = np.asarray(y); s = np.asarray(score, dtype=float)
    pos = s[y==1]; neg = s[y==0]
    return float(np.mean([(a>b)+0.5*(a==b) for a in pos for b in neg]))
for name, sc, sign in [('sigma_0 (V-Dem polyarchy 2000)', df.sigma0, -1),
                       ('f_0 (V-Dem corruption 2000)', df.f0, +1),
                       ('k_0', df.k0, -1), ('rho_0', df.rho0, +1),
                       ('k_eff_0', df.k_eff0, -1)]:
    a = auc(sign*sc.values, df.PO1.values)
    print(f'  {name:32s} AUC={a:.3f}  (0.5 = chance; direction: {"lower=riskier" if sign<0 else "higher=riskier"})')

# ---- 4. Degenerate flags: already past threshold at t=0 ----
print("\n=== Flags that fire because the initial state is ALREADY past the collapse threshold ===")
deg = df[(df.pred==1)]
deg = deg.assign(f0_gt_0p8=deg.f0>0.8, sigma0_lt_0p2=deg.sigma0<0.2)
print(deg[['sigma0','f0','pred_year','f0_gt_0p8','sigma0_lt_0p2']].round(3).to_string())
print(f"  {int(deg.f0_gt_0p8.sum())} of {len(deg)} flags have f_0 > 0.8 at initialisation "
      f"(engine's own collapse threshold), i.e. collapse is asserted at t=0, not predicted.")

# ---- 5. rule characterisation ----
print("\n=== Engine rule reduces to a threshold on (sigma_0, f_0) ===")
print("  sigma(t)=sigma_0-0.02t  -> collapse (sigma<0.2) within 20y iff sigma_0 < 0.60")
print("  f(t)=f_0+0.005t         -> collapse (f>0.8)     within 20y iff f_0 > 0.70")
chk = ((df.sigma0 < 0.60) | (df.f0 > 0.70)).astype(int)
print(f"  agreement of that closed form with the actual engine runs: "
      f"{int((chk==df.pred).sum())}/{len(df)}")
