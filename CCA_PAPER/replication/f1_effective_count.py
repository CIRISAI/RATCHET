"""F-1: does k/lambda_max agree with the Kish k_eff on the same data?
Prereg f6aea87d9e6e65edef4812df8792c8f12660f77288b044a5f19657fee1171a0a"""
import json, numpy as np
d = json.load(open('/home/emoore/CIRISArray/experiments/expC1_results.json'))
k = 16
rows = []
for s in d['sweep']:
    C = np.array(s['correlation_matrix'])
    C = (C + C.T) / 2                       # symmetrize against float asymmetry
    off = C[np.triu_indices(k, 1)]
    rho = float(off.mean())                 # signed mean, the Kish input
    lam = float(np.linalg.eigvalsh(C).max())
    kish = k / (1 + rho * (k - 1)) if rho > -1/(k-1) else np.nan
    spec = k / lam
    rows.append((s['sync'], rho, kish, spec, abs(spec-kish)/kish))

print(f"{'sync':>5} {'rho':>7} {'Kish k_eff':>11} {'k/lam_max':>10} {'rel diff':>9}")
print("-"*48)
for sy, rho, kish, spec, rel in rows:
    print(f"{sy:>5.2f} {rho:>7.4f} {kish:>11.3f} {spec:>10.3f} {rel:>8.1%}")

rel_all = np.array([r[4] for r in rows])
mask = np.array([r[1] >= 0.04 for r in rows])
rel_ok = rel_all[mask]
print(f"\nfull sweep      n={len(rel_all):2d}  median rel diff = {np.median(rel_all):.1%}")
print(f"rho >= 0.04     n={mask.sum():2d}  median rel diff = {np.median(rel_ok):.1%}  <-- adjudicated")
print(f"\nF1-a  fires if median(restricted) > 25%: {'FIRES' if np.median(rel_ok)>0.25 else 'does not fire'}")
below = sum(1 for r in rows if r[3] <= r[2])
print(f"F1-c  k/lambda_max <= Kish pointwise: {below}/{len(rows)}")
lo = [r[4] for r in rows if r[1] < 0.10]; hi = [r[4] for r in rows if r[1] >= 0.20]
print(f"F1-b  median rel diff  rho<0.10: {np.median(lo):.1%}   rho>=0.20: {np.median(hi):.1%}")
json.dump([{'sync':a,'rho':b,'kish':c,'spectral':d_,'rel':e} for a,b,c,d_,e in rows],
          open('f1_results.json','w'), indent=2)
