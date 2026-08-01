"""Asymmetry test, stratified: sample honest and deceptive regions separately
and measure each one's survival rate, so neither is starved of points."""
import numpy as np
RNG = np.random.default_rng(117)
D_DIM, N, OFFSET = 8, 60_000, 0.35
H_CENTER = np.zeros(D_DIM); H_CENTER[0] = 0.5
R_H = 0.25

def in_ball(n, d, center=None, r=1.0):
    X = RNG.standard_normal((n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X * RNG.random(n)[:, None] ** (1.0 / d) * r
    return X + (center if center is not None else 0.0)

def rates(truth_tracking, k_max=40, trials=8):
    surv_h, surv_d = np.zeros(k_max), np.zeros(k_max)
    for _ in range(trials):
        H = in_ball(N, D_DIM, H_CENTER, R_H)                 # honest points
        Dp = in_ball(N, D_DIM)                               # ambient (deceptive proxy)
        Dp = Dp[np.linalg.norm(Dp - H_CENTER, axis=1) > R_H]
        ah, ad = np.ones(len(H), bool), np.ones(len(Dp), bool)
        for k in range(k_max):
            while True:
                n = RNG.standard_normal(D_DIM); n /= np.linalg.norm(n)
                if not truth_tracking or (H_CENTER @ n) + R_H <= OFFSET:
                    break
            ah &= (H @ n <= OFFSET); ad &= (Dp @ n <= OFFSET)
            surv_h[k] += ah.mean() / trials
            surv_d[k] += ad.mean() / trials
    return surv_h, surv_d

for name, tt in (("Case A: i.i.d. constraints -- the theorem's assumption (a)", False),
                 ("Case B: truth-tracking constraints -- assumed, not derived", True)):
    sh, sd = rates(tt)
    print(f"\n{name}")
    print(f"{'k':>3} {'surv honest':>12} {'surv decept':>12} {'D/H ratio':>11} {'vs k=1':>8}")
    print("-" * 52)
    base = sd[0] / sh[0]
    for k in (1, 5, 10, 20, 40):
        r = sd[k-1] / sh[k-1] if sh[k-1] > 0 else float('nan')
        print(f"{k:>3} {sh[k-1]:>12.4f} {sd[k-1]:>12.4f} {r:>11.4f} {r/base:>8.3f}")
    lh = -np.log(max(sh[39], 1e-12))/40; ld = -np.log(max(sd[39], 1e-12))/40
    print(f"  decay rates: lambda_H={lh:.4f}  lambda_D={ld:.4f}  gap={ld-lh:+.4f}")
