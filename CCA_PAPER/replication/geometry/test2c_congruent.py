"""Isolate the source of the asymmetry.

Case A gave a falling D/H ratio under i.i.d. constraints -- but H was small and
central while D was large and peripheral, so the regions were not congruent.
Redo with H and D CONGRUENT: same radius, same distance from origin, different
directions. If the theorem alone produces safety, the ratio must still fall.
"""
import numpy as np
RNG = np.random.default_rng(311)
D_DIM, N, OFFSET, R = 8, 60_000, 0.35, 0.25
H_C = np.zeros(D_DIM); H_C[0] = 0.5      # honest ball
D_C = np.zeros(D_DIM); D_C[1] = 0.5      # deceptive ball: congruent, rotated

def ball(n, center, r):
    X = RNG.standard_normal((n, D_DIM)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X * RNG.random(n)[:, None] ** (1.0 / D_DIM) * r + center

def run(truth_tracking, k_max=40, trials=8):
    sh, sd = np.zeros(k_max), np.zeros(k_max)
    for _ in range(trials):
        H, Dp = ball(N, H_C, R), ball(N, D_C, R)
        ah, ad = np.ones(N, bool), np.ones(N, bool)
        for k in range(k_max):
            while True:
                n = RNG.standard_normal(D_DIM); n /= np.linalg.norm(n)
                if not truth_tracking or (H_C @ n) + R <= OFFSET:
                    break
            ah &= (H @ n <= OFFSET); ad &= (Dp @ n <= OFFSET)
            sh[k] += ah.mean()/trials; sd[k] += ad.mean()/trials
    return sh, sd

for name, tt in (("CONGRUENT regions, i.i.d. constraints (theorem assumption (a) alone)", False),
                 ("CONGRUENT regions, truth-tracking constraints", True)):
    sh, sd = run(tt)
    base = sd[0]/sh[0]
    print(f"\n{name}")
    print(f"{'k':>3} {'surv H':>9} {'surv D':>9} {'D/H':>9} {'vs k=1':>8}")
    print("-"*44)
    for k in (1,5,10,20,40):
        r = sd[k-1]/sh[k-1] if sh[k-1]>0 else float('nan')
        print(f"{k:>3} {sh[k-1]:>9.4f} {sd[k-1]:>9.4f} {r:>9.4f} {r/base:>8.3f}")
    lh=-np.log(max(sh[39],1e-12))/40; ld=-np.log(max(sd[39],1e-12))/40
    print(f"  lambda_H={lh:.4f}  lambda_D={ld:.4f}  gap={ld-lh:+.4f}")
