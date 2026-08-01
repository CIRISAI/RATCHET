"""
Does the volume-decay theorem give a SAFETY result?

Safety needs the ratio |deceptive| / |honest| to fall as constraints accumulate.
The theorem cuts a bounded convex region with i.i.d. halfspaces and says nothing
about which points are honest. Test what that buys.

Case A: constraints i.i.d. from a distribution with no reference to H or D
        (exactly the theorem's assumption (a)).
Case B: constraints drawn truth-tracking -- rejected unless satisfied by the
        honest region. Not assumed by the theorem; the alignment problem itself.
"""
import numpy as np

RNG = np.random.default_rng(117)
D_DIM, NMC, OFFSET = 8, 120_000, 0.35

def sample_ball(n, d):
    X = RNG.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X * RNG.random(n)[:, None] ** (1.0 / d)

# Honest region: a ball of radius r_h around a point. Deceptive: everything else.
H_CENTER = np.zeros(D_DIM); H_CENTER[0] = 0.5
R_H = 0.25

def label(X):
    honest = np.linalg.norm(X - H_CENTER, axis=1) <= R_H
    return honest, ~honest

def run(truth_tracking, k_max=40, trials=6):
    hist = []
    for _ in range(trials):
        X = sample_ball(NMC, D_DIM)
        honest, decept = label(X)
        alive = np.ones(len(X), bool)
        row = []
        for k in range(1, k_max + 1):
            while True:
                n = RNG.standard_normal(D_DIM); n /= np.linalg.norm(n)
                if not truth_tracking:
                    break
                # truth-tracking: reject any constraint that would cut the honest region
                if (H_CENTER @ n) + R_H <= OFFSET:
                    break
            alive &= (X @ n <= OFFSET)
            h = (alive & honest).sum(); d = (alive & decept).sum()
            row.append((h, d))
        hist.append(row)
    return np.array(hist, float).mean(axis=0)

for name, tt in (("Case A: i.i.d. constraints (the theorem's assumption)", False),
                 ("Case B: truth-tracking constraints (assumed, not derived)", True)):
    r = run(tt)
    h0, d0 = r[0]
    print(f"\n{name}")
    print(f"{'k':>3} {'honest':>9} {'deceptive':>10} {'D/H ratio':>11} {'vs k=1':>8}")
    print("-" * 46)
    base = (d0 / h0) if h0 else float('nan')
    for k in (1, 5, 10, 20, 40):
        h, d = r[k - 1]
        ratio = d / h if h else float('nan')
        print(f"{k:>3} {h:>9.0f} {d:>10.0f} {ratio:>11.3f} {ratio/base:>8.3f}")
