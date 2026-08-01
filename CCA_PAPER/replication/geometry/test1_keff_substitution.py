"""
Does volume decay actually follow k_eff = k/(1+rho(k-1))?

Theorem 2.1 substitutes k_eff into the exponent by appeal to the effective
sample size formula. Kish governs the VARIANCE of a mean of equicorrelated
variables. The volume argument needs the EXPECTATION of a sum of log-reduction
factors. Test whether the substitution holds anyway.

Setup: bounded convex region = unit ball in R^D. Cut by k halfspaces whose
normals are equicorrelated with correlation rho. Measure -ln(V_k/V_0) by
Monte Carlo, and ask what effective count that implies vs the Kish prediction.
"""
import numpy as np

RNG = np.random.default_rng(2026)
D = 8          # ambient dimension
NMC = 200_000  # MC points for volume estimation
OFFSET = 0.35  # halfspace offset: keep <n,x> <= OFFSET (so each cuts real volume)


def correlated_normals(k, rho, D):
    """k unit normals in R^D with pairwise correlation ~rho (equicorrelated)."""
    common = RNG.standard_normal(D)
    out = []
    for _ in range(k):
        idio = RNG.standard_normal(D)
        v = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def surviving_fraction(k, rho, D, nmc=NMC):
    """Fraction of the unit ball satisfying all k halfspace constraints."""
    X = RNG.standard_normal((nmc, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X *= RNG.random(nmc)[:, None] ** (1.0 / D)      # uniform in ball
    N = correlated_normals(k, rho, D)
    return float(np.all(X @ N.T <= OFFSET, axis=1).mean())


# lambda: decay per constraint at rho=0, calibrated from k=1
f1 = np.mean([surviving_fraction(1, 0.0, D) for _ in range(12)])
lam = -np.log(f1)
print(f"D={D}, offset={OFFSET}: single-constraint survival {f1:.4f} -> lambda={lam:.4f}\n")

print(f"{'k':>3} {'rho':>5} {'Kish k_eff':>11} {'implied k_eff':>14} {'ratio':>7}")
print("-" * 46)
rows = []
for k in (4, 8, 16, 32):
    for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
        fr = np.mean([surviving_fraction(k, rho, D) for _ in range(8)])
        if fr <= 0:
            continue
        implied = -np.log(fr) / lam            # effective count the geometry actually shows
        kish = k / (1 + rho * (k - 1))
        rows.append((k, rho, kish, implied, implied / kish))
        print(f"{k:>3} {rho:>5.2f} {kish:>11.2f} {implied:>14.2f} {implied/kish:>7.2f}")

r = np.array([x[4] for x in rows])
print(f"\nratio implied/Kish: min {r.min():.2f}  max {r.max():.2f}  spread {r.max()/r.min():.1f}x")
print("If the substitution were valid this column would sit at 1.0 throughout.")
