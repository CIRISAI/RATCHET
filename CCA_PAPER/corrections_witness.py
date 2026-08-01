#!/usr/bin/env python3
"""
Machine witness for CORRECTIONS_v3.md — Coherence Collapse Analysis v3 (Zenodo 18217688).

Each check reproduces one defect claimed in the corrections note, from the paper's
own stated assumptions. Run: python3 corrections_witness.py

Exit status 0 means every defect reproduced as documented.
"""

import sympy as sp

FAILURES = []


def check(label, condition, detail=""):
    status = "OK  " if condition else "FAIL"
    if not condition:
        FAILURES.append(label)
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))


k, rho, lam, sig, alpha, d = sp.symbols(
    "k rho lambda sigma alpha d", positive=True
)
keff = k / (1 + rho * (k - 1))


def stability_bracket(j_has_one_minus_rho):
    """dJ/dt under the paper's assumptions (i)-(iii), divided by its positive prefactor.

    (i)   dk_eff/dt = alpha / (1 + rho(k-1))
    (ii)  drho/dt   = 0                        (quasi-static)
    (iii) dsigma/dt = -d * sigma
          lambda constant
    """
    diversity = (1 - rho) if j_has_one_minus_rho else 1
    dkeff_dt = alpha / (1 + rho * (k - 1))
    dsig_dt = -d * sig
    dJdt = lam * (dkeff_dt * diversity * sig + keff * diversity * dsig_dt)
    prefactor = lam * diversity * sig
    return sp.simplify(dJdt / prefactor)


print("\nC-1 — Theorem 2.3 stability criterion")
bracket = stability_bracket(j_has_one_minus_rho=True)
boundary = sp.solve(sp.Eq(bracket, 0), alpha)
print(f"        dJ/dt sign expression : {bracket}")
print(f"        stability boundary    : alpha = {boundary[0]}")
check(
    "boundary is alpha = d*k (criterion alpha/k >= d), not alpha/k_eff >= d",
    sp.simplify(boundary[0] - d * k) == 0,
)
gap = sp.simplify((alpha / keff) / (alpha / k))
check(
    "stated criterion too permissive by exactly the design effect 1+rho(k-1)",
    sp.simplify(gap - (1 + rho * (k - 1))) == 0,
    f"ratio = {gap}",
)

# Concrete system the published criterion calls stable while J is strictly decreasing.
sub = {k: 100, rho: sp.Rational(1, 2), d: sp.Rational(1, 10)}
keff_v = float(keff.subs(sub))
a_stated = float((d * keff).subs(sub))
a_true = float((d * k).subs(sub))
a_test = (a_stated + a_true) / 2
bracket_v = float(bracket.subs({**sub, alpha: a_test}))
print(
    f"        counterexample k=100, rho=0.5, d=0.1, alpha={a_test:.4f}: "
    f"k_eff={keff_v:.4f}, alpha/k_eff={a_test/keff_v:.4f} > d  (published: STABLE)"
)
check(
    "counterexample: published criterion says stable while dJ/dt < 0",
    a_test / keff_v > 0.1 and bracket_v < 0,
    f"actual dJ/dt proportional to {bracket_v:.5f}",
)

print("\nC-1b — correction is invariant to the J-form drift (C-6)")
bracket_no_div = stability_bracket(j_has_one_minus_rho=False)
check(
    "same boundary alpha = d*k with or without the (1-rho) factor",
    sp.simplify(sp.solve(sp.Eq(bracket_no_div, 0), alpha)[0] - d * k) == 0,
)

print("\nC-2 — Table 8, Venezuela row")
ven = 0.667 / (1 + 0.299 * (0.667 - 1))
print(f"        k=0.667, rho=0.299 -> k_eff = {ven:.4f}   (published: 0.55)")
check("printed k_eff 0.55 does not match the formula", abs(ven - 0.55) > 0.15)
check("k = 0.667 < 1 is outside the Kish domain (k is a count >= 1)", 0.667 < 1)

print("\nC-3 — Table 8, Turkey row is degenerate")
turkey = [1.0 / (1 + r * (1.0 - 1)) for r in (0.0, 0.3, 0.9)]
print(f"        k=1 gives k_eff = {turkey} for rho = 0.0, 0.3, 0.9")
check(
    "at k=1 the formula is constant in rho, so rho=0.000 is unconstrained by k_eff",
    len(set(turkey)) == 1,
)

print("\nC-7 — Corollary 2.4 fails under the published criterion, holds under the correction")
rho_v, d_v, alpha_v = 0.5, 0.1, 1.0
ceiling = alpha_v * rho_v  # limit of alpha/k_eff as k -> infinity is alpha*rho
print(
    f"        rho=0.5, d=0.1, alpha=1: alpha/k_eff -> {ceiling} (floor), alpha/k -> 0"
)
check(
    "published criterion is never violated as k grows (alpha/k_eff floors above d)",
    ceiling > d_v,
)
check(
    "corrected criterion is eventually violated (alpha/k -> 0 < d)",
    alpha_v / 1e9 < d_v,
)

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) DID NOT REPRODUCE: {FAILURES}")
    raise SystemExit(1)
print("All defects reproduced as documented in CORRECTIONS_v3.md.")
