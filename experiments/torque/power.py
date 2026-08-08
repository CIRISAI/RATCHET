#!/usr/bin/env python3
"""Power for TORQUE's paired concordance contrasts. Every term MEASURED.

  self-discordance  0.037   same model, same item, same prompt, temp 0.7, n=80
  baseline          0.68    axiotic_primary, Llama-4 Scout, n=40
  run-to-run SD     0.035   72 prior HE-300 runs at n=300, 8 models, 5-11 repeats
  binomial-only SD  0.028   at p=0.62, n=300
  excess variance   1.2x on SD -> ICC ~ 0.05 for 10-item conversations

The contrasts are PAIRED: every arm sees the same items, so item difficulty
cancels and the estimator is McNemar on discordant pairs.

    N_items = (z_a/2 + z_b)^2 * pi_d / delta^2   * designeffect
    pi_d    ~ delta + 2*self_discordance          (discordance from effect + noise)
    DE      = 1 + (m-1)*ICC,  m = 10 items per conversation
"""
import math

Z = {0.80: 0.8416, 0.90: 1.2816}
SELF_DISC = 0.037
M = 10

def n_items(delta, power=0.80, alpha=0.05, icc=0.05, equiv=False):
    za = 1.6449 if equiv else 1.9600            # TOST is one-sided per side
    zb = Z[power]
    pi_d = delta + 2 * SELF_DISC
    de = 1 + (M - 1) * icc
    return math.ceil((za + zb) ** 2 * pi_d / delta ** 2 * de)

if __name__ == "__main__":
    USD_PER_ITEM = 334 / 1200          # measured: full six-arm run / staked items
    print("PAIRED DIFFERENCE CONTRASTS (pipeline_effect, values_effect, …)\n")
    print(f"{'delta':>6} | " + " | ".join(f"ICC {i:<4}" for i in (0.05, 0.10, 0.20)))
    print("-" * 52)
    for d in (0.20, 0.15, 0.12, 0.10, 0.08, 0.05):
        cells = []
        for icc in (0.05, 0.10, 0.20):
            n = n_items(d, icc=icc)
            cells.append(f"{n:>4}i/{math.ceil(n/M):>3}c")
        print(f"{d:>6.2f} | " + " | ".join(cells))
    print("\n  i = items, c = 10-item conversations, per contrast, 80% power, alpha .05")

    print("\n\nWHAT A GIVEN BUDGET BUYS  (six arms, all four strata)\n")
    print(f"{'items':>6} {'conv':>5} {'cost':>7}   {'detectable delta at ICC 0.05 / 0.10 / 0.20':<44}")
    print("-" * 74)
    for items in (400, 600, 800, 1200):
        row = []
        for icc in (0.05, 0.10, 0.20):
            lo, hi = 0.01, 0.50
            for _ in range(60):
                mid = (lo + hi) / 2
                if n_items(mid, icc=icc) > items: lo = mid
                else: hi = mid
            row.append(f"{hi:.3f}")
        print(f"{items:>6} {items//M:>5} {items*USD_PER_ITEM:>6.0f}$   "
              + "  /  ".join(row))

    print("\n\nEQUIVALENCE KILLS (reversion_null, form_vs_content_null), bound 0.15\n")
    for icc in (0.05, 0.10, 0.20):
        n = n_items(0.15, icc=icc, equiv=True)
        print(f"  ICC {icc:<5} {n:>4} items = {math.ceil(n/M):>3} conversations")
    print("\n  NOTE the regime says equivalence needs MORE than a difference test at the")
    print("  same bound. At these parameters it needs FEWER: TOST uses z=1.645 per side")
    print("  against 1.96 two-sided, so (1.645+0.84)^2 = 6.2 vs (1.96+0.84)^2 = 7.8.")
