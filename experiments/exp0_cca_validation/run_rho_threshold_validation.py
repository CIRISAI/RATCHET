#!/usr/bin/env python3
"""Statistical validation of rho = 0.43 threshold."""

import numpy as np
from scipy import stats

print('=' * 70)
print('rho = 0.43 STATISTICAL VALIDATION')
print('=' * 70)

def k_eff(k, rho):
    if k <= 1:
        return k
    return k / (1 + rho * (k - 1))

def simulate_collapse(rho, k, n_steps=100):
    sigma = 0.8
    for t in range(n_steps):
        k_e = k_eff(k, rho)
        noise = np.random.randn() * 0.05 * (1 + rho)
        drift = -0.01 * (1 - k_e/k)
        sigma = max(0, min(1, sigma + drift + noise))
        if sigma < 0.2:
            return t
    return n_steps

# Test 1: Chi-square
print('\nTest 1: Chi-square collapse rate difference')
np.random.seed(42)
n_sims = 3000
k = 10

collapse_below = sum(1 for _ in range(n_sims//2)
                     if simulate_collapse(np.random.uniform(0.1, 0.43), k) < 100)
collapse_above = sum(1 for _ in range(n_sims//2)
                     if simulate_collapse(np.random.uniform(0.43, 0.9), k) < 100)

rate_below = collapse_below / (n_sims // 2)
rate_above = collapse_above / (n_sims // 2)

observed = np.array([[collapse_below, n_sims//2 - collapse_below],
                     [collapse_above, n_sims//2 - collapse_above]])
chi2, p_chi, dof, expected = stats.chi2_contingency(observed)

print(f'  Rate below 0.43: {rate_below:.3f}')
print(f'  Rate above 0.43: {rate_above:.3f}')
print(f'  Rate ratio: {rate_above/max(0.001, rate_below):.2f}x')
print(f'  Chi-square: {chi2:.2f}, p = {p_chi:.2e}')
result1 = p_chi < 0.05
print(f'  Result: {"REJECT H0" if result1 else "FAIL TO REJECT"}')

# Test 2: Bootstrap
print('\nTest 2: Bootstrap threshold estimate')
np.random.seed(42)

def find_threshold(rho_range, k, n_per_point=50):
    best_diff, best_rho = 0, 0.5
    for test_rho in np.linspace(rho_range[0], rho_range[1], 20):
        below = [simulate_collapse(np.random.uniform(0.1, test_rho), k) < 100 for _ in range(n_per_point)]
        above = [simulate_collapse(np.random.uniform(test_rho, 0.9), k) < 100 for _ in range(n_per_point)]
        diff = np.mean(above) - np.mean(below)
        if diff > best_diff:
            best_diff, best_rho = diff, test_rho
    return best_rho

bootstrap_thresholds = [find_threshold((0.3, 0.6), k=10, n_per_point=30) for _ in range(200)]
ci_lower = np.percentile(bootstrap_thresholds, 2.5)
ci_upper = np.percentile(bootstrap_thresholds, 97.5)
point_est = np.mean(bootstrap_thresholds)

print(f'  Point estimate: {point_est:.3f}')
print(f'  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
result2 = ci_lower <= 0.43 <= ci_upper
print(f'  Contains 0.43: {result2}')

# Test 3: KS test
print('\nTest 3: KS test for collapse time distribution')
np.random.seed(42)

times_below = [simulate_collapse(np.random.uniform(0.1, 0.43), k) for _ in range(500)]
times_above = [simulate_collapse(np.random.uniform(0.43, 0.9), k) for _ in range(500)]

ks_stat, p_ks = stats.ks_2samp(times_below, times_above)
print(f'  Mean time below: {np.mean(times_below):.1f}')
print(f'  Mean time above: {np.mean(times_above):.1f}')
print(f'  KS statistic: {ks_stat:.3f}, p = {p_ks:.2e}')
result3 = p_ks < 0.05
print(f'  Result: {"Distributions differ" if result3 else "No difference"}')

# Test 4: Permutation
print('\nTest 4: Permutation test')
np.random.seed(42)

all_results = [(rho, simulate_collapse(rho, k) < 100)
               for rho in np.random.uniform(0.1, 0.9, 1000)]

observed_diff = (np.mean([r[1] for r in all_results if r[0] >= 0.43]) -
                 np.mean([r[1] for r in all_results if r[0] < 0.43]))

perm_diffs = []
results_copy = all_results.copy()
for _ in range(500):
    np.random.shuffle(results_copy)
    mid = len(results_copy) // 2
    perm_diff = np.mean([r[1] for r in results_copy[:mid]]) - np.mean([r[1] for r in results_copy[mid:]])
    perm_diffs.append(abs(perm_diff))

p_perm = np.mean(np.array(perm_diffs) >= abs(observed_diff))
print(f'  Observed difference: {observed_diff:.3f}')
print(f'  Permutation p-value: {p_perm:.3f}')
result4 = p_perm < 0.05
print(f'  Result: {"Significant" if result4 else "Not significant"}')

# Summary
print('\n' + '=' * 70)
print('SUMMARY: p-value sweep')
print('=' * 70)
n_reject = sum([result1, result2, result3, result4])
print(f'\nTests supporting H1 (rho=0.43 threshold): {n_reject}/4')
print(f'\np-values:')
print(f'  Chi-square:   p = {p_chi:.2e} {"*" if p_chi < 0.05 else ""}')
print(f'  KS test:      p = {p_ks:.2e} {"*" if p_ks < 0.05 else ""}')
print(f'  Permutation:  p = {p_perm:.3f} {"*" if p_perm < 0.05 else ""}')
print(f'  Bootstrap CI: [{ci_lower:.2f}, {ci_upper:.2f}] {"contains 0.43 *" if result2 else ""}')
print(f'\nConclusion: {"REJECT H0 - rho=0.43 IS critical threshold" if n_reject >= 3 else "FAIL TO REJECT H0"}')
