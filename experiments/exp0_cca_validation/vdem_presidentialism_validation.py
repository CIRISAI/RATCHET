"""
RATCHET Validation: V-Dem-style Presidentialism as rho

Uses Polity5 executive constraints (xconst) inverted as a presidentialism proxy.

Concept:
- V-Dem's presidentialism index (v2xnp_pres) measures concentration of power
- Polity5's xconst measures executive constraints (inverse concept)
- We map: rho = (8 - xconst) / 7
  - xconst=1 (unlimited power) -> rho=1.0 (max constraint correlation)
  - xconst=7 (executive parity) -> rho=0.14 (independent constraints)

This tests whether DIRECT measurement of executive concentration improves
k_eff prediction over the WGI variance-based proxy (AUC=0.48).

Target: Beat sigma baseline (AUC=0.65)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# WGI indicator columns
WGI_INDICATORS = ['CC.EST', 'GE.EST', 'PV.EST', 'RQ.EST', 'RL.EST', 'VA.EST']


def compute_keff(k: float, rho: float) -> float:
    """Compute effective constraints using Kish (1965) formula."""
    if k <= 1:
        return k
    if rho >= 1.0:
        return 1.0
    if rho <= 0.0:
        return k
    return k / (1 + rho * (k - 1))


def compute_pres_rho(xconst: float) -> float:
    """
    Convert executive constraints to presidentialism (rho).

    xconst 1-7 scale:
    1 = Unlimited authority (no constraints) -> rho = 1.0
    7 = Executive parity/subordination -> rho ~ 0.14

    Maps to: rho = (8 - xconst) / 7
    """
    if pd.isna(xconst) or xconst < 1 or xconst > 7:
        return 0.5  # Default for missing/invalid
    return (8 - xconst) / 7.0


def compute_sigma(row: pd.Series, indicators: list) -> float:
    """Compute sustainability as mean governance quality."""
    values = [row[ind] for ind in indicators if pd.notna(row.get(ind))]
    if len(values) == 0:
        return 0.5
    normalized = [(v + 2.5) / 5.0 for v in values]
    return max(0.0, min(1.0, np.mean(normalized)))


def compute_wgi_rho(row: pd.Series, indicators: list) -> float:
    """Compute rho from WGI indicator variance (OLD METHOD - for comparison)."""
    values = [row[ind] for ind in indicators if pd.notna(row.get(ind))]
    if len(values) < 2:
        return 0.5
    values = [(v + 2.5) / 5.0 for v in values]
    values = [max(0, min(1, v)) for v in values]
    mean_val = np.mean(values)
    std_val = np.std(values)
    if mean_val == 0:
        return 0.5
    cv = std_val / (mean_val + 0.01)
    rho = 1.0 - min(1.0, cv * 2)
    return max(0.0, min(1.0, rho))


# Country code mapping (Polity scode -> WGI country_code)
CODE_MAP = {
    'USA': 'USA', 'GBR': 'GBR', 'RUS': 'RUS', 'CHN': 'CHN',
    'FRN': 'FRA', 'GMY': 'DEU', 'ITA': 'ITA', 'JPN': 'JPN',
    'CAN': 'CAN', 'AUL': 'AUS', 'BRA': 'BRA', 'IND': 'IND',
    'MEX': 'MEX', 'ARG': 'ARG', 'TAW': 'TWN', 'KOR': 'KOR',
    'SAF': 'ZAF', 'TUR': 'TUR', 'POL': 'POL', 'ESP': 'ESP',
    'NTH': 'NLD', 'BEL': 'BEL', 'SWD': 'SWE', 'NOR': 'NOR',
    'DEN': 'DNK', 'FIN': 'FIN', 'SWZ': 'CHE', 'AUS': 'AUT',
    'GRC': 'GRC', 'POR': 'PRT', 'IRE': 'IRL', 'CZE': 'CZE',
    'HUN': 'HUN', 'ROM': 'ROU', 'BUL': 'BGR', 'UKR': 'UKR',
    'BLR': 'BLR', 'SRB': 'SRB', 'CRO': 'HRV', 'SLO': 'SVN',
    'SLV': 'SVK', 'LIT': 'LTU', 'LAT': 'LVA', 'EST': 'EST',
    'GRG': 'GEO', 'ARM': 'ARM', 'AZE': 'AZE', 'KAZ': 'KAZ',
    'UZB': 'UZB', 'TAJ': 'TJK', 'KYR': 'KGZ', 'TKM': 'TKM',
    'AFG': 'AFG', 'PAK': 'PAK', 'BNG': 'BGD', 'MYA': 'MMR',
    'THI': 'THA', 'MAL': 'MYS', 'SIN': 'SGP', 'INS': 'IDN',
    'PHI': 'PHL', 'VIE': 'VNM', 'CAM': 'KHM', 'LAO': 'LAO',
    'NEP': 'NPL', 'SRI': 'LKA', 'IRN': 'IRN', 'IRQ': 'IRQ',
    'SAU': 'SAU', 'YEM': 'YEM', 'SYR': 'SYR', 'JOR': 'JOR',
    'LEB': 'LBN', 'ISR': 'ISR', 'EGY': 'EGY', 'LIB': 'LBY',
    'TUN': 'TUN', 'ALG': 'DZA', 'MOR': 'MAR', 'NIG': 'NER',
    'NIA': 'NGA', 'ETH': 'ETH', 'KEN': 'KEN', 'TAN': 'TZA',
    'UGA': 'UGA', 'RWA': 'RWA', 'DRC': 'COD', 'ZAM': 'ZMB',
    'ZIM': 'ZWE', 'ANG': 'AGO', 'MZM': 'MOZ', 'GHA': 'GHA',
    'CIV': 'CIV', 'SEN': 'SEN', 'MLI': 'MLI', 'BFO': 'BFA',
    'SUD': 'SDN', 'CHA': 'TCD', 'CMR': 'CMR', 'CON': 'COG',
    'GAB': 'GAB', 'MAG': 'MDG', 'BOT': 'BWA', 'NAM': 'NAM',
    'LES': 'LSO', 'SWA': 'SWZ', 'MAW': 'MWI', 'TOG': 'TGO',
    'BEN': 'BEN', 'GUI': 'GIN', 'LBR': 'LBR', 'SIE': 'SLE',
    'GAM': 'GMB', 'MRT': 'MRT', 'CAO': 'CAF', 'EQG': 'GNQ',
    'HAI': 'HTI', 'DOM': 'DOM', 'JAM': 'JAM', 'CUB': 'CUB',
    'GUA': 'GTM', 'HON': 'HND', 'SAL': 'SLV', 'NIC': 'NIC',
    'COS': 'CRI', 'PAN': 'PAN', 'COL': 'COL', 'VEN': 'VEN',
    'ECU': 'ECU', 'PER': 'PER', 'BOL': 'BOL', 'PAR': 'PRY',
    'CHL': 'CHL', 'URU': 'URY', 'ALB': 'ALB', 'MAC': 'MKD',
    'BOS': 'BIH', 'MNT': 'MNE', 'KOS': 'XKX', 'MOL': 'MDA',
    'MNG': 'MNG', 'PRK': 'PRK', 'BHU': 'BTN',
}


def main():
    print("=" * 80)
    print("RATCHET VALIDATION: Presidentialism-based rho (V-Dem proxy)")
    print("=" * 80)
    print()
    print("Hypothesis: Using executive concentration (1 - xconst/7) as rho")
    print("should outperform WGI variance-based rho (AUC=0.48)")
    print()
    print("Target: Beat sigma baseline (AUC=0.65)")
    print()

    base_path = Path(__file__).parent.parent

    # Load data
    print("-" * 80)
    print("1. LOADING DATA")
    print("-" * 80)

    polity = pd.read_excel(base_path / 'data' / 'validation' / 'polity5.xls')
    wgi = pd.read_csv(base_path / 'data' / 'institutional' / 'wgi_data.csv')

    print(f"Polity5: {len(polity):,} observations")
    print(f"WGI: {len(wgi):,} observations")

    # Filter to valid xconst and WGI era
    polity_valid = polity[
        (polity['xconst'] >= 1) & (polity['xconst'] <= 7) &
        (polity['year'] >= 1996) & (polity['year'] <= 2020)
    ].copy()

    polity_valid['wgi_code'] = polity_valid['scode'].map(CODE_MAP)
    print(f"Polity5 valid (xconst 1-7, 1996-2020): {len(polity_valid):,}")

    # Get adverse transitions
    adverse = polity[
        polity['regtrans'].isin([-2, -1]) &
        (polity['year'] >= 1996) & (polity['year'] <= 2020)
    ].copy()
    adverse['wgi_code'] = adverse['scode'].map(CODE_MAP)
    print(f"Adverse regime transitions: {len(adverse)}")

    # Merge
    print()
    print("-" * 80)
    print("2. MERGING DATA")
    print("-" * 80)

    merged = pd.merge(
        polity_valid[['scode', 'wgi_code', 'country', 'year', 'xconst']],
        wgi[['country_code', 'year'] + WGI_INDICATORS],
        left_on=['wgi_code', 'year'],
        right_on=['country_code', 'year'],
        how='inner'
    )
    print(f"Merged observations: {len(merged):,}")
    print(f"Countries: {merged['country'].nunique()}")

    # Compute variables
    print()
    print("-" * 80)
    print("3. COMPUTING RATCHET VARIABLES")
    print("-" * 80)

    # Presidentialism-based rho (NEW METHOD)
    merged['rho_pres'] = merged['xconst'].apply(compute_pres_rho)

    # WGI variance-based rho (OLD METHOD - for comparison)
    merged['rho_wgi'] = merged.apply(lambda r: compute_wgi_rho(r, WGI_INDICATORS), axis=1)

    # Sigma (governance quality)
    merged['sigma'] = merged.apply(lambda r: compute_sigma(r, WGI_INDICATORS), axis=1)

    # k = 6 (number of WGI indicators as constraint count)
    k = 6

    # k_eff with presidentialism rho
    merged['k_eff_pres'] = merged['rho_pres'].apply(lambda rho: compute_keff(k, rho))

    # k_eff with WGI rho (baseline comparison)
    merged['k_eff_wgi'] = merged['rho_wgi'].apply(lambda rho: compute_keff(k, rho))

    print(f"\nrho_pres (presidentialism-based):")
    print(f"  min: {merged['rho_pres'].min():.3f}")
    print(f"  max: {merged['rho_pres'].max():.3f}")
    print(f"  mean: {merged['rho_pres'].mean():.3f}")

    print(f"\nrho_wgi (variance-based):")
    print(f"  min: {merged['rho_wgi'].min():.3f}")
    print(f"  max: {merged['rho_wgi'].max():.3f}")
    print(f"  mean: {merged['rho_wgi'].mean():.3f}")

    print(f"\nk_eff_pres:")
    print(f"  min: {merged['k_eff_pres'].min():.2f}")
    print(f"  max: {merged['k_eff_pres'].max():.2f}")
    print(f"  mean: {merged['k_eff_pres'].mean():.2f}")

    # Create collapse labels
    print()
    print("-" * 80)
    print("4. CREATING COLLAPSE LABELS")
    print("-" * 80)

    collapse_set = set()
    for _, row in adverse.iterrows():
        if pd.notna(row['wgi_code']):
            collapse_set.add((row['wgi_code'], int(row['year'])))
        collapse_set.add((row['scode'], int(row['year'])))

    def has_collapse_ahead(row, lookahead=3):
        for future_year in range(int(row['year']) + 1, int(row['year']) + lookahead + 1):
            if (row['wgi_code'], future_year) in collapse_set:
                return 1
            if (row['scode'], future_year) in collapse_set:
                return 1
        return 0

    merged['collapse_3yr'] = merged.apply(lambda r: has_collapse_ahead(r, 3), axis=1)

    print(f"Positive cases (collapse in 3 years): {merged['collapse_3yr'].sum()}")
    print(f"Base rate: {merged['collapse_3yr'].mean():.2%}")

    # If insufficient positives, use sigma-drop proxy
    if merged['collapse_3yr'].sum() < 30:
        print("\nUsing sigma-drop proxy for additional collapse events...")
        merged = merged.sort_values(['country', 'year'])
        merged['sigma_prev'] = merged.groupby('country')['sigma'].shift(1)
        merged['sigma_drop'] = merged['sigma_prev'] - merged['sigma']
        threshold = merged['sigma_drop'].quantile(0.90)
        merged['is_sigma_collapse'] = (merged['sigma_drop'] > threshold).astype(int)

        def sigma_collapse_ahead(group, lookahead=3):
            labels = []
            collapse_years = set(group[group['is_sigma_collapse'] == 1]['year'])
            for _, row in group.iterrows():
                label = 0
                for future_year in range(int(row['year']) + 1, int(row['year']) + lookahead + 1):
                    if future_year in collapse_years:
                        label = 1
                        break
                labels.append(label)
            group = group.copy()
            group['collapse_sigma'] = labels
            return group

        merged = merged.groupby('country', group_keys=False).apply(
            lambda g: sigma_collapse_ahead(g, 3)
        )

        # Combine: collapse if either Polity OR sigma-drop
        merged['label'] = ((merged['collapse_3yr'] == 1) |
                          (merged.get('collapse_sigma', 0) == 1)).astype(int)
        print(f"Combined positive cases: {merged['label'].sum()}")
    else:
        merged['label'] = merged['collapse_3yr']

    # Validation
    print()
    print("-" * 80)
    print("5. VALIDATION RESULTS")
    print("-" * 80)

    valid_df = merged.dropna(subset=['label', 'k_eff_pres', 'k_eff_wgi', 'sigma', 'rho_pres'])
    print(f"\nValidation set: {len(valid_df):,} observations")
    print(f"Positive labels: {int(valid_df['label'].sum())}")
    print(f"Base rate: {valid_df['label'].mean():.2%}")

    if valid_df['label'].sum() < 5:
        print("\nERROR: Insufficient positive cases")
        return

    y_true = valid_df['label'].values
    results = {}

    # 1. Sigma baseline (low sigma = high risk)
    sigma_prob = 1 - valid_df['sigma'].values
    try:
        sigma_auc = roc_auc_score(y_true, sigma_prob)
    except:
        sigma_auc = 0.5
    results['sigma_baseline'] = sigma_auc

    # 2. k_eff with presidentialism rho (NEW - low k_eff = high risk)
    keff_pres_prob = 1 - (valid_df['k_eff_pres'].values / 6.0)
    try:
        keff_pres_auc = roc_auc_score(y_true, keff_pres_prob)
    except:
        keff_pres_auc = 0.5
    results['k_eff_presidentialism'] = keff_pres_auc

    # 3. k_eff with WGI rho (OLD - for comparison)
    keff_wgi_prob = 1 - (valid_df['k_eff_wgi'].values / 6.0)
    try:
        keff_wgi_auc = roc_auc_score(y_true, keff_wgi_prob)
    except:
        keff_wgi_auc = 0.5
    results['k_eff_wgi_variance'] = keff_wgi_auc

    # 4. Raw presidentialism rho (high = high risk)
    rho_pres_prob = valid_df['rho_pres'].values
    try:
        rho_pres_auc = roc_auc_score(y_true, rho_pres_prob)
    except:
        rho_pres_auc = 0.5
    results['rho_presidentialism'] = rho_pres_auc

    # 5. Combined: low k_eff + low sigma (additive risk)
    combined_risk = (1 - valid_df['k_eff_pres'].values / 6.0) * 0.5 + \
                   (1 - valid_df['sigma'].values) * 0.5
    try:
        combined_auc = roc_auc_score(y_true, combined_risk)
    except:
        combined_auc = 0.5
    results['combined_keff_sigma'] = combined_auc

    # 6. Deterioration: declining k_eff trend
    valid_df = valid_df.sort_values(['country', 'year'])
    valid_df['k_eff_prev'] = valid_df.groupby('country')['k_eff_pres'].shift(1)
    valid_df['k_eff_delta'] = valid_df['k_eff_pres'] - valid_df['k_eff_prev']

    valid_with_delta = valid_df.dropna(subset=['k_eff_delta'])
    if len(valid_with_delta) > 50 and valid_with_delta['label'].sum() > 5:
        y_delta = valid_with_delta['label'].values
        delta_prob = -valid_with_delta['k_eff_delta'].values
        delta_prob = (delta_prob - delta_prob.min()) / (delta_prob.max() - delta_prob.min() + 1e-6)
        try:
            delta_auc = roc_auc_score(y_delta, delta_prob)
        except:
            delta_auc = 0.5
        results['k_eff_decline'] = delta_auc

    # Print results
    print()
    print("=" * 60)
    print("AUC COMPARISON")
    print("=" * 60)
    print(f"\n{'Predictor':<25} {'AUC':>10} {'vs 0.65':>12} {'vs 0.48':>12}")
    print("-" * 60)

    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        vs_sigma = auc - 0.65
        vs_wgi = auc - 0.48
        sign_sigma = '+' if vs_sigma >= 0 else ''
        sign_wgi = '+' if vs_wgi >= 0 else ''
        print(f"{name:<25} {auc:>10.3f} {sign_sigma}{vs_sigma:>11.3f} {sign_wgi}{vs_wgi:>11.3f}")

    # Temporal holdout
    print()
    print("-" * 60)
    print("TEMPORAL HOLDOUT (Train <= 2015, Test >= 2016)")
    print("-" * 60)

    train_df = valid_df[valid_df['year'] <= 2015]
    test_df = valid_df[valid_df['year'] >= 2016]

    print(f"\nTrain: {len(train_df)} obs, {train_df['label'].sum():.0f} positives")
    print(f"Test: {len(test_df)} obs, {test_df['label'].sum():.0f} positives")

    if len(test_df) > 0 and test_df['label'].sum() > 0:
        y_test = test_df['label'].values

        print(f"\n{'Predictor':<25} {'AUC':>10}")
        print("-" * 40)

        for name, col, invert in [
            ('sigma_baseline', 'sigma', True),
            ('k_eff_presidentialism', 'k_eff_pres', True),
            ('k_eff_wgi_variance', 'k_eff_wgi', True),
            ('rho_presidentialism', 'rho_pres', False),
        ]:
            if col not in test_df.columns:
                continue
            if invert:
                prob = 1 - (test_df[col].values / test_df[col].max())
            else:
                prob = test_df[col].values
            try:
                auc = roc_auc_score(y_test, prob)
            except:
                auc = 0.5
            print(f"{name:<25} {auc:>10.3f}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best = max(results.items(), key=lambda x: x[1])
    pres_auc = results.get('k_eff_presidentialism', 0.5)
    wgi_auc = results.get('k_eff_wgi_variance', 0.5)
    sigma_auc = results.get('sigma_baseline', 0.5)

    print(f"""
Key Results:
  Sigma baseline AUC:            {sigma_auc:.3f}
  k_eff (presidentialism) AUC:   {pres_auc:.3f}
  k_eff (WGI variance) AUC:      {wgi_auc:.3f}
  Best predictor: {best[0]} (AUC = {best[1]:.3f})

Improvement Analysis:
  Presidentialism vs WGI variance: {'+' if pres_auc > wgi_auc else ''}{pres_auc - wgi_auc:.3f}
  Presidentialism vs sigma (0.65): {'+' if pres_auc > 0.65 else ''}{pres_auc - 0.65:.3f}

Conclusion:
  {'PRESIDENTIALISM RHO IMPROVES over WGI variance' if pres_auc > wgi_auc else 'No improvement over WGI variance'}
  {'BEATS SIGMA BASELINE' if pres_auc > 0.65 else 'Does not beat sigma baseline'}
""")

    # Save results
    results_df = pd.DataFrame([
        {'predictor': k, 'auc': v, 'vs_sigma': v - 0.65, 'vs_wgi': v - 0.48}
        for k, v in results.items()
    ])
    results_path = base_path / 'experiments' / 'vdem_presidentialism_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
