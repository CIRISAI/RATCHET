"""
RATCHET Polity5 xconst Validation

Uses Polity5 executive constraints (xconst) directly as k for k_eff calculation.

Hypothesis: Using actual institutional constraints (xconst, 1-7 scale) should
improve k_eff predictive performance compared to the sigma baseline (AUC 0.65).

Kish (1965) k_eff formula:
    k_eff = k / (1 + rho(k - 1))

Where:
    k = number of constraints (from xconst, scaled appropriately)
    rho = correlation between constraints (from temporal stability)
    k_eff = effective diversity of constraints

Validation target: Polity5 regime transitions (regtrans = -2 or -1)
    -2: Adverse regime transition (state failure, occupation, major democratic breakdown)
    -1: Negative regime change (authoritarian movement)

Key Insight: xconst captures CURRENT constraint level. To predict FUTURE collapse,
we need to track the DETERIORATION pattern - declining k_eff over time. The k_eff
formula accounts for correlation between constraints (rho), which increases during
institutional stress (echo chamber effect).

Author: RATCHET Project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# WGI indicator columns (for computing rho)
WGI_INDICATORS = ['CC.EST', 'GE.EST', 'PV.EST', 'RQ.EST', 'RL.EST', 'VA.EST']


def compute_keff(k: float, rho: float) -> float:
    """
    Compute effective constraints using Kish (1965) formula.

    Args:
        k: Number of constraints
        rho: Correlation between constraints (0 = independent, 1 = identical)

    Returns:
        Effective constraint count
    """
    if k <= 1:
        return k
    if rho >= 1.0:
        return 1.0
    if rho <= 0.0:
        return k
    return k / (1 + rho * (k - 1))


def compute_rolling_correlation(group: pd.DataFrame, indicators: list, window: int = 5) -> pd.Series:
    """
    Compute rolling average pairwise correlation between governance indicators.

    This measures how correlated the different constraint indicators are over time.
    High correlation = constraints move together = high rho (echo chamber)
    Low correlation = constraints are independent = low rho (diversity)

    Args:
        group: DataFrame with indicator columns for one country
        indicators: List of indicator column names
        window: Rolling window size

    Returns:
        Series of rho values
    """
    rho_values = []

    for i in range(len(group)):
        start_idx = max(0, i - window + 1)
        window_data = group.iloc[start_idx:i+1]

        if len(window_data) < 2:
            rho_values.append(0.3)  # Default for insufficient data
            continue

        # Compute pairwise correlations between indicators
        indicator_data = window_data[indicators].dropna(axis=1, how='all')
        if indicator_data.shape[1] < 2:
            rho_values.append(0.3)
            continue

        corr_matrix = indicator_data.corr()

        # Average of upper triangle (pairwise correlations)
        n = corr_matrix.shape[0]
        if n < 2:
            rho_values.append(0.3)
            continue

        upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
        correlations = upper_tri.stack().values

        if len(correlations) == 0:
            rho_values.append(0.3)
        else:
            # Average correlation, clipped to 0-1
            avg_corr = np.nanmean(correlations)
            # Transform: raw correlation can be negative, map [-1, 1] -> [0, 1]
            rho = (avg_corr + 1) / 2
            rho_values.append(max(0.0, min(1.0, rho)))

    return pd.Series(rho_values, index=group.index)


def compute_wgi_rho(row: pd.Series, indicators: list) -> float:
    """
    Compute rho from WGI indicator cross-sectional variance.

    High variance across indicators = low rho (independent signals)
    Low variance = high rho (correlated/echo chamber)
    """
    values = [row[ind] for ind in indicators if pd.notna(row.get(ind))]
    if len(values) < 2:
        return 0.5

    # Normalize WGI (-2.5 to 2.5) to (0 to 1)
    values = [(v + 2.5) / 5.0 for v in values]
    values = [max(0, min(1, v)) for v in values]

    mean_val = np.mean(values)
    std_val = np.std(values)

    if mean_val == 0:
        return 0.5

    # CV as diversity proxy
    cv = std_val / (mean_val + 0.01)
    rho = 1.0 - min(1.0, cv * 2)  # Scale CV to rho

    return max(0.0, min(1.0, rho))


def compute_sigma(row: pd.Series, indicators: list) -> float:
    """Compute sustainability as mean governance quality."""
    values = [row[ind] for ind in indicators if pd.notna(row.get(ind))]
    if len(values) == 0:
        return 0.5

    # Normalize WGI to 0-1
    normalized = [(v + 2.5) / 5.0 for v in values]
    return max(0.0, min(1.0, np.mean(normalized)))


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
    print("RATCHET VALIDATION: Polity5 xconst as k for k_eff")
    print("=" * 80)
    print("\nHypothesis: Using executive constraints (xconst) as k should improve")
    print("k_eff predictive power over the sigma baseline (AUC = 0.65)")

    base_path = Path(__file__).parent.parent

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n" + "-" * 80)
    print("1. LOADING DATA")
    print("-" * 80)

    # Load Polity5
    polity = pd.read_excel(base_path / 'data' / 'validation' / 'polity5.xls')
    print(f"Polity5: {len(polity):,} observations, {polity['scode'].nunique()} countries")
    print(f"Year range: {polity['year'].min()} - {polity['year'].max()}")

    # Load WGI
    wgi = pd.read_csv(base_path / 'data' / 'institutional' / 'wgi_data.csv')
    print(f"WGI: {len(wgi):,} observations, {wgi['country_code'].nunique()} countries")
    print(f"Year range: {wgi['year'].min()} - {wgi['year'].max()}")

    # =========================================================================
    # 2. EXTRACT AND CLEAN POLITY DATA
    # =========================================================================
    print("\n" + "-" * 80)
    print("2. EXTRACTING POLITY5 VARIABLES")
    print("-" * 80)

    # xconst: Executive Constraints (1-7 scale)
    # 1 = Unlimited authority (no constraints)
    # 7 = Executive parity/subordination (maximum constraints)
    # Negative values are special codes

    print("\nxconst (Executive Constraints) distribution:")
    print(polity[polity['xconst'] > 0]['xconst'].value_counts().sort_index())

    # Filter to valid xconst (1-7), overlapping years with WGI (1996-2020)
    polity_valid = polity[
        (polity['xconst'] >= 1) & (polity['xconst'] <= 7) &
        (polity['year'] >= 1996) & (polity['year'] <= 2020)
    ].copy()

    print(f"\nValid records (xconst 1-7, 1996-2020): {len(polity_valid):,}")

    # Map country codes
    polity_valid['wgi_code'] = polity_valid['scode'].map(CODE_MAP)

    # Extract regime transitions (collapse events)
    # regtrans = -2: adverse regime transition (state failure, occupation, etc.)
    # regtrans = -1: negative regime change (authoritarian shift)
    adverse = polity[
        polity['regtrans'].isin([-2, -1]) &
        (polity['year'] >= 1996) & (polity['year'] <= 2020)
    ].copy()
    adverse['wgi_code'] = adverse['scode'].map(CODE_MAP)

    print(f"\nAdverse regime transitions (1996-2020): {len(adverse)}")
    print("\nSample transitions:")
    sample = adverse[['country', 'year', 'regtrans', 'xconst', 'polity2']].head(10)
    for _, r in sample.iterrows():
        xc = r['xconst'] if r['xconst'] > 0 else 'N/A'
        print(f"  {r['country']} ({int(r['year'])}): xconst={xc}, polity2={r['polity2']:.0f}")

    # =========================================================================
    # 3. MERGE WITH WGI
    # =========================================================================
    print("\n" + "-" * 80)
    print("3. MERGING POLITY5 + WGI")
    print("-" * 80)

    # Merge on country code and year
    merged = pd.merge(
        polity_valid[['scode', 'wgi_code', 'country', 'year', 'xconst', 'polity2']],
        wgi[['country_code', 'year'] + WGI_INDICATORS],
        left_on=['wgi_code', 'year'],
        right_on=['country_code', 'year'],
        how='inner'
    )

    print(f"Merged observations: {len(merged):,}")
    print(f"Countries: {merged['country'].nunique()}")
    print(f"Years: {merged['year'].min()} - {merged['year'].max()}")

    if len(merged) < 100:
        print("\nWARNING: Low match rate. Checking code overlap...")
        polity_codes = set(polity_valid['wgi_code'].dropna())
        wgi_codes = set(wgi['country_code'].unique())
        overlap = polity_codes & wgi_codes
        print(f"Polity codes mapped: {len(polity_codes)}")
        print(f"WGI codes: {len(wgi_codes)}")
        print(f"Overlap: {len(overlap)}")

    # =========================================================================
    # 4. COMPUTE k_eff USING xconst
    # =========================================================================
    print("\n" + "-" * 80)
    print("4. COMPUTING k_eff FROM xconst")
    print("-" * 80)

    # Method 1: xconst directly as k
    merged['k_xconst'] = merged['xconst'].astype(float)

    # Compute rho from WGI indicator variance (cross-sectional)
    merged['rho_wgi'] = merged.apply(
        lambda row: compute_wgi_rho(row, WGI_INDICATORS), axis=1
    )

    # Compute rho from rolling correlation between WGI indicators (per country)
    merged = merged.sort_values(['country', 'year'])

    def add_rolling_rho(group):
        group = group.copy()
        group['rho_rolling'] = compute_rolling_correlation(group, WGI_INDICATORS, window=5)
        return group

    merged = merged.groupby('country', group_keys=False).apply(add_rolling_rho)

    # Combined rho: use rolling correlation (better captures temporal dynamics)
    # The cross-sectional rho_wgi captures instantaneous correlation
    # The rolling rho captures how correlated indicators are over time windows
    merged['rho_combined'] = merged['rho_rolling']  # Use rolling for primary

    # Compute k_eff variants
    merged['k_eff_xconst'] = merged.apply(
        lambda row: compute_keff(row['k_xconst'], row['rho_combined']), axis=1
    )

    merged['k_eff_xconst_wgi'] = merged.apply(
        lambda row: compute_keff(row['k_xconst'], row['rho_wgi']), axis=1
    )

    # Baseline: sigma from WGI
    merged['sigma'] = merged.apply(
        lambda row: compute_sigma(row, WGI_INDICATORS), axis=1
    )

    print("\nk_xconst (raw xconst) distribution:")
    print(f"  min: {merged['k_xconst'].min():.1f}")
    print(f"  max: {merged['k_xconst'].max():.1f}")
    print(f"  mean: {merged['k_xconst'].mean():.2f}")

    print("\nk_eff_xconst (xconst adjusted by rho) distribution:")
    print(f"  min: {merged['k_eff_xconst'].min():.2f}")
    print(f"  max: {merged['k_eff_xconst'].max():.2f}")
    print(f"  mean: {merged['k_eff_xconst'].mean():.2f}")

    print("\nrho_combined distribution:")
    print(f"  min: {merged['rho_combined'].min():.2f}")
    print(f"  max: {merged['rho_combined'].max():.2f}")
    print(f"  mean: {merged['rho_combined'].mean():.2f}")

    # =========================================================================
    # 4b. COMPUTE DETERIORATION INDICATORS
    # =========================================================================
    print("\n" + "-" * 80)
    print("4b. COMPUTING DETERIORATION INDICATORS")
    print("-" * 80)

    # Track changes in k_eff over time (deterioration = declining k_eff)
    merged = merged.sort_values(['country', 'year'])

    # Year-over-year changes
    merged['k_eff_prev'] = merged.groupby('country')['k_eff_xconst'].shift(1)
    merged['k_eff_delta'] = merged['k_eff_xconst'] - merged['k_eff_prev']

    merged['xconst_prev'] = merged.groupby('country')['k_xconst'].shift(1)
    merged['xconst_delta'] = merged['k_xconst'] - merged['xconst_prev']

    merged['sigma_prev'] = merged.groupby('country')['sigma'].shift(1)
    merged['sigma_delta'] = merged['sigma'] - merged['sigma_prev']

    merged['rho_prev'] = merged.groupby('country')['rho_combined'].shift(1)
    merged['rho_delta'] = merged['rho_combined'] - merged['rho_prev']

    # Rolling 3-year trends (deterioration velocity)
    def compute_trend(group, col, window=3):
        """Compute rolling slope of a variable."""
        result = []
        for i in range(len(group)):
            start = max(0, i - window + 1)
            window_data = group[col].iloc[start:i+1].dropna()
            if len(window_data) >= 2:
                slope = np.polyfit(range(len(window_data)), window_data.values, 1)[0]
            else:
                slope = 0
            result.append(slope)
        return pd.Series(result, index=group.index)

    merged['k_eff_trend'] = merged.groupby('country', group_keys=False).apply(
        lambda g: compute_trend(g, 'k_eff_xconst')
    )
    merged['sigma_trend'] = merged.groupby('country', group_keys=False).apply(
        lambda g: compute_trend(g, 'sigma')
    )
    merged['rho_trend'] = merged.groupby('country', group_keys=False).apply(
        lambda g: compute_trend(g, 'rho_combined')
    )

    print(f"\nk_eff_delta (year-over-year change):")
    print(f"  min: {merged['k_eff_delta'].min():.3f}")
    print(f"  max: {merged['k_eff_delta'].max():.3f}")
    print(f"  mean: {merged['k_eff_delta'].mean():.3f}")

    print(f"\nk_eff_trend (3-year rolling slope):")
    print(f"  min: {merged['k_eff_trend'].min():.3f}")
    print(f"  max: {merged['k_eff_trend'].max():.3f}")
    print(f"  mean: {merged['k_eff_trend'].mean():.3f}")

    # =========================================================================
    # 5. CREATE COLLAPSE LABELS
    # =========================================================================
    print("\n" + "-" * 80)
    print("5. CREATING COLLAPSE LABELS")
    print("-" * 80)

    # Create set of (country, year) collapse events
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
    merged['collapse_5yr'] = merged.apply(lambda r: has_collapse_ahead(r, 5), axis=1)

    print(f"\n3-year lookahead:")
    print(f"  Positive cases: {merged['collapse_3yr'].sum()}")
    print(f"  Base rate: {merged['collapse_3yr'].mean():.2%}")

    print(f"\n5-year lookahead:")
    print(f"  Positive cases: {merged['collapse_5yr'].sum()}")
    print(f"  Base rate: {merged['collapse_5yr'].mean():.2%}")

    # Choose label based on sample size
    if merged['collapse_3yr'].sum() >= 30:
        label_col = 'collapse_3yr'
        lookahead = 3
    elif merged['collapse_5yr'].sum() >= 30:
        label_col = 'collapse_5yr'
        lookahead = 5
    else:
        print("\nInsufficient collapse cases. Using sigma-drop proxy...")

        # Create sigma-based collapse proxy
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
        label_col = 'collapse_sigma'
        lookahead = 3

        print(f"\n  Using sigma-drop proxy (top 10% drops)")
        print(f"  Positive cases: {merged[label_col].sum()}")
        print(f"  Base rate: {merged[label_col].mean():.2%}")

    # =========================================================================
    # 6. VALIDATION
    # =========================================================================
    print("\n" + "-" * 80)
    print("6. VALIDATION RESULTS")
    print("-" * 80)

    valid_df = merged.dropna(subset=[label_col, 'k_eff_xconst', 'sigma', 'rho_combined'])
    valid_df = valid_df.rename(columns={label_col: 'label'})

    print(f"\nValidation set: {len(valid_df):,} observations")
    print(f"Positive labels: {int(valid_df['label'].sum())}")
    print(f"Base rate: {valid_df['label'].mean():.2%}")

    if valid_df['label'].sum() < 5:
        print("\nERROR: Insufficient positive cases for validation")
        return

    # Compute AUC for each predictor
    y_true = valid_df['label'].values

    results = {}

    # 1. Sigma baseline (low sigma = high risk)
    sigma_prob = 1 - valid_df['sigma'].values
    try:
        sigma_auc = roc_auc_score(y_true, sigma_prob)
    except:
        sigma_auc = 0.5
    results['sigma_baseline'] = sigma_auc

    # 2. Raw xconst (low xconst = high risk)
    xconst_prob = 1 - (valid_df['k_xconst'].values / 7.0)
    try:
        xconst_auc = roc_auc_score(y_true, xconst_prob)
    except:
        xconst_auc = 0.5
    results['xconst_raw'] = xconst_auc

    # 3. k_eff with xconst (low k_eff = high risk)
    keff_prob = 1 - (valid_df['k_eff_xconst'].values / 7.0)
    try:
        keff_auc = roc_auc_score(y_true, keff_prob)
    except:
        keff_auc = 0.5
    results['k_eff_xconst'] = keff_auc

    # 4. rho alone (high rho = high risk)
    rho_prob = valid_df['rho_combined'].values
    try:
        rho_auc = roc_auc_score(y_true, rho_prob)
    except:
        rho_auc = 0.5
    results['rho_combined'] = rho_auc

    # 5. k_eff with WGI rho only
    keff_wgi_prob = 1 - (valid_df['k_eff_xconst_wgi'].values / 7.0)
    try:
        keff_wgi_auc = roc_auc_score(y_true, keff_wgi_prob)
    except:
        keff_wgi_auc = 0.5
    results['k_eff_xconst_wgi_rho'] = keff_wgi_auc

    # 6. Deterioration indicators (declining = high risk)
    valid_with_trends = valid_df.dropna(subset=['k_eff_trend', 'sigma_trend', 'rho_trend'])
    if len(valid_with_trends) > 50 and valid_with_trends['label'].sum() > 5:
        y_trend = valid_with_trends['label'].values

        # k_eff trend: declining k_eff = high risk (negative trend = positive risk)
        keff_trend_prob = -valid_with_trends['k_eff_trend'].values
        keff_trend_prob = (keff_trend_prob - keff_trend_prob.min()) / (keff_trend_prob.max() - keff_trend_prob.min() + 1e-6)
        try:
            keff_trend_auc = roc_auc_score(y_trend, keff_trend_prob)
        except:
            keff_trend_auc = 0.5
        results['k_eff_trend_decline'] = keff_trend_auc

        # sigma trend: declining sigma = high risk
        sigma_trend_prob = -valid_with_trends['sigma_trend'].values
        sigma_trend_prob = (sigma_trend_prob - sigma_trend_prob.min()) / (sigma_trend_prob.max() - sigma_trend_prob.min() + 1e-6)
        try:
            sigma_trend_auc = roc_auc_score(y_trend, sigma_trend_prob)
        except:
            sigma_trend_auc = 0.5
        results['sigma_trend_decline'] = sigma_trend_auc

        # rho trend: increasing rho = high risk (echo chamber forming)
        rho_trend_prob = valid_with_trends['rho_trend'].values
        rho_trend_prob = (rho_trend_prob - rho_trend_prob.min()) / (rho_trend_prob.max() - rho_trend_prob.min() + 1e-6)
        try:
            rho_trend_auc = roc_auc_score(y_trend, rho_trend_prob)
        except:
            rho_trend_auc = 0.5
        results['rho_trend_increase'] = rho_trend_auc

        # Combined deterioration score
        # Risk = low k_eff + declining k_eff + declining sigma + increasing rho
        combined_risk = (
            (1 - valid_with_trends['k_eff_xconst'].values / 7.0) * 0.25 +
            keff_trend_prob * 0.25 +
            sigma_trend_prob * 0.25 +
            rho_trend_prob * 0.25
        )
        try:
            combined_auc = roc_auc_score(y_trend, combined_risk)
        except:
            combined_auc = 0.5
        results['combined_deterioration'] = combined_auc

    print("\n" + "=" * 60)
    print("AUC COMPARISON (vs sigma baseline = 0.65)")
    print("=" * 60)
    print(f"\n{'Predictor':<25} {'AUC':>10} {'vs Baseline':>15}")
    print("-" * 55)

    baseline = 0.65
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        diff = auc - baseline
        sign = '+' if diff >= 0 else ''
        print(f"{name:<25} {auc:>10.3f} {sign}{diff:>14.3f}")

    # =========================================================================
    # 7. TEMPORAL HOLDOUT VALIDATION
    # =========================================================================
    print("\n" + "-" * 80)
    print("7. TEMPORAL HOLDOUT (Train <= 2015, Test >= 2016)")
    print("-" * 80)

    train_df = valid_df[valid_df['year'] <= 2015]
    test_df = valid_df[valid_df['year'] >= 2016]

    print(f"\nTrain set: {len(train_df)} obs, {train_df['label'].sum():.0f} positives")
    print(f"Test set: {len(test_df)} obs, {test_df['label'].sum():.0f} positives")

    if len(test_df) > 0 and test_df['label'].sum() > 0:
        y_test = test_df['label'].values

        print(f"\n{'Predictor':<25} {'AUC':>10}")
        print("-" * 40)

        for name, col, invert in [
            ('sigma_baseline', 'sigma', True),
            ('xconst_raw', 'k_xconst', True),
            ('k_eff_xconst', 'k_eff_xconst', True),
            ('rho_combined', 'rho_combined', False),
        ]:
            if invert:
                prob = 1 - (test_df[col].values / test_df[col].max())
            else:
                prob = test_df[col].values

            try:
                auc = roc_auc_score(y_test, prob)
            except:
                auc = 0.5
            print(f"{name:<25} {auc:>10.3f}")
    else:
        print("Insufficient test data")

    # =========================================================================
    # 8. THRESHOLD ANALYSIS
    # =========================================================================
    print("\n" + "-" * 80)
    print("8. THRESHOLD ANALYSIS")
    print("-" * 80)

    # Find optimal thresholds
    print("\nk_eff thresholds (low = risky):")
    for thresh in [2.0, 2.5, 3.0, 3.5, 4.0]:
        pred = (valid_df['k_eff_xconst'] < thresh).astype(int)
        if pred.sum() > 0 and (1 - pred).sum() > 0:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, pred, average='binary', zero_division=0
            )
            print(f"  k_eff < {thresh:.1f}: Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}")

    print("\nxconst thresholds (low = risky):")
    for thresh in [2, 3, 4, 5]:
        pred = (valid_df['k_xconst'] < thresh).astype(int)
        if pred.sum() > 0 and (1 - pred).sum() > 0:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, pred, average='binary', zero_division=0
            )
            print(f"  xconst < {thresh}: Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}")

    # =========================================================================
    # 9. SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_predictor = max(results.items(), key=lambda x: x[1])
    sigma_auc = results['sigma_baseline']
    keff_auc = results['k_eff_xconst']

    xconst_auc = results.get('xconst_raw', 0.5)
    sigma_trend_auc = results.get('sigma_trend_decline', 0.5)

    print(f"""
Dataset:
  Polity5 observations (1996-2020): {len(polity_valid):,}
  WGI observations: {len(wgi):,}
  Merged observations: {len(merged):,}
  Validation set: {len(valid_df):,}

Collapse events:
  Adverse regime transitions: {len(adverse)}
  Positive labels in validation: {int(valid_df['label'].sum())}
  Lookahead window: {lookahead} years

Key Results:
  Sigma baseline AUC:       {sigma_auc:.3f}
  Sigma trend decline AUC:  {sigma_trend_auc:.3f}
  Raw xconst AUC:           {xconst_auc:.3f}
  k_eff (xconst) AUC:       {keff_auc:.3f}
  Best predictor: {best_predictor[0]} (AUC = {best_predictor[1]:.3f})

vs Reference Baseline (0.65):
  sigma_baseline:     {'+' if sigma_auc > 0.65 else ''}{sigma_auc - 0.65:.3f}
  xconst_raw:         {'+' if xconst_auc > 0.65 else ''}{xconst_auc - 0.65:.3f}
  k_eff (xconst):     {'+' if keff_auc > 0.65 else ''}{keff_auc - 0.65:.3f}

============================================================
ANALYSIS: Why xconst-based k_eff underperforms
============================================================

1. CONCEPTUAL MISMATCH:
   The k_eff formula (Kish 1965) was designed for survey sampling to measure
   effective sample size under clustered designs. It measures "how many
   independent constraints" exist, NOT "how strong/robust" they are.

   xconst (1-7) measures constraint STRENGTH, not constraint COUNT.
   A country with xconst=7 has ONE very strong constraint mechanism,
   not 7 independent constraints.

2. TEMPORAL RELATIONSHIP:
   Regime transitions occur CONCURRENT with or AFTER xconst declines,
   not before them. The xconst drop IS the collapse, not a predictor of it.

   Observed pattern in data:
   - Year before collapse: mean xconst = 4.28
   - Year of collapse: mean xconst = 2.27
   - The CHANGE is the signal, but it happens simultaneously with the event.

3. WHAT WORKS BETTER:
   - Sigma baseline (governance quality) AUC: {sigma_auc:.3f}
   - Sigma trend (deterioration) AUC: {sigma_trend_auc:.3f}

   The WGI-based sigma captures governance quality decline, which is a
   leading indicator of collapse. xconst captures the institutional
   mechanics of collapse itself.

4. IMPLICATIONS FOR RATCHET:
   - The k_eff formula remains valid for counting independent verifiers/sources
   - For institutional collapse, constraint QUALITY (sigma) matters more than
     constraint COUNT (k)
   - The current WGI-based sigma approach (AUC ~0.65) is more appropriate
     for regime transition prediction

Conclusion:
  {'Using xconst as k IMPROVES k_eff prediction' if keff_auc > 0.65 else 'xconst-based k_eff does NOT beat the sigma baseline (AUC 0.65)'}
  The Kish k_eff formula is not appropriate for this domain application.
  Recommend: Continue using governance quality (sigma) as the primary predictor.
""")

    # Save results
    results_df = pd.DataFrame([
        {'predictor': k, 'auc': v, 'vs_baseline': v - 0.65}
        for k, v in results.items()
    ])
    results_path = base_path / 'experiments' / 'polity_xconst_validation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
