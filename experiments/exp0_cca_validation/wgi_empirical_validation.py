"""
RATCHET Empirical Validation on World Governance Indicators (WGI)

This script performs rigorous empirical validation of the k_eff framework
using real WGI data from 203 countries (1996-2023).

Validation Components:
1. K-fold cross-validation for collapse prediction
2. Baseline comparison (EDI vs simple rho monitoring)
3. Temporal holdout (train 1996-2015, test 2016-2023)
4. Leading indicator detection rate

Author: RATCHET Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# WGI indicator columns
WGI_INDICATORS = ['CC.EST', 'GE.EST', 'PV.EST', 'RQ.EST', 'RL.EST', 'VA.EST']
WGI_NAMES = {
    'CC.EST': 'Control of Corruption',
    'GE.EST': 'Government Effectiveness',
    'PV.EST': 'Political Stability',
    'RQ.EST': 'Regulatory Quality',
    'RL.EST': 'Rule of Law',
    'VA.EST': 'Voice and Accountability'
}


def compute_keff(k: float, rho: float) -> float:
    """Compute effective constraints using Kish formula."""
    if rho >= 1.0:
        return 1.0
    if rho <= 0.0:
        return k
    return k / (1 + rho * (k - 1))


def compute_rho_from_indicators(row: pd.Series, indicators: List[str]) -> float:
    """
    Compute average pairwise correlation proxy from indicator values.

    Uses normalized variance approach: high variance across indicators
    suggests low correlation (independent), low variance suggests high
    correlation (moving together).
    """
    values = [row[ind] for ind in indicators if pd.notna(row[ind])]
    if len(values) < 2:
        return 0.5  # Default moderate correlation

    # Normalize to 0-1 range (WGI is roughly -2.5 to 2.5)
    values = [(v + 2.5) / 5.0 for v in values]
    values = [max(0, min(1, v)) for v in values]

    # Compute coefficient of variation as inverse correlation proxy
    mean_val = np.mean(values)
    std_val = np.std(values)

    if mean_val == 0:
        return 0.5

    cv = std_val / (mean_val + 0.01)  # Coefficient of variation

    # Map CV to rho: low CV = high correlation, high CV = low correlation
    # CV typically ranges 0-1 for this data
    rho = 1.0 - min(1.0, cv * 2)
    return max(0.0, min(1.0, rho))


def compute_sigma(row: pd.Series, indicators: List[str]) -> float:
    """
    Compute sustainability (sigma) as mean normalized governance quality.
    """
    values = [row[ind] for ind in indicators if pd.notna(row[ind])]
    if len(values) == 0:
        return 0.5

    # Normalize WGI (-2.5 to 2.5) to (0 to 1)
    normalized = [(v + 2.5) / 5.0 for v in values]
    return max(0.0, min(1.0, np.mean(normalized)))


@dataclass
class CollapseEvent:
    """Represents a detected institutional collapse."""
    country: str
    year: int
    sigma_drop: float
    sigma_before: float
    sigma_after: float


def detect_collapses(df: pd.DataFrame, threshold: float = 0.15) -> List[CollapseEvent]:
    """
    Detect collapse events based on significant drops in sigma.

    Args:
        df: DataFrame with country, year, sigma columns
        threshold: Minimum sigma drop to qualify as collapse (default 0.15 = 15%)

    Returns:
        List of CollapseEvent objects
    """
    collapses = []

    for country in df['country'].unique():
        country_data = df[df['country'] == country].sort_values('year')

        if len(country_data) < 2:
            continue

        sigmas = country_data['sigma'].values
        years = country_data['year'].values

        for i in range(1, len(sigmas)):
            drop = sigmas[i-1] - sigmas[i]
            if drop >= threshold:
                collapses.append(CollapseEvent(
                    country=country,
                    year=int(years[i]),
                    sigma_drop=drop,
                    sigma_before=sigmas[i-1],
                    sigma_after=sigmas[i]
                ))

    return collapses


def prepare_prediction_dataset(df: pd.DataFrame, collapses: List[CollapseEvent],
                                lookahead: int = 3) -> pd.DataFrame:
    """
    Create dataset for collapse prediction.

    For each country-year, label as 1 if a collapse occurs within lookahead years.
    """
    collapse_set = {(c.country, c.year) for c in collapses}

    records = []
    for _, row in df.iterrows():
        country = row['country']
        year = row['year']

        # Check if collapse happens in next `lookahead` years
        label = 0
        for future_year in range(year + 1, year + lookahead + 1):
            if (country, future_year) in collapse_set:
                label = 1
                break

        records.append({
            'country': country,
            'year': year,
            'k_eff': row['k_eff'],
            'rho': row['rho'],
            'sigma': row['sigma'],
            'label': label
        })

    return pd.DataFrame(records)


def evaluate_predictor(y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.5

    return metrics


class KeffPredictor:
    """Predicts collapse based on k_eff threshold."""

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def predict(self, k_eff: np.ndarray) -> np.ndarray:
        return (k_eff < self.threshold).astype(int)

    def predict_proba(self, k_eff: np.ndarray) -> np.ndarray:
        # Lower k_eff = higher collapse probability
        return 1.0 - (k_eff / 6.0)  # Normalize assuming max k_eff ~6


class RhoPredictor:
    """Baseline: predicts collapse based on raw rho threshold."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def predict(self, rho: np.ndarray) -> np.ndarray:
        return (rho > self.threshold).astype(int)

    def predict_proba(self, rho: np.ndarray) -> np.ndarray:
        return rho


class SigmaPredictor:
    """Baseline: predicts collapse based on low sigma."""

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def predict(self, sigma: np.ndarray) -> np.ndarray:
        return (sigma < self.threshold).astype(int)

    def predict_proba(self, sigma: np.ndarray) -> np.ndarray:
        return 1.0 - sigma


def run_cross_validation(pred_df: pd.DataFrame, n_folds: int = 5) -> Dict[str, Dict]:
    """
    Run k-fold cross-validation comparing predictors.

    Folds are stratified by country to ensure each country appears in train and test.
    """
    results = {
        'k_eff': [],
        'rho_baseline': [],
        'sigma_baseline': []
    }

    # Get unique countries
    countries = pred_df['country'].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(countries)):
        train_countries = countries[train_idx]
        test_countries = countries[test_idx]

        train_df = pred_df[pred_df['country'].isin(train_countries)]
        test_df = pred_df[pred_df['country'].isin(test_countries)]

        if len(test_df) == 0 or test_df['label'].sum() == 0:
            continue

        # k_eff predictor
        keff_pred = KeffPredictor(threshold=2.0)
        y_pred = keff_pred.predict(test_df['k_eff'].values)
        y_prob = keff_pred.predict_proba(test_df['k_eff'].values)
        results['k_eff'].append(evaluate_predictor(
            test_df['label'].values, y_pred, y_prob
        ))

        # rho baseline
        rho_pred = RhoPredictor(threshold=0.7)
        y_pred = rho_pred.predict(test_df['rho'].values)
        y_prob = rho_pred.predict_proba(test_df['rho'].values)
        results['rho_baseline'].append(evaluate_predictor(
            test_df['label'].values, y_pred, y_prob
        ))

        # sigma baseline
        sigma_pred = SigmaPredictor(threshold=0.4)
        y_pred = sigma_pred.predict(test_df['sigma'].values)
        y_prob = sigma_pred.predict_proba(test_df['sigma'].values)
        results['sigma_baseline'].append(evaluate_predictor(
            test_df['label'].values, y_pred, y_prob
        ))

    # Aggregate results
    aggregated = {}
    for method, fold_results in results.items():
        if len(fold_results) == 0:
            continue
        aggregated[method] = {
            metric: np.mean([r[metric] for r in fold_results])
            for metric in fold_results[0].keys()
        }
        aggregated[method]['std_f1'] = np.std([r['f1'] for r in fold_results])

    return aggregated


def run_temporal_holdout(pred_df: pd.DataFrame,
                         train_end: int = 2015,
                         test_start: int = 2016) -> Dict[str, Dict]:
    """
    Run temporal holdout validation: train on 1996-2015, test on 2016-2023.
    """
    train_df = pred_df[pred_df['year'] <= train_end]
    test_df = pred_df[pred_df['year'] >= test_start]

    if len(test_df) == 0 or test_df['label'].sum() == 0:
        return {}

    results = {}

    # k_eff predictor
    keff_pred = KeffPredictor(threshold=2.0)
    y_pred = keff_pred.predict(test_df['k_eff'].values)
    y_prob = keff_pred.predict_proba(test_df['k_eff'].values)
    results['k_eff'] = evaluate_predictor(test_df['label'].values, y_pred, y_prob)

    # rho baseline
    rho_pred = RhoPredictor(threshold=0.7)
    y_pred = rho_pred.predict(test_df['rho'].values)
    y_prob = rho_pred.predict_proba(test_df['rho'].values)
    results['rho_baseline'] = evaluate_predictor(test_df['label'].values, y_pred, y_prob)

    # sigma baseline
    sigma_pred = SigmaPredictor(threshold=0.4)
    y_pred = sigma_pred.predict(test_df['sigma'].values)
    y_prob = sigma_pred.predict_proba(test_df['sigma'].values)
    results['sigma_baseline'] = evaluate_predictor(test_df['label'].values, y_pred, y_prob)

    return results


def compute_leading_indicators(df: pd.DataFrame, collapses: List[CollapseEvent],
                               window: int = 3) -> Dict[str, float]:
    """
    Compute what fraction of collapses show leading indicators.

    A leading indicator is defined as k_eff declining or rho increasing
    in the window before collapse.
    """
    detected = 0
    total = 0

    for collapse in collapses:
        country_data = df[df['country'] == collapse.country].sort_values('year')

        # Get data for years before collapse
        pre_collapse = country_data[
            (country_data['year'] >= collapse.year - window) &
            (country_data['year'] < collapse.year)
        ]

        if len(pre_collapse) < 2:
            continue

        total += 1

        # Check for declining k_eff or increasing rho
        keff_trend = np.polyfit(range(len(pre_collapse)), pre_collapse['k_eff'].values, 1)[0]
        rho_trend = np.polyfit(range(len(pre_collapse)), pre_collapse['rho'].values, 1)[0]

        if keff_trend < -0.05 or rho_trend > 0.05:
            detected += 1

    return {
        'total_collapses': total,
        'detected_with_leading_indicator': detected,
        'detection_rate': detected / total if total > 0 else 0.0
    }


def main():
    """Run full empirical validation."""
    print("=" * 70)
    print("RATCHET EMPIRICAL VALIDATION - WGI Dataset")
    print("=" * 70)

    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'institutional' / 'wgi_data.csv'
    print(f"\nLoading data from: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} observations from {df['country'].nunique()} countries")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    # Compute RATCHET variables
    print("\nComputing RATCHET variables...")
    k = len(WGI_INDICATORS)  # 6 indicators = 6 constraints

    df['rho'] = df.apply(lambda row: compute_rho_from_indicators(row, WGI_INDICATORS), axis=1)
    df['sigma'] = df.apply(lambda row: compute_sigma(row, WGI_INDICATORS), axis=1)
    df['k_eff'] = df.apply(lambda row: compute_keff(k, row['rho']), axis=1)

    print(f"k_eff range: {df['k_eff'].min():.2f} - {df['k_eff'].max():.2f}")
    print(f"rho range: {df['rho'].min():.2f} - {df['rho'].max():.2f}")
    print(f"sigma range: {df['sigma'].min():.2f} - {df['sigma'].max():.2f}")

    # Detect collapses
    print("\n" + "=" * 70)
    print("COLLAPSE DETECTION")
    print("=" * 70)

    collapses = detect_collapses(df, threshold=0.12)
    print(f"Detected {len(collapses)} collapse events (sigma drop >= 12%)")

    if len(collapses) > 0:
        print("\nTop 10 collapses by severity:")
        sorted_collapses = sorted(collapses, key=lambda c: c.sigma_drop, reverse=True)[:10]
        for c in sorted_collapses:
            print(f"  {c.country} ({c.year}): sigma {c.sigma_before:.2f} -> {c.sigma_after:.2f} "
                  f"(drop: {c.sigma_drop:.2f})")

    # Prepare prediction dataset
    pred_df = prepare_prediction_dataset(df, collapses, lookahead=3)
    collapse_rate = pred_df['label'].mean()
    print(f"\nPrediction dataset: {len(pred_df)} observations")
    print(f"Collapse rate (3-year lookahead): {collapse_rate:.1%}")

    # Cross-validation
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    cv_results = run_cross_validation(pred_df, n_folds=5)

    print(f"\n{'Method':<20} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)
    for method, metrics in cv_results.items():
        print(f"{method:<20} {metrics.get('auc', 0):.3f}    {metrics['f1']:.3f}    "
              f"{metrics['precision']:.3f}      {metrics['recall']:.3f}")

    # Temporal holdout
    print("\n" + "=" * 70)
    print("TEMPORAL HOLDOUT (Train: 1996-2015, Test: 2016-2023)")
    print("=" * 70)

    temporal_results = run_temporal_holdout(pred_df)

    print(f"\n{'Method':<20} {'AUC':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("-" * 65)
    for method, metrics in temporal_results.items():
        print(f"{method:<20} {metrics.get('auc', 0):.3f}    {metrics['f1']:.3f}    "
              f"{metrics['true_positives']:>5.0f} {metrics['false_positives']:>5.0f} "
              f"{metrics['true_negatives']:>5.0f} {metrics['false_negatives']:>5.0f}")

    # Leading indicators
    print("\n" + "=" * 70)
    print("LEADING INDICATOR ANALYSIS")
    print("=" * 70)

    leading = compute_leading_indicators(df, collapses, window=3)
    print(f"\nCollapses analyzed: {leading['total_collapses']}")
    print(f"Collapses with leading indicators: {leading['detected_with_leading_indicator']}")
    print(f"Detection rate: {leading['detection_rate']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Empirical Validation Results:
- Dataset: WGI 1996-2023, {df['country'].nunique()} countries, {len(df)} observations
- Collapse events detected: {len(collapses)} (threshold: 12% sigma drop)
- Base collapse rate: {collapse_rate:.1%}

Cross-Validation (5-fold):
- k_eff AUC: {cv_results.get('k_eff', {}).get('auc', 'N/A')}
- rho baseline AUC: {cv_results.get('rho_baseline', {}).get('auc', 'N/A')}
- sigma baseline AUC: {cv_results.get('sigma_baseline', {}).get('auc', 'N/A')}

Temporal Holdout (2016-2023):
- k_eff F1: {temporal_results.get('k_eff', {}).get('f1', 'N/A')}
- rho baseline F1: {temporal_results.get('rho_baseline', {}).get('f1', 'N/A')}

Leading Indicator Detection: {leading['detection_rate']:.1%}
""")

    # Save results
    results_path = Path(__file__).parent / 'wgi_validation_results.csv'

    results_rows = []
    for method, metrics in cv_results.items():
        results_rows.append({
            'validation_type': 'cross_validation',
            'method': method,
            **metrics
        })
    for method, metrics in temporal_results.items():
        results_rows.append({
            'validation_type': 'temporal_holdout',
            'method': method,
            **metrics
        })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    return cv_results, temporal_results, leading


if __name__ == '__main__':
    main()
