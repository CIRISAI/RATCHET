"""
RATCHET Omega Module - Report Generation

Generates comprehensive reports on omega analysis results,
summarizing null hypothesis tests, distribution analysis,
correlations, and validation results.
"""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np

from .residuals import OmegaTimeSeries, DomainType
from .null_test import NullHypothesisBattery, run_null_hypothesis_battery
from .distribution import DistributionStats, compute_distribution_stats, compare_to_null_distribution
from .correlations import CorrelationResult, correlation_matrix
from .outliers import OutlierResult, ChangepointResult, detect_all_outliers, detect_changepoints
from .validity import ValidationResult


@dataclass
class OmegaReport:
    """
    Comprehensive report on omega analysis.

    Attributes:
        title: Report title
        generated_at: Timestamp of report generation
        domain: Domain type analyzed
        n_observations: Total observations
        summary: Executive summary
        null_hypothesis: Null hypothesis test results
        distribution: Distribution analysis results
        outliers: Outlier detection results
        changepoints: Changepoint detection results
        correlations: Cross-domain correlation results (if multiple series)
        validation: Validation results
        conclusion: Overall conclusion
        recommendations: List of recommendations
    """
    title: str = "Omega Residual Analysis Report"
    generated_at: str = ""
    domain: Optional[DomainType] = None
    n_observations: int = 0
    summary: str = ""
    null_hypothesis: Optional[Dict[str, Any]] = None
    distribution: Optional[Dict[str, Any]] = None
    outliers: Optional[Dict[str, Any]] = None
    changepoints: Optional[Dict[str, Any]] = None
    correlations: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    conclusion: str = ""
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set generation timestamp."""
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'title': self.title,
            'generated_at': self.generated_at,
            'domain': self.domain.value if self.domain else None,
            'n_observations': self.n_observations,
            'summary': self.summary,
            'null_hypothesis': self.null_hypothesis,
            'distribution': self.distribution,
            'outliers': self.outliers,
            'changepoints': self.changepoints,
            'correlations': self.correlations,
            'validation': self.validation,
            'conclusion': self.conclusion,
            'recommendations': self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        # Header
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Generated:** {self.generated_at}")
        if self.domain:
            lines.append(f"**Domain:** {self.domain.value}")
        lines.append(f"**Observations:** {self.n_observations}")
        lines.append("")

        # Summary
        if self.summary:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(self.summary)
            lines.append("")

        # Null Hypothesis Tests
        if self.null_hypothesis:
            lines.append("## Null Hypothesis Tests")
            lines.append("")
            lines.append("Testing H0: omega is random noise (mean=0, no structure)")
            lines.append("")

            if 'results' in self.null_hypothesis:
                lines.append("| Test | Statistic | p-value | Result |")
                lines.append("|------|-----------|---------|--------|")

                for name, result in self.null_hypothesis['results'].items():
                    stat = result.get('statistic', 'N/A')
                    pval = result.get('p_value', 'N/A')
                    reject = "REJECT" if result.get('reject_null', False) else "ACCEPT"
                    lines.append(f"| {name} | {stat:.3f} | {pval:.4f} | {reject} |")
                lines.append("")

            if 'overall_interpretation' in self.null_hypothesis:
                lines.append(f"**Interpretation:** {self.null_hypothesis['overall_interpretation']}")
                lines.append("")

        # Distribution Analysis
        if self.distribution:
            lines.append("## Distribution Analysis")
            lines.append("")

            dist = self.distribution
            if 'mean' in dist:
                lines.append(f"- Mean omega: {dist['mean']:.4f}")
            if 'std' in dist:
                lines.append(f"- Std omega: {dist['std']:.4f}")
            if 'skewness' in dist:
                lines.append(f"- Skewness: {dist['skewness']:.3f} ({dist.get('skewness_interpretation', '')})")
            if 'kurtosis' in dist:
                lines.append(f"- Kurtosis: {dist['kurtosis']:.3f} ({dist.get('kurtosis_interpretation', '')})")
            if 'is_approximately_normal' in dist:
                normal_str = "Yes" if dist['is_approximately_normal'] else "No"
                lines.append(f"- Approximately normal: {normal_str}")
            lines.append("")

        # Outliers
        if self.outliers:
            lines.append("## Outlier Detection")
            lines.append("")

            for method, result in self.outliers.items():
                n_out = result.get('n_outliers', 0)
                frac = result.get('fraction_outliers', 0) * 100
                lines.append(f"- **{method}:** {n_out} outliers ({frac:.1f}%)")
            lines.append("")

        # Changepoints
        if self.changepoints:
            lines.append("## Changepoint Detection")
            lines.append("")

            n_cp = self.changepoints.get('n_changepoints', 0)
            lines.append(f"- Changepoints detected: {n_cp}")

            if n_cp > 0 and 'changepoint_timestamps' in self.changepoints:
                lines.append(f"- Changepoint locations: {self.changepoints['changepoint_timestamps']}")
            lines.append("")

        # Correlations
        if self.correlations:
            lines.append("## Cross-Domain Correlations")
            lines.append("")
            lines.append("(Only applicable when analyzing multiple domain series)")
            lines.append("")

        # Validation
        if self.validation:
            lines.append("## Validation Results")
            lines.append("")

            valid_str = "PASSED" if self.validation.get('valid', False) else "FAILED"
            lines.append(f"- Validation status: {valid_str}")

            if 'rmse' in self.validation:
                lines.append(f"- RMSE: {self.validation['rmse']:.4f}")
            if 'r_squared' in self.validation:
                lines.append(f"- R-squared: {self.validation['r_squared']:.4f}")

            if 'issues' in self.validation and self.validation['issues']:
                lines.append("- Issues:")
                for issue in self.validation['issues']:
                    lines.append(f"  - {issue}")
            lines.append("")

        # Conclusion
        if self.conclusion:
            lines.append("## Conclusion")
            lines.append("")
            lines.append(self.conclusion)
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)


def generate_omega_report(
    omega_series: OmegaTimeSeries,
    title: Optional[str] = None,
    run_full_analysis: bool = True,
    alpha: float = 0.05,
) -> OmegaReport:
    """
    Generate a comprehensive omega analysis report.

    Args:
        omega_series: OmegaTimeSeries to analyze
        title: Report title (auto-generated if None)
        run_full_analysis: Whether to run all analyses
        alpha: Significance level for tests

    Returns:
        OmegaReport object
    """
    n = len(omega_series)
    domain = omega_series.domain

    if title is None:
        domain_str = domain.value if domain else "unknown"
        title = f"Omega Residual Analysis Report - {domain_str.capitalize()} Domain"

    # Initialize report
    report = OmegaReport(
        title=title,
        domain=domain,
        n_observations=n,
    )

    if n < 5:
        report.summary = "Insufficient data for analysis (n < 5)"
        report.conclusion = "Cannot draw conclusions due to insufficient data."
        report.recommendations = ["Collect more observations before analysis."]
        return report

    omega_values = omega_series.omega_values

    # Null hypothesis tests
    if run_full_analysis:
        null_battery = run_null_hypothesis_battery(omega_series, alpha=alpha)
        report.null_hypothesis = null_battery.to_dict()

    # Distribution analysis
    dist_stats = compute_distribution_stats(omega_series)
    report.distribution = dist_stats.to_dict()

    # Add null comparison
    if run_full_analysis:
        null_comparison = compare_to_null_distribution(omega_series)
        report.distribution['null_comparison'] = null_comparison

    # Outlier detection
    if run_full_analysis:
        outlier_results = detect_all_outliers(omega_series)
        report.outliers = {k: v.to_dict() for k, v in outlier_results.items()}

    # Changepoint detection
    if run_full_analysis and n >= 20:
        cp_result = detect_changepoints(omega_series)
        report.changepoints = cp_result.to_dict()

    # Generate summary
    summary_parts = []

    mean_omega = float(np.mean(omega_values))
    std_omega = float(np.std(omega_values, ddof=1))

    summary_parts.append(
        f"Analysis of {n} omega observations from the {domain.value if domain else 'unknown'} domain."
    )
    summary_parts.append(
        f"Mean omega = {mean_omega:.4f} (std = {std_omega:.4f})."
    )

    if report.null_hypothesis:
        n_rejected = report.null_hypothesis.get('summary', {}).get('n_rejected', 0)
        n_tests = report.null_hypothesis.get('summary', {}).get('n_tests', 0)
        if n_rejected == 0:
            summary_parts.append(
                "All null hypothesis tests passed, suggesting omega is consistent with random noise."
            )
        else:
            summary_parts.append(
                f"{n_rejected}/{n_tests} null hypothesis tests rejected, suggesting omega has detectable structure."
            )

    report.summary = " ".join(summary_parts)

    # Generate conclusion
    conclusion_parts = []

    # Check if H0 is rejected
    h0_rejected = False
    if report.null_hypothesis:
        h0_rejected = report.null_hypothesis.get('overall_reject', False)

    if h0_rejected:
        conclusion_parts.append(
            "The omega residuals show statistically significant structure, "
            "suggesting that CCA predictions capture real dynamics beyond simple curve-fitting."
        )
    else:
        conclusion_parts.append(
            "The omega residuals are consistent with random noise centered at zero, "
            "which is consistent with the null hypothesis that CCA is performing curve-fitting."
        )

    # Add distribution interpretation
    if dist_stats.is_approximately_normal:
        conclusion_parts.append(
            "The distribution of omega is approximately normal."
        )
    else:
        conclusion_parts.append(
            f"The distribution of omega deviates from normality "
            f"(skewness={dist_stats.skewness:.2f}, kurtosis={dist_stats.kurtosis:.2f})."
        )

    report.conclusion = " ".join(conclusion_parts)

    # Generate recommendations
    recommendations = []

    if n < 50:
        recommendations.append(
            "Collect more observations (n >= 50 recommended) for more reliable statistical tests."
        )

    if abs(mean_omega) > 0.1:
        recommendations.append(
            f"Investigate systematic bias in predictions (mean omega = {mean_omega:.3f})."
        )

    if report.outliers:
        max_outliers = max(r.get('n_outliers', 0) for r in report.outliers.values())
        if max_outliers > 0:
            recommendations.append(
                f"Investigate {max_outliers} detected outliers for potential model failures or regime changes."
            )

    if report.changepoints and report.changepoints.get('n_changepoints', 0) > 0:
        n_cp = report.changepoints['n_changepoints']
        recommendations.append(
            f"Investigate {n_cp} detected changepoint(s) for structural breaks in the system."
        )

    if not recommendations:
        recommendations.append("Continue monitoring omega residuals over time.")

    report.recommendations = recommendations

    return report


def generate_comparison_report(
    omega_series_list: List[OmegaTimeSeries],
    title: str = "Cross-Domain Omega Comparison Report",
) -> OmegaReport:
    """
    Generate a report comparing omega across multiple domains.

    Args:
        omega_series_list: List of OmegaTimeSeries from different domains
        title: Report title

    Returns:
        OmegaReport object
    """
    report = OmegaReport(title=title)

    if len(omega_series_list) == 0:
        report.summary = "No omega series provided for comparison."
        return report

    if len(omega_series_list) == 1:
        return generate_omega_report(omega_series_list[0], title=title)

    # Total observations
    total_n = sum(len(s) for s in omega_series_list)
    report.n_observations = total_n

    # Compute correlation matrix
    corr_df, p_df = correlation_matrix(omega_series_list)

    report.correlations = {
        'correlation_matrix': corr_df.to_dict(),
        'p_value_matrix': p_df.to_dict(),
        'n_series': len(omega_series_list),
    }

    # Per-domain statistics
    domain_stats = []
    for series in omega_series_list:
        stats = compute_distribution_stats(series)
        domain_stats.append({
            'domain': series.domain.value if series.domain else 'unknown',
            'n': len(series),
            'mean_omega': stats.mean,
            'std_omega': stats.std,
            'skewness': stats.skewness,
        })

    report.distribution = {
        'per_domain': domain_stats,
    }

    # Summary
    domains = [s.domain.value if s.domain else 'unknown' for s in omega_series_list]
    report.summary = (
        f"Comparison of omega residuals across {len(omega_series_list)} domains: {', '.join(domains)}. "
        f"Total observations: {total_n}."
    )

    # Conclusion based on correlations
    if not corr_df.empty:
        # Find significant correlations
        sig_corrs = []
        for i in range(len(corr_df)):
            for j in range(i + 1, len(corr_df)):
                if p_df.iloc[i, j] < 0.05:
                    sig_corrs.append((corr_df.index[i], corr_df.columns[j], corr_df.iloc[i, j]))

        if sig_corrs:
            report.conclusion = (
                f"Found {len(sig_corrs)} significant cross-domain correlation(s) in omega residuals, "
                "suggesting shared structural dynamics across domains."
            )
        else:
            report.conclusion = (
                "No significant cross-domain correlations found in omega residuals. "
                "Domain dynamics appear independent."
            )

    return report


__all__ = [
    'OmegaReport',
    'generate_omega_report',
    'generate_comparison_report',
]
