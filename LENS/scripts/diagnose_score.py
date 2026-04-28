#!/usr/bin/env python3
"""
CIRIS Score Diagnostic Tool

Usage:
    python scripts/diagnose_score.py <agent_name>
    python scripts/diagnose_score.py --all
    python scripts/diagnose_score.py --fleet
"""

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, "/app" if "/app" in sys.path or True else ".")


async def get_db_pool():
    """Get database pool from main module."""
    import main
    if main.db_pool is None:
        await main.startup()
    return main.db_pool


async def diagnose_agent(pool: Any, agent_name: str, verbose: bool = True) -> dict:
    """Run comprehensive diagnostics on an agent's score."""
    from ciris_scoring import (
        calculate_ciris_score,
        calculate_factor_C,
        calculate_factor_I_int,
        calculate_factor_R,
        calculate_factor_I_inc,
        calculate_factor_S,
        PARAMS,
    )

    window_end = datetime.now(UTC)
    window_start = window_end - timedelta(days=PARAMS["default_window_days"])

    async with pool.acquire() as conn:
        # Get trace statistics
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_traces,
                COUNT(*) FILTER (WHERE selected_action IN ('SPEAK', 'TOOL', 'MEMORIZE', 'FORGET')) as non_exempt,
                MIN(timestamp) as oldest,
                MAX(timestamp) as newest,
                AVG(EXTRACT(epoch FROM (NOW() - timestamp)) / 86400) as avg_age_days,
                COUNT(*) FILTER (WHERE coherence_passed IS NOT NULL) as has_coherence,
                COUNT(*) FILTER (WHERE coherence_passed = true) as coherence_passed,
                COUNT(*) FILTER (WHERE coherence_passed = false) as coherence_failed
            FROM cirislens.accord_traces
            WHERE agent_name = $1
            AND timestamp BETWEEN $2 AND $3
        """, agent_name, window_start, window_end)

        # Calculate full score
        result = await calculate_ciris_score(pool, agent_name)
        composite = result.C.score * result.I_int.score * result.R.score * result.I_inc.score * result.S.score

        # Identify issues
        issues = []

        # Check each factor
        factors = {
            'C': result.C,
            'I_int': result.I_int,
            'R': result.R,
            'I_inc': result.I_inc,
            'S': result.S
        }

        for name, factor in factors.items():
            if factor.score < 0.7:
                issues.append(f"{name}={factor.score:.2f} is LOW")

        # Check S_base vs raw_coherence
        if 'S_base' in result.S.components and 'raw_coherence_rate' in result.S.components:
            s_base = result.S.components['S_base']
            raw_coh = result.S.components['raw_coherence_rate']
            if raw_coh > 0.9 and s_base < 0.7:
                decay_impact = raw_coh - s_base
                issues.append(f"Decay reducing S_base by {decay_impact:.0%} (traces avg {stats['avg_age_days']:.1f} days old)")

        # Check R drift
        if 'absolute_change' in result.R.components:
            change = result.R.components['absolute_change']
            if change > 0.05:
                issues.append(f"R: {change:.1%} CSDMA drift from baseline")

        # Check trace count
        if stats['total_traces'] < 30:
            issues.append(f"Only {stats['total_traces']} traces (need 30+ for reliable scoring)")

        # Check coherence data coverage
        if stats['non_exempt'] > 0:
            coverage = stats['has_coherence'] / stats['non_exempt'] if stats['non_exempt'] > 0 else 0
            if coverage < 0.8:
                issues.append(f"Only {coverage:.0%} of non-exempt traces have coherence data")

        diagnostics = {
            'agent': agent_name,
            'composite': composite,
            'factors': {name: factor.score for name, factor in factors.items()},
            'lowest_factor': min(factors.items(), key=lambda x: x[1].score),
            'trace_stats': {
                'total': stats['total_traces'],
                'non_exempt': stats['non_exempt'],
                'avg_age_days': round(stats['avg_age_days'] or 0, 1),
                'has_coherence': stats['has_coherence'],
                'coherence_passed': stats['coherence_passed'],
            },
            'issues': issues,
            'components': {
                'C': dict(result.C.components),
                'I_int': dict(result.I_int.components),
                'R': dict(result.R.components),
                'I_inc': dict(result.I_inc.components),
                'S': dict(result.S.components),
            }
        }

        if verbose:
            print_diagnostics(diagnostics)

        return diagnostics


def print_diagnostics(d: dict):
    """Pretty print diagnostics."""
    print(f"\n{'='*60}")
    print(f"CIRIS SCORE DIAGNOSTICS: {d['agent']}")
    print(f"{'='*60}")

    # Overall score
    composite = d['composite']
    if composite >= 0.7:
        status = "✓ HEALTHY"
    elif composite >= 0.5:
        status = "⚠ MODERATE"
    else:
        status = "✗ LOW"
    print(f"\nComposite Score: {composite:.1%} {status}")

    # Factor breakdown
    print(f"\nFactor Breakdown:")
    for name, score in d['factors'].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        flag = " ◄ LOW" if score < 0.7 else ""
        print(f"  {name:5} [{bar}] {score:.1%}{flag}")

    # Trace stats
    stats = d['trace_stats']
    print(f"\nTrace Statistics (30-day window):")
    print(f"  Total traces: {stats['total']}")
    print(f"  Non-exempt (SPEAK/TOOL/etc): {stats['non_exempt']}")
    print(f"  Average age: {stats['avg_age_days']} days")
    print(f"  With coherence data: {stats['has_coherence']}")
    print(f"  Coherence passed: {stats['coherence_passed']}")

    # Issues
    if d['issues']:
        print(f"\n⚠ Issues Detected:")
        for issue in d['issues']:
            print(f"  • {issue}")
    else:
        print(f"\n✓ No issues detected")

    # Lowest factor details
    lowest_name, lowest_factor = d['lowest_factor']
    print(f"\nLowest Factor Details ({lowest_name}):")
    for k, v in d['components'][lowest_name].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


async def diagnose_fleet(pool: Any):
    """Diagnose all agents with recent traces."""
    async with pool.acquire() as conn:
        agents = await conn.fetch("""
            SELECT DISTINCT agent_name
            FROM cirislens.accord_traces
            WHERE timestamp > NOW() - INTERVAL '30 days'
            ORDER BY agent_name
        """)

    print(f"\n{'='*60}")
    print("FLEET CIRIS SCORE SUMMARY")
    print(f"{'='*60}\n")

    results = []
    for row in agents:
        agent = row['agent_name']
        try:
            d = await diagnose_agent(pool, agent, verbose=False)
            results.append(d)
        except Exception as e:
            print(f"  {agent}: ERROR - {e}")

    # Sort by composite score
    results.sort(key=lambda x: x['composite'], reverse=True)

    # Print summary table
    print(f"{'Agent':<20} {'Score':>7} {'C':>6} {'I_int':>6} {'R':>6} {'I_inc':>6} {'S':>6}  Issues")
    print("-" * 90)

    for d in results:
        f = d['factors']
        issues = len(d['issues'])
        issue_str = f"⚠ {issues}" if issues > 0 else "✓"
        print(f"{d['agent']:<20} {d['composite']:>6.1%} {f['C']:>6.2f} {f['I_int']:>6.2f} {f['R']:>6.2f} {f['I_inc']:>6.2f} {f['S']:>6.2f}  {issue_str}")

    # Show agents with issues
    agents_with_issues = [d for d in results if d['issues']]
    if agents_with_issues:
        print(f"\n\nAgents needing attention:")
        for d in agents_with_issues:
            print(f"\n  {d['agent']} ({d['composite']:.1%}):")
            for issue in d['issues']:
                print(f"    • {issue}")


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pool = await get_db_pool()

    if sys.argv[1] == '--fleet' or sys.argv[1] == '--all':
        await diagnose_fleet(pool)
    else:
        agent_name = sys.argv[1]
        await diagnose_agent(pool, agent_name)


if __name__ == "__main__":
    asyncio.run(main())
