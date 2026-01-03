#!/usr/bin/env python3
"""
Visualization of deception complexity results.
Creates plots showing cost growth for honest vs deceptive agents.
"""

import json
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from deception_complexity import run_simulation


def plot_operations_vs_statements():
    """Plot: Operations count vs number of statements (n)."""
    m_values = [8, 10]
    k = 3
    n_values = list(range(1, 11))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, m in enumerate(m_values):
        honest_ops = []
        deceptive_ops = []

        for n in n_values:
            result = run_simulation(m=m, n=n, k=k, seed=42)
            honest_ops.append(result['honest_total_ops'])
            deceptive_ops.append(result['deceptive_total_ops'])

        ax = axes[idx]
        ax.plot(n_values, honest_ops, 'o-', label='Honest', linewidth=2, markersize=8)
        ax.plot(n_values, deceptive_ops, 's-', label='Deceptive', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Statements (n)', fontsize=12)
        ax.set_ylabel('Total Operations', fontsize=12)
        ax.set_title(f'Operations Growth (m={m} facts, k={k} literals)', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/home/emoore/RATCHET/simulation/ops_vs_statements.png', dpi=150)
    print("Saved: ops_vs_statements.png")
    plt.close()


def plot_operations_vs_world_size():
    """Plot: Operations count vs world model size (m)."""
    m_values = [6, 8, 10, 12]
    n = 5
    k = 3

    honest_ops = []
    deceptive_ops = []

    for m in m_values:
        result = run_simulation(m=m, n=n, k=k, seed=42)
        honest_ops.append(result['honest_total_ops'])
        deceptive_ops.append(result['deceptive_total_ops'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(m_values))
    width = 0.35

    ax.bar(x - width/2, honest_ops, width, label='Honest', alpha=0.8)
    ax.bar(x + width/2, deceptive_ops, width, label='Deceptive', alpha=0.8)

    ax.set_xlabel('World Size (m = number of facts)', fontsize=12)
    ax.set_ylabel('Total Operations (log scale)', fontsize=12)
    ax.set_title(f'Computational Cost vs World Size (n={n} statements, k={k} literals)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(m_values)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/emoore/RATCHET/simulation/ops_vs_world_size.png', dpi=150)
    print("Saved: ops_vs_world_size.png")
    plt.close()


def plot_cost_ratio():
    """Plot: Deception/Honest cost ratio over time."""
    m_values = [8, 10, 12]
    k = 3
    n_values = list(range(1, 11))

    fig, ax = plt.subplots(figsize=(10, 6))

    for m in m_values:
        ratios = []

        for n in n_values:
            result = run_simulation(m=m, n=n, k=k, seed=42)
            ratios.append(result['ops_ratio'])

        ax.plot(n_values, ratios, 'o-', label=f'm={m} facts', linewidth=2, markersize=8)

    # Add 10x threshold line
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10x threshold')

    ax.set_xlabel('Number of Statements (n)', fontsize=12)
    ax.set_ylabel('Cost Ratio (Deceptive/Honest)', fontsize=12)
    ax.set_title('When Does Deception Cost 10x More?', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/home/emoore/RATCHET/simulation/cost_ratio.png', dpi=150)
    print("Saved: cost_ratio.png")
    plt.close()


def plot_theoretical_complexity():
    """Plot: Theoretical complexity curves."""
    n_values = np.arange(1, 21)
    k = 3

    fig, ax = plt.subplots(figsize=(10, 6))

    # Honest: O(n*k)
    honest_complexity = n_values * k

    # Deceptive for different m values: O(n^2 * 2^m)
    m_values = [6, 8, 10, 12]

    ax.plot(n_values, honest_complexity, 'k-', linewidth=3, label='Honest: O(n·k)')

    for m in m_values:
        deceptive_complexity = n_values**2 * (2**m)
        ax.plot(n_values, deceptive_complexity, '--', linewidth=2,
                label=f'Deceptive (m={m}): O(n²·2^{m})')

    ax.set_xlabel('Number of Statements (n)', fontsize=12)
    ax.set_ylabel('Theoretical Operations', fontsize=12)
    ax.set_title('Theoretical Complexity: Truth vs Deception', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([1, 1e8])

    plt.tight_layout()
    plt.savefig('/home/emoore/RATCHET/simulation/theoretical_complexity.png', dpi=150)
    print("Saved: theoretical_complexity.png")
    plt.close()


def generate_summary_table():
    """Generate a summary table of key results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE: Break-even Analysis")
    print("="*80)
    print(f"{'m (facts)':<12} {'n=2':<15} {'n=5':<15} {'n=10':<15}")
    print("-"*80)

    for m in [6, 8, 10, 12]:
        row = [f"{m}"]
        for n in [2, 5, 10]:
            if m > 12 and n > 5:
                row.append("--")
                continue
            try:
                result = run_simulation(m=m, n=n, k=3, seed=42)
                ratio = result['ops_ratio']
                row.append(f"{ratio:.1f}x")
            except:
                row.append("--")

        print(f"{row[0]:<12} {row[1]:<15} {row[2]:<15} {row[3] if len(row) > 3 else '--':<15}")

    print("="*80)
    print("Note: Values show deception cost ratio (operations compared to honest agent)")
    print("      Bold indicates >10x threshold reached")
    print("="*80)


def main():
    """Generate all visualizations."""
    print("\nGenerating visualizations for deception complexity analysis...\n")

    # Suppress simulation output for cleaner viz output
    import sys
    import os

    # Redirect stdout temporarily
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        plot_theoretical_complexity()
        plot_operations_vs_statements()
        plot_operations_vs_world_size()
        plot_cost_ratio()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    print("\nAll visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - theoretical_complexity.png")
    print("  - ops_vs_statements.png")
    print("  - ops_vs_world_size.png")
    print("  - cost_ratio.png")

    # Generate summary table with output
    generate_summary_table()


if __name__ == "__main__":
    main()
