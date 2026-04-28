#!/usr/bin/env python3
"""
Visualize the 7D constraint space in 3D using dimensionality reduction.

Three approaches:
1. PCA - Linear projection preserving variance
2. Direct mapping - p, a, c as axes with k/fragile/conscience as color/size
3. UMAP - Non-linear embedding preserving local structure

Usage:
    python scripts/visualize_constraint_space.py [--method pca|direct|umap]
"""

import argparse
import sys
from io import StringIO

import numpy as np
import pandas as pd

# Check for required packages
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install plotly scikit-learn pandas numpy")
    sys.exit(1)


def load_data_from_db():
    """Load trace data from database via SSH."""
    import subprocess

    cmd = '''ssh -i ~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519 root@108.61.242.236 "docker exec cirislens-db psql -U cirislens -d cirislens -c \\"COPY (
SELECT
    trace_id,
    csdma_plausibility_score as p,
    dsdma_domain_alignment as a,
    coherence_level as c,
    COALESCE(idma_k_eff, 1.0) as k,
    CASE WHEN idma_fragility_flag THEN 1 ELSE 0 END as fragile,
    CASE WHEN conscience_passed THEN 1 ELSE 0 END as conscience,
    CASE WHEN action_was_overridden THEN 1 ELSE 0 END as overridden,
    dsdma_domain as domain,
    selected_action as action
FROM cirislens.accord_traces
WHERE signature_verified = true
AND csdma_plausibility_score IS NOT NULL
AND dsdma_domain_alignment IS NOT NULL
AND coherence_level IS NOT NULL
) TO STDOUT WITH CSV HEADER\\""'''

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching data: {result.stderr}")
        sys.exit(1)

    return pd.read_csv(StringIO(result.stdout))


def classify_edge_case(row):
    """Classify trace into edge case categories."""
    p, a, c, k = row['p'], row['a'], row['c'], row['k']
    fragile = row['fragile']
    conscience = row['conscience']
    overridden = row['overridden']

    if p < 0.4 and conscience == 1:
        return 'CONSCIENCE_BYPASS'
    if p > 0.9 and a < 0.3:
        return 'FALSE_CONFIDENCE'
    if fragile == 1 and c > 0.7:
        return 'FRAGILE_PARADOX'
    if k > 2.0:
        return 'HIGH_K_EFF'
    if abs(p - a) > 0.5:
        return 'SCORE_DIVERGENCE'
    if overridden == 1:
        return 'OVERRIDDEN'
    return 'NORMAL'


def visualize_direct(df):
    """Direct 3D mapping: p, a, c as spatial axes."""
    df['edge_case'] = df.apply(classify_edge_case, axis=1)
    df['marker_size'] = df['k'].clip(1, 3) * 5  # Size by k_eff

    # Color by domain, shape by edge case
    fig = px.scatter_3d(
        df,
        x='p', y='a', z='c',
        color='domain',
        symbol='edge_case',
        size='marker_size',
        hover_data=['trace_id', 'k', 'fragile', 'conscience', 'action'],
        title='CIRIS Constraint Space (Direct: Plausibility × Alignment × Coherence)',
        labels={
            'p': 'Plausibility',
            'a': 'Domain Alignment',
            'c': 'Coherence',
        },
        opacity=0.7,
    )

    # Add constraint surface wireframe (valid region boundary)
    # The "valid" region is roughly p > 0.5, a > 0.5, c > 0.5
    fig.add_trace(go.Mesh3d(
        x=[0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1],
        y=[0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5],
        z=[0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1],
        opacity=0.1,
        color='green',
        name='Valid Region'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Plausibility (CSDMA)',
            yaxis_title='Alignment (DSDMA)',
            zaxis_title='Coherence',
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
            zaxis=dict(range=[0, 1.05]),
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def visualize_pca(df):
    """PCA projection of 7D space to 3D."""
    df['edge_case'] = df.apply(classify_edge_case, axis=1)

    # Prepare feature matrix
    features = ['p', 'a', 'c', 'k', 'fragile', 'conscience', 'overridden']
    X = df[features].fillna(df[features].median())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    df['PC3'] = X_pca[:, 2]

    # Explained variance
    var_exp = pca.explained_variance_ratio_

    fig = px.scatter_3d(
        df,
        x='PC1', y='PC2', z='PC3',
        color='domain',
        symbol='edge_case',
        hover_data=['trace_id', 'p', 'a', 'c', 'k', 'action'],
        title=f'CIRIS Constraint Space (PCA: {var_exp.sum()*100:.1f}% variance explained)',
        labels={
            'PC1': f'PC1 ({var_exp[0]*100:.1f}%)',
            'PC2': f'PC2 ({var_exp[1]*100:.1f}%)',
            'PC3': f'PC3 ({var_exp[2]*100:.1f}%)',
        },
        opacity=0.7,
    )

    # Print component loadings
    print("\nPCA Component Loadings:")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2', 'PC3'],
        index=features
    )
    print(loadings.round(3))

    return fig


def visualize_umap(df):
    """UMAP non-linear embedding to 3D."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        print("Falling back to PCA...")
        return visualize_pca(df)

    df['edge_case'] = df.apply(classify_edge_case, axis=1)

    features = ['p', 'a', 'c', 'k', 'fragile', 'conscience', 'overridden']
    X = df[features].fillna(df[features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15)
    X_umap = reducer.fit_transform(X_scaled)

    df['UMAP1'] = X_umap[:, 0]
    df['UMAP2'] = X_umap[:, 1]
    df['UMAP3'] = X_umap[:, 2]

    fig = px.scatter_3d(
        df,
        x='UMAP1', y='UMAP2', z='UMAP3',
        color='domain',
        symbol='edge_case',
        hover_data=['trace_id', 'p', 'a', 'c', 'k', 'action'],
        title='CIRIS Constraint Space (UMAP embedding)',
        opacity=0.7,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize CIRIS constraint space')
    parser.add_argument('--method', choices=['direct', 'pca', 'umap'], default='direct',
                        help='Visualization method')
    parser.add_argument('--output', default='constraint_space.html',
                        help='Output HTML file')
    args = parser.parse_args()

    print("Loading trace data from production database...")
    df = load_data_from_db()
    print(f"Loaded {len(df)} traces")

    # Remove duplicates
    df = df.drop_duplicates(subset=['trace_id'])
    print(f"After dedup: {len(df)} unique traces")

    # Fill missing domains
    df['domain'] = df['domain'].fillna('Unknown')

    # Generate visualization
    print(f"\nGenerating {args.method.upper()} visualization...")

    if args.method == 'direct':
        fig = visualize_direct(df)
    elif args.method == 'pca':
        fig = visualize_pca(df)
    elif args.method == 'umap':
        fig = visualize_umap(df)

    # Summary stats
    edge_cases = df.apply(classify_edge_case, axis=1).value_counts()
    print("\nEdge Case Distribution:")
    print(edge_cases)

    # Save
    output_path = f"/home/emoore/CIRISLens/{args.output}"
    fig.write_html(output_path)
    print(f"\nSaved to: {output_path}")
    print(f"Open in browser: file://{output_path}")


if __name__ == '__main__':
    main()
