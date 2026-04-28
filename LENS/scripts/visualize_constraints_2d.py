#!/usr/bin/env python3
"""
Clearer 2D visualizations of the constraint space.
- Parallel coordinates (all 7 dimensions)
- Pairwise scatter matrix
- Density contours showing "valid region"
"""

import subprocess
import sys
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data():
    """Load trace data from production."""
    cmd = '''ssh -i ~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519 root@108.61.242.236 "docker exec cirislens-db psql -U cirislens -d cirislens -c \\"COPY (
SELECT
    csdma_plausibility_score as p,
    dsdma_domain_alignment as a,
    coherence_level as c,
    COALESCE(idma_k_eff, 1.0) as k,
    CASE WHEN idma_fragility_flag THEN 1 ELSE 0 END as fragile,
    CASE WHEN conscience_passed THEN 1 ELSE 0 END as conscience,
    CASE WHEN action_was_overridden THEN 1 ELSE 0 END as overridden,
    dsdma_domain as domain
FROM cirislens.accord_traces
WHERE signature_verified = true
AND csdma_plausibility_score IS NOT NULL
AND dsdma_domain_alignment IS NOT NULL
AND coherence_level IS NOT NULL
) TO STDOUT WITH CSV HEADER\\""'''

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    df = pd.read_csv(StringIO(result.stdout))
    df = df.drop_duplicates()
    df['domain'] = df['domain'].fillna('Unknown')
    return df


def parallel_coordinates_view(df):
    """
    Parallel coordinates: each vertical axis is a dimension.
    Lines connect values for each trace across all dimensions.
    """
    # Normalize k_eff to 0-1 range for visual consistency
    df_viz = df.copy()
    df_viz['k_norm'] = (df_viz['k'] - 1) / (df_viz['k'].max() - 1)  # Normalize k_eff

    fig = px.parallel_coordinates(
        df_viz,
        dimensions=['p', 'a', 'c', 'k_norm', 'fragile', 'conscience'],
        color='conscience',
        color_continuous_scale=[[0, 'red'], [1, 'green']],
        labels={
            'p': 'Plausibility',
            'a': 'Alignment',
            'c': 'Coherence',
            'k_norm': 'k_eff (norm)',
            'fragile': 'Fragile',
            'conscience': 'Conscience'
        },
        title='Constraint Space: Parallel Coordinates (green=conscience passed)'
    )

    fig.update_layout(height=500)
    return fig


def pairwise_density(df):
    """
    2x2 grid of key pairwise relationships with density contours.
    Shows where "valid" traces cluster.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Plausibility vs Alignment',
            'Plausibility vs Coherence',
            'k_eff vs Coherence',
            'Plausibility vs k_eff'
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )

    # Color by conscience passed
    colors = df['conscience'].map({1: 'green', 0: 'red'})

    # Panel 1: p vs a
    fig.add_trace(
        go.Scatter(x=df['p'], y=df['a'], mode='markers',
                   marker=dict(color=colors, size=5, opacity=0.5),
                   name='traces'),
        row=1, col=1
    )
    # Add density contour
    fig.add_trace(
        go.Histogram2dContour(x=df['p'], y=df['a'],
                               colorscale='Blues', showscale=False,
                               contours=dict(coloring='none'),
                               line=dict(width=2)),
        row=1, col=1
    )

    # Panel 2: p vs c
    fig.add_trace(
        go.Scatter(x=df['p'], y=df['c'], mode='markers',
                   marker=dict(color=colors, size=5, opacity=0.5),
                   showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram2dContour(x=df['p'], y=df['c'],
                               colorscale='Blues', showscale=False,
                               contours=dict(coloring='none'),
                               line=dict(width=2)),
        row=1, col=2
    )

    # Panel 3: k vs c
    fig.add_trace(
        go.Scatter(x=df['k'], y=df['c'], mode='markers',
                   marker=dict(color=colors, size=5, opacity=0.5),
                   showlegend=False),
        row=2, col=1
    )

    # Panel 4: p vs k
    fig.add_trace(
        go.Scatter(x=df['p'], y=df['k'], mode='markers',
                   marker=dict(color=colors, size=5, opacity=0.5),
                   showlegend=False),
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Plausibility", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Alignment", row=1, col=1, range=[0, 1.05])
    fig.update_xaxes(title_text="Plausibility", row=1, col=2, range=[0, 1.05])
    fig.update_yaxes(title_text="Coherence", row=1, col=2, range=[0, 1.05])
    fig.update_xaxes(title_text="k_eff", row=2, col=1)
    fig.update_yaxes(title_text="Coherence", row=2, col=1, range=[0, 1.05])
    fig.update_xaxes(title_text="Plausibility", row=2, col=2, range=[0, 1.05])
    fig.update_yaxes(title_text="k_eff", row=2, col=2)

    fig.update_layout(
        height=700,
        width=900,
        title_text="Constraint Surface: Pairwise Views (green=passed, red=overridden)",
        showlegend=False
    )

    return fig


def domain_profiles(df):
    """
    Radar charts showing average score profiles by domain.
    """
    domains = df['domain'].unique()

    fig = go.Figure()

    categories = ['Plausibility', 'Alignment', 'Coherence', 'Conscience Rate', '1/k_eff']

    for domain in domains:
        domain_df = df[df['domain'] == domain]
        values = [
            domain_df['p'].mean(),
            domain_df['a'].mean(),
            domain_df['c'].mean(),
            domain_df['conscience'].mean(),
            1 / domain_df['k'].mean(),  # Invert so higher = less correlated = better
        ]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=f'{domain} (n={len(domain_df)})',
            fill='toself',
            opacity=0.5
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Domain Profiles: Average Scores',
        height=500
    )

    return fig


def edge_case_view(df):
    """
    Highlight specific edge cases on p vs a plot.
    """
    # Classify
    def classify(row):
        if row['p'] < 0.4 and row['conscience'] == 1:
            return 'Low plausibility + passed'
        elif row['p'] > 0.9 and row['a'] < 0.3:
            return 'High p, low a'
        elif row['overridden'] == 1:
            return 'Overridden'
        else:
            return 'Normal'

    df['case'] = df.apply(classify, axis=1)

    fig = px.scatter(
        df, x='p', y='a',
        color='case',
        color_discrete_map={
            'Normal': 'lightgray',
            'Low plausibility + passed': 'red',
            'High p, low a': 'orange',
            'Overridden': 'purple'
        },
        opacity=0.6,
        title='Edge Cases: Plausibility vs Alignment',
        labels={'p': 'Plausibility', 'a': 'Alignment'}
    )

    # Add quadrant annotations
    fig.add_shape(type="line", x0=0.4, x1=0.4, y0=0, y1=1,
                  line=dict(color="red", dash="dash", width=1))
    fig.add_shape(type="line", x0=0, x1=1, y0=0.3, y1=0.3,
                  line=dict(color="orange", dash="dash", width=1))

    fig.add_annotation(x=0.2, y=0.8, text="Low P zone", showarrow=False,
                       font=dict(color="red", size=10))
    fig.add_annotation(x=0.95, y=0.15, text="Low A zone", showarrow=False,
                       font=dict(color="orange", size=10))

    fig.update_layout(height=500)
    fig.update_xaxes(range=[0, 1.05])
    fig.update_yaxes(range=[0, 1.05])

    return fig


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} traces")

    print("\nGenerating parallel coordinates...")
    fig1 = parallel_coordinates_view(df)
    fig1.write_html("/home/emoore/CIRISLens/constraint_parallel.html")

    print("Generating pairwise density...")
    fig2 = pairwise_density(df)
    fig2.write_html("/home/emoore/CIRISLens/constraint_pairwise.html")

    print("Generating domain profiles...")
    fig3 = domain_profiles(df)
    fig3.write_html("/home/emoore/CIRISLens/constraint_domains.html")

    print("Generating edge case view...")
    fig4 = edge_case_view(df)
    fig4.write_html("/home/emoore/CIRISLens/constraint_edges.html")

    print("\nFiles created:")
    print("  - constraint_parallel.html  (all dimensions)")
    print("  - constraint_pairwise.html  (2D density plots)")
    print("  - constraint_domains.html   (radar by domain)")
    print("  - constraint_edges.html     (edge cases highlighted)")


if __name__ == '__main__':
    main()
