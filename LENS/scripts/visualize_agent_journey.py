#!/usr/bin/env python3
"""
Agent Journey Visualization - Trace ethical paths through time.

Each agent draws a line through ethical space as it processes observations.
This visualization shows:
- Timeline of thoughts within tasks
- Position in (P, A, C) constraint space at each step
- Pipeline step durations
- Observation weight (memory, context, alternatives)
- Outcome coloring: green=positive, yellow=complete, red=reject/incomplete
"""

import subprocess
import sys
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data():
    """Load enriched trace data from production."""
    cmd = '''ssh -i ~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519 root@108.61.242.236 "docker exec cirislens-db psql -U cirislens -d cirislens -c \\\"COPY (
SELECT
    trace_id,
    thought_id,
    task_id,
    agent_id_hash,
    agent_name,
    -- Ethical position
    csdma_plausibility_score as p,
    dsdma_domain_alignment as a,
    coherence_level as c,
    COALESCE(idma_k_eff, 1.0) as k_eff,
    COALESCE(idma_fragility_flag, false) as fragile,
    -- Outcome
    selected_action as verb,
    COALESCE(has_positive_moment, false) as positive,
    COALESCE(action_success, false) as success,
    COALESCE(conscience_passed, true) as conscience_passed,
    COALESCE(action_was_overridden, false) as overridden,
    -- Step timestamps
    thought_start_at,
    snapshot_at,
    dma_results_at,
    aspdma_at,
    idma_at,
    tsaspdma_at,
    conscience_at,
    action_result_at,
    -- Observation weight
    COALESCE(memory_count, 0) as memory_count,
    COALESCE(context_tokens, 0) as context_tokens,
    COALESCE(conversation_turns, 0) as conversation_turns,
    COALESCE(alternatives_considered, 0) as alternatives,
    COALESCE(conscience_checks_count, 0) as conscience_checks,
    -- Overall timing
    started_at,
    completed_at
FROM cirislens.accord_traces
WHERE signature_verified = true
AND agent_name IS NOT NULL
ORDER BY agent_name, task_id, thought_start_at
) TO STDOUT WITH CSV HEADER\\\"\"'''

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    df = pd.read_csv(StringIO(result.stdout))

    # Parse timestamps
    timestamp_cols = ['thought_start_at', 'snapshot_at', 'dma_results_at', 'aspdma_at',
                      'idma_at', 'tsaspdma_at', 'conscience_at', 'action_result_at',
                      'started_at', 'completed_at']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Use started_at as fallback for thought_start_at if missing
    if 'thought_start_at' in df.columns and 'started_at' in df.columns:
        df['thought_start_at'] = df['thought_start_at'].fillna(df['started_at'])

    return df


def is_true(val):
    """Handle boolean fields that may come as strings or booleans."""
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ('true', 't', '1', 'yes')
    return bool(val)


def classify_task(task_df):
    """Classify entire task outcome. Green if TASK_COMPLETE has positive moment."""
    # Check if any TASK_COMPLETE in this task has positive moment
    task_complete = task_df[task_df['verb'].str.upper() == 'TASK_COMPLETE'] if 'verb' in task_df.columns else pd.DataFrame()

    if len(task_complete) > 0:
        if task_complete['positive'].apply(is_true).any():
            return 'green'
        else:
            return 'yellow'

    # Check for rejections
    if task_df['verb'].str.upper().eq('REJECT').any():
        return 'red'

    # Check for overrides
    if task_df['overridden'].apply(is_true).any():
        return 'red'

    # Has activity but no TASK_COMPLETE
    if task_df['success'].apply(is_true).any():
        return 'yellow'

    return 'red'


def classify_outcome(row):
    """Classify individual trace outcome for coloring (fallback)."""
    verb = str(row['verb']).upper() if pd.notna(row['verb']) else ''

    overridden = is_true(row.get('overridden'))
    positive = is_true(row.get('positive'))
    success = is_true(row.get('success'))

    if verb == 'REJECT' or overridden:
        return 'red'
    elif positive:
        return 'green'
    elif verb == 'TASK_COMPLETE' or success:
        return 'yellow'
    elif verb in ['SPEAK', 'OBSERVE', 'TOOL', 'MEMORIZE', 'RECALL', 'PONDER', 'DEFER']:
        return 'yellow'
    else:
        return 'red'


def compute_step_durations(row):
    """Compute duration of each pipeline step in ms."""
    steps = []
    timestamps = [
        ('thought_start', row.get('thought_start_at')),
        ('snapshot', row.get('snapshot_at')),
        ('dma', row.get('dma_results_at')),
        ('aspdma', row.get('aspdma_at')),
        ('idma', row.get('idma_at')),
        ('tsaspdma', row.get('tsaspdma_at')),
        ('conscience', row.get('conscience_at')),
        ('action', row.get('action_result_at')),
    ]

    prev_ts = None
    for name, ts in timestamps:
        if pd.notna(ts):
            if prev_ts is not None:
                duration_ms = (ts - prev_ts).total_seconds() * 1000
                steps.append((name, duration_ms))
            prev_ts = ts

    return steps


def create_agent_timeline(df, agent_name):
    """Create timeline visualization for a single agent."""
    agent_df = df[df['agent_name'] == agent_name].copy()

    if len(agent_df) == 0:
        return None

    # Add outcome color
    agent_df['color'] = agent_df.apply(classify_outcome, axis=1)

    # Get unique tasks
    tasks = agent_df['task_id'].dropna().unique()

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'{agent_name}: Ethical Position Over Time',
            'Pipeline Step Durations',
            'Observation Weight'
        ),
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08
    )

    color_map = {'green': '#2ecc71', 'yellow': '#f1c40f', 'red': '#e74c3c'}

    # Panel 1: Ethical position (P, A, C) over time
    for task_id in tasks:
        task_df = agent_df[agent_df['task_id'] == task_id].sort_values('thought_start_at')

        if len(task_df) == 0 or task_df['thought_start_at'].isna().all():
            continue

        # Use first valid timestamp as x
        x_vals = task_df['thought_start_at'].dropna()
        if len(x_vals) == 0:
            continue

        # Plot P (plausibility) - solid line
        fig.add_trace(
            go.Scatter(
                x=task_df['thought_start_at'],
                y=task_df['p'],
                mode='lines+markers',
                name=f'P ({task_id[:8] if task_id else "?"}...)',
                line=dict(width=2),
                marker=dict(
                    size=8,
                    color=[color_map.get(c, '#95a5a6') for c in task_df['color']],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>P</b>: %{y:.2f}<br>Verb: %{customdata}<extra></extra>',
                customdata=task_df['verb'],
                legendgroup=task_id,
                showlegend=True
            ),
            row=1, col=1
        )

        # Plot A (alignment) - dashed
        fig.add_trace(
            go.Scatter(
                x=task_df['thought_start_at'],
                y=task_df['a'],
                mode='lines+markers',
                name=f'A',
                line=dict(width=2, dash='dash'),
                marker=dict(size=6, color=[color_map.get(c, '#95a5a6') for c in task_df['color']]),
                hovertemplate='<b>A</b>: %{y:.2f}<extra></extra>',
                legendgroup=task_id,
                showlegend=False
            ),
            row=1, col=1
        )

        # Plot C (coherence) - dotted
        fig.add_trace(
            go.Scatter(
                x=task_df['thought_start_at'],
                y=task_df['c'],
                mode='lines+markers',
                name=f'C',
                line=dict(width=2, dash='dot'),
                marker=dict(size=6, color=[color_map.get(c, '#95a5a6') for c in task_df['color']]),
                hovertemplate='<b>C</b>: %{y:.2f}<extra></extra>',
                legendgroup=task_id,
                showlegend=False
            ),
            row=1, col=1
        )

    # Panel 2: Pipeline step durations as stacked bars
    step_names = ['snapshot', 'dma', 'aspdma', 'idma', 'tsaspdma', 'conscience', 'action']
    step_colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#34495e', '#e74c3c', '#2ecc71']

    for idx, row in agent_df.iterrows():
        steps = compute_step_durations(row)
        if not steps:
            continue

        x_pos = row['thought_start_at']
        if pd.isna(x_pos):
            continue

        cumulative = 0
        for step_name, duration in steps:
            if step_name in step_names:
                color_idx = step_names.index(step_name)
                fig.add_trace(
                    go.Bar(
                        x=[x_pos],
                        y=[duration],
                        base=cumulative,
                        marker_color=step_colors[color_idx],
                        name=step_name,
                        showlegend=(idx == agent_df.index[0]),
                        legendgroup=step_name,
                        hovertemplate=f'<b>{step_name}</b>: %{{y:.0f}}ms<extra></extra>',
                        width=60000  # 1 minute width
                    ),
                    row=2, col=1
                )
                cumulative += duration

    # Panel 3: Observation weight
    fig.add_trace(
        go.Scatter(
            x=agent_df['thought_start_at'],
            y=agent_df['memory_count'],
            mode='lines+markers',
            name='Memories',
            marker=dict(size=8),
            line=dict(width=2),
            hovertemplate='<b>Memories</b>: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=agent_df['thought_start_at'],
            y=agent_df['context_tokens'] / 100,  # Scale down
            mode='lines+markers',
            name='Context (x100 tokens)',
            marker=dict(size=6),
            line=dict(width=2, dash='dash'),
            hovertemplate='<b>Context</b>: %{customdata} tokens<extra></extra>',
            customdata=agent_df['context_tokens']
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=agent_df['thought_start_at'],
            y=agent_df['alternatives'],
            mode='lines+markers',
            name='Alternatives',
            marker=dict(size=6),
            line=dict(width=2, dash='dot'),
            hovertemplate='<b>Alternatives</b>: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        title=dict(
            text=f"Agent Journey: {agent_name}",
            font=dict(size=20)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='stack'
    )

    fig.update_yaxes(title_text="Score (0-1)", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Duration (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    return fig


def create_task_flow(df, agent_name):
    """Create task summary - one row per task showing journey and outcome."""
    agent_df = df[df['agent_name'] == agent_name].copy()

    if len(agent_df) == 0:
        return None

    color_map = {'green': '#2ecc71', 'yellow': '#f1c40f', 'red': '#e74c3c'}

    # Group by task and compute task-level metrics
    tasks_data = []
    for task_id, task_df in agent_df.groupby('task_id'):
        if pd.isna(task_id):
            continue

        task_df = task_df.sort_values('thought_start_at')
        task_color = classify_task(task_df)

        # Get task metrics
        thoughts = len(task_df)
        start = task_df['thought_start_at'].min()
        end = task_df['thought_start_at'].max()
        duration = (end - start).total_seconds() if pd.notna(start) and pd.notna(end) else 0

        avg_p = task_df['p'].mean()
        avg_a = task_df['a'].mean()
        avg_c = task_df['c'].mean()

        # Get final verb
        final_verb = task_df.iloc[-1]['verb'] if len(task_df) > 0 else '?'

        tasks_data.append({
            'task_id': task_id[:20] + '...' if len(str(task_id)) > 20 else task_id,
            'thoughts': thoughts,
            'start': start,
            'duration': duration,
            'avg_p': avg_p,
            'avg_a': avg_a,
            'avg_c': avg_c,
            'final_verb': final_verb,
            'color': task_color
        })

    if not tasks_data:
        return None

    tasks_df = pd.DataFrame(tasks_data).sort_values('start', ascending=False).head(30)

    fig = go.Figure()

    # Create horizontal bar chart - each task is a bar
    fig.add_trace(go.Bar(
        y=tasks_df['task_id'],
        x=tasks_df['thoughts'],
        orientation='h',
        marker=dict(
            color=[color_map.get(c, '#95a5a6') for c in tasks_df['color']],
            line=dict(width=1, color='white')
        ),
        text=tasks_df['final_verb'],
        textposition='inside',
        hovertemplate=(
            '<b>Task</b>: %{y}<br>'
            '<b>Thoughts</b>: %{x}<br>'
            '<b>Final verb</b>: %{text}<br>'
            '<b>Avg P/A/C</b>: %{customdata[0]:.2f}/%{customdata[1]:.2f}/%{customdata[2]:.2f}<extra></extra>'
        ),
        customdata=list(zip(tasks_df['avg_p'], tasks_df['avg_a'], tasks_df['avg_c']))
    ))

    fig.update_layout(
        height=max(300, len(tasks_df) * 25 + 100),
        title=f"Recent Tasks: {agent_name} (bar = thought count, color = outcome)",
        xaxis_title="Thoughts in Task",
        yaxis_title="",
        showlegend=False
    )

    return fig


def create_3d_journey(df, agent_name):
    """Create 3D visualization of agent journey through (P, A, C) space."""
    agent_df = df[df['agent_name'] == agent_name].copy()

    if len(agent_df) == 0:
        return None

    # Filter to rows with valid P, A, C
    valid = agent_df[['p', 'a', 'c']].notna().all(axis=1)
    agent_df = agent_df[valid].copy()

    if len(agent_df) == 0:
        return None

    # Classify by task, not individual thought
    task_colors = {}
    for task_id, task_df in agent_df.groupby('task_id'):
        task_colors[task_id] = classify_task(task_df)

    agent_df['task_color'] = agent_df['task_id'].map(task_colors)

    color_map = {'green': '#2ecc71', 'yellow': '#f1c40f', 'red': '#e74c3c'}

    fig = go.Figure()

    # Plot points by color for better legend
    for color_name, color_hex in color_map.items():
        mask = agent_df['task_color'] == color_name
        if mask.sum() == 0:
            continue

        subset = agent_df[mask]
        fig.add_trace(
            go.Scatter3d(
                x=subset['p'].tolist(),
                y=subset['a'].tolist(),
                z=subset['c'].tolist(),
                mode='markers',
                marker=dict(
                    size=8,
                    color=color_hex,
                    opacity=0.9
                ),
                text=subset['verb'].fillna('').tolist(),
                hovertemplate=(
                    '<b>P</b>: %{x:.2f}<br>'
                    '<b>A</b>: %{y:.2f}<br>'
                    '<b>C</b>: %{z:.2f}<br>'
                    '<b>Verb</b>: %{text}<extra></extra>'
                ),
                name=color_name.capitalize()
            )
        )

    # Add "valid region" wireframe (P > 0.5, A > 0.5, C > 0.5)
    fig.add_trace(
        go.Mesh3d(
            x=[0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1],
            y=[0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5],
            z=[0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1],
            opacity=0.1,
            color='green',
            name='Valid Region',
            showlegend=False
        )
    )

    fig.update_layout(
        height=600,
        title=f"3D Ethical Space: {agent_name}",
        scene=dict(
            xaxis_title='Plausibility (P)',
            yaxis_title='Alignment (A)',
            zaxis_title='Coherence (C)',
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
            zaxis=dict(range=[0, 1.05]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )
    )

    return fig


def create_verb_sunburst(df, agent_name):
    """Create sunburst showing verb distribution by outcome."""
    agent_df = df[df['agent_name'] == agent_name].copy()

    if len(agent_df) == 0:
        return None

    agent_df['color'] = agent_df.apply(classify_outcome, axis=1)

    # Group by verb and outcome
    grouped = agent_df.groupby(['verb', 'color']).size().reset_index(name='count')

    # Build sunburst data
    labels = ['All']
    parents = ['']
    values = [len(agent_df)]
    colors = ['#95a5a6']

    color_map = {'green': '#2ecc71', 'yellow': '#f1c40f', 'red': '#e74c3c'}

    for verb in grouped['verb'].unique():
        if pd.isna(verb):
            continue
        verb_total = grouped[grouped['verb'] == verb]['count'].sum()
        labels.append(verb)
        parents.append('All')
        values.append(verb_total)
        colors.append('#3498db')

        for _, row in grouped[grouped['verb'] == verb].iterrows():
            labels.append(f"{verb} ({row['color']})")
            parents.append(verb)
            values.append(row['count'])
            colors.append(color_map.get(row['color'], '#95a5a6'))

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        branchvalues='total'
    ))

    fig.update_layout(
        height=500,
        title=f"Verb Distribution: {agent_name}"
    )

    return fig


def create_dashboard(df):
    """Create interactive dashboard with agent selector."""
    agents = sorted(df['agent_name'].dropna().unique())

    if len(agents) == 0:
        print("No agents found in data")
        return

    # Create HTML with dropdown
    html_parts = ['''
<!DOCTYPE html>
<html>
<head>
    <title>Agent Journey Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .header h1 { margin: 0; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        select { padding: 10px; font-size: 16px; border-radius: 5px; border: 1px solid #ddd; min-width: 300px; }
        .legend { display: flex; gap: 20px; margin-top: 10px; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
        .green { background: #2ecc71; }
        .yellow { background: #f1c40f; }
        .red { background: #e74c3c; }
        .viz-container { display: none; }
        .viz-container.active { display: block; }
        .panel { background: white; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; padding: 15px; background: white; border-radius: 8px; margin-bottom: 20px; }
        .stat { text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .stat-label { font-size: 12px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Agent Journey Visualization</h1>
        <p>Trace ethical paths through constraint space over time</p>
    </div>

    <div class="controls">
        <label for="agent-select"><strong>Select Agent:</strong></label>
        <select id="agent-select" data-testid="agent-select" onchange="showAgent(this.value)">
            <option value="">-- Choose an agent --</option>
''']

    for agent in agents:
        count = len(df[df['agent_name'] == agent])
        html_parts.append(f'            <option value="{agent}">{agent} ({count} traces)</option>\n')

    html_parts.append('''
        </select>
        <button id="btn-prev" data-testid="prev-agent" onclick="navigateAgent(-1)" style="margin-left:10px;padding:8px 12px;">← Prev</button>
        <button id="btn-next" data-testid="next-agent" onclick="navigateAgent(1)" style="padding:8px 12px;">Next →</button>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot green"></div> Positive moment</div>
            <div class="legend-item"><div class="legend-dot yellow"></div> Completed (no positive)</div>
            <div class="legend-item"><div class="legend-dot red"></div> Rejected / incomplete</div>
        </div>
    </div>

    <div id="placeholder" style="text-align: center; padding: 50px; color: #7f8c8d;">
        Select an agent to view their journey through ethical space
    </div>
''')

    # Generate visualizations for each agent
    for agent in agents:
        agent_df = df[df['agent_name'] == agent]

        # Stats
        total = len(agent_df)
        green = len(agent_df[agent_df.apply(classify_outcome, axis=1) == 'green'])
        yellow = len(agent_df[agent_df.apply(classify_outcome, axis=1) == 'yellow'])
        red = len(agent_df[agent_df.apply(classify_outcome, axis=1) == 'red'])
        tasks = agent_df['task_id'].nunique()
        avg_p = agent_df['p'].mean()
        avg_a = agent_df['a'].mean()
        avg_c = agent_df['c'].mean()

        html_parts.append(f'''
    <div id="agent-{agent.replace(" ", "_")}" class="viz-container">
        <div class="stats">
            <div class="stat"><div class="stat-value">{total}</div><div class="stat-label">Total Traces</div></div>
            <div class="stat"><div class="stat-value">{tasks}</div><div class="stat-label">Tasks</div></div>
            <div class="stat"><div class="stat-value" style="color:#2ecc71">{green}</div><div class="stat-label">Positive</div></div>
            <div class="stat"><div class="stat-value" style="color:#f1c40f">{yellow}</div><div class="stat-label">Completed</div></div>
            <div class="stat"><div class="stat-value" style="color:#e74c3c">{red}</div><div class="stat-label">Rejected</div></div>
            <div class="stat"><div class="stat-value">{avg_p:.2f}</div><div class="stat-label">Avg P</div></div>
            <div class="stat"><div class="stat-value">{avg_a:.2f}</div><div class="stat-label">Avg A</div></div>
            <div class="stat"><div class="stat-value">{avg_c:.2f}</div><div class="stat-label">Avg C</div></div>
        </div>
''')

        # Timeline
        fig_timeline = create_agent_timeline(df, agent)
        if fig_timeline:
            html_parts.append(f'        <div class="panel" id="timeline-{agent.replace(" ", "_")}">')
            html_parts.append(fig_timeline.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append('        </div>\n')

        # 3D journey
        fig_3d = create_3d_journey(df, agent)
        if fig_3d:
            html_parts.append(f'        <div class="panel" id="journey3d-{agent.replace(" ", "_")}">')
            html_parts.append(fig_3d.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append('        </div>\n')

        # Task flow
        fig_flow = create_task_flow(df, agent)
        if fig_flow:
            html_parts.append(f'        <div class="panel" id="flow-{agent.replace(" ", "_")}">')
            html_parts.append(fig_flow.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append('        </div>\n')

        # Verb sunburst
        fig_sunburst = create_verb_sunburst(df, agent)
        if fig_sunburst:
            html_parts.append(f'        <div class="panel" id="sunburst-{agent.replace(" ", "_")}">')
            html_parts.append(fig_sunburst.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append('        </div>\n')

        html_parts.append('    </div>\n')

    html_parts.append('''
    <script>
        function showAgent(agent) {
            // Hide all
            document.querySelectorAll('.viz-container').forEach(el => el.classList.remove('active'));
            document.getElementById('placeholder').style.display = agent ? 'none' : 'block';

            if (agent) {
                const containerId = 'agent-' + agent.replace(/ /g, '_');
                const container = document.getElementById(containerId);
                if (container) {
                    container.classList.add('active');
                    // Trigger resize for plotly
                    window.dispatchEvent(new Event('resize'));
                }
            }
        }

        function navigateAgent(direction) {
            const select = document.getElementById('agent-select');
            const options = Array.from(select.options).filter(o => o.value);
            const currentIdx = options.findIndex(o => o.value === select.value);
            let newIdx = currentIdx + direction;
            if (newIdx < 0) newIdx = options.length - 1;
            if (newIdx >= options.length) newIdx = 0;
            select.value = options[newIdx].value;
            showAgent(select.value);
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') navigateAgent(-1);
            if (e.key === 'ArrowRight') navigateAgent(1);
        });
    </script>
</body>
</html>
''')

    return ''.join(html_parts)


def main():
    print("Loading enriched trace data...")
    df = load_data()
    print(f"Loaded {len(df)} traces from {df['agent_name'].nunique()} agents")

    if len(df) == 0:
        print("No data to visualize")
        return

    # Show data summary
    print("\nData Summary:")
    print(f"  Agents: {sorted(df['agent_name'].dropna().unique())}")
    print(f"  Tasks: {df['task_id'].nunique()}")
    print(f"  Date range: {df['thought_start_at'].min()} to {df['thought_start_at'].max()}")

    # Check for new fields
    has_timestamps = df['thought_start_at'].notna().sum()
    has_memory = df['memory_count'].notna().sum()
    print(f"\n  Step timestamps: {has_timestamps} traces ({has_timestamps/len(df)*100:.1f}%)")
    print(f"  Memory count: {has_memory} traces ({has_memory/len(df)*100:.1f}%)")

    print("\nGenerating dashboard...")
    html = create_dashboard(df)

    output_path = "/home/emoore/CIRISLens/agent_journey.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nSaved to: {output_path}")
    print(f"Open in browser: file://{output_path}")


if __name__ == '__main__':
    main()
