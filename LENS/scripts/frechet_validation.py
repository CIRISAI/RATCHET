#!/usr/bin/env python3
"""
Fréchet Distance Validation for Trace Anomaly Detection.

Tests whether canonical routes emerge from the trace corpus by:
1. Computing discrete Fréchet distance between all trace paths
2. Clustering with HDBSCAN to find dense routes
3. Validating held-out traces against the baseline

The key question: do held-out traces look canonical, or does the detector overfit?
"""

import subprocess
import sys
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute discrete Fréchet distance between two paths.

    The Fréchet distance asks: if a dog walks path P and another walks path Q,
    what's the shortest leash that keeps them connected? This respects the
    temporal ordering of points, unlike Hausdorff.

    Args:
        P: Path 1 as (n, d) array of n points in d dimensions
        Q: Path 2 as (m, d) array of m points in d dimensions

    Returns:
        Discrete Fréchet distance (float)
    """
    n, m = len(P), len(Q)

    if n == 0 or m == 0:
        return float('inf')

    # Compute pairwise distances between all points
    dist = cdist(P, Q, metric='euclidean')

    # Dynamic programming table
    # ca[i,j] = Fréchet distance for P[0:i+1] and Q[0:j+1]
    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        """Recursive computation with memoization."""
        if ca[i, j] > -0.5:
            return ca[i, j]

        if i == 0 and j == 0:
            ca[i, j] = dist[0, 0]
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), dist[0, j])
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), dist[i, 0])
        else:
            ca[i, j] = max(
                min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)),
                dist[i, j]
            )
        return ca[i, j]

    # Use iterative version to avoid recursion limit
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                ca[i, j] = dist[0, 0]
            elif i == 0:
                ca[i, j] = max(ca[0, j - 1], dist[0, j])
            elif j == 0:
                ca[i, j] = max(ca[i - 1, 0], dist[i, 0])
            else:
                ca[i, j] = max(
                    min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                    dist[i, j]
                )

    return ca[n - 1, m - 1]


def load_traces():
    """Load trace data from production, grouped by task."""
    print("Loading trace data from production...")

    query = """COPY (
SELECT
    trace_id,
    task_id,
    agent_name,
    csdma_plausibility_score as p,
    dsdma_domain_alignment as a,
    coherence_level as c,
    selected_action as verb,
    thought_start_at
FROM cirislens.accord_traces
WHERE signature_verified = true
  AND agent_name IS NOT NULL
  AND task_id IS NOT NULL
  AND csdma_plausibility_score IS NOT NULL
  AND dsdma_domain_alignment IS NOT NULL
  AND coherence_level IS NOT NULL
ORDER BY agent_name, task_id, thought_start_at
) TO STDOUT WITH CSV HEADER"""

    # Build command with proper escaping
    ssh_key = "~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519"
    host = "root@108.61.242.236"
    docker_cmd = f'docker exec cirislens-db psql -U cirislens -d cirislens -c "{query}"'
    cmd = f'ssh -i {ssh_key} {host} \'{docker_cmd}\''

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    df = pd.read_csv(StringIO(result.stdout))
    df['thought_start_at'] = pd.to_datetime(df['thought_start_at'], errors='coerce')

    return df


def extract_task_paths(df: pd.DataFrame) -> dict:
    """
    Extract paths for each task.

    A path is a sequence of (P, A, C) points ordered by time.
    Returns dict mapping task_id -> (agent_name, path_array, verb_sequence)
    """
    paths = {}

    for (agent, task_id), group in df.groupby(['agent_name', 'task_id']):
        group = group.sort_values('thought_start_at')

        # Extract (P, A, C) coordinates
        coords = group[['p', 'a', 'c']].values
        verbs = group['verb'].tolist()

        if len(coords) >= 2:  # Need at least 2 points for a path
            paths[task_id] = {
                'agent': agent,
                'path': coords,
                'verbs': verbs,
                'n_steps': len(coords)
            }

    return paths


def compute_pairwise_frechet(paths: dict, task_ids: list) -> np.ndarray:
    """Compute pairwise Fréchet distance matrix."""
    n = len(task_ids)
    dist_matrix = np.zeros((n, n))

    total = n * (n - 1) // 2
    computed = 0

    print(f"Computing {total} pairwise Fréchet distances...")

    for i in range(n):
        for j in range(i + 1, n):
            path_i = paths[task_ids[i]]['path']
            path_j = paths[task_ids[j]]['path']

            d = discrete_frechet(path_i, path_j)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

            computed += 1
            if computed % 500 == 0:
                print(f"  {computed}/{total} ({100*computed/total:.1f}%)")

    return dist_matrix


def cluster_routes(dist_matrix: np.ndarray, min_cluster_size: int = 5):
    """Cluster paths using HDBSCAN on precomputed distances."""
    try:
        import hdbscan
    except ImportError:
        print("Installing hdbscan...")
        subprocess.run([sys.executable, "-m", "pip", "install", "hdbscan"],
                      capture_output=True)
        import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='precomputed',
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(dist_matrix)

    return labels, clusterer


def find_nearest_cluster_distance(test_path: np.ndarray,
                                   train_paths: dict,
                                   train_ids: list,
                                   labels: np.ndarray) -> tuple:
    """
    Find distance from test path to nearest canonical cluster.

    Returns (min_distance, nearest_cluster_id, nearest_task_id)
    """
    min_dist = float('inf')
    nearest_cluster = -1
    nearest_task = None

    for i, task_id in enumerate(train_ids):
        if labels[i] == -1:  # Skip noise points
            continue

        d = discrete_frechet(test_path, train_paths[task_id]['path'])
        if d < min_dist:
            min_dist = d
            nearest_cluster = labels[i]
            nearest_task = task_id

    return min_dist, nearest_cluster, nearest_task


def run_validation():
    """Run the full validation pipeline."""

    # Load data
    df = load_traces()
    print(f"Loaded {len(df)} trace records")

    # Extract task paths
    paths = extract_task_paths(df)
    print(f"Extracted {len(paths)} task paths (tasks with 2+ steps)")

    # Get task IDs and shuffle for random split
    task_ids = list(paths.keys())
    np.random.seed(42)  # Reproducible
    np.random.shuffle(task_ids)

    # Split: 80% train, 20% test
    n_train = int(len(task_ids) * 0.8)
    train_ids = task_ids[:n_train]
    test_ids = task_ids[n_train:]

    print(f"\nSplit: {len(train_ids)} train, {len(test_ids)} test")

    # Compute pairwise Fréchet for training set
    print("\n=== Computing Training Set Distances ===")
    train_dist_matrix = compute_pairwise_frechet(paths, train_ids)

    # Cluster to find canonical routes
    print("\n=== Clustering Canonical Routes ===")
    labels, clusterer = cluster_routes(train_dist_matrix, min_cluster_size=3)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"Found {n_clusters} canonical route clusters")
    print(f"Noise points (isolated traces): {n_noise} ({100*n_noise/len(train_ids):.1f}%)")

    # Analyze clusters
    print("\nCluster sizes:")
    cluster_counts = defaultdict(int)
    for l in labels:
        if l >= 0:
            cluster_counts[l] += 1
    for cluster_id, count in sorted(cluster_counts.items(), key=lambda x: -x[1]):
        print(f"  Cluster {cluster_id}: {count} traces")

    # Compute distance distribution within training set
    train_distances_to_cluster = []
    for i, task_id in enumerate(train_ids):
        if labels[i] >= 0:  # In a cluster
            # Distance to other members of same cluster
            same_cluster = [j for j, l in enumerate(labels) if l == labels[i] and j != i]
            if same_cluster:
                dists = [train_dist_matrix[i, j] for j in same_cluster]
                train_distances_to_cluster.append(min(dists))

    if train_distances_to_cluster:
        threshold_p95 = np.percentile(train_distances_to_cluster, 95)
        threshold_p99 = np.percentile(train_distances_to_cluster, 99)
        print(f"\nIntra-cluster distance distribution:")
        print(f"  Mean: {np.mean(train_distances_to_cluster):.4f}")
        print(f"  P95:  {threshold_p95:.4f}")
        print(f"  P99:  {threshold_p99:.4f}")
    else:
        threshold_p95 = 0.5  # Default
        threshold_p99 = 1.0

    # Validate held-out traces
    print("\n=== Validating Held-Out Traces ===")

    results = []
    for task_id in test_ids:
        test_path = paths[task_id]['path']
        agent = paths[task_id]['agent']
        n_steps = paths[task_id]['n_steps']

        min_dist, nearest_cluster, nearest_task = find_nearest_cluster_distance(
            test_path, paths, train_ids, labels
        )

        # Classify
        if nearest_cluster == -1:
            status = 'no_cluster'  # No canonical clusters to compare to
        elif min_dist <= threshold_p95:
            status = 'canonical'
        elif min_dist <= threshold_p99:
            status = 'variant'
        else:
            status = 'novel'

        results.append({
            'task_id': task_id,
            'agent': agent,
            'n_steps': n_steps,
            'min_distance': min_dist,
            'nearest_cluster': nearest_cluster,
            'status': status
        })

    results_df = pd.DataFrame(results)

    # Report
    print("\nValidation Results:")
    status_counts = results_df['status'].value_counts()
    for status, count in status_counts.items():
        pct = 100 * count / len(results_df)
        print(f"  {status}: {count} ({pct:.1f}%)")

    # False positive floor = traces flagged as novel that shouldn't be
    # (assuming held-out set is mostly normal)
    novel_rate = (results_df['status'] == 'novel').mean()
    print(f"\n=== Key Metrics ===")
    print(f"False-positive floor (novel rate on held-out): {100*novel_rate:.1f}%")
    print(f"Canonical detection rate: {100*(results_df['status'] == 'canonical').mean():.1f}%")

    # Show some examples
    print("\n=== Novel Traces (potential anomalies) ===")
    novel = results_df[results_df['status'] == 'novel'].sort_values('min_distance', ascending=False)
    for _, row in novel.head(5).iterrows():
        print(f"  {row['agent']}/{row['task_id'][:20]}... "
              f"dist={row['min_distance']:.3f} steps={row['n_steps']}")

    print("\n=== Canonical Traces (baseline matches) ===")
    canonical = results_df[results_df['status'] == 'canonical'].sort_values('min_distance')
    for _, row in canonical.head(5).iterrows():
        print(f"  {row['agent']}/{row['task_id'][:20]}... "
              f"dist={row['min_distance']:.3f} cluster={row['nearest_cluster']}")

    # By agent breakdown
    print("\n=== By Agent ===")
    for agent in results_df['agent'].unique():
        agent_df = results_df[results_df['agent'] == agent]
        canonical_pct = 100 * (agent_df['status'] == 'canonical').mean()
        novel_pct = 100 * (agent_df['status'] == 'novel').mean()
        print(f"  {agent}: {len(agent_df)} traces, "
              f"{canonical_pct:.0f}% canonical, {novel_pct:.0f}% novel")

    return results_df, labels, train_ids, paths


if __name__ == '__main__':
    results_df, labels, train_ids, paths = run_validation()
