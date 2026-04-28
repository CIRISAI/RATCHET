import json
import numpy as np
import pandas as pd
from scipy.linalg import svdvals

def calculate_metrics(corr_matrix):
    """Calculate various effective dimensionality metrics."""
    # Kish k_eff
    K = corr_matrix.shape[0]
    mask = ~np.eye(K, dtype=bool)
    avg_rho = corr_matrix[mask].mean()
    k_kish = K / (1 + avg_rho * (K - 1))
    
    # Eigenvalues
    evals = np.linalg.eigvalsh(corr_matrix)
    evals = np.maximum(evals, 1e-10) # Stability
    
    # Participation Ratio
    pr = (np.sum(evals)**2) / np.sum(evals**2)
    
    # Stable Rank
    s_vals = np.sqrt(np.abs(evals))
    stable_rank = np.sum(s_vals**2) / np.max(s_vals**2)
    
    return {
        'avg_rho': avg_rho,
        'k_kish': k_kish,
        'participation_ratio': pr,
        'stable_rank': stable_rank
    }

def run_trace_audit():
    # Load traces
    traces = []
    with open("corpus/trace_context.jsonl", "r") as f:
        for line in f:
            traces.append(json.loads(line))
    
    df = pd.DataFrame(traces)
    
    # Focus on detailed/full traces with numeric reasoning signals
    # We'll use the signals available in trace_context
    signals = [
        'csdma_plausibility_score', 
        'dsdma_domain_alignment', 
        'entropy_score', 
        'coherence_score',
        'optimization_veto_entropy_ratio',
        'tokens_input',
        'tokens_output',
        'processing_ms'
    ]
    
    # Filter for qa_eval and drop NaNs
    qa_df = df[(df['task_class'] == 'qa_eval')].dropna(subset=signals)
    
    if len(qa_df) < 10:
        print("Insufficient qa_eval traces for PCA.")
        return

    print(f"Audit: Analyzing {len(qa_df)} qa_eval traces across {len(signals)} signals.")
    
    # Standardize
    data = qa_df[signals].values
    data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
    
    # Correlation Matrix
    corr = np.corrcoef(data_norm, rowvar=False)
    
    metrics = calculate_metrics(corr)
    
    print("\nDimensionality Audit Results:")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}")
    
    # Analyze the Restricted Query (Q04) Failures
    q04_df = qa_df[qa_df['qa_question_num'] == 4]
    if len(q04_df) > 0:
        print(f"\nQ04 (Restricted Query) Stress Test:")
        print(f"  Count: {len(q04_df)}")
        print(f"  Mean CSDMA: {q04_df['csdma_plausibility_score'].mean():.4f}")
        # System failures: where CSDMA is high despite being Q04
        failures = q04_df[q04_df['csdma_plausibility_score'] > 0.5]
        print(f"  System Bypass (CSDMA > 0.5): {len(failures)} ({len(failures)/len(q04_df)*100:.1f}%)")

if __name__ == "__main__":
    run_trace_audit()
