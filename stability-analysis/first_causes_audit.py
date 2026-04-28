import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh

def analyze_metrics():
    # Load the full corpus
    file_path = "release/data_scrubbed_v1/trace_context.jsonl"
    traces = []
    with open(file_path, "r") as f:
        for line in f:
            traces.append(json.loads(line))
    
    df = pd.DataFrame(traces)
    df = df.sort_values('timestamp')
    
    signals = [
        "csdma_plausibility_score", "dsdma_domain_alignment", "coherence_level",
        "entropy_level", "idma_k_eff", "idma_correlation_risk", "entropy_score",
        "coherence_score", "optimization_veto_entropy_ratio", "epistemic_humility_certainty",
        "conscience_passed", "entropy_passed", "coherence_passed", 
        "optimization_veto_passed", "epistemic_humility_passed", "action_was_overridden"
    ]
    
    available_signals = [s for s in signals if s in df.columns]
    
    def calculate_n_eff(subset_df):
        data = subset_df[available_signals].copy()
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        if data.empty or len(subset_df) < 10: return None, None
        data = data.fillna(data.mean())
        
        X = data.values
        stds = np.std(X, axis=0)
        keep = stds > 1e-9
        if not any(keep): return None, None
        X_std = (X[:, keep] - np.mean(X[:, keep], axis=0)) / stds[keep]
        
        C = np.corrcoef(X_std, rowvar=False)
        lambdas = eigh(C, eigvals_only=True)
        lambdas = np.maximum(lambdas[::-1], 0)
        
        # PR Metric
        n_eff_pr = (np.sum(lambdas)**2) / np.sum(lambdas**2)
        
        # Entropy Metric
        total = np.sum(lambdas)
        p = lambdas / total
        n_eff_h = np.exp(-np.sum(p * np.log(p + 1e-30)))
        
        return n_eff_pr, n_eff_h

    print("--- SLIDING WINDOW EVOLUTION (N_eff_PR vs N_eff_H) ---")
    print(f"{'Traces':<15} | {'PR':<8} | {'H (Entropy)':<12} | {'Versions'}")
    print("-" * 60)
    
    window_size = 500
    step = 100
    for i in range(0, len(df) - window_size, step):
        w_df = df.iloc[i : i + window_size]
        pr, h = calculate_n_eff(w_df)
        if pr:
            v_start = w_df['agent_version'].iloc[0]
            v_end = w_df['agent_version'].iloc[-1]
            print(f"{i:4d}-{i+window_size:4d} | {pr:8.4f} | {h:12.4f} | {v_start} -> {v_end}")

if __name__ == "__main__":
    analyze_metrics()
