import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

def measure_hitting_prob(D, r, n_samples=10000):
    """
    Measure probability that a random hyperplane intersects a ball of radius r
    centered in a unit hypercube [0,1]^D.
    """
    # Center of the hypercube
    center = np.ones(D) * 0.5
    
    hits = 0
    for _ in range(n_samples):
        # Generate random normal vector (unit sphere)
        normal = np.random.randn(D)
        normal /= np.linalg.norm(normal)
        
        # Random offset d. 
        # For the plane to intersect the unit cube, the offset d must be in range:
        # [min(<n, x>), max(<n, x>)] where x in [0,1]^D
        # min(<n,x>) = sum(min(0, n_i))
        # max(<n,x>) = sum(max(0, n_i))
        
        min_proj = np.sum(np.minimum(0, normal))
        max_proj = np.sum(np.maximum(0, normal))
        
        # Sample offset uniformly from the range that hits the cube
        offset = np.random.uniform(min_proj, max_proj)
        
        # Distance from cube center to plane: |<n, center> - offset|
        dist = np.abs(np.dot(normal, center) - offset)
        
        if dist <= r:
            hits += 1
            
    return hits / n_samples

def run_audit():
    r = 0.2
    dims = [2, 3, 5, 10, 20, 50, 100, 256, 512, 1024]
    results = []
    
    print(f"Adversarial Audit: Hitting Probability p(r={r}) vs Dimension D")
    print(f"{'D':>5} | {'p_obs':>8} | {'p_theory (2r)':>12} | {'ratio':>10}")
    print("-" * 45)
    
    for D in dims:
        p_obs = measure_hitting_prob(D, r)
        p_theory = 2 * r
        ratio = p_obs / p_theory
        results.append({'D': D, 'p_obs': p_obs, 'ratio': ratio})
        print(f"{D:5d} | {p_obs:8.4f} | {p_theory:12.4f} | {ratio:10.4f}")
    
    df = pd.DataFrame(results)
    
    # Check scaling hypothesis: p ~ 2r / sqrt(D)
    # Actually, in this sampling, the denominator is the width of the cube projection
    # Width = max_proj - min_proj = sum(|n_i|)
    # For a random normal n_i, E[|n_i|] = sqrt(2/pi)
    # Expected Width = D * sqrt(2/pi)
    
    print("\nScaling Analysis:")
    df['expected_width'] = df['D'] * np.sqrt(2 / np.pi)
    df['predicted_p'] = (2 * r) / (df['D'] / np.sqrt(df['D'])) # Heuristic: diag is sqrt(D)
    
    # Save results
    df.to_csv("stability-analysis/geometric_audit_results.csv", index=False)
    print("\nResults saved to stability-analysis/geometric_audit_results.csv")

if __name__ == "__main__":
    run_audit()
