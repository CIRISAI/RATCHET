import numpy as np
def measure_hitting_prob(D, r, n_samples=100000):
    center = np.ones(D) * 0.5
    hits = 0
    for _ in range(n_samples):
        normal = np.random.randn(D)
        normal /= np.linalg.norm(normal)
        min_proj = np.sum(np.minimum(0, normal))
        max_proj = np.sum(np.maximum(0, normal))
        offset = np.random.uniform(min_proj, max_proj)
        dist = np.abs(np.dot(normal, center) - offset)
        if dist <= r:
            hits += 1
    return hits / n_samples
p = measure_hitting_prob(11, 0.325)
print(f"p = {p}")
print(f"k_req = {-np.log(0.01) / p}")
