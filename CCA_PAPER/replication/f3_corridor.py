"""F-3: is fragility U-shaped in rho, or monotone? Tests the rigidity arm,
never measured by the F-series (max rho 0.171 vs a 0.43 edge).
Prereg f6aea87d9e6e65edef4812df8792c8f12660f77288b044a5f19657fee1171a0a"""
import json, time, subprocess
import cupy as cp, numpy as np

N_SENS, W, M = 16, 256, 60
RNG = np.random.default_rng(43)

def temp():
    r = subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return float(r.stdout.strip().split('\n')[0])

class Arr:
    def __init__(self):
        self.st = [cp.cuda.Stream() for _ in range(N_SENS)]
        self.w  = [cp.random.rand(W, W) for _ in range(N_SENS)]
        self.pert = cp.random.rand(W*3, W*3)
    def measure(self, mode, perturb):
        out = np.zeros((N_SENS, M))
        if mode == "barrier":
            for s in range(M):
                cp.cuda.stream.get_current_stream().synchronize()
                if perturb: self.pert = cp.sin(self.pert)
                for i in range(N_SENS):
                    with self.st[i]:
                        t0 = time.perf_counter_ns()
                        self.w[i] = cp.sin(self.w[i]); self.st[i].synchronize()
                        out[i, s] = time.perf_counter_ns() - t0
        else:
            for i in range(N_SENS):
                with self.st[i]:
                    for s in range(M):
                        if perturb and s % 5 == 0: self.pert = cp.sin(self.pert)
                        t0 = time.perf_counter_ns()
                        self.w[i] = cp.sin(self.w[i]); self.st[i].synchronize()
                        out[i, s] = time.perf_counter_ns() - t0
        return out

def rho_of(t):
    C = np.corrcoef(t); C = np.nan_to_num(C)
    return float(np.mean(np.abs(C[np.triu_indices(N_SENS,1)])))

a = Arr()
a.measure("independent", False)          # discard first post-idle run (protocol)
pts = []
for trial in range(6):
    for mode in ("independent", "barrier"):
        base = a.measure(mode, False)
        pert = a.measure(mode, True)
        r = rho_of(base)
        resp = abs(pert.mean() - base.mean()) / base.mean()
        pts.append({"trial":trial,"mode":mode,"rho":r,"response":resp,"temp":temp()})
        print(f"  t{trial} {mode:<12} rho={r:.3f}  response={resp:>7.1%}")

json.dump(pts, open("f3_results.json","w"), indent=2)
rhos = np.array([p["rho"] for p in pts]); resp = np.array([p["response"] for p in pts])
print(f"\nachieved rho range: {rhos.min():.3f} - {rhos.max():.3f}")
print(f"F3-a  rigidity arm reached (rho > 0.43)? {'YES' if rhos.max() > 0.43 else 'NO -> UNTESTED'}")
for lo, hi, lab in ((0,0.10,"chaos"),(0.10,0.43,"healthy"),(0.43,1.01,"rigidity")):
    m = (rhos>=lo)&(rhos<hi)
    if m.sum(): print(f"  {lab:<9} n={m.sum():2d}  mean response {resp[m].mean():>7.1%}  sd {resp[m].std():.1%}")
