"""F-3 under recovery-time fragility.

tau from AR(1) lag-1 autocorrelation of the recovery segment -- the standard
critical-slowing-down statistic. Temporal, so it does not degrade as the array
loses spatial discrimination the way response amplitude does.

Sampling is synchronous in EVERY condition (required for time-resolved array
state). rho is varied by memory sharing, which is independent of sampling.
"""
import json, time, subprocess
import cupy as cp, numpy as np

N, W = 16, 256
BASE, DRIVE, REC, TRIALS = 30, 40, 120, 8
RNG = np.random.default_rng(891)

def gtemp():
    r = subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return float(r.stdout.strip().split('\n')[0])

class Arr:
    def __init__(self):
        self.st  = [cp.cuda.Stream() for _ in range(N)]
        self.iso = [cp.random.rand(W, W) for _ in range(N)]   # isolated buffers
        self.shr = cp.random.rand(W, W)                       # one shared buffer
        self.load = cp.random.rand(W*6, W*6)                  # perturbation workload

    def step(self, shared, drive):
        """One synchronous sweep across all sensors -> (N,) timings."""
        out = np.empty(N)
        if drive:
            self.load = cp.sin(self.load); self.load = cp.sin(self.load)
        for i in range(N):
            with self.st[i]:
                t0 = time.perf_counter_ns()
                if shared: self.shr = cp.sin(self.shr)
                else:      self.iso[i] = cp.sin(self.iso[i])
                self.st[i].synchronize()
                out[i] = time.perf_counter_ns() - t0
        return out

def ar1_tau(x):
    """AR(1) lag-1 autocorrelation -> tau = -1/ln(r1). Returns (tau, r1)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    if x.std() < 1e-12: return None, None
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not np.isfinite(r1) or r1 <= 0: return 0.0, r1
    r1 = min(r1, 0.999)
    return float(-1.0 / np.log(r1)), r1

a = Arr()
for _ in range(10): a.step(False, False)          # discard post-idle warmup

rows = []
for trial in range(TRIALS):
    for shared in (False, True):
        base = np.array([a.step(shared, False) for _ in range(BASE)])
        C = np.nan_to_num(np.corrcoef(base.T))
        rho = float(np.mean(np.abs(C[np.triu_indices(N, 1)])))
        for _ in range(DRIVE): a.step(shared, True)             # perturb
        rec = np.array([a.step(shared, False) for _ in range(REC)])   # release + observe
        one = int(RNG.integers(N))
        tau_a, r1a = ar1_tau(rec.mean(axis=1))
        tau_s, r1s = ar1_tau(rec[:, one])
        rows.append({"trial":trial, "shared":shared, "rho":rho,
                     "tau_array":tau_a, "ar1_array":r1a,
                     "tau_single":tau_s, "ar1_single":r1s, "temp":gtemp()})
        print(f"  t{trial} {'shared ' if shared else 'isolated'}  rho={rho:.3f}  "
              f"AR1={r1a:+.3f}  tau_array={tau_a:6.2f}  tau_single={tau_s:6.2f}")

json.dump(rows, open("f3_recovery2_results.json","w"), indent=2)

def bin_(lo, hi, key):
    return np.array([r[key] for r in rows if r[key] is not None and lo <= r["rho"] < hi])

print()
for lo, hi, lab in ((0,0.10,"chaos"),(0.10,0.43,"healthy"),(0.43,1.01,"rigidity")):
    A, S = bin_(lo,hi,"tau_array"), bin_(lo,hi,"tau_single")
    if len(A):
        print(f"  {lab:<9} n={len(A):2d}  tau_array {A.mean():6.2f} (sd {A.std():5.2f})  "
              f"tau_single {S.mean():6.2f}  ratio {A.mean()/S.mean() if S.mean() else float('nan'):.2f}")

H, R = bin_(0.10,0.43,"tau_array"), bin_(0.43,1.01,"tau_array")
HS, RS = bin_(0.10,0.43,"tau_single"), bin_(0.43,1.01,"tau_single")
if len(H) >= 2 and len(R) >= 2:
    pooled = np.sqrt((H.var(ddof=1) + R.var(ddof=1)) / 2)
    diff = R.mean() - H.mean()
    ratios = [x for x in (H.mean()/HS.mean() if HS.mean() else np.nan,
                          R.mean()/RS.mean() if RS.mean() else np.nan) if np.isfinite(x)]
    ok = all(0.5 <= x <= 2.0 for x in ratios)
    print(f"\nR-a  array/single tau ratios {['%.2f'%x for x in ratios]} -> "
          f"{'CONFIRMED, measure admissible' if ok else 'REFUTED -> UNRESOLVED'}")
    print(f"R-b  tau(rigidity) - tau(healthy) = {diff:+.2f}, pooled sd {pooled:.2f} -> "
          f"{'corridor SURVIVES' if diff >= pooled else 'F-3 FIRES'}")
else:
    print(f"\nInsufficient coverage: healthy n={len(H)}, rigidity n={len(R)} -> UNRESOLVED")
