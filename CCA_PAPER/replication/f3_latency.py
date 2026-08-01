"""F-3 under detection latency at fixed false-positive rate.

The array IS a detector. Fragility = degraded function. Measure how many samples
it needs to detect a perturbation, with the decision threshold calibrated per
condition to a fixed FP rate on that condition's OWN baseline.

That calibration is what makes this k_eff-independent: a collapsed array has a
different variance scale, but its threshold adapts to it, so what remains is
detection capability rather than variance magnitude. Both prior measures failed
exactly here -- response amplitude and AR(1) tau both inherit the variance scale.
"""
import json, time, subprocess
import cupy as cp, numpy as np

N, W = 16, 256
CAL, WINDOW, EPISODES, TRIALS = 400, 60, 40, 6
FP_TARGET = 0.05
RNG = np.random.default_rng(1729)

def gtemp():
    r = subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return float(r.stdout.strip().split('\n')[0])

class Arr:
    def __init__(self):
        self.st=[cp.cuda.Stream() for _ in range(N)]
        self.iso=[cp.random.rand(W,W) for _ in range(N)]
        self.shr=cp.random.rand(W,W)
        self.load=cp.random.rand(W*6,W*6)
    def step(self, shared, drive):
        out=np.empty(N)
        if drive: self.load=cp.sin(self.load); self.load=cp.sin(self.load)
        for i in range(N):
            with self.st[i]:
                t0=time.perf_counter_ns()
                if shared: self.shr=cp.sin(self.shr)
                else:      self.iso[i]=cp.sin(self.iso[i])
                self.st[i].synchronize()
                out[i]=time.perf_counter_ns()-t0
        return out

a=Arr()
for _ in range(10): a.step(False,False)

rows=[]
for trial in range(TRIALS):
    for shared in (False,True):
        cal=np.array([a.step(shared,False) for _ in range(CAL)])
        C=np.nan_to_num(np.corrcoef(cal.T))
        rho=float(np.mean(np.abs(C[np.triu_indices(N,1)])))
        stat=cal.mean(axis=1)                                  # detector statistic
        thr=float(np.quantile(stat,1-FP_TARGET))               # per-condition threshold
        fp=float((stat>thr).mean())                            # realized FP, sanity
        lat=[]; censored=0
        for _ in range(EPISODES):
            hit=None
            for s in range(WINDOW):
                if a.step(shared,True).mean()>thr: hit=s+1; break
            if hit is None: censored+=1; lat.append(WINDOW)    # censored at window
            else: lat.append(hit)
        rows.append({"trial":trial,"shared":shared,"rho":rho,"thr":thr,"fp":fp,
                     "latency_mean":float(np.mean(lat)),"latency_median":float(np.median(lat)),
                     "censored":censored,"temp":gtemp()})
        print(f"  t{trial} {'shared ' if shared else 'isolated'} rho={rho:.3f} "
              f"FP={fp:.3f} latency={np.mean(lat):5.2f} censored={censored}/{EPISODES}")

json.dump(rows,open("f3_latency_results.json","w"),indent=2)
def b(lo,hi,k): return np.array([r[k] for r in rows if lo<=r["rho"]<hi])
print()
for lo,hi,lab in ((0,0.10,"chaos"),(0.10,0.43,"healthy"),(0.43,1.01,"rigidity")):
    L=b(lo,hi,"latency_mean")
    if len(L): print(f"  {lab:<9} n={len(L):2d}  latency {L.mean():5.2f} (sd {L.std():4.2f})  "
                     f"censored {b(lo,hi,'censored').sum():.0f}  FP {b(lo,hi,'fp').mean():.3f}")
H,R=b(0.10,0.43,"latency_mean"),b(0.43,1.01,"latency_mean")
if len(H)>=2 and len(R)>=2:
    pooled=np.sqrt((H.var(ddof=1)+R.var(ddof=1))/2)
    diff=R.mean()-H.mean()
    cens=b(0,1.01,"censored").sum()
    print(f"\ncensoring total {cens:.0f}/{EPISODES*len(rows)}  "
          f"({'OK, measure not window-limited' if cens<0.1*EPISODES*len(rows) else 'HIGH -- window-limited, inadmissible'})")
    print(f"FP realized {b(0,1.01,'fp').mean():.3f} vs target {FP_TARGET} "
          f"({'calibrated' if abs(b(0,1.01,'fp').mean()-FP_TARGET)<0.02 else 'MISCALIBRATED'})")
    print(f"latency(rigidity)-latency(healthy) = {diff:+.2f}, pooled sd {pooled:.2f} -> "
          f"{'corridor SURVIVES (rigidity degrades detection)' if diff>=pooled else 'F-3 FIRES (no rigidity-side degradation)'}")
else: print(f"\ncoverage healthy n={len(H)} rigidity n={len(R)} -> UNRESOLVED")
