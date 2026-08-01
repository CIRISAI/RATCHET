"""F-3 re-run under recovery-time tau. Prereg aab996847e46...
tau is temporal, so it does not degrade as the array loses spatial discrimination."""
import json, time, subprocess
import cupy as cp, numpy as np

N, W = 16, 256
BASE, PERT, REC = 40, 25, 70      # samples: baseline, perturbed, recovery
RNG = np.random.default_rng(89)

def gtemp():
    r = subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return float(r.stdout.strip().split('\n')[0])

class Arr:
    def __init__(self):
        self.st=[cp.cuda.Stream() for _ in range(N)]
        self.w=[cp.random.rand(W,W) for _ in range(N)]
        self.load=cp.random.rand(W*4,W*4)
    def sweep(self, mode, n, perturb):
        """one pass over all sensors; returns (N,) timings for this time index"""
        out=np.zeros(N)
        if mode=="barrier": cp.cuda.stream.get_current_stream().synchronize()
        if perturb: self.load=cp.sin(self.load)
        for i in range(N):
            with self.st[i]:
                t0=time.perf_counter_ns()
                self.w[i]=cp.sin(self.w[i]); self.st[i].synchronize()
                out[i]=time.perf_counter_ns()-t0
        return out

def fit_tau(series, base):
    """tau from log-linear fit of |x - base| decay; None if no usable decay."""
    d=np.abs(np.asarray(series,float)-base)
    d=d[:max(4,len(d))]
    good=d>1e-9
    if good.sum()<5: return None
    t=np.arange(len(d))[good]; y=np.log(d[good])
    sl,_=np.polyfit(t,y,1)
    if sl>=0: return None            # not decaying
    return float(-1.0/sl)

a=Arr(); [a.sweep("independent",0,False) for _ in range(8)]   # discard post-idle warmup
rows=[]
for trial in range(6):
    for mode in ("independent","barrier"):
        base=np.array([a.sweep(mode,0,False) for _ in range(BASE)])       # (BASE,N)
        C=np.corrcoef(base.T); C=np.nan_to_num(C)
        rho=float(np.mean(np.abs(C[np.triu_indices(N,1)])))
        mu_base=base.mean()
        one=RNG.integers(N)
        s1_base=base[:,one].mean()
        for _ in range(PERT): a.sweep(mode,0,True)                        # drive
        rec=np.array([a.sweep(mode,0,False) for _ in range(REC)])         # release
        tau_arr=fit_tau(rec.mean(axis=1), mu_base)
        tau_one=fit_tau(rec[:,one],      s1_base)
        rows.append({"trial":trial,"mode":mode,"rho":rho,
                     "tau_array":tau_arr,"tau_single":tau_one,"temp":gtemp()})
        ta="none" if tau_arr is None else f"{tau_arr:6.2f}"
        to="none" if tau_one is None else f"{tau_one:6.2f}"
        print(f"  t{trial} {mode:<12} rho={rho:.3f}  tau_array={ta}  tau_single={to}")

json.dump(rows,open("f3_recovery_results.json","w"),indent=2)
def agg(lo,hi,key):
    v=[r[key] for r in rows if r[key] is not None and lo<=r["rho"]<hi]
    return np.array(v)
print()
for lo,hi,lab in ((0,0.10,"chaos"),(0.10,0.43,"healthy"),(0.43,1.01,"rigidity")):
    A,S=agg(lo,hi,"tau_array"),agg(lo,hi,"tau_single")
    if len(A): print(f"  {lab:<9} n={len(A)}  tau_array {A.mean():6.2f} (sd {A.std():5.2f})   tau_single {S.mean() if len(S) else float('nan'):6.2f}")
H,R=agg(0.10,0.43,"tau_array"),agg(0.43,1.01,"tau_array")
HS,RS=agg(0.10,0.43,"tau_single"),agg(0.43,1.01,"tau_single")
if len(H) and len(R):
    pooled=np.sqrt((H.var()+R.var())/2)
    print(f"\nR-b  tau(rigidity)-tau(healthy) = {R.mean()-H.mean():+.2f}, pooled sd {pooled:.2f}"
          f"  -> {'SURVIVES' if R.mean()-H.mean()>=pooled else 'FIRES'}")
    ratios=[]
    for lab,A,S in (("healthy",H,HS),("rigidity",R,RS)):
        if len(A) and len(S): ratios.append(A.mean()/S.mean()); print(f"R-a  {lab}: tau_array/tau_single = {A.mean()/S.mean():.2f}")
    if ratios: print(f"R-a  {'CONFIRMED (all in [0.5,2.0])' if all(0.5<=x<=2.0 for x in ratios) else 'REFUTED -> F-3 UNRESOLVED'}")
