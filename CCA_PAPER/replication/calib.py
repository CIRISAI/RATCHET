"""Quick calibration: find (a) a perturbation strength giving latency dynamic range,
(b) whether heterogeneous workloads lower rho under synchronous sampling."""
import time, numpy as np, cupy as cp
N,W=16,256
st=[cp.cuda.Stream() for _ in range(N)]
def mk(het):
    return [cp.random.rand(W+ (i*40 if het else 0), W) for i in range(N)]
def step(buf, load, drive, reps):
    out=np.empty(N)
    if drive:
        for _ in range(reps): load[0]=cp.sin(load[0])
    for i in range(N):
        with st[i]:
            t0=time.perf_counter_ns(); buf[i]=cp.sin(buf[i]); st[i].synchronize()
            out[i]=time.perf_counter_ns()-t0
    return out
for het in (False,True):
    buf=mk(het); load=[cp.random.rand(W*2,W*2)]
    for _ in range(8): step(buf,load,False,0)
    base=np.array([step(buf,load,False,0) for _ in range(200)])
    C=np.nan_to_num(np.corrcoef(base.T)); rho=np.mean(np.abs(C[np.triu_indices(N,1)]))
    stat=base.mean(axis=1); thr=np.quantile(stat,0.95)
    print(f"het={het}  rho={rho:.3f}")
    for reps in (0,1,2,4):
        hits=[]
        for _ in range(12):
            for s in range(40):
                if step(buf,load,True,reps).mean()>thr: hits.append(s+1); break
            else: hits.append(40)
        print(f"    reps={reps}: mean latency {np.mean(hits):5.2f}  (want 3-20)")
