# Exploratory (CAL calibration stories only — training side, never eval): parallax between the READING direction
# (probe for "next word is ADJ" at the post-mlp_L state) and the WRITING direction (gradient of the ADJ-word target
# through the full network w.r.t. mlp_L's write at the same positions), and the fine-tune's realised write change.
import os, sys, json, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/emoore/RATCHET/experiments/kindscope")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import edit1 as X
from ks import engine as E
cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); sd0, _ = X.load_params(cache.info["model_path"])
stories, windows, toks = X.tag_train_core(os.path.join(X.SCRATCH, "gw_data/TinyStories-valid.txt"))
rng = np.random.default_rng(20261115); order = rng.permutation(len(stories))
cal = sorted(order[: len(stories)//10].tolist()); pool = order[len(stories)//10:]
prs = X.pairs(windows, toks)
dn, da = np.zeros(len(stories)), np.zeros(len(stories))
for s, a, b, y0, y1, f in prs:
    if y0 == X.DET: dn[s] += 1; da[s] += y1 == X.ADJ
elig = [s for s in pool if dn[s] >= 5]; rate = {s: da[s]/dn[s] for s in elig}
adj_rich = sorted(sorted(elig, key=lambda s: -rate[s])[:len(pool)//5])
sub = cal[:600]; sub_set = set(sub)
det_next = [p for p in prs if p[0] in sub_set and p[3] == X.DET]
adj_sub = [p for p in det_next if p[4] == X.ADJ]
def T(sd): return float(np.nanmean(X.evaluate(sd, windows, sub, adj_sub)[0]))
base = T(sd0)
sdA2, _, _ = X.finetune(sd0, X.stream(windows, adj_rich), 1)
gainA2 = T(sdA2) - base
print(f"base {base:.3f}  A2 seed1 gain on CAL subset {gainA2:+.3f}  (n adj words {len(adj_sub)})", flush=True)
D = 192
def fwd_add(sd, idx, L, add):
    B, Tn = idx.shape
    x = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(Tn, device=E.DEV)][None]
    mask = torch.ones(Tn, Tn, dtype=torch.bool, device=E.DEV).tril()
    ln = lambda x, n: F.layer_norm(x, (D,), sd[f"{n}.weight"], sd[f"{n}.bias"], 1e-5)
    for i in range(6):
        h = ln(x, f"blocks.{i}.ln1")
        q, k, v = (h @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
        q, k, v = (t.view(B, Tn, 4, D//4).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D//4)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + (att @ v).transpose(1, 2).reshape(B, Tn, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
        x = x + F.gelu(ln(x, f"blocks.{i}.ln2") @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate="tanh") @ sd[f"blocks.{i}.mlp.proj.weight"].T
        if i == L: x = x + add
    return ln(x, "ln_f") @ sd["wte.weight"].T
by = {}
for s, a, b, y0, y1, f in det_next: by.setdefault(s, []).append((a, b, y1))
out = {"gainA2_cal_subset": gainA2, "layers": {}}
for L in range(6):
    g_sum = torch.zeros(D, device=E.DEV, dtype=torch.float64); S_list, Y_list, W0, W2 = [], [], [], []
    for ss, idx in X.batches(windows, sub, bs=32):
        add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
        lp = fwd_add(sd0, idx, L, add).log_softmax(-1)
        J = 0
        rows, cols = [], []
        for r, s in enumerate(ss):
            for a, b, y1 in by.get(s, []):
                rows.append(r); cols.append(a - 1)
                if y1 == X.ADJ:
                    p = torch.arange(a - 1, b, device=E.DEV); J = J + lp[r, p, idx[r, p + 1]].sum()
        J.backward()
        rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
        g_sum += add.grad[rt, ct].double().sum(0)
        with torch.no_grad():
            _, h0, st0 = X.fwd(sd0, idx, want_hidden=L); _, h2, _ = X.fwd(sdA2, idx, want_hidden=L)
            S_list.append(st0[L][rt, ct].double().cpu().numpy())
            Y_list += [int(y1 == X.ADJ) for r, s in enumerate(ss) for a, b, y1 in by.get(s, [])]
            W0.append((h0[rt, ct] @ sd0[f"blocks.{L}.mlp.proj.weight"].T).double()); W2.append((h2[rt, ct] @ sdA2[f"blocks.{L}.mlp.proj.weight"].T).double())
    Xs = np.concatenate(S_list); yv = np.array(Y_list)
    mu, sdv = Xs.mean(0), np.where(Xs.std(0) > 0, Xs.std(0), 1)
    th, _, _, why = E.newton(torch.from_numpy((Xs - mu)/sdv).to(E.DEV, E.DT), torch.from_numpy(yv)[None])
    d_read = torch.from_numpy(th[0, :-1].cpu().numpy() / sdv).to(E.DEV); d_read /= d_read.norm()
    d_write = g_sum / g_sum.norm()
    W0, W2 = torch.cat(W0), torch.cat(W2)
    d_ft = (W2 - W0).mean(0); d_ft_n = d_ft / d_ft.norm()
    cos = lambda a, b: float(a @ b)
    wn = float(W0.norm(dim=1).mean())
    rec = {"cos_read_write": cos(d_read, d_write), "cos_ft_write": cos(d_ft_n, d_write), "cos_ft_read": cos(d_ft_n, d_read),
           "ft_change_over_write": float(d_ft.norm()) / wn, "probe_exit": why[0]}
    # a single-direction ADDITIVE edit at those positions only (no key selectivity): how much does each direction buy?
    for name, d in (("read", d_read), ("write", d_write), ("ft", d_ft_n)):
        cur = []
        for a in (0.25, 0.5, 1.0, 2.0):
            vals = []
            with torch.no_grad():
                tot, n = 0.0, 0
                for ss, idx in X.batches(windows, sub, bs=64):
                    add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV)
                    for r, s in enumerate(ss):
                        for aa, bb, y1 in by.get(s, []):
                            add[r, aa - 1] += (a * wn) * d.float()
                    lp = fwd_add(sd0, idx, L, add).log_softmax(-1)
                    for r, s in enumerate(ss):
                        for aa, bb, y1 in by.get(s, []):
                            if y1 == X.ADJ:
                                p = torch.arange(aa - 1, bb, device=E.DEV); tot += float(lp[r, p, idx[r, p + 1]].sum()); n += 1
            cur.append((a, round(tot / n - base, 4)))
        rec[f"curve_{name}"] = cur
    out["layers"][L] = rec
    print(L, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(X.SCRATCH, "diag_parallax.json"), "w"), indent=1)
