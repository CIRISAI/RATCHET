"""Per-item traces for the Holon page (format `holon-trace 1`, CIRISHolon trace_check) plus a readings sidecar.

Usage: python make_traces.py <TinyStories-valid.txt> <fresh slice> <out dir>
Items: FRESH-1 kept stories 20,000-20,007 (unread by any analysis), first 256 bytes each (the GoldWorm window).
Writes, in the order the model adds them (float32): embed (wte + wpe), attn0, mlp0, ..., attn5, mlp5; then the
pre-norm state the model held. Readings: projections of the state after each mlp_L onto unit directions fitted on CAL
0-799 exactly as PARALLAX-H (probe, write, resid, bytes16), and the C4 residual of the logits rebuilt from the writes
with the final norm frozen.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, hashlib, json, struct, sys  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import edit2 as Y  # noqa: E402
import mlp0_1 as M  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402

FRESH_SHA = "3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a"
NAMES = ["embed"] + [f"{p}{i}" for i in range(6) for p in ("attn", "mlp")]
D = 192


def directions(sd0, windows, toks, cal):
    """Probe, write, resid and bytes16 per layer, fitted on CAL 0-799 as in parallaxh.py."""
    dir_set = set(cal[:800])
    dkey = collections.defaultdict(list)
    for s, a, b, y0, y1, f in X.pairs(windows, toks):
        if y0 == X.DET and s in dir_set:
            dkey[s].append((a, b, y1))
    first = collections.Counter(int(windows[s][a]) for s in dkey for a, b, y1 in dkey[s])
    TOPB = [c for c, _ in first.most_common(16)]
    clouds = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_clouds.json")))
    out = {}
    for L in range(6):
        gA = torch.zeros(D, device=E.DEV, dtype=torch.float64)
        gB = {c: torch.zeros(D, device=E.DEV, dtype=torch.float64) for c in TOPB}
        Xs, ys = [], []
        for ss, idx in X.batches(windows, cal[:800], bs=32):
            add = torch.zeros(idx.shape[0], idx.shape[1], D, device=E.DEV, requires_grad=True)
            logits, _, st, _ = Y.fwd_full(sd0, idx, {("mlp", L): add}); lp = logits.log_softmax(-1)
            rows, cols, Ja, Jb = [], [], 0, {c: 0 for c in TOPB}
            for r, s in enumerate(ss):
                for a, b, y1 in dkey.get(s, []):
                    rows.append(r); cols.append(a - 1); ys.append(int(y1 == X.ADJ))
                    if y1 == X.ADJ:
                        p = torch.arange(a - 1, b, device=E.DEV); Ja = Ja + lp[r, p, idx[r, p + 1]].sum()
                    for c in TOPB:
                        Jb[c] = Jb[c] + lp[r, a - 1, c]
            if not rows:
                continue
            rt, ct = torch.tensor(rows, device=E.DEV), torch.tensor(cols, device=E.DEV)
            gA += torch.autograd.grad(Ja, add, retain_graph=True)[0][rt, ct].double().sum(0)
            for c in TOPB:
                gB[c] += torch.autograd.grad(Jb[c], add, retain_graph=True)[0][rt, ct].double().sum(0)
            Xs.append(st[L][rt, ct].detach().double().cpu().numpy())
        Xa = np.concatenate(Xs); yv = np.array(ys)
        mu, sd = Xa.mean(0), np.where(Xa.std(0) > 0, Xa.std(0), 1)
        th, _, _, _ = E.newton(torch.from_numpy((Xa - mu) / sd).to(E.DEV, E.DT), torch.from_numpy(yv)[None])
        u = lambda v: (v / np.linalg.norm(v)).tolist()  # noqa: E731
        out[L] = {"probe": u(th[0, :-1].cpu().numpy() / sd), "write": u(gA.cpu().numpy()),
                  "resid": u(np.array(clouds["layers"][str(L)]["resid_dir"])),
                  "bytes16": {chr(c): u(gB[c].cpu().numpy()) for c in TOPB}}
    return out


def main(valid, fresh, odir):
    torch.use_deterministic_algorithms(True)
    assert V.sha256(fresh) == FRESH_SHA
    os.makedirs(odir, exist_ok=True)
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    sd0, _ = X.load_params(cache.info["model_path"])
    stories, windows, toks = X.tag_train_core(valid)
    rng = np.random.default_rng(20261115); order = rng.permutation(len(stories))
    cal = sorted(order[: len(stories) // 10].tolist())
    dirs = directions(sd0, windows, toks, cal)
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    sd = port.sd
    manifest = []
    for k, story in enumerate(kept[20000:20008]):
        w = (b"<|endoftext|>\n\n" + story)[:256]
        idx = torch.tensor(list(w), dtype=torch.long, device=E.DEV)[None]
        with torch.no_grad():
            logits, writes = M.forward(port, idx)
            embed = writes["wte"] + sd["wpe.weight"][torch.arange(idx.shape[1], device=E.DEV)][None]
            order_w = [embed] + [writes[n] for n in NAMES[1:]]
            x = torch.zeros_like(embed)
            for t in order_w:          # the model's own order, float32
                x = x + t
            # the pre-norm state the model itself held: recompute it through the model's forward path
            st = Y.fwd_full(sd0, idx)[2][5]
            assert torch.equal(x, st), "cumulative sum of writes differs from the model's own state"
            # C4 residual per position: logits rebuilt from the writes with the final norm frozen
            mu = st.mean(-1, keepdim=True); r = (st.var(-1, unbiased=False, keepdim=True) + 1e-5).rsqrt()
            g, b, H = sd["ln_f.weight"], sd["ln_f.bias"], sd["head.weight"]
            rebuilt = sum(((t - t.mean(-1, keepdim=True)) * r * g) @ H.T for t in order_w) + b @ H.T
            c4 = (rebuilt - logits).abs().amax(-1)[0].cpu().numpy()
        P = idx.shape[1]
        path = os.path.join(odir, f"gw_item{k}.trace")
        with open(path, "wb") as f:
            f.write(f"holon-trace 1 positions {P} width {D} writes {len(NAMES)} {' '.join(NAMES)}\n".encode())
            W = torch.stack(order_w, 2)[0].cpu().numpy().astype("<f4")   # [P, K, D]
            S = st[0].cpu().numpy().astype("<f4")                         # [P, D]
            for p in range(P):
                f.write(W[p].tobytes()); f.write(S[p].tobytes())
        states = Y.fwd_full(sd0, idx)[2]
        readings = {}
        for L in range(6):
            s_ = states[L][0].cpu().double().numpy()
            readings[f"mlp{L}"] = {n: (s_ @ np.array(v)).round(5).tolist() for n, v in dirs[L].items() if n != "bytes16"}
            readings[f"mlp{L}"]["bytes16"] = {c: (s_ @ np.array(v)).round(5).tolist() for c, v in dirs[L]["bytes16"].items()}
        side = {"item": k, "source": "FRESH-1 kept story index %d" % (20000 + k), "bytes": list(w),
                "text": w.decode("ascii"), "readings_after_mlp": readings, "c4_max_abs_logit_residual": c4.round(7).tolist(),
                "next_byte_logprob_true": logits[0, :-1].log_softmax(-1).gather(-1, idx[0, 1:, None])[:, 0].cpu().numpy()
                .round(5).tolist()}
        json.dump(side, open(os.path.join(odir, f"gw_item{k}.readings.json"), "w"))
        manifest.append({"item": k, "trace": os.path.basename(path), "trace_sha256": V.sha256(path),
                         "positions": P, "c4_max": float(c4.max())})
        print(manifest[-1], flush=True)
    json.dump({"model": "GoldWorm v6 base", "model_sha256": cache.info["model_sha256"], "fresh_sha256": FRESH_SHA,
               "writes": NAMES, "directions_from": "CAL stories 0-799, as parallaxh.py", "items": manifest},
              open(os.path.join(odir, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
