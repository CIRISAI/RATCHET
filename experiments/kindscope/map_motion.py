"""Map and motion, one table (GoldWorm v6 base). EXPLORATORY.

MAP: for each of the 13 states and each of 10 part-of-speech categories, at a word's LAST byte: two unit directions — the
one-vs-rest probe's weight direction (a READING direction; not where the model writes) and the class mean-difference
direction — with held-out accuracy and base rate on the explore split.
MOTION: for each real equal-length one-word pair (explore_wake.json), at each byte from the changed word's first byte to
12 bytes past its end, and each state: the a-minus-b state difference projected onto every category direction, its total
norm, and the share of its squared norm inside the span of the ten directions.
NULL: the same for 200 pairs of unrelated openings at the same byte positions.

Usage: python map_motion.py <out.json>
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import mlp0_1 as M  # noqa: E402
from ks import engine as E  # noqa: E402
from ks.lang import POS, STATES  # noqa: E402

PRE = b"<|endoftext|>\n\n"
AFTER = 12


def states13(port, text_bytes):
    idx = torch.tensor(list(text_bytes), dtype=torch.long, device=E.DEV)[None]
    _, w = M.forward(port, idx)
    x = w["wte"] + port.sd["wpe.weight"][torch.arange(idx.shape[1], device=E.DEV)][None]
    out = [x]
    for i in range(6):
        for p in ("attn", "mlp"):
            x = x + w[f"{p}{i}"]; out.append(x)
    return torch.stack(out, 0)[:, 0].double().cpu().numpy()        # [13, T, 192]


def main(out_path):
    torch.set_grad_enabled(False)
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    tr, te = cache.tr, cache.split["explore"]
    dirs = {"probe": np.zeros((13, 10, 192)), "meandiff": np.zeros((13, 10, 192))}
    score = {}
    for L in range(13):
        F = np.asarray(cache.feats["last"][L], dtype=np.float64)
        mu, sd = F[tr].mean(0), np.where(F[tr].std(0) > 0, F[tr].std(0), 1.0)
        Xtr = torch.from_numpy((F[tr] - mu) / sd).to(E.DEV, E.DT); Xte = torch.from_numpy((F[te] - mu) / sd).to(E.DEV, E.DT)
        Y = torch.from_numpy(np.stack([(cache.y[tr] == c).astype(np.int64) for c in range(10)]))
        th, gn, it, why = E.newton(Xtr, Y)
        pred = E.predict(Xte, th)
        for c in range(10):
            w = th[c, :-1].cpu().numpy() / sd
            dirs["probe"][L, c] = w / np.linalg.norm(w)
            m = F[tr][cache.y[tr] == c].mean(0) - F[tr][cache.y[tr] != c].mean(0)
            dirs["meandiff"][L, c] = m / np.linalg.norm(m)
            yte = (cache.y[te] == c)
            score[f"{STATES[L]}/{POS[c]}"] = {"acc": float((pred[c].astype(bool) == yte).mean()),
                                              "base_rate": float(max(yte.mean(), 1 - yte.mean())), "fit_exit": why[c],
                                              "cos_probe_meandiff": float(dirs["probe"][L, c] @ dirs["meandiff"][L, c])}
        print(STATES[L], "acc", [round(score[f"{STATES[L]}/{POS[c]}"]["acc"], 2) for c in range(10)], flush=True)
    Q = [np.linalg.qr(dirs["probe"][L].T)[0] for L in range(13)]       # span of the ten probe directions per state

    def motion(a_bytes, b_bytes, start, end, wend):
        Sa, Sb = states13(port, a_bytes), states13(port, b_bytes)
        lo, hi = start, min(end + AFTER, wend, Sa.shape[1], Sb.shape[1])
        d = Sa[:, lo:hi] - Sb[:, lo:hi]                                  # [13, n, 192]
        norm = np.linalg.norm(d, axis=-1)
        rec = {"from_byte": int(lo), "norm": np.round(norm, 4).tolist(),
               "share_in_map": np.round(np.stack([((d[L] @ Q[L]) ** 2).sum(-1) for L in range(13)]) / np.maximum(norm ** 2, 1e-30), 4).tolist()}
        for k in dirs:
            rec[f"proj_{k}"] = np.round(np.einsum("lnd,lcd->lnc", d, dirs[k]), 4).tolist()   # [13, n, 10]
        return rec
    wake = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_wake.json")))
    pairs = []
    for r in wake["pairs"]:
        a = PRE + r["a"].encode("ascii"); s, e = r["changed_span"]
        b = a[:s] + r["b_word"].encode("ascii") + a[e:]
        assert len(a) == len(b)
        pairs.append({"pair_id": r["pair_id"], "sort": r["sort"], "a_word": r["a_word"], "b_word": r["b_word"],
                      "changed_span": [s, e], "window_end": r["window_end"], "text_a": r["a"], **motion(a, b, s, e, r["window_end"])})
    rng = np.random.default_rng(7)
    nulls = []
    texts = [PRE + r["a"].encode("ascii") for r in wake["pairs"]]
    for _ in range(200):
        i, j = rng.choice(len(texts), 2, replace=False)
        n = min(len(texts[i]), len(texts[j]))
        s = int(rng.integers(len(PRE) + 20, max(len(PRE) + 21, n - AFTER - 2)))
        m = motion(texts[i][:n], texts[j][:n], s, s + 1, n)
        nulls.append({"share_in_map": m["share_in_map"], "norm": m["norm"],
                      "abs_proj_probe_mean": np.round(np.abs(np.array(m["proj_probe"])).mean(1), 4).tolist()})
    summary = {"share_in_map_pairs_by_state": np.round(np.mean([np.mean(p["share_in_map"], axis=1) for p in pairs], 0), 4).tolist(),
               "share_in_map_null_by_state": np.round(np.mean([np.mean(p["share_in_map"], axis=1) for p in nulls], 0), 4).tolist(),
               "random_10_of_192_expectation": round(10 / 192, 4)}
    json.dump({"model": "GoldWorm v6 base", "device": str(E.DEV), "states": STATES, "categories": POS,
               "map": {"site": "a word's last byte", "fit": "train split of outer_val", "scored_on": "explore split",
                       "directions": {k: np.round(v, 5).tolist() for k, v in dirs.items()}, "scores": score,
                       "note": "probe = one-vs-rest logistic weight direction (a reading direction, NOT a writing direction — "
                               "KS-READ-NOT-WRITE-EARLY); meandiff = class mean minus rest mean."},
               "motion": {"pairs": pairs, "null_unrelated_openings": nulls, "summary": summary,
                          "note": "projection of (member a state - member b state) onto each category direction, per state and byte "
                                  "from the changed word's first byte; share_in_map = squared norm inside the ten probe directions' span."},
               "status": "EXPLORATORY; sort is a tagger's description of what differs, not a kind"}, open(out_path, "w"))
    print("summary", summary)


if __name__ == "__main__":
    main(sys.argv[1])
