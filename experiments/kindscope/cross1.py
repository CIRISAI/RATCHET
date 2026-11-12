"""CROSS-1: does a change of category move the state across the map? Frozen design: PREREG_CROSS1.md.
Usage: python cross1.py <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import map_motion as MM  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402
from ks.lang import POS, STATES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = b"<|endoftext|>\n\n"


def main(out_path):
    torch.set_grad_enabled(False)
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    mm = json.load(open(os.path.join(HERE, "map_motion.json")))
    D = {k: np.array(v) for k, v in mm["map"]["directions"].items()}       # [13, 10, 192]
    Q = [np.linalg.qr(D["probe"][L].T)[0] for L in range(13)]
    rng = np.random.default_rng(20261123)

    def last_state(sentence, k):
        words = sentence.split(" ")
        end = len(PRE) + len(" ".join(words[:k + 1]))                      # one past the word's last byte
        return MM.states13(port, PRE + sentence.encode("ascii"))[:, end - 1]   # [13, 192]

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))
    cross = json.load(open(os.path.join(HERE, "cross_pairs.json")))
    rows = []
    for r in cross:
        sa, sb = last_state(r["a"], r["word_index"]), last_state(r["b"], r["word_index"])
        A, B = POS.index(r["a_pos"]), POS.index(r["b_pos"]); d = sa - sb
        rows.append({"pair": "/".join(sorted((r["a_pos"], r["b_pos"]))), "equal_length": r["equal_length"],
                     "a_word": r["a_word"], "b_word": r["b_word"], "a_pos": r["a_pos"], "b_pos": r["b_pos"],
                     "align_meandiff": [cos(d[L], D["meandiff"][L, A] - D["meandiff"][L, B]) for L in range(13)],
                     "align_probe": [cos(d[L], D["probe"][L, A] - D["probe"][L, B]) for L in range(13)],
                     "share_in_map": [float(((d[L] @ Q[L]) ** 2).sum() / (d[L] @ d[L] + 1e-30)) for L in range(13)]})
    # floor: same-category pairs against an axis they do not cross
    same = []
    for p in mm["motion"]["pairs"]:
        a = p["text_a"]; s, e = p["changed_span"]
        b = (PRE + a.encode("ascii"))[:s] + p["b_word"].encode("ascii") + (PRE + a.encode("ascii"))[e:]
        Sa = MM.states13(port, PRE + a.encode("ascii"))[:, e - 1]; Sb = MM.states13(port, b)[:, e - 1]
        same.append(Sa - Sb)
    cat_of = {"name": "PROPN", "thing": "NOUN", "quality word": "ADJ", "action word": "VERB"}
    same_cat = [cat_of.get(p["sort"]) for p in mm["motion"]["pairs"]]
    keep = [i for i, c in enumerate(same_cat) if c]
    floor_draws = np.zeros((200, 13))
    for t in range(200):
        vals = []
        for i in keep:
            A = POS.index(same_cat[i]); B = int(rng.choice([c for c in range(10) if c != A]))
            vals.append([cos(same[i][L], D["meandiff"][L, A] - D["meandiff"][L, B]) for L in range(13)])
        floor_draws[t] = np.mean(vals, 0)

    def boot(v):
        v = np.array(v); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        return {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(v))}
    res = {"states": STATES, "n_cross": len(rows), "n_floor_pairs": len(keep), "by_state": {}}
    x1 = x2 = True
    for L in range(13):
        c = boot([r["align_meandiff"][L] for r in rows])
        fl = {"mean_abs_of_draw_means": float(np.abs(floor_draws[:, L]).mean()), "p95_of_draw_means": float(np.percentile(floor_draws[:, L], 95)),
              "mean_of_draw_means": float(floor_draws[:, L].mean())}
        rec = {"cross": c, "floor": fl, "probe_dirs": boot([r["align_probe"][L] for r in rows]),
               "share_in_map_cross": float(np.mean([r["share_in_map"][L] for r in rows])),
               "equal_length_only": boot([r["align_meandiff"][L] for r in rows if r["equal_length"]]),
               "by_pair": {k: boot([r["align_meandiff"][L] for r in rows if r["pair"] == k]) for k in ("AUX/VERB", "ADJ/NOUN")}}
        rec["by_pair"]["rest"] = boot([r["align_meandiff"][L] for r in rows if r["pair"] not in ("AUX/VERB", "ADJ/NOUN")])
        rec["X1"] = bool(c["ci95"][0] > 0); rec["X2"] = bool(c["ci95"][0] > fl["p95_of_draw_means"] and c["mean"] - fl["mean_abs_of_draw_means"] > 0)
        if L >= 2:
            x1 &= rec["X1"]; x2 &= rec["X2"]
        res["by_state"][STATES[L]] = rec
        print(f"{STATES[L]:6} cross {c['mean']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] floor p95 {fl['p95_of_draw_means']:+.3f} "
              f"| probe dirs {rec['probe_dirs']['mean']:+.3f} | share in map {rec['share_in_map_cross']:.3f} | AUX/VERB {rec['by_pair']['AUX/VERB']['mean']:+.3f} "
              f"ADJ/NOUN {rec['by_pair']['ADJ/NOUN']['mean']:+.3f} rest {rec['by_pair']['rest']['mean']:+.3f} | eq-len {rec['equal_length_only']['mean']:+.3f}", flush=True)
    floor_ok = all(abs(res["by_state"][STATES[L]]["floor"]["mean_of_draw_means"]) <= 0.05 for L in range(2, 13))
    res["plant_floor"] = {"ok": bool(floor_ok), "rule": "|mean of floor draw means| <= 0.05 at mlp0..mlp5 states (declared here: the floor must sit at 0)"}
    res["X1"] = bool(x1 and floor_ok); res["X2"] = bool(x2 and floor_ok)
    res["pairs"] = rows
    res.update({"prereg_sha256": V.sha256(os.path.join(HERE, "PREREG_CROSS1.md")), "script_sha256": V.sha256(os.path.abspath(__file__)), "device": str(E.DEV)})
    json.dump(res, open(out_path, "w"))
    print("plant floor", res["plant_floor"]["ok"], "X1", res["X1"], "X2", res["X2"])


if __name__ == "__main__":
    main(sys.argv[1])
