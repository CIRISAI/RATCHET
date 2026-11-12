"""CROSS-1 exploratory re-read (declared after the result; NOT a new stake): remove word pairs whose tagger labels a
reader would dispute, one row per DISTINCT word pair, and emit by-byte wake rows for the explorer.
The dispute list is ONE reader's judgement (this session), written out below so anyone can contest a line.
Usage: python cross1_reread.py <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X, map_motion as MM
from ks import engine as E
from ks.lang import POS, STATES
HERE = os.path.dirname(os.path.abspath(__file__)); PRE = b"<|endoftext|>\n\n"
# words the tagger mislabelled in these openings, by this reader: (word, tagger label) -> why
DISPUTED = {("Lily", "ADV"): "a name", ("Lily", "ADJ"): "a name", ("Molly", "ADV"): "a name", ("jolly", "ADV"): "an adjective here",
            ("pretty", "ADV"): "an adjective here ('a pretty girl')", ("kind", "NOUN"): "an adjective here ('a kind girl')",
            ("chubby", "NOUN"): "an adjective", ("chubby", "PROPN"): "an adjective", ("mummy", "ADJ"): "a noun", ("ant", "ADJ"): "a noun",
            ("kindly", "ADV"): "an adjective here", ("love", "NOUN"): "a verb here", ("do", "AUX"): "a main verb here",
            ("organized", "VERB"): "an adjective here", ("broken", "VERB"): "an adjective here (participle)"}
def main(out_path):
    torch.set_grad_enabled(False)
    r = json.load(open(os.path.join(HERE, "cross1_result.json")))
    groups = collections.OrderedDict()
    for p in r["pairs"]:
        groups.setdefault((p["a_word"], p["a_pos"], p["b_word"], p["b_pos"]), []).append(p)
    rows = []
    for (a, pa, b, pb), ps in groups.items():
        why = DISPUTED.get((a.strip(".,"), pa)) or DISPUTED.get((b.strip(".,"), pb))
        rows.append({"a_word": a, "a_pos": pa, "b_word": b, "b_pos": pb, "occurrences": len(ps), "disputed": why,
                     "align_meandiff": np.mean([p["align_meandiff"] for p in ps], 0).round(4).tolist(),
                     "align_probe": np.mean([p["align_probe"] for p in ps], 0).round(4).tolist()})
    rng = np.random.default_rng(5)
    def boot(sel, L):
        v = np.array([x["align_meandiff"][L] for x in sel])
        if len(v) < 3: return {"mean": float(v.mean()) if len(v) else None, "n": int(len(v))}
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        return {"mean": round(float(v.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "n": int(len(v))}
    an = lambda x: tuple(sorted((x["a_pos"], x["b_pos"]))) == ("ADJ", "NOUN")
    summ = {}
    for L in (0, 2, 6, 12):
        summ[STATES[L]] = {"adj_noun_undisputed": boot([x for x in rows if an(x) and not x["disputed"]], L),
                           "adj_noun_disputed": boot([x for x in rows if an(x) and x["disputed"]], L),
                           "all_undisputed": boot([x for x in rows if not x["disputed"]], L),
                           "all_disputed": boot([x for x in rows if x["disputed"]], L)}
        print(STATES[L], json.dumps(summ[STATES[L]]))
    # by-byte wake rows, one opening per distinct word pair
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    mm = json.load(open(os.path.join(HERE, "map_motion.json")))
    D = {k: np.array(v) for k, v in mm["map"]["directions"].items()}; Q = [np.linalg.qr(D["probe"][L].T)[0] for L in range(13)]
    cross = json.load(open(os.path.join(HERE, "cross_pairs.json"))); seen = set(); wakes = []
    for c in cross:
        key = (c["a_word"], c["b_word"])
        if key in seen: continue
        seen.add(key)
        k = c["word_index"]; wa = c["a"].split(" "); wb = c["b"].split(" ")
        sa = len(PRE) + len(" ".join(wa[:k])) + (1 if k else 0); ea = sa + len(wa[k]); eb = sa + len(wb[k])
        A, Bt = PRE + c["a"].encode("ascii"), PRE + c["b"].encode("ascii")
        Sa, Sb = MM.states13(port, A), MM.states13(port, Bt)
        n = min(len(A) - ea, len(Bt) - eb, 12)
        # aligned from the END of the changed word: offset 0 = the word's last byte in each member
        d = np.stack([Sa[:, ea - 1 + o] - Sb[:, eb - 1 + o] for o in range(0, n + 1)], 1)       # [13, n+1, 192]
        norm = np.linalg.norm(d, axis=-1)
        wakes.append({"a_word": c["a_word"], "b_word": c["b_word"], "a_pos": c["a_pos"], "b_pos": c["b_pos"], "text_a": c["a"], "text_b": c["b"],
                      "changed_span_a": [sa, ea], "changed_span_b": [sa, eb], "offsets_from_word_end": list(range(0, n + 1)),
                      "disputed": DISPUTED.get((c["a_word"].strip(".,"), c["a_pos"])) or DISPUTED.get((c["b_word"].strip(".,"), c["b_pos"])),
                      "norm": norm.round(4).tolist(),
                      "share_in_map": (np.stack([((d[L] @ Q[L]) ** 2).sum(-1) for L in range(13)]) / np.maximum(norm ** 2, 1e-30)).round(4).tolist(),
                      "proj_meandiff": np.einsum("lnd,lcd->lnc", d, D["meandiff"]).round(4).tolist()})
    json.dump({"status": "EXPLORATORY re-read of CROSS-1; the dispute list is one reader's judgement", "states": STATES, "categories": POS,
               "disputed_rule": {f"{w}:{t}": why for (w, t), why in DISPUTED.items()}, "distinct_word_pairs": rows, "summary": summ,
               "wakes": wakes, "counts": {"distinct": len(rows), "disputed": sum(1 for x in rows if x["disputed"])}}, open(out_path, "w"))
    print("distinct", len(rows), "disputed", sum(1 for x in rows if x["disputed"]), "wakes", len(wakes))
main(sys.argv[1])
