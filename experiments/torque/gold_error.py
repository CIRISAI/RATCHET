#!/usr/bin/env python3
"""Estimate the GOLD-LABEL ERROR FLOOR by re-labelling a seeded sample.

WHY THIS BLOCKS THE RUN. Concordance is agreement with the ETHICS annotator pool.
If that pool's labels carry error rate e, no predictor can exceed ~1-e, and a
contrast of delta 0.05-0.07 measured against an unmeasured e plausibly of the
same order is, in the reviewer's words, "a coin toss with a confidence interval
printed on it". The floor has to exist BEFORE any concordance number is reported.

WHAT THIS IS AND IS NOT. Two INDEPENDENT re-labels of the same seeded items,
under the same question the campaign asks, with the gold hidden. Their
disagreement with each other bounds annotation reliability on this material; their
disagreement with gold bounds the shipped labels.

It is a PROXY. The re-labellers here are models, not the crowdworkers Hendrycks
used, so this measures "how reproducible is this judgement" and not "how wrong is
the pool". Stated as a proxy, published as a floor, never laundered into a human
agreement rate. Two different model families are used precisely so the number is
not one model agreeing with itself.

    gold_error.py --n 150 --seed 7
"""
from __future__ import annotations
import argparse, importlib.util, json, os, pathlib, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score
_s = importlib.util.spec_from_file_location("bh", HERE / "build_he300_arcs.py")
bh = importlib.util.module_from_spec(_s); _s.loader.exec_module(bh)
URL = "https://api.deepinfra.com/v1/openai/chat/completions"

#: Two DIFFERENT families, so the agreement number is not one model with itself.
ANNOTATORS = ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "openai/gpt-oss-120b")

def ask(model, key, cat, text):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 200,
        "messages": [{"role": "user", "content": f"{bh.QUESTION[cat]}\n\n{text}"}]}).encode()
    req = urllib.request.Request(URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(2)
    return ""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--key-file", default=os.path.expanduser("~/.deepinfra_key"))
    ap.add_argument("--out", default=str(HERE / "gold_error.json"))
    a = ap.parse_args()
    key = pathlib.Path(a.key_file).read_text().strip()

    rng = random.Random(a.seed)
    items, per = [], max(1, a.n // 4)
    for s in ("axiotic_primary", "axiotic_secondary", "deontic_held", "discriminant_control"):
        pool = bh.load_items(s); rng.shuffle(pool)
        items += [(s, it) for it in pool[:per]]

    def one(pair):
        s, it = pair
        out = {"stratum": s, "item_id": it["item_id"], "gold": it["gold"]}
        for i, m in enumerate(ANNOTATORS):
            out[f"a{i}"] = score.extract(ask(m, key, it["category"], it["text"]), it["category"])[0]
        return out
    with ThreadPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(one, items))

    both = [r for r in rows if r["a0"] is not None and r["a1"] is not None]
    agree = [r for r in both if r["a0"] == r["a1"]]
    vs_gold = [r for r in both if r["a0"] == r["gold"]] , [r for r in both if r["a1"] == r["gold"]]
    # Where the two INDEPENDENT re-labels agree with each other and BOTH differ
    # from gold is the strongest available signal that the shipped label is wrong.
    both_disagree_gold = [r for r in agree if r["a0"] != r["gold"]]

    print(f"{len(rows)} items, {len(both)} labelled by both annotators\n")
    print(f"  annotator-annotator agreement : {len(agree)/max(1,len(both)):.1%}")
    print(f"  A vs gold                     : {len(vs_gold[0])/max(1,len(both)):.1%}")
    print(f"  B vs gold                     : {len(vs_gold[1])/max(1,len(both)):.1%}")
    print(f"  BOTH agree AND differ from gold: {len(both_disagree_gold)}/{len(both)}"
          f" = {len(both_disagree_gold)/max(1,len(both)):.1%}   <- declared floor")
    print(f"\n{'stratum':22s} {'n':>4} {'a-a agree':>10} {'both≠gold':>10}")
    for s in ("axiotic_primary", "axiotic_secondary", "deontic_held", "discriminant_control"):
        ss = [r for r in both if r["stratum"] == s]
        ag = [r for r in ss if r["a0"] == r["a1"]]
        bd = [r for r in ag if r["a0"] != r["gold"]]
        print(f"{s:22s} {len(ss):>4} {len(ag)/max(1,len(ss)):>9.1%} {len(bd)/max(1,len(ss)):>9.1%}")
    pathlib.Path(a.out).write_text(json.dumps({
        "n": len(rows), "both": len(both), "seed": a.seed,
        "annotators": list(ANNOTATORS),
        "annotator_agreement": len(agree)/max(1,len(both)),
        "declared_floor_both_differ_from_gold": len(both_disagree_gold)/max(1,len(both)),
        "is_a_proxy": "model re-labellers, not the crowdworkers Hendrycks used — "
                      "this bounds reproducibility of the judgement, not the pool's error",
        "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
