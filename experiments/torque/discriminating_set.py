#!/usr/bin/env python3
"""Measure the DISCRIMINATING SET: items where the two accords disagree.

WHY. `accord_swap_effect` (formerly `values_effect`) swaps two specific
documents. Power computed over all 600 items assumes every item can move. It
cannot: an item is only movable if CIRIS's ordering and Anthropic's ordering
yield DIFFERENT verdicts on it, and both are humane constitutions that mostly
rank the same things the same way. Every non-discriminating item is noise in the
power denominator.

So measure the base rate before staking, rather than discovering it in the
residuals. This runs both accords over a seeded sample as PLAIN SYSTEM CONTENT —
no pipeline — because the question is whether the DOCUMENTS disagree, not whether
the harness does. That is a lower bound on the discriminating set: the pipeline
can only add divergence, not remove the documents' agreement.

    discriminating_set.py --n 120 --seed 42
"""
from __future__ import annotations
import argparse, importlib.util, json, os, pathlib, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score
_s = importlib.util.spec_from_file_location("bh", HERE / "build_he300_arcs.py")
bh = importlib.util.module_from_spec(_s); _s.loader.exec_module(bh)

URL = "https://api.deepinfra.com/v1/openai/chat/completions"

def ask(model, key, system, cat, text, timeout=90):
    body = json.dumps({"model": model, "temperature": 0.7, "top_p": 1.0,
                       "max_tokens": 400,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user",
                                     "content": f"{bh.QUESTION[cat]}\n\n{text}"}]}).encode()
    req = urllib.request.Request(URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(2)
    return ""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    ap.add_argument("--key-file", default=os.path.expanduser("~/.deepinfra_key"))
    ap.add_argument("--ciris", default="/tmp/a2913/accord_1.2b.txt")
    ap.add_argument("--alt", default=str(HERE / "corpora/values-alt/A-accord-FINAL.txt"))
    ap.add_argument("--out", default=str(HERE / "discriminating_set.json"))
    a = ap.parse_args()

    key = pathlib.Path(a.key_file).read_text().strip()
    sysA = pathlib.Path(a.ciris).read_text(encoding="utf-8")
    sysB = pathlib.Path(a.alt).read_text(encoding="utf-8")

    import random
    rng = random.Random(a.seed)
    items = []
    per = max(1, a.n // 4)
    for s in ("axiotic_primary", "axiotic_secondary", "deontic_held", "discriminant_control"):
        pool = bh.load_items(s); rng.shuffle(pool)
        items += [(s, it) for it in pool[:per]]

    def one(pair):
        s, it = pair
        ra = ask(a.model, key, sysA, it["category"], it["text"])
        rb = ask(a.model, key, sysB, it["category"], it["text"])
        la = score.extract(ra, it["category"])[0]
        lb = score.extract(rb, it["category"])[0]
        return {"stratum": s, "item_id": it["item_id"], "gold": it["gold"],
                "ciris": la, "alt": lb,
                "discriminating": (la is not None and lb is not None and la != lb)}
    with ThreadPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(one, items))

    scored = [r for r in rows if r["ciris"] is not None and r["alt"] is not None]
    disc = [r for r in scored if r["discriminating"]]
    print(f"{len(rows)} items, {len(scored)} scored on both accords")
    print(f"DISCRIMINATING SET: {len(disc)}/{len(scored)} = {len(disc)/max(1,len(scored)):.1%}\n")
    print(f"{'stratum':22s} {'n':>4} {'disc':>5} {'rate':>7}")
    for s in ("axiotic_primary", "axiotic_secondary", "deontic_held", "discriminant_control"):
        ss = [r for r in scored if r["stratum"] == s]
        dd = [r for r in ss if r["discriminating"]]
        print(f"{s:22s} {len(ss):>4} {len(dd):>5} {len(dd)/max(1,len(ss)):>6.1%}")
    pathlib.Path(a.out).write_text(json.dumps(
        {"n": len(rows), "scored": len(scored), "discriminating": len(disc),
         "rate": len(disc)/max(1,len(scored)), "seed": a.seed, "rows": rows},
        indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    print("\nThis is a LOWER BOUND on the discriminating set: the documents are")
    print("compared as plain system content, so the pipeline can add divergence")
    print("but cannot remove the documents' agreement.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
