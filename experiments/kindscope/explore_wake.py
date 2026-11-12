"""Exploration: does a one-word change reach the model's OUTPUT later in the sentence, and which stage carries it there?

Real pairs only: unseen TinyStories openings differing in exactly one word of EQUAL byte length (so every later byte is at
the same position in both members). Per later position p (after the changed word, to the sentence end):
  tv[p]      = total variation between the two members' next-byte distributions
  closure[w] = 1 - tv(member b with write w at p replaced by member a's write w at p, member a) / tv[p]
               for each of the 12 block writes: how much of the output difference at p that one write carries.
No trace files are written here (the per-pair rows in the output carry the texts and spans).

Usage: python explore_wake.py <TinyStories-valid.txt> <fresh slice> <pairs dir> <out.json>
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, sys  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_pair_traces as P  # noqa: E402
import edit1 as X  # noqa: E402
import mlp0_1 as M  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402

WRITES = [f"{p}{i}" for i in range(6) for p in ("attn", "mlp")]
CAP = 40


def main(valid, fresh, odir, out_path):
    torch.set_grad_enabled(False)
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    firsts = {i: P.first_sentence(s) for i, s in enumerate(kept)}
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, f in firsts.items():
        if f and len(f) <= 200:
            w = f.split(" ")
            if 6 <= len(w) <= 25:
                for k in range(len(w) - 1):          # not the last word: something must follow the change
                    idx[(k, " ".join(w[:k] + ["\x00"] + w[k + 1:]))][w[k]].append(i)
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

    def pos_of(sent, k):
        words = sent.split(" "); start = len(" ".join(words[:k])) + (1 if k else 0)
        for t in nlp(sent):
            if t.idx <= start < t.idx + len(t.text):
                return t.pos_
        return "X"
    cands = collections.defaultdict(list)
    for (k, masked), words in sorted(idx.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())):
        ws = sorted(words, key=lambda w: (-len(words[w]), w))
        got = None
        for a in ws[:6]:
            for b in ws[:6]:
                if a < b and len(a) == len(b):
                    got = (a, b); break
            if got:
                break
        if not got:
            continue
        a, b = got; ia, ib = words[a][0], words[b][0]
        strip = lambda w: w.strip(".,!?;:\"'")  # noqa: E731
        if strip(a) == strip(b):
            sort = "punctuation"
        else:
            pa, pb = pos_of(firsts[ia], k), pos_of(firsts[ib], k)
            sort = {"PROPN": "name", "NUM": "number", "ADJ": "quality word", "NOUN": "thing", "VERB": "action word"}.get(pa, "other") if pa == pb else "other"
        if sort != "other":
            cands[sort].append((k, ia, ib, a, b))
    available = {s: len(v) for s, v in cands.items()}
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    pre = len(P.PRE)
    pairs, rows = [], []
    for sort in sorted(cands):
        for n, (k, ia, ib, a, b) in enumerate(cands[sort][:CAP]):
            sa, sb = firsts[ia], firsts[ib]
            assert len(sa) == len(sb)
            L = pre + len(sa)
            words = sa.split(" "); start = pre + len(" ".join(words[:k])) + (1 if k else 0); end = start + len(a)
            ids = [torch.tensor(list((P.PRE + kept[i].encode("ascii"))[:L]), dtype=torch.long, device=E.DEV)[None] for i in (ia, ib)]
            (la, wa), (lb, wb) = M.forward(port, ids[0]), M.forward(port, ids[1])
            pa_, pb_ = la[0].softmax(-1).double(), lb[0].softmax(-1).double()
            tv = 0.5 * (pa_ - pb_).abs().sum(-1).cpu().numpy()                    # [L]
            later = [p for p in range(end, L - 1)]                                # positions after the changed word
            clos = {}
            for w in WRITES:
                c = np.full(L, np.nan)
                for p in later:
                    if tv[p] < 0.02:
                        continue
                    lg, _ = M.forward(port, ids[1], (w, torch.tensor([0], device=E.DEV), torch.tensor([p], device=E.DEV), wa[w][0, p][None]))
                    c[p] = 1 - float(0.5 * (lg[0, p].softmax(-1).double() - pa_[p]).abs().sum()) / tv[p]
                clos[w] = c
            pid = f"eq-{sort.replace(' ', '_')}-{n:03d}"
            rows.append({"pair_id": pid, "sort": sort, "a": sa, "b_word": b, "a_word": a, "changed_span": [start, end], "window_end": L,
                         "tv": [round(float(x), 4) for x in tv],
                         "next_byte_after": [chr(int(c_)) for c_ in ids[0][0].cpu().numpy()[1:]] ,
                         "closure": {w: [None if np.isnan(x) else round(float(x), 3) for x in c] for w, c in clos.items()}})
    # ---- summaries
    def summarise(sel):
        prof = collections.defaultdict(list); best = collections.Counter(); late = []
        for r in sel:
            s, e, L = r["changed_span"][0], r["changed_span"][1], r["window_end"]
            for p in range(e - 1, L - 1):
                prof[p - (e - 1)].append(r["tv"][p])
            for p in range(e + 2, L - 1):
                if r["tv"][p] >= 0.05:
                    cl = {w: r["closure"][w][p] for w in WRITES if r["closure"][w][p] is not None}
                    if cl:
                        best[max(cl, key=cl.get)] += 1
                        late.append({w: cl[w] for w in cl})
        mean_cl = {w: round(float(np.mean([d[w] for d in late if w in d])), 3) for w in WRITES} if late else {}
        return {"pairs": len(sel), "tv_by_offset_from_last_changed_byte": {int(k): round(float(np.mean(v)), 4) for k, v in sorted(prof.items())[:25]},
                "late_positions(tv>=0.05, >=3 bytes after the word)": len(late), "which_write_closes_most": dict(best.most_common()),
                "mean_closure_by_write": mean_cl}
    summary = {s: summarise([r for r in rows if r["sort"] == s]) for s in sorted({r["sort"] for r in rows})}
    summary["all"] = summarise(rows)
    json.dump({"available_equal_length_per_sort": available, "cap": CAP, "device": str(E.DEV), "summary": summary, "pairs": rows,
               "note": "EXPLORATORY. sort is a tagger's description of what differs, not a kind."}, open(out_path, "w"))
    print("available", available)
    for s, v in summary.items():
        print(s, "pairs", v["pairs"], "| tv by offset 0..8:", [v["tv_by_offset_from_last_changed_byte"].get(i) for i in range(9)])
        print("   late positions", v["late_positions(tv>=0.05, >=3 bytes after the word)"], "closes most:", v["which_write_closes_most"])
        print("   mean closure:", v["mean_closure_by_write"])


if __name__ == "__main__":
    main(*sys.argv[1:5])
