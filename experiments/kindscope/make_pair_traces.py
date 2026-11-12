"""Paired traces for 'a change travelling through the model' (GoldWorm v6 base).

Pairs are REAL: two unseen TinyStories openings that differ in exactly one word (nothing constructed). Each member is
written as a holon-trace 1 file over the opening sentence only (positions 0 .. window_end), plus pairs.json.
`sort` describes the change by what differs; it is NOT a Constitution kind.

Usage: python make_pair_traces.py <TinyStories-valid.txt> <fresh slice> <out dir>
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, hashlib, json, re, sys  # noqa: E401,E402
import numpy as np  # noqa: E402
import spacy  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X  # noqa: E402
import edit2 as Y  # noqa: E402
import mlp0_1 as M  # noqa: E402
import vocab1 as V  # noqa: E402
from ks import engine as E  # noqa: E402

FRESH_SHA = "3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a"
NAMES = ["embed"] + [f"{p}{i}" for i in range(6) for p in ("attn", "mlp")]
PRE = b"<|endoftext|>\n\n"
PER_SORT, N_NULL, D = 30, 8, 192


def first_sentence(s):
    m = re.match(r"^(.+?[.!?])(\s|$)", s)
    return m.group(1) if m and "\n" not in m.group(1) else None


def main(valid, fresh, odir):
    torch.set_grad_enabled(False)
    assert V.sha256(fresh) == FRESH_SHA
    os.makedirs(odir, exist_ok=True)
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    firsts = {i: first_sentence(s) for i, s in enumerate(kept)}
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    same = collections.defaultdict(list)
    for i, f in firsts.items():
        if not f or len(f) > 200:
            continue
        same[f].append(i)
        w = f.split(" ")
        if 6 <= len(w) <= 25:
            for k in range(len(w)):
                idx[(k, " ".join(w[:k] + ["\x00"] + w[k + 1:]))][w[k]].append(i)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

    def pos_of(sent, k):
        words = sent.split(" ")
        start = len(" ".join(words[:k])) + (1 if k else 0)
        for t in nlp(sent):
            if t.idx <= start < t.idx + len(t.text):
                return t.pos_
        return "X"

    cands = collections.defaultdict(list)
    for (k, masked), words in sorted(idx.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())):
        if len(words) < 2:
            continue
        a, b = sorted(words, key=lambda w: (-len(words[w]), w))[:2]
        ia, ib = words[a][0], words[b][0]
        sa, sb = firsts[ia], firsts[ib]
        strip = lambda w: w.strip(".,!?;:\"'")  # noqa: E731
        if strip(a) == strip(b):
            sort = "punctuation"
        else:
            pa, pb = pos_of(sa, k), pos_of(sb, k)
            sort = ({"PROPN": "name", "NUM": "number", "ADJ": "quality word", "NOUN": "thing", "VERB": "action word"}.get(pa, "other")
                    if pa == pb else "other")
        cands[sort].append((k, ia, ib, a, b))
    available = {s: len(v) for s, v in cands.items()}
    chosen = [(s, c) for s in sorted(cands) if s != "other" for c in cands[s][:PER_SORT]]
    nulls = [(f, v[:2]) for f, v in sorted(same.items(), key=lambda kv: -len(kv[1])) if len(v) >= 2
             and kept[v[0]] != kept[v[1]]][:N_NULL]
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    sd0, _ = X.load_params(cache.info["model_path"]); sd = port.sd

    def trace(story_i, n_pos, name):
        w = (PRE + kept[story_i].encode("ascii"))[:256]
        n_pos = min(n_pos, len(w))
        idxt = torch.tensor(list(w), dtype=torch.long, device=E.DEV)[None]
        _, writes = M.forward(port, idxt)
        embed = writes["wte"] + sd["wpe.weight"][torch.arange(idxt.shape[1], device=E.DEV)][None]
        order = [embed] + [writes[n] for n in NAMES[1:]]
        st = Y.fwd_full(sd0, idxt)[2][5]
        x = torch.zeros_like(embed)
        for t in order:
            x = x + t
        assert torch.equal(x, st)
        W = torch.stack(order, 2)[0, :n_pos].cpu().numpy().astype("<f4"); S = st[0, :n_pos].cpu().numpy().astype("<f4")
        path = os.path.join(odir, name)
        with open(path, "wb") as f:
            f.write(f"holon-trace 1 positions {n_pos} width {D} writes {len(NAMES)} {' '.join(NAMES)}\n".encode())
            for p in range(n_pos):
                f.write(W[p].tobytes()); f.write(S[p].tobytes())
        return {"trace": name, "sha256": V.sha256(path), "bytes": list(w[:n_pos]), "text": w[:n_pos].decode("ascii"),
                "fresh_story_index": int(story_i), "positions": n_pos}

    out = []
    for n, (sort, (k, ia, ib, a, b)) in enumerate(chosen):
        pid = f"{sort.replace(' ', '_')}-{n:03d}"
        mem = []
        for tag, i, wd in (("a", ia, a), ("b", ib, b)):
            sent = firsts[i]; words = sent.split(" ")
            start = len(PRE) + len(" ".join(words[:k])) + (1 if k else 0)
            m = trace(i, len(PRE) + len(sent), f"{pid}.{tag}.trace")
            m.update({"changed_word": wd, "changed_span": [start, start + len(wd)], "window_end": len(PRE) + len(sent)})
            mem.append(m)
        out.append({"pair_id": pid, "sort": sort, "word_index": k, "aligned_until": mem[0]["changed_span"][0] - 1,
                    "alignment_after_change": "from the END of the changed word, position by position, to each member's window_end",
                    "members": mem})
    for n, (f, (ia, ib)) in enumerate(nulls):
        pid = f"null-{n:03d}"
        mem = [dict(trace(i, len(PRE) + len(f), f"{pid}.{tag}.trace"), window_end=len(PRE) + len(f)) for tag, i in (("a", ia), ("b", ib))]
        out.append({"pair_id": pid, "sort": "null (identical opening sentence, stories diverge after it)", "members": mem,
                    "aligned_until": len(PRE) + len(f) - 1})
    json.dump({"model": "GoldWorm v6 base", "model_sha256": cache.info["model_sha256"], "fresh_sha256": FRESH_SHA,
               "writes": NAMES, "device": str(E.DEV), "available_per_sort": available, "per_sort_cap": PER_SORT,
               "note": "sort describes what differs between two REAL openings; it is not a Constitution kind. Traces cover the opening "
                       "sentence only. Story indices are into FRESH-1's kept list and overlap stories read by earlier runs.",
               "pairs": out}, open(os.path.join(odir, "pairs.json"), "w"), indent=1)
    print("pairs", len(out), "available per sort", available)


if __name__ == "__main__":
    main(*sys.argv[1:4])
