"""Mine real openings differing in exactly one word whose two words have DIFFERENT parts of speech (both among the ten).
Usage: python cross_mine.py <valid> <fresh> <out.json>"""
import collections, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spacy
import make_pair_traces as P
from ks.lang import POS
def main(valid, fresh, out):
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    firsts = {i: P.first_sentence(s) for i, s in enumerate(kept)}
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, f in firsts.items():
        if f and len(f) <= 200:
            w = f.split(" ")
            if 6 <= len(w) <= 25:
                for k in range(len(w) - 1):
                    idx[(k, " ".join(w[:k] + ["\x00"] + w[k + 1:]))][w[k]].append(i)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    def pos_of(sent, k):
        words = sent.split(" "); start = len(" ".join(words[:k])) + (1 if k else 0)
        for t in nlp(sent):
            if t.idx <= start < t.idx + len(t.text): return t.pos_
        return "X"
    rows = []
    for (k, masked), words in sorted(idx.items()):
        ws = sorted(words, key=lambda w: (-len(words[w]), w))[:6]
        done = False
        for a in ws:
            for b in ws:
                if a < b and a.strip(".,!?;:\"'") != b.strip(".,!?;:\"'"):
                    ia, ib = words[a][0], words[b][0]
                    pa, pb = pos_of(firsts[ia], k), pos_of(firsts[ib], k)
                    if pa != pb and pa in POS and pb in POS:
                        rows.append({"word_index": k, "a": firsts[ia], "b": firsts[ib], "a_word": a, "b_word": b, "a_pos": pa, "b_pos": pb,
                                     "equal_length": len(a) == len(b), "stories": [ia, ib]}); done = True; break
            if done: break
    json.dump(rows, open(out, "w"), indent=0)
    c = collections.Counter(tuple(sorted((r["a_pos"], r["b_pos"]))) for r in rows)
    print(len(rows), "cross-category pairs;", sum(r["equal_length"] for r in rows), "equal length; by category pair:", c.most_common(12))
main(*sys.argv[1:4])
