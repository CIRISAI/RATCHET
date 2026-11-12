"""How reliable are the MAP's labels? Agreement between the tagger that labels the map (spaCy en_core_web_sm 3.8.0) and an
independent one (Stanza 1.10.1, English default package) on the map's own tokens (outer_val stories, the loop cache).
Agreement between two taggers bounds label noise from above; it is not ground truth. EXPLORATORY measurement.
Usage: python tagger_agreement.py <stanza model dir> <TinyStories-valid.txt> <out.json>"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stanza
import edit1 as X
import vocab1 as V
from ks import engine as E
from ks.lang import POS
def main(mdir, corpus, out):
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache"))
    stories = V.load_stories(corpus)
    nlp = stanza.Pipeline("en", dir=mdir, processors="tokenize,pos", verbose=False, download_method=None)
    P0 = len(V.PREFIX)
    by = collections.defaultdict(dict)
    for n, (s, a, b, y, f) in enumerate(cache.toks):
        by[int(s)][(int(a) - P0, int(b) - P0 + 1)] = n
    st = np.full(len(cache.toks), -1)       # stanza label index, -1 = no token with the same span, -2 = outside the ten
    for si in sorted(by):
        doc = nlp(stories[si][:260])
        for sent in doc.sentences:
            for w in sent.words:
                key = (w.start_char, w.end_char)
                if key in by[si]:
                    st[by[si][key]] = POS.index(w.upos) if w.upos in POS else -2
    y = cache.y; m = st != -1
    agree = (st == y) & m
    res = {"tokens": int(len(y)), "span_matched": int(m.sum()), "agreement_on_matched": float(agree.sum() / m.sum()),
           "by_spacy_label": {}, "confusions": {}, "top_disagreeing_forms": []}
    for c, name in enumerate(POS):
        sel = m & (y == c)
        other = collections.Counter(("other" if t == -2 else POS[t]) for t in st[sel & ~agree])
        res["by_spacy_label"][name] = {"n": int(sel.sum()), "agreement": float(agree[sel].mean()) if sel.any() else None,
                                       "stanza_says_instead": dict(other.most_common(4))}
    dis = collections.Counter(); tot = collections.Counter()
    for n in np.where(m)[0]:
        tot[cache.forms[n]] += 1
        if not agree[n]: dis[(cache.forms[n], POS[y[n]], "other" if st[n] == -2 else POS[st[n]])] += 1
    res["top_disagreeing_forms"] = [{"form": f, "spacy": a, "stanza": b, "count": c, "of": tot[f]} for (f, a, b), c in dis.most_common(30)]
    json.dump(res, open(out, "w"), indent=1)
    print("agreement", round(res["agreement_on_matched"], 4), "on", res["span_matched"], "of", res["tokens"])
    for k, v in res["by_spacy_label"].items(): print(f"  {k:6} n={v['n']:6} agree {v['agreement']:.3f}  stanza instead: {v['stanza_says_instead']}")
    print("  top disagreeing forms:", [(d["form"], d["spacy"], d["stanza"], d["count"], d["of"]) for d in res["top_disagreeing_forms"][:14]])
main(*sys.argv[1:4])
