"""Logic against training, first picture (GoldWorm v6 base; EXPLORATORY).

Case: a story names its character ("... named X.") and X occurs again inside the 256-byte window. At each recurrence:
  logic      : the name is X (1.0)
  statistics : X's share among names in the model's own training text (train_core), i.e. what frequency alone gives
  weights    : the model's probability of X's bytes at that position, given that a capitalised word starts there
               (P(bytes of X) / P(first byte is A-Z)), and the same quantity for the corpus's most frequent name
x-axis: how often X is named in train_core. Real stories only: FRESH-1 kept stories 12,000-15,999 (unread).
Usage: python torque_names.py <valid> <fresh> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import collections, json, math, re, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit1 as X, mlp0_1 as M
from ks import engine as E
PRE = b"<|endoftext|>\n\n"; NAMED = re.compile(r"\bnamed ([A-Z][a-z]+)\b")
def main(valid, fresh, out):
    torch.set_grad_enabled(False)
    raw = open(valid, "rb").read()
    train = raw[:17502553].decode("ascii", "ignore")
    freq = collections.Counter(NAMED.findall(train)); total = sum(freq.values()); top = freq.most_common(1)[0][0]
    vset = set(p.strip() for p in raw.split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip().decode("ascii") for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    cache = E.Cache(os.path.join(X.SCRATCH, "ks_cache")); port = cache.port()
    upper = torch.tensor(list(range(65, 91)), device=E.DEV)
    rows = []
    for si in range(12000, 16000):
        s = kept[si]; m = NAMED.search(s[:120])
        if not m: continue
        name = m.group(1); w = (PRE + s.encode("ascii"))[:256]; text = w.decode("ascii")
        occ = [mm.start() for mm in re.finditer(r"(?<=[ \n\"])" + name + r"\b", text) if mm.start() > len(PRE) + m.end()]
        if not occ: continue
        lp = M.forward(port, torch.tensor(list(w), dtype=torch.long, device=E.DEV)[None])[0][0].log_softmax(-1).double()
        def name_prob(p, nm):   # P(bytes of nm | context up to p) using the true context for the first byte only
            # first byte from the real context; later bytes need nm in context, so only exact for nm == name
            return float(lp[p - 1, ord(nm[0])])
        for k, p in enumerate(occ):
            if p + len(name) > len(w): continue
            full = sum(float(lp[p - 1 + j, w[p + j]]) for j in range(len(name)))          # log P(X bytes), teacher-forced
            cap = float(torch.logsumexp(lp[p - 1, upper], 0))                              # log P(first byte is A-Z)
            rows.append({"story": si, "name": name, "train_count": freq.get(name, 0), "occurrence": k + 1, "position": p,
                         "logp_name": full, "logp_capital": cap, "p_name_given_capital": math.exp(full - cap),
                         "p_first_byte_given_capital": math.exp(float(lp[p - 1, ord(name[0])]) - cap),
                         "p_top_name_first_byte_given_capital": math.exp(float(lp[p - 1, ord(top[0])]) - cap),
                         "same_first_byte_as_top": name[0] == top[0]})
    # bins by training count
    edges = [0, 1, 3, 10, 30, 100, 300, 1000, 10**9]; bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["train_count"] < hi]
        if not sel: continue
        names = {r["name"] for r in sel}
        bins.append({"train_count_range": [lo, hi if hi < 10**9 else None], "recurrences": len(sel), "distinct_names": len(names),
                     "weights_p_name": float(np.mean([r["p_name_given_capital"] for r in sel])),
                     "weights_p_first_byte": float(np.mean([r["p_first_byte_given_capital"] for r in sel])),
                     "statistics_share": float(np.mean([r["train_count"] / total for r in sel])),
                     "weights_p_top_first_byte_when_different": float(np.mean([r["p_top_name_first_byte_given_capital"] for r in sel if not r["same_first_byte_as_top"]] or [float("nan")])),
                     "logic": 1.0})
    json.dump({"status": "EXPLORATORY", "model": "GoldWorm v6 base", "device": str(E.DEV), "train_named_total": total, "top_name": top,
               "top_names": freq.most_common(8), "n_recurrences": len(rows), "n_stories": len({r["story"] for r in rows}),
               "bins": bins, "rows": rows}, open(out, "w"))
    print("named in train_core:", total, "distinct", len(freq), "top", freq.most_common(5))
    print("recurrences", len(rows), "stories", len({r["story"] for r in rows}), "distinct names", len({r["name"] for r in rows}))
    for b in bins: print(b["train_count_range"], "n", b["recurrences"], "names", b["distinct_names"], "| statistics", round(b["statistics_share"], 4),
                         "| weights P(name)", round(b["weights_p_name"], 3), "P(first byte)", round(b["weights_p_first_byte"], 3),
                         "| P(top name's first byte) when different", round(b["weights_p_top_first_byte_when_different"], 3))
main(*sys.argv[1:4])
