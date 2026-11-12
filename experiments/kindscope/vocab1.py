"""VOCAB-1: where does GoldWorm v6 carry parts of speech? Frozen design: PREREG_VOCAB1.md.

Usage: python vocab1.py <model.safetensors> <TinyStories-valid.txt> <out.json>
"""

import hashlib
import json
import platform
import sys
import time

import numpy as np
import spacy
import torch
from safetensors.torch import load_file

from cal1 import OUTER, Port, sha256

SEED = 20261113
CLASSES = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX", "CCONJ", "PROPN"]
PREFIX = b"<|endoftext|>\n\n"
CTX = 256
N_NULL, N_CTRL, N_BOOT, N_RAND, INLP_ROUNDS = 200, 20, 1000, 20, 3
LBFGS_ITERS = 300
SITES = ("last", "first", "pre")
STATE_NAMES = ["embed"] + [f"{p}{i}" for i in range(6) for p in ("attn", "mlp")]


# ---------------------------------------------------------------- data

def load_stories(corpus_path):
    d = open(corpus_path, "rb").read()[OUTER[0]:OUTER[1]]
    parts = d.split(b"<|endoftext|>")[1:-1]
    out = []
    for p in parts:
        s = p.strip()
        if s and all(b < 128 for b in s):
            out.append(s.decode("ascii"))
    return out


def tag(stories):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    P = len(PREFIX)
    windows, toks = [], []  # toks: (story, first_byte, last_byte, label, form)
    for si, doc in enumerate(nlp.pipe(stories, batch_size=64)):
        w = (PREFIX + stories[si].encode("ascii"))[:CTX]
        windows.append(np.frombuffer(w, dtype=np.uint8).copy())
        for t in doc:
            if t.pos_ not in CLASSES or not t.text.strip():
                continue
            a, b = P + t.idx, P + t.idx + len(t.text) - 1
            if b >= len(w):
                continue
            toks.append((si, a, b, CLASSES.index(t.pos_), t.text.lower()))
    return windows, toks


# ---------------------------------------------------------------- model

def states_and_forward(port, idx, proj=None):
    """Run the port; return logits and the 13 cumulative residual states.
    proj: (state_index, mu[D], Q[D,r]) applied mean-preservingly to that state before continuing."""
    sd = port.sd
    B, T = idx.shape
    D = 192
    import math
    import torch.nn.functional as F
    x = sd["wte.weight"][idx] + sd["wpe.weight"][torch.arange(T)][None]
    states = []

    def maybe(x, j):
        if proj is not None and proj[0] == j:
            mu, Q = proj[1], proj[2]
            x = x - ((x - mu) @ Q) @ Q.T
        return x

    x = maybe(x, 0)
    states.append(x)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    for i in range(port.n_layer):
        h = port.ln(x, f"blocks.{i}.ln1")
        q, k, v = (h @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
        q, k, v = (t.view(B, T, 4, D // 4).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // 4)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
        x = maybe(x, 1 + 2 * i)
        states.append(x)
        h = port.ln(x, f"blocks.{i}.ln2")
        x = x + F.gelu(h @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate=port.approx) @ sd[f"blocks.{i}.mlp.proj.weight"].T
        x = maybe(x, 2 + 2 * i)
        states.append(x)
    logits = port.ln(x, "ln_f") @ sd["head.weight"].T
    return logits, states


def batches(windows, sel, bs=64):
    for s in range(0, len(sel), bs):
        ids = sel[s:s + bs]
        L = max(len(windows[i]) for i in ids)
        idx = torch.zeros(len(ids), L, dtype=torch.long)
        for j, i in enumerate(ids):
            idx[j, :len(windows[i])] = torch.from_numpy(windows[i].astype(np.int64))
        yield ids, idx


@torch.no_grad()
def collect(port, windows, toks):
    """feats[site] -> float32 [13, N, 192]; also per-state means over all positions of every story (train subset later)."""
    by_story = {}
    for n, (si, a, b, _, _) in enumerate(toks):
        by_story.setdefault(si, []).append((n, a, b))
    N = len(toks)
    feats = {s: np.zeros((13, N, 192), dtype=np.float32) for s in SITES}
    pos_sum = np.zeros((len(windows), 13, 192))  # per-story sum over positions, for train means
    pos_cnt = np.zeros(len(windows))
    for ids, idx in batches(windows, list(range(len(windows)))):
        _, st = states_and_forward(port, idx)
        S = torch.stack(st, 0)  # [13,B,T,D]
        for j, si in enumerate(ids):
            L = len(windows[si])
            pos_sum[si] = S[:, j, :L].sum(1).double().numpy()
            pos_cnt[si] = L
            for n, a, b in by_story.get(si, []):
                feats["last"][:, n] = S[:, j, b].numpy()
                feats["first"][:, n] = S[:, j, a].numpy()
                feats["pre"][:, n] = S[:, j, a - 1].numpy()
    return feats, pos_sum, pos_cnt


# ---------------------------------------------------------------- probe

def fit(X, Y, K=10):
    """Batched multinomial LR, sklearn L2 objective at C=1: sum CE + 0.5||W||^2 (bias unpenalised).
    X [n,d] float64 (already standardised), Y [B,n] long. Returns W [B,d,K], b [B,K], final grad norm per draw."""
    B = Y.shape[0]
    n, d = X.shape
    W = torch.zeros(B, d, K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(B, K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=LBFGS_ITERS, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = torch.einsum("nd,bdk->bnk", X, W) + b[:, None]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, K), Y.reshape(-1), reduction="sum") + 0.5 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    closure()
    gn = torch.sqrt(W.grad.pow(2).sum((1, 2)) + b.grad.pow(2).sum(1)).detach()
    return W.detach(), b.detach(), gn


def predict(X, W, b):
    return (torch.einsum("nd,bdk->bnk", X, W) + b[:, None]).argmax(-1)


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (torch.from_numpy((Xtr - mu) / sd), torch.from_numpy((Xte - mu) / sd), mu, sd)


def fit_chunked(Xtr, Y, chunk=50):
    Ws, bs, gns = [], [], []
    for s in range(0, Y.shape[0], chunk):
        W, b, gn = fit(Xtr, Y[s:s + chunk])
        Ws.append(W); bs.append(b); gns.append(gn)
    return torch.cat(Ws), torch.cat(bs), torch.cat(gns)


def onehot_bytes(windows, toks, site, k):
    rows = []
    for si, a, b, _, _ in toks:
        end = {"last": b, "first": a, "pre": a - 1}[site]
        w = windows[si]
        v = np.zeros(k * 256)
        for j in range(k):
            p = end - (k - 1) + j
            if p >= 0:
                v[j * 256 + int(w[p])] = 1.0
        rows.append(v)
    return np.array(rows)


def boot_ci(diff_tok, story_of, rng, n=N_BOOT):
    """diff_tok: per-test-token (state_correct - surface_correct) on novel tokens; resample stories."""
    stories = np.unique(story_of)
    idx_by = {s: np.where(story_of == s)[0] for s in stories}
    tot = np.array([diff_tok[idx_by[s]].sum() for s in stories])
    cnt = np.array([len(idx_by[s]) for s in stories])
    vals = []
    for _ in range(n):
        r = rng.integers(0, len(stories), len(stories))
        vals.append(tot[r].sum() / max(cnt[r].sum(), 1))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------- amnesic

def removed_basis(W_std, sd):
    """Rowspace directions of a standardised-space probe, in raw state space: columns of diag(1/sd) W, centred over
    classes (softmax invariance), orthonormalised; rank <= K-1."""
    Wr = W_std / sd[:, None]
    Wr = Wr - Wr.mean(1, keepdims=True)
    U, S, _ = np.linalg.svd(Wr, full_matrices=False)
    r = int((S > S.max() * 1e-8).sum())
    return U[:, :r]


@torch.no_grad()
def loss_sets(port, windows, test_sids, Wpos, proj=None):
    """Mean next-byte loss at W positions and O positions over test stories."""
    sW = nW = sO = nO = 0.0
    P = len(PREFIX)
    for ids, idx in batches(windows, test_sids):
        logits, _ = states_and_forward(port, idx, proj)
        lp = logits.log_softmax(-1)
        for j, si in enumerate(ids):
            L = len(windows[si])
            t = torch.arange(P - 1, L - 1)
            nll = -lp[j, t, idx[j, t + 1]].double().numpy()
            isW = np.isin(t.numpy(), Wpos.get(si, []))
            sW += nll[isW].sum(); nW += isW.sum()
            sO += nll[~isW].sum(); nO += (~isW).sum()
    return sW / nW, sO / nO


# ---------------------------------------------------------------- main

def main(model_path, corpus_path, out_path):
    torch.set_num_threads(16)
    torch.manual_seed(SEED)
    t0 = time.time()
    port = Port({k: v.float() for k, v in load_file(model_path).items()}, "tanh")
    stories = load_stories(corpus_path)
    windows, toks = tag(stories)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(stories))
    train_s = set(perm[: int(0.8 * len(stories))].tolist())
    story_of = np.array([t[0] for t in toks])
    y = np.array([t[3] for t in toks])
    forms = [t[4] for t in toks]
    tr = np.array([s in train_s for s in story_of])
    te = ~tr
    train_forms = {f for f, m in zip(forms, tr) if m}
    novel = te & np.array([f not in train_forms for f in forms])
    nov_in_te = novel[te]
    ytr, yte = torch.from_numpy(y[tr]), torch.from_numpy(y[te])
    print(f"stories {len(stories)} tokens {len(toks)} train {tr.sum()} test {te.sum()} novel {novel.sum()}", flush=True)

    feats, pos_sum, pos_cnt = collect(port, windows, toks)
    train_ids = sorted(train_s)
    state_mu = pos_sum[train_ids].sum(0) / pos_cnt[train_ids].sum()  # [13, D]

    # labels for nulls and controls
    null_Y = torch.stack([ytr[torch.from_numpy(rng.permutation(len(ytr)))] for _ in range(N_NULL)])
    prior = np.bincount(y[tr], minlength=10) / tr.sum()
    types = sorted(set(forms))
    ctrl_lab = [dict(zip(types, rng.choice(10, size=len(types), p=prior))) for _ in range(N_CTRL)]
    ctrl_all = np.array([[c[f] for f in forms] for c in ctrl_lab])  # [N_CTRL, N]
    ctrl_Ytr = torch.from_numpy(ctrl_all[:, tr])
    ctrl_Yte = torch.from_numpy(ctrl_all[:, te])

    surface = {}
    for site, k in (("last", 4), ("first", 5), ("pre", 4)):
        Xs = onehot_bytes(windows, toks, site, k)
        Xtr_, Xte_, _, _ = standardise(Xs[tr], Xs[te])
        W, b, gn = fit(Xtr_, ytr[None])
        pred = predict(Xte_, W, b)[0]
        surface[site] = {"correct": (pred == yte).numpy(), "acc_all": float((pred == yte).double().mean()),
                         "acc_novel": float((pred == yte)[torch.from_numpy(nov_in_te)].double().mean()),
                         "grad_norm": float(gn[0])}
        print(f"surface {site}: all {surface[site]['acc_all']:.3f} novel {surface[site]['acc_novel']:.3f}", flush=True)

    cells = {}
    boot_rng = np.random.default_rng(SEED + 1)
    te_story = story_of[te]
    for site in SITES:
        for j, sn in enumerate(STATE_NAMES):
            X = feats[site][j].astype(np.float64)
            Xtr_, Xte_, mu, sd = standardise(X[tr], X[te])
            W, b, gn = fit(Xtr_, ytr[None])
            pred = predict(Xte_, W, b)[0]
            corr = (pred == yte).numpy()
            acc = float(corr.mean())
            acc_nov = float(corr[nov_in_te].mean())
            Wn, bn, gnn = fit_chunked(Xtr_, null_Y)
            null_acc = (predict(Xte_, Wn, bn) == yte[None]).double().mean(1).numpy()
            Wc, bc, gnc = fit_chunked(Xtr_, ctrl_Ytr)
            ctrl_acc = (predict(Xte_, Wc, bc) == ctrl_Yte).double().mean(1).numpy()
            diff = corr.astype(float) - surface[site]["correct"].astype(float)
            ci = boot_ci(diff[nov_in_te], te_story[nov_in_te], boot_rng)
            c = {"acc_all": acc, "acc_novel": acc_nov, "grad_norm": float(gn[0]),
                 "V1": {"null_p95": float(np.percentile(null_acc, 95)), "null_max": float(null_acc.max()),
                        "null_grad_norm_max": float(gnn.max()), "pass": acc > float(np.percentile(null_acc, 95))},
                 "V2": {"ctrl_accs": ctrl_acc.tolist(), "ctrl_max": float(ctrl_acc.max()), "selectivity": acc - float(ctrl_acc.mean()),
                        "ctrl_grad_norm_max": float(gnc.max()), "pass": acc > float(ctrl_acc.max())},
                 "V3": {"surface_acc_novel": surface[site]["acc_novel"], "diff_novel": acc_nov - surface[site]["acc_novel"],
                        "ci95": ci, "pass": ci[0] > 0}}
            c["carried_beyond_surface"] = c["V1"]["pass"] and c["V2"]["pass"] and c["V3"]["pass"]
            cells[f"{site}/{sn}"] = c
            print(f"{site}/{sn}: acc {acc:.3f} nov {acc_nov:.3f} null95 {c['V1']['null_p95']:.3f} ctrlmax {c['V2']['ctrl_max']:.3f} "
                  f"Δnov {c['V3']['diff_novel']:+.3f} ci {ci[0]:+.3f},{ci[1]:+.3f} -> {c['carried_beyond_surface']}  [{time.time()-t0:.0f}s]", flush=True)

    # amnesic test, pre-site probe, every state
    test_sids = sorted(set(range(len(stories))) - train_s)
    Wpos = {}
    for (si, a, _, _, _), m in zip(toks, te):
        if m:
            Wpos.setdefault(si, []).append(a - 1)
    base_W, base_O = loss_sets(port, windows, test_sids, Wpos)
    amnesic = {"base": {"W": base_W, "O": base_O}}
    rrng = np.random.default_rng(SEED + 2)
    for j, sn in enumerate(STATE_NAMES):
        X = feats["pre"][j].astype(np.float64)
        Xtr_raw, Xte_raw = X[tr].copy(), X[te].copy()
        mu_state = state_mu[j]
        Qs = np.zeros((192, 0))
        for _ in range(INLP_ROUNDS):
            Xtr_, _, _, sd = standardise(Xtr_raw, Xte_raw)
            W, b, _ = fit(Xtr_, ytr[None])
            U = removed_basis(W[0].numpy(), sd)
            U = U - Qs @ (Qs.T @ U)
            Qn, _ = np.linalg.qr(U)
            Qs = np.concatenate([Qs, Qn], 1)
            Xtr_raw = X[tr] - ((X[tr] - mu_state) @ Qs) @ Qs.T
            Xte_raw = X[te] - ((X[te] - mu_state) @ Qs) @ Qs.T
        Xtr_, Xte_, _, _ = standardise(Xtr_raw, Xte_raw)
        W, b, _ = fit(Xtr_, ytr[None])
        post_acc = float((predict(Xte_, W, b)[0] == yte).double().mean())
        r = Qs.shape[1]
        mu_t = torch.from_numpy(mu_state).float()
        dW, dO = loss_sets(port, windows, test_sids, Wpos, (j, mu_t, torch.from_numpy(Qs).float()))
        rand = []
        for _ in range(N_RAND):
            Qr, _ = np.linalg.qr(rrng.standard_normal((192, r)))
            rand.append(loss_sets(port, windows, test_sids, Wpos, (j, mu_t, torch.from_numpy(Qr).float())))
        rW = np.array([a for a, _ in rand]) - base_W
        rO = np.array([o for _, o in rand]) - base_O
        realW, realO = dW - base_W, dO - base_O
        a1 = bool(realW > rW.max())
        a2 = bool((realW - rW.mean()) > (realO - rO.mean()))
        amnesic[sn] = {"rank_removed": r, "probe_acc_after": post_acc, "dW": realW, "dO": realO,
                       "rand_dW": rW.tolist(), "rand_dO": rO.tolist(), "A1": a1, "A2": a2}
        print(f"amnesic {sn}: rank {r} after {post_acc:.3f} dW {realW:+.4f} (rand max {rW.max():+.4f}) dO {realO:+.4f} "
              f"(rand mean {rO.mean():+.4f}) A1 {a1} A2 {a2}  [{time.time()-t0:.0f}s]", flush=True)

    res = {
        "prereg_sha256": sha256(__file__.replace("vocab1.py", "PREREG_VOCAB1.md")),
        "script_sha256": sha256(__file__),
        "model_sha256": sha256(model_path), "corpus_sha256": sha256(corpus_path),
        "counts": {"stories": len(stories), "tokens": len(toks), "train": int(tr.sum()), "test": int(te.sum()),
                   "novel_test": int(novel.sum()), "label_counts_train": dict(zip(CLASSES, np.bincount(y[tr], minlength=10).tolist())),
                   "label_counts_test": dict(zip(CLASSES, np.bincount(y[te], minlength=10).tolist()))},
        "majority_acc_test": float((yte == int(np.bincount(y[tr]).argmax())).double().mean()),
        "surface": {s: {k: v for k, v in d.items() if k != "correct"} for s, d in surface.items()},
        "cells": cells, "amnesic": amnesic,
        "env": {"torch": torch.__version__, "numpy": np.__version__, "spacy": spacy.__version__,
                "spacy_model": spacy.load("en_core_web_sm").meta["version"], "python": platform.python_version(),
                "platform": platform.platform(), "threads": torch.get_num_threads()},
        "wall_sec": time.time() - t0,
    }
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
