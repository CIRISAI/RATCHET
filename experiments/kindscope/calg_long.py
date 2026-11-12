"""CAL-G-LONG: Gemma instrument at the real stage-1 prompt length with a cached prefix. Frozen design: PREREG_CALG_LONG.md.
Usage: python calg_long.py <gemma4 dir> <rendered dir> <out.json>"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import copy, hashlib, json, math, sys, time  # noqa: E401,E402
import torch  # noqa: E402

PREFIX_SHA = "92f645c0accf794501e8bd490f12b1c6765a6563d138c8269283b5522dc4faef"
FIT_SHA = "3b4c9325db39df64dafe5f643ee6c664dbcc0723c60303f200c162abf0d0624c"
N_ITEMS = 8


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main(mdir, rdir, out_path):
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time()-t0:.0f}s]", flush=True)  # noqa: E731
    res = {"L1": {"prefix": sha(os.path.join(rdir, "prefix.json")) == PREFIX_SHA,
                  "fit": sha(os.path.join(rdir, "fit.jsonl")) == FIT_SHA}}
    res["L1"]["pass"] = all(res["L1"].values())
    prefix = json.load(open(os.path.join(rdir, "prefix.json")))
    rows = []
    with open(os.path.join(rdir, "fit.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == N_ITEMS:
                break
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(mdir)
    ids = [tok.apply_chat_template(prefix + [{"role": "user", "content": r["user"]}], add_generation_prompt=True,
                                   tokenize=True, return_dict=False) for r in rows]
    ids = [x["input_ids"] if isinstance(x, dict) else list(x) for x in ids]
    P = 0
    while all(len(x) > P and x[P] == ids[0][P] for x in ids):
        P += 1
    res["tokens"] = {"shared_prefix": P, "items": [len(x) for x in ids]}
    log(f"L1 {res['L1']} tokens {res['tokens']}")
    full = Gemma4ForConditionalGeneration.from_pretrained(mdir, dtype=torch.bfloat16, device_map="cpu")
    text = full.model.language_model
    # Amendment 3: SDPA after all (eager's full-attention score matrices peaked the GPU at 15.96 GB); L0 bridges kernels
    def short_logits(impl, toks):
        text.config._attn_implementation = impl
        t = torch.tensor([toks], dtype=torch.long)
        emb = text.embed_tokens(t.to(dev))
        pl = ple(t).reshape(1, len(toks), text.config.num_hidden_layers, text.config.hidden_size_per_layer_input).to(dev, torch.float32)
        return text(inputs_embeds=emb, per_layer_inputs=pl, use_cache=False).last_hidden_state[0] @ text.embed_tokens.weight.detach().T
    ple = text.embed_tokens_per_layer; text.embed_tokens_per_layer = None
    softcap = full.config.get_text_config().final_logit_softcapping
    dev = torch.device("cuda")
    text.to(dev, torch.float32)
    # ---- L0: SDPA vs eager at CAL-G's regime (first 2 stories of CAL-G's FRESH-1 set, 256 tokens, BOS explicit)
    vset_path = os.environ.get("KS_VALID", "/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad/gw_data/TinyStories-valid.txt")
    fresh_path = os.environ.get("KS_FRESH", "/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad/gw_fresh/TinyStories-train.head20MiB.txt")
    vset = set(p.strip() for p in open(vset_path, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh_path, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    l0 = 0.0
    for st_ in kept[8000:8002]:
        tk = ([tok.bos_token_id] + tok(st_.decode("ascii"), add_special_tokens=False)["input_ids"])[:256]
        a_ = short_logits("eager", tk); b_ = short_logits("sdpa", tk)
        l0 = max(l0, float((a_ - b_).abs().max())); del a_, b_
    text.config._attn_implementation = "sdpa"
    res["L0"] = {"max_abs_sdpa_vs_eager_256": l0, "pass": l0 <= 1e-3}
    print(f"L0 {res['L0']}", flush=True)
    torch.cuda.empty_cache()
    head = text.embed_tokens.weight.detach()
    nl = text.config.num_hidden_layers
    hpl = text.config.hidden_size_per_layer_input
    scal = [float(text.layers[i].layer_scalar.item()) for i in range(nl)]
    tail = [math.prod(scal[k:]) for k in range(nl)]
    del full

    def run(tok_ids, cache=None, use_cache=False, last=32):
        """Logits only for the last `last` positions (Amendment 2: full-vocab logits over 20k positions are 21 GB)."""
        t = torch.tensor([tok_ids], dtype=torch.long)
        emb = text.embed_tokens(t.to(dev))
        pl = ple(t).reshape(1, len(tok_ids), nl, hpl).to(dev, torch.float32)
        out = text(inputs_embeds=emb, per_layer_inputs=pl, past_key_values=cache, use_cache=use_cache)
        lg = None if last == 0 else out.last_hidden_state[:, -last:] @ head.T
        return emb, lg, out

    def prefill(tok_ids, chunk=2048):
        """Amendment 4: chunked prefill through the cache."""
        cache = None
        for s0 in range(0, len(tok_ids), chunk):
            _, _, o = run(tok_ids[s0:s0 + chunk], cache=cache, use_cache=True, last=0)
            cache = o.past_key_values
            del o; torch.cuda.empty_cache()
        return cache

    # ---- shared prefix, once (chunked)
    torch.cuda.reset_peak_memory_stats()
    pcache = prefill(ids[0][:P])
    res["prefix_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    log(f"prefix cached ({P} tokens), peak {res['prefix_peak_gb']:.2f} GB")
    # ---- L2: cache equivalence on 2 items
    l2 = 0.0
    for k in range(2):  # Amendment 4: at 6,144 tokens, chunked prefill of 6,112 + cached last 32 vs one uncached pass
        short = ids[k][:6144]
        c_ = prefill(short[:6112])
        _, pre_c, _ = run(short[6112:], cache=c_, use_cache=True)
        del c_; torch.cuda.empty_cache()
        _, pre_f, _ = run(short, use_cache=False)
        l2 = max(l2, float((pre_c[0] - pre_f[0]).abs().max()))  # both are the last 32 positions
        del pre_c, pre_f
        torch.cuda.empty_cache()
    res["L2"] = {"max_abs": l2, "pass": l2 <= 1e-3}
    log(f"L2 {res['L2']} peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # ---- L3/L4/L5 on 8 items, cached, with taps on the item's own positions
    taps, handles = {}, []

    def tap(name):
        def f(mod, inp, out):
            taps[name] = out.detach().double()
        return f
    for i, layer in enumerate(text.layers[:nl]):
        handles.append(layer.post_attention_layernorm.register_forward_hook(tap(("a", i))))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(tap(("m", i))))
        handles.append(layer.post_per_layer_input_norm.register_forward_hook(tap(("p", i))))
    handles.append(text.norm.register_forward_pre_hook(lambda mod, args: taps.__setitem__("prenorm", args[0].detach().double())))
    num3 = num3w = den = c4 = 0.0
    det = True
    g = text.norm.weight.detach().float(); eps = text.norm.eps
    secs = []
    for k in range(N_ITEMS):
        ts = time.time()
        emb, pre, _ = run(ids[k][P:], cache=copy.deepcopy(pcache), use_cache=True)
        secs.append(time.time() - ts)
        x0 = emb.double()
        recon = tail[0] * x0; wrong = x0.clone()
        W = {i: taps[("a", i)] + taps[("m", i)] + taps[("p", i)] for i in range(nl)}
        for i in range(nl):
            recon = recon + tail[i] * W[i]; wrong = wrong + W[i]
        pn = taps["prenorm"]
        # the item positions' prenorm includes the prefix's contribution through attention, but the frame is per position:
        # x_final(t) = (prod s) x0(t) + sum_L (prod_{k>=L} s_k) w_L(t), with w_L(t) the writes at t (which READ the prefix)
        den = max(den, float(pn.abs().max()))
        num3 = max(num3, float((recon - pn).abs().max())); num3w = max(num3w, float((wrong - pn).abs().max()))
        sel = torch.arange(max(0, pn.shape[1] - 8), pn.shape[1], device=dev)
        xs = pn[0, sel]; rr = (xs.pow(2).mean(-1, keepdim=True) + eps).pow(-0.5)
        terms = tail[0] * x0[0, sel] + sum(tail[i] * W[i][0, sel] for i in range(nl))
        c4 = max(c4, float((((terms * rr).float() * g) @ head.T - pre[0, -8:]).abs().max()))
        _, pre2, _ = run(ids[k][P:], cache=copy.deepcopy(pcache), use_cache=True)
        det &= bool(torch.equal(pre, pre2))
        del emb, pre, pre2, W, recon, wrong, pn, x0
        taps.clear(); torch.cuda.empty_cache()
        log(f"item {k}: {len(ids[k]) - P} own tokens, {secs[-1]:.1f}s")
    for h in handles:
        h.remove()
    res["L3"] = {"max_rel": num3 / den, "pass": num3 / den <= 1e-5}
    res["L3_wrong_frame"] = {"max_rel": num3w / den, "fails": num3w / den > 1e-5}
    res["L4"] = {"max_abs": c4, "pass": c4 <= 1e-3, "positions": 8 * N_ITEMS}
    res["L5"] = {"identical": det, "pass": det}
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["seconds_per_item_cached"] = secs
    res["PASS"] = all(res[c]["pass"] for c in ("L0", "L1", "L2", "L3", "L4", "L5")) and res["L3_wrong_frame"]["fails"]
    res["attn_implementation"] = str(getattr(text.config, "_attn_implementation", None))
    res.update({"prereg_sha256": sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_CALG_LONG.md")),
                "script_sha256": sha(os.path.abspath(__file__)), "wall_sec": time.time() - t0})
    log(json.dumps({k: res[k] for k in ("L2", "L3", "L3_wrong_frame", "L4", "L5", "peak_gpu_gb", "PASS")}))
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:4])
