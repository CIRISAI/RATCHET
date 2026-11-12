"""CAL-G: calibrate the instrument on Gemma-4-E2B-it. Frozen design: PREREG_CALG.md.
Usage: python calg.py <gemma4 dir> <TinyStories-valid.txt> <fresh slice> <out.json>

Read-only taps on the reference implementation (transformers 5.22.2). The per-layer token embedding table stays on the
CPU (bf16 weights; lookups upcast); everything else runs on the GPU in float32 if it fits (C0), else bfloat16.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib, json, math, sys, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_SHA = "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
TOK_SHA = "ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2"
FRESH_SHA = "3232df12d6bdd8f83071f0d8196ad10128624168e228aaf9a23f6a1f8431ee8a"
N_STORIES, CTX = 64, 256


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main(mdir, valid, fresh, out_path):
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time()-t0:.0f}s]", flush=True)  # noqa: E731
    res = {"C1": {"model": sha(os.path.join(mdir, "model.safetensors")) == MODEL_SHA,
                  "tokenizer": sha(os.path.join(mdir, "tokenizer.json")) == TOK_SHA, "fresh": sha(fresh) == FRESH_SHA}}
    res["C1"]["pass"] = all(res["C1"].values())
    log(f"C1 {res['C1']}")
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    import transformers
    tok = AutoTokenizer.from_pretrained(mdir)
    full = Gemma4ForConditionalGeneration.from_pretrained(mdir, dtype=torch.bfloat16, device_map="cpu")
    text = full.model.language_model
    head_w = full.lm_head.weight  # tied to text.embed_tokens.weight
    softcap = full.config.get_text_config().final_logit_softcapping
    ple = text.embed_tokens_per_layer  # stays on CPU
    text.embed_tokens_per_layer = None
    dev = torch.device("cuda")
    free = torch.cuda.mem_get_info()[0]
    nbytes = sum(p.numel() for n, p in text.named_parameters()) * 4
    dtype = torch.float32 if nbytes < 0.8 * free else torch.bfloat16
    text.to(dev, dtype)
    head = head_w.to(dev, dtype)
    res["C0"] = {"dtype": str(dtype), "text_params_on_gpu": sum(p.numel() for p in text.parameters()),
                 "gpu_free_bytes_before": int(free), "transformers": transformers.__version__, "torch": torch.__version__,
                 "gpu": torch.cuda.get_device_name(0)}
    log(f"C0 {res['C0']}")
    del full
    # ---- text: FRESH-1 kept stories 8000..8063
    vset = set(p.strip() for p in open(valid, "rb").read().split(b"<|endoftext|>") if p.strip())
    parts = open(fresh, "rb").read().split(b"<|endoftext|>")[1:-1]
    kept = [p.strip() for p in parts if p.strip() and all(c < 128 for c in p.strip()) and p.strip() not in vset]
    stories = [p.decode("ascii") for p in kept[8000:8000 + N_STORIES]]
    enc = [tok(s, add_special_tokens=True)["input_ids"][:CTX] for s in stories]
    assert all(e[0] == tok.bos_token_id for e in enc), "BOS not prepended"
    L = max(len(e) for e in enc)
    ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(enc), L, dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e); mask[i, :len(e)] = 1
    lens = [len(e) for e in enc]

    nl = text.config.num_hidden_layers
    scal = [float(text.layers[i].layer_scalar.item()) for i in range(nl)]
    res["layer_scalars"] = scal
    log(f"layer_scalar: min {min(scal):.4f} max {max(scal):.4f} prod {math.prod(scal):.3e}")

    taps, handles = {}, []

    def tap(name):
        def f(mod, inp, out):
            taps.setdefault(name, []).append(out.detach().float().cpu())
        return f

    def install():
        for i, layer in enumerate(text.layers[:nl]):
            handles.append(layer.post_attention_layernorm.register_forward_hook(tap(("a", i))))
            handles.append(layer.post_feedforward_layernorm.register_forward_hook(tap(("m", i))))
            handles.append(layer.post_per_layer_input_norm.register_forward_hook(tap(("p", i))))
            handles.append(layer.register_forward_hook(tap(("out", i))))
            handles.append(layer.register_forward_pre_hook(lambda mod, args, i=i: taps.setdefault(("in", i), []).append(
                args[0].detach().float().cpu())))
        handles.append(text.norm.register_forward_pre_hook(lambda mod, args: taps.setdefault("prenorm", []).append(
            args[0].detach().float().cpu())))

    @torch.no_grad()
    def run(batch_ids, batch_mask, zero_layer=None):
        emb = text.embed_tokens(batch_ids.to(dev))
        pl = ple(batch_ids).reshape(*batch_ids.shape, nl, text.config.hidden_size_per_layer_input).to(dev, dtype)
        hz = []
        if zero_layer is not None:
            lay = text.layers[zero_layer]
            for m in (lay.post_attention_layernorm, lay.post_feedforward_layernorm, lay.post_per_layer_input_norm):
                hz.append(m.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out)))
        out = text(inputs_embeds=emb, per_layer_inputs=pl, attention_mask=batch_mask.to(dev), use_cache=False)
        for h in hz:
            h.remove()
        h = out.last_hidden_state
        pre = h @ head.T
        return emb, pre, torch.tanh(pre / softcap) * softcap

    B = 8
    valid = torch.zeros(ids.shape, dtype=torch.bool)
    for i, n in enumerate(lens):
        valid[i, :n] = True
    gsel = torch.Generator().manual_seed(1)
    vpos = torch.nonzero(valid)
    pos4 = {tuple(x) for x in vpos[torch.randperm(len(vpos), generator=gsel)[:256]].tolist()}
    m_next = valid.clone(); m_next[:, -1] = False
    for i, n in enumerate(lens):
        m_next[i, n - 1] = False
    npos = torch.nonzero(m_next)
    rsel = {tuple(x) for x in npos[torch.randperm(len(npos), generator=torch.Generator().manual_seed(2))[:512]].tolist()}
    tail = [math.prod(scal[k:]) for k in range(nl)]
    g = text.norm.weight.detach().float()
    eps = text.norm.eps
    bar = 1e-5 if dtype == torch.float32 else 1e-2
    bar4 = 1e-3 if dtype == torch.float32 else 1e-1
    st = {"c2": True, "c5": True, "c3_num": 0.0, "c3w_num": 0.0, "den": 0.0, "local": 0.0, "c4": 0.0,
          "nll_sum": 0.0, "nll_n": 0, "r1_lin": {i: [] for i in range(nl)}, "r1_base": [], "r1_where": []}
    for s0 in range(0, len(enc), B):
        bi, bm = ids[s0:s0 + B], mask[s0:s0 + B]
        _, _, ref = run(bi, bm)
        taps.clear(); install()
        emb, pre, lg = run(bi, bm)
        for h in handles:
            h.remove()
        handles.clear()
        st["c2"] &= torch.equal(ref, lg)
        st["c5"] &= torch.equal(ref, run(bi, bm)[2])
        v = valid[s0:s0 + B].to(dev)
        x0 = emb.double(); prenorm = taps["prenorm"][0].double().to(dev)
        recon = tail[0] * x0; wrong = x0.clone()
        W = {}
        for i in range(nl):
            w = sum(taps[(k, i)][0].double().to(dev) for k in ("a", "m", "p"))
            W[i] = w
            recon += tail[i] * w; wrong += w
            loc = scal[i] * (taps[("in", i)][0].double().to(dev) + w) - taps[("out", i)][0].double().to(dev)
            st["local"] = max(st["local"], float(loc[v].abs().max() / taps[("out", i)][0].double().to(dev)[v].abs().max()))
        st["den"] = max(st["den"], float(prenorm[v].abs().max()))
        st["c3_num"] = max(st["c3_num"], float((recon - prenorm)[v].abs().max()))
        st["c3w_num"] = max(st["c3w_num"], float((wrong - prenorm)[v].abs().max()))
        # C4 on the selected positions in this batch
        sel = [(r, p) for r in range(bi.shape[0]) for p in range(bi.shape[1]) if (s0 + r, p) in pos4]
        if sel:
            rt = torch.tensor([a for a, _ in sel], device=dev); pt = torch.tensor([b for _, b in sel], device=dev)
            xs = prenorm[rt, pt]
            rr = (xs.pow(2).mean(-1, keepdim=True) + eps).pow(-0.5)
            terms = tail[0] * x0[rt, pt] + sum(tail[i] * W[i][rt, pt] for i in range(nl))
            contrib = ((terms * rr).float() * g) @ head.T.float()
            st["c4"] = max(st["c4"], float((contrib.double() - pre[rt, pt].double()).abs().max()))
        # NLL and R1 (direct prediction per layer) on this batch
        lpf = torch.log_softmax(ref.double(), -1)
        for r in range(bi.shape[0]):
            n = lens[s0 + r]
            t = bi[r, 1:n].to(dev)
            st["nll_sum"] += float(-lpf[r, :n - 1].gather(-1, t[:, None]).sum()); st["nll_n"] += n - 1
        sel = [(r, p) for r in range(bi.shape[0]) for p in range(bi.shape[1]) if (s0 + r, p) in rsel]
        if sel:
            rt = torch.tensor([a for a, _ in sel], device=dev); pt = torch.tensor([b for _, b in sel], device=dev)
            tgt = bi.to(dev)[rt, pt + 1]
            st["r1_base"].append(lpf[rt, pt, tgt].cpu()); st["r1_where"] += [(s0 + a, b) for a, b in sel]
            xs = prenorm[rt, pt]; rr = (xs.pow(2).mean(-1, keepdim=True) + eps).pow(-0.5)
            for i in range(nl):
                pre_lin = pre[rt, pt].double() - ((tail[i] * W[i][rt, pt] * rr).float() * g @ head.T.float()).double()
                st["r1_lin"][i].append(torch.log_softmax(torch.tanh(pre_lin / softcap) * softcap, -1)
                                       [torch.arange(len(sel), device=dev), tgt].cpu())
        del W, recon, wrong, prenorm, x0
        taps.clear()
        log(f"batch {s0 // B + 1}/{math.ceil(len(enc) / B)}")
    res["C2"] = {"identical": st["c2"], "pass": st["c2"]}
    res["C5"] = {"identical": st["c5"], "pass": st["c5"]}
    c3, c3w = st["c3_num"] / st["den"], st["c3w_num"] / st["den"]
    res["C3"] = {"max_rel_err": c3, "per_layer_local_max_rel_err": st["local"], "pass": c3 <= bar, "bar": bar}
    res["C3_wrong_frame"] = {"max_rel_err": c3w, "fails": c3w > bar, "all_scalars_one": all(abs(x - 1) < 1e-6 for x in scal)}
    res["C4"] = {"max_abs_err": st["c4"], "pass": st["c4"] <= bar4, "bar": bar4}
    res["mean_nll_nats"] = st["nll_sum"] / st["nll_n"]
    log(f"C2 {res['C2']} C5 {res['C5']} C3 {res['C3']} wrong {res['C3_wrong_frame']} C4 {res['C4']} "
        f"NLL {res['mean_nll_nats']:.4f}")
    # R1 realised: zero each layer's three writes, rerun all batches
    base = torch.cat(st["r1_base"]).numpy()
    where = st["r1_where"]
    r1 = {}
    for i in range(nl):
        real = []
        for s0 in range(0, len(enc), B):
            bi, bm = ids[s0:s0 + B], mask[s0:s0 + B]
            z = torch.log_softmax(run(bi, bm, zero_layer=i)[2].double(), -1)
            for (a, b) in where:
                if s0 <= a < s0 + B:
                    real.append(float(z[a - s0, b, int(ids[a, b + 1])]))
        lin = torch.cat(st["r1_lin"][i]).numpy()
        a_, b_ = lin - base, np.array(real) - base
        r1[i] = {"corr": float(np.corrcoef(a_, b_)[0, 1]), "mean_abs_real": float(np.abs(b_).mean()),
                 "mean_abs_direct": float(np.abs(a_).mean())}
    res["R1"] = r1
    log("R1 corr: " + " ".join(f"{i}:{v['corr']:.2f}" for i, v in r1.items()))
    res["PASS"] = all(res[c]["pass"] for c in ("C1", "C2", "C3", "C4", "C5"))
    res["gguf_top1"] = "not run: llama.cpp runtime not installed in this environment"
    res.update({"prereg_sha256": sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PREREG_CALG.md")),
                "script_sha256": sha(os.path.abspath(__file__)), "wall_sec": time.time() - t0})
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    log(f"PASS {res['PASS']}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
