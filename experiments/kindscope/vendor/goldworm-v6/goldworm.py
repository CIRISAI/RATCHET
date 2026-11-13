"""GoldWorm v6, self-contained: load the vendored weights and run them. Needs only torch and safetensors; runs on CPU.

This is the kindscope PORT, written from the checkpoint's tensor layout (no upstream code is used or executed):
pre-LN GPT-2 style, bytes in (vocab 256), context 256, width 192, 6 blocks, 4 heads x 48, bias-free linears,
tanh GELU, LayerNorm eps 1e-5, output table tied to the input embedding. CAL-1 (../../CAL1_RESULTS.md) matched the
published bits per byte with this port.

  python goldworm.py check                 verify the weight file's sha256 and print its tensors
  python goldworm.py score FILE            bits per byte of a text file (non-overlapping 256-byte windows)
  python goldworm.py generate "Once upon"  greedy or sampled continuation (--n BYTES, --temp T)

In code:
  import goldworm as gw
  sd = gw.load()                                   # dict of tensors; edit them freely
  logits, writes = gw.forward(sd, gw.encode("Once upon a time"))   # logits [1, T, 256]; writes = each block's
                                                                   # addition to the stream, by name
"""
import hashlib, math, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "model.safetensors")
SHA256 = "31729a9deb9160b045d399e2775c18972ab7ed7465eb8411b05eb801b3753521"
CTX, D, NL, NH = 256, 192, 6, 4


def sha256(path=WEIGHTS):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path=WEIGHTS, device="cpu", dtype=torch.float32, verify=True):
    from safetensors.torch import load_file
    if verify and path == WEIGHTS and sha256(path) != SHA256:
        raise SystemExit("goldworm: model.safetensors does not match the pinned sha256")
    sd = {k: v.to(device=device, dtype=dtype) for k, v in load_file(path).items()}
    if "head.weight" in sd and torch.equal(sd["head.weight"], sd["wte.weight"]):
        del sd["head.weight"]                      # tied: one table serves as input embedding and output table
    return sd


def encode(text, device="cpu"):
    b = text if isinstance(text, bytes) else text.encode("utf-8")
    return torch.tensor(list(b[-CTX:]), dtype=torch.long, device=device)[None]


def forward(sd, idx):
    """idx [B, T<=256] of byte values. Returns (logits [B, T, 256], writes: name -> [B, T, 192])."""
    B, T = idx.shape
    ln = lambda x, n: F.layer_norm(x, (D,), sd[f"{n}.weight"], sd[f"{n}.bias"], 1e-5)  # noqa: E731
    writes = {"wte": sd["wte.weight"][idx], "wpe": sd["wpe.weight"][torch.arange(T, device=idx.device)][None]}
    x = writes["wte"] + writes["wpe"]
    mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    for i in range(NL):
        q, k, v = (ln(x, f"blocks.{i}.ln1") @ sd[f"blocks.{i}.attn.qkv.weight"].T).split(D, dim=2)
        q, k, v = (t.view(B, T, NH, D // NH).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // NH)).masked_fill(~mask, float("-inf")).softmax(-1)
        writes[f"attn{i}"] = (att @ v).transpose(1, 2).reshape(B, T, D) @ sd[f"blocks.{i}.attn.proj.weight"].T
        x = x + writes[f"attn{i}"]
        h = F.gelu(ln(x, f"blocks.{i}.ln2") @ sd[f"blocks.{i}.mlp.fc.weight"].T, approximate="tanh")
        writes[f"mlp{i}"] = h @ sd[f"blocks.{i}.mlp.proj.weight"].T
        x = x + writes[f"mlp{i}"]
    head = sd["head.weight"] if "head.weight" in sd else sd["wte.weight"]
    return ln(x, "ln_f") @ head.T, writes


@torch.no_grad()
def bits_per_byte(sd, data):
    """Mean -log2 p(next byte) over non-overlapping 256-byte windows (the first byte of each window is not scored)."""
    tot, n = 0.0, 0
    for s in range(0, len(data) - 1, CTX):
        w = data[s:s + CTX]
        if len(w) < 2:
            break
        idx = torch.tensor(list(w), dtype=torch.long)[None]
        lg = forward(sd, idx[:, :-1])[0]
        tot += float(F.cross_entropy(lg[0].double(), idx[0, 1:], reduction="sum")); n += idx.shape[1] - 1
    return tot / n / math.log(2)


@torch.no_grad()
def generate(sd, prompt, n=200, temp=0.0, seed=0):
    g = torch.Generator().manual_seed(seed); out = bytearray(prompt.encode("utf-8"))
    for _ in range(n):
        lg = forward(sd, encode(bytes(out)))[0][0, -1]
        nb = int(lg.argmax()) if temp <= 0 else int(torch.multinomial((lg / temp).softmax(-1), 1, generator=g))
        out.append(nb)
    return out.decode("utf-8", "replace")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "check":
        got = sha256(); print("sha256", got, "OK" if got == SHA256 else "MISMATCH")
        sd = load(); print("tensors", len(sd), "parameters", sum(v.numel() for v in sd.values()))
        for k, v in sd.items(): print(f"  {k:32s} {tuple(v.shape)}")
    elif a[0] == "score":
        print(f"{bits_per_byte(load(), open(a[1], 'rb').read()):.4f} bits per byte")
    elif a[0] == "generate":
        n = int(a[a.index("--n") + 1]) if "--n" in a else 200; t = float(a[a.index("--temp") + 1]) if "--temp" in a else 0.0
        print(generate(load(), a[1], n=n, temp=t))
