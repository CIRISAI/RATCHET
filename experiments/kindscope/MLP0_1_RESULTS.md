# MLP0-1 results — mlp0 is a lexical lookup at a word's last byte

Preregistration `PREREG_MLP0_1.md` (sha256 in `PREREG_MLP0_1.sha256`, frozen at RATCHET `ab73e23`); script `mlp0_1.py`;
raw `mlp0_1_result.json`; figures `figs/mlp0_1_*.png` (`figs_mlp0_1.py`). All **measured**: GoldWorm v6 base, one
checkpoint, outer_val EXPLORE split only (CONFIRM unread), CUDA RTX 4090 Laptop (float32 forwards, float64 statistics).

| prediction | measured | verdict |
|---|---|---|
| P1 η²(mlp0) > η²(attn0) by word form | 0.844 vs 0.589; diff +0.254 [0.244, 0.259]; null 95th pct 0.014 | **holds** |
| Plant P-WTE (patch machinery shows both outcomes) | same-word wte patch Δ exactly 0; other-word +0.551 nats | **fired** |
| P2 mlp0 more context-invariant than attn0 | same-word Δ: attn0 0.0056, mlp0 0.0014 nats; paired diff +0.0042 [0.0035, 0.0051] | **holds** |
| P3 mlp0 carries word identity | mlp0 other-word 0.320 vs same-word 0.0014 nats; diff +0.318 [0.307, 0.330] | **holds** |

Floor: self-donor patches returned exactly 0 for all three writes (bar 10 × 1e-6).

**Reading (measured, this checkpoint).** At a word's last byte the alternation is clean: η² by word form goes byte
embedding 1.00 → attn0 0.59 → mlp0 0.84 → attn1 0.45 → mlp1 0.80 (`figs/mlp0_1_eta.png`). Attention mixes context
in; each MLP writes word identity back. mlp0's write is close to a lookup: swapping in the same word's write from
another story moves the next 8 bytes by 0.0014 nats, a different same-POS same-length word by 0.32 — 235× more
(`figs/mlp0_1_patch.png`). In the PCA of the 8 most frequent forms (`figs/mlp0_1_scatter.png`) attn0 smears each word
along a context arc, while mlp0 snaps each to a tight cluster. That explains VOCAB-1's mlp0 cell: the lookup makes word
identity (and so the control task) far easier and returns a weaker code for words it has not stored, so novel-word POS
dips there.

**M3 (descriptive).** POS stays readable after mlp0 (native probes 0.91–1.00) but an attn0-trained probe read after
mlp0 falls, often below base rate (AUX 0.98 → 0.54). That shows the attn0 reading does not transfer; it does **not**
separate a rotation from a shift of offset or scale — the figure title says only what was measured.

## New theory from the image (open — staked next as MLP0-2)

In the mlp0 panel "the", "he" and "she" overlap, and all three end in "-he". **H-SUFFIX:** mlp0's lookup is keyed on
the word's final bytes, not the whole word. Predicts: η² by last-k-byte suffix close to η² by form; a donor sharing
the last 2–3 bytes patches nearly as cleanly as the same word.
