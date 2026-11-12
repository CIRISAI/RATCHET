# VOCAB-1 — where does GoldWorm v6 carry parts of speech? (preregistration)

Frozen before any probe is trained or any state is read for this purpose. Part of the kindscope series (`GOAL.md`);
follows CAL-1 (PASS, `CAL1_RESULTS.md`) and must honour its candidate misfit **KS-DIRECT-ONLY** (direct reads explain
only the last block; early writes act through downstream blocks) — so every *causal* claim below is measured by running
the full network forward, never by a direct read through `ln_f` and the head.

## Question

At which of the 13 residual states (embedding, then after each attention and each MLP write in blocks 0–5) does the
base model carry the part of speech of (a) the word it is reading and (b) the word it is about to write — beyond what
word identity alone and the word's surface bytes give — and does removing that structure hurt prediction where part of
speech should matter more than where it should not?

## Materials (pinned)

- Model: CAL-1's port (`cal1.Port`, tanh GELU — the form CAL-1 recorded), weights sha256 `31729a9d…3521` (full hash in
  `PREREG_CAL1.md`).
- Text: the same `TinyStories-valid.txt` (sha256 `94e43181…38b4`), restricted to outer_val `[17764697, 19447282)`.
  Split on the literal `<|endoftext|>`; drop the first and last fragment (cut by the span boundary); keep stories that
  are pure ASCII. Each story's input window is `b"<|endoftext|>\n\n" + story`, truncated to 256 bytes (the corpus's own
  separator form). Real text only; nothing generated.
- Tagger: spaCy 3.8.7, `en_core_web_sm` 3.8.0 (wheel sha256
  `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`), run on the whole story with components `parser`, `ner`, `lemmatizer` disabled (POS comes from `tagger` +
  `attribute_ruler`); tokens mapped to byte
  spans by character offset (ASCII ⇒ char = byte). Labels: coarse UPOS restricted to the 10 classes NOUN, VERB, ADJ,
  ADV, PRON, DET, ADP, AUX, CCONJ, PROPN; tokens with any other tag, or whose span is not wholly inside the window, are
  dropped. The tagger is the label source, not ground truth; its error rate is a known floor on every accuracy.
- Probe: multinomial logistic regression with scikit-learn's L2 objective at `C=1.0` (`Σ cross-entropy + ½‖W‖²`, bias
  unpenalised), on per-feature standardised states (mean/std from the training split only). Fitted in torch (full batch,
  `torch.optim.LBFGS`, strong-Wolfe line search, 300 iterations, float64) so the 200 null draws can be fitted as one
  batched problem; the same fitter is used for every probe (real, null, control, surface, INLP). No hyper-parameter is
  tuned.

## Sites (per word token w)

- **last** — the state at w's final byte (the word has been read).
- **first** — the state at w's first byte.
- **pre** — the state at the byte *before* w's first byte (the model is about to write w; label = POS of w).

## Splits

Stories are split 80/20 by a seeded permutation (seed 20261113). Test accuracy is reported two ways: **all-test** and
**novel-type** (test tokens whose lower-cased word form never occurs in the training split).

## Measures and bars (bars from nulls or prior art, none from this read)

For each site × state (3 × 13 = 39 cells):

| id | measure | bar for the claim | source of bar |
|---|---|---|---|
| V1 | all-test accuracy vs **shuffled-label null** (labels permuted within the training split, 200 draws, same probe) | real > null 95th percentile | GOAL.md statistical misfit rule (the null that killed closure on Gemma) |
| V2 | **selectivity** = real − control-task accuracy; control task = each training word type gets a label drawn once from the empirical label distribution, applied to all its tokens (Hewitt & Liang 2019); 20 draws | real > max of the 20 control draws | prior art (control tasks) |
| V3 | novel-type accuracy vs a **surface baseline**: the same probe on one-hot bytes (last site: last 4 bytes of w; first site: first byte of w plus the 4 bytes before it; pre site: the 4 bytes before w) | 95% bootstrap CI (1,000 resamples by story) of (state − surface) excludes 0, positive | arithmetic / resampling |

**"Carried beyond surface" at a cell** = V1 ∧ V2 ∧ V3 all hold. Expect V1 to hold almost everywhere (word identity alone
predicts POS well); V2 and V3 are the informative ones.

## Causal test (amnesic, full forward — honours KS-DIRECT-ONLY)

For each of the 13 states, using the **pre**-site probe: iterative nullspace projection (INLP; Ravfogel et al. 2020,
amnesic probing per Elazar et al. 2021) — **3 rounds, fixed**, each round removing the rowspace of a freshly trained
probe on the training split (rank ≤ 9 per round, ≤ 27 total). The projection is mean-preserving, `x ← x − (x − μ)QQᵀ` with `Q` the orthonormal removed basis and `μ` that state's
mean over all positions of the training stories, so only variation along `Q` is removed. It is applied to that residual
state at every position of every test story, and the network is run forward from there unchanged. Control: 20 random orthonormal
subspaces of the same rank (seed 20261113), same procedure.

Report Δ mean next-byte loss (nats) on test stories at two position sets: **W** = positions predicting the first byte of
a kept word (the pre-site positions), **O** = all other positions.

| id | claim | bar |
|---|---|---|
| A1 | removal hurts at W more than a random rank-matched removal | real Δ_W > max of 20 random Δ_W |
| A2 | the harm is POS-specific rather than general | (real Δ_W − mean random Δ_W) > (real Δ_O − mean random Δ_O) |

Also reported, not gated: probe accuracy after the 3 INLP rounds (how much was actually removed).

## Outcomes and what each means

- A layer profile of V1–V3 per site (the map's first row). Cells where V1 holds but V2/V3 fail are registered as
  **candidate misfit KS-LEXICAL** at that cell ("POS readable only as word identity").
- A1 ∧ A2 at a state ⇒ the carried POS structure is *used* there (measured, this checkpoint).
- A cell that is "carried beyond surface" but fails A1 ⇒ **candidate misfit KS-READ-NOT-USED** ("readable, not used"),
  the Gemma hand-over finding in miniature.
- Candidates become misfits only on recurrence on the forks (FORK-1) under the shuffled-label rule.

## Outputs

`vocab1.py` (committed before its first run), `vocab1_result.json` (all 39 cells, nulls, bootstrap CIs, the amnesic
table, environment, hashes, token/label counts), `VOCAB1_RESULTS.md`. Local CPU; stateless (GOAL.md "Running").
