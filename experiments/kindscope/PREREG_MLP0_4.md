# MLP0-4 — H-SUFFIX patching, raw-byte donors, fresh stories (preregistration)

Frozen 2026-11-12, before any MLP0-4 quantity is computed. Re-stakes MLP0-3 (VOID by its letter: 2 of 71,195 donors matched on lower-cased forms had a different raw last byte — KS-CASEFOLD-DONOR; earlier, MLP0-2's exact-zero
plant read summation-order noise across batch shapes — KS-EXACT-ZERO-ACROSS-BATCH).

misfits: KS-CASEFOLD-DONOR, KS-EXACT-ZERO-ACROSS-BATCH, KS-DIRECT-ONLY, M-PLANT-OBS, M-FLOOR-UNSTAKED, G-SECOND-READ, M-DEVICE-CLASS, M-BAR-FROM-THE-READ

plants: P-WTE carrier: GoldWorm v6 base, the byte-embedding part of the embed write at a word's last byte — a donor sharing the last 3 bytes must patch it to within the self-donor floor, a donor sharing only the first 3 bytes must move the loss by more than 10 x that floor

## Data

FRESH-1 (`PREREG_PARALLAX2.md`: TinyStories-train first 20 MiB, sha256 `3232df12…31ee8a`, 100 validation duplicates
excluded), stories 16,000–19,999 of the kept list (unread; 0–3,999 PARALLAX-2, 4,000–7,999 MLP0-3, 8,000–8,063 CAL-G), tagged as VOCAB-1.

## Design (MLP0-2 M2, changed only where the misfit requires)

Word forms are the RAW bytes (no case-folding); the plant precondition (same raw last byte for same-word and suffix donors) is asserted over every job before any forward. Tokens of length ≥ 4; donors from another story: same word, same last 3 bytes (different form), same first 3 bytes
(different form); tokens lacking any donor kind excluded. Writes patched at the token's last byte: wte, mlp0, mlp1.
Δ = mean NLL of the next 8 bytes, patched − unpatched. **Change:** for every batch of jobs the unpatched forward is run
in that same batch (identical shapes and layout), so self-donor Δ is exactly 0 and the floor is that maximum |Δ| (or
1e-6 if larger).

- Plant P-WTE as above; else VOID.
- **P2 (H-SUFFIX over H-PREFIX):** mlp0, paired per token, Δ(prefix-3) − Δ(suffix-3) > 0, 95% story-bootstrap CI
  excludes 0, |mean| > 10 × floor.
- Descriptive: Δ(suffix-3)/Δ(same word) for mlp0 and mlp1.

## Outputs

`mlp0_4.py` (committed before its first run), `mlp0_4_result.json`, appended to `EDIT_RESULTS.md`.
