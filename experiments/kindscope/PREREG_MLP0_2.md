# MLP0-2 — is mlp0's lookup keyed on the word's final bytes? (preregistration)

Frozen 2026-11-11, before any MLP0-2 quantity is computed. Follows MLP0-1 (RATCHET `0782815`).

misfits: G-SECOND-READ, KS-DIRECT-ONLY, M-PLANT-OBS, M-FLOOR-UNSTAKED, M-BAR-FROM-THE-READ, M-ONE-MODEL-DELTA, M-DEVICE-CLASS

plants: P-WTE carrier: GoldWorm v6 base, the byte-embedding part of the embed write at the last byte — a donor sharing only the last byte must patch it to exactly 0, a donor sharing only the first bytes must not

## Why, and what was known

MLP0-1 measured mlp0 as a lexical lookup at a word's last byte (P1–P3 held, EXPLORE split). Its PCA image showed "the",
"he", "she" overlapping after mlp0 — all end in "-he". **H-SUFFIX:** the lookup is keyed on the word's final bytes.
The theory was formed by LOOKING at EXPLORE data, so it is tested **only on the CONFIRM split**, unread until now
(G-SECOND-READ). Competing hypothesis **H-PREFIX** (keyed on the first bytes) is the built-in alternative, so no bar
needs a number from anywhere: the prediction is a comparison between the two.

## Measures (CONFIRM split; same cache, model, positions and device class as MLP0-1)

**M1 — η² of the mlp0 write by grouping, null-adjusted.** Tokens whose form has ≥ 10 CONFIRM occurrences and length
≥ 4 bytes (so suffix and prefix of length k ≤ 3 do not coincide). Groupings: form; last-k bytes; first-k bytes, k = 1,
2, 3. Each η² minus its permuted-label null mean (200 draws) — this corrects for different group counts.
- **P1 (H-SUFFIX over H-PREFIX):** for each k ∈ {2, 3}, adjusted η²(last-k) − adjusted η²(first-k) > 0, 95% story-
  bootstrap CI (1,000) excludes 0. (k = 1 reported, not staked: the last byte is the current input byte, so it is
  partly the embedding's own share.)

**M2 — patching through the full network (KS-DIRECT-ONLY).** For each eligible CONFIRM token t (length ≥ 4), replace the
mlp0 write at t's last byte by a donor's mlp0 write (from another story, also CONFIRM) and run forward; Δ = mean NLL of
the next 8 bytes minus unpatched. Donors: **same word**; **suffix-3**: different form, same last 3 bytes; **prefix-3**:
different form, same first 3 bytes; tokens lacking either kind of donor are excluded from both (paired). Floor:
self-donor Δ must be exactly 0.
- Plant P-WTE: the same donors patched into the **wte** write: suffix-3 donor Δ exactly 0 (same last byte), prefix-3
  donor Δ > 0 with CI excluding 0. Else M2 is VOID.
- **P2 (H-SUFFIX):** paired per token, Δ(prefix-3) − Δ(suffix-3) > 0, CI excludes 0.
- Descriptive: Δ(suffix-3) / Δ(same word), and the same three donors for the **mlp1** write (does the second MLP key
  on more of the word?).

## Outcomes

P1 ∧ P2 ⇒ H-SUFFIX holds over H-PREFIX (measured, CONFIRM split, this checkpoint). Either failing ⇒ H-SUFFIX is not
supported; the image's overlap gets a different explanation, recorded as a loop miss.

## Outputs

`mlp0_2.py` (committed before its first run), `mlp0_2_result.json`, `MLP0_2_RESULTS.md`, figures.
