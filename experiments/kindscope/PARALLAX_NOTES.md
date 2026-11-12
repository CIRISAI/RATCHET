# What the parallax is — exploration notes (CAL calibration stories only; not verdicts)

Scripts and raw: `explore_parallax.py`/`.json`, `explore_parallax2.py`/`.json`, `explore_parallax3.py`/`.json`.
Figures: `figs/explore_parallax_forest.png`, `figs/explore_parallax_plane.png`, `figs/explore_parallax_meaning.png`.
GoldWorm v6 base; states at the position before a word that follows a determiner; target = log-prob of following
adjective words. Everything here was formed by looking, so every claim it suggests is tested on unread data.

1. **Reading (probe) and writing (pulled-back gradient) directions are as unrelated as random directions** at every
   MLP (82–90°; the 95% band for two random directions in R¹⁹² is 81.9°–98.1°).
2. **Not the decoder-vs-pattern effect (Haufe et al. 2014).** The probe's pattern (Σ·r) and the class mean difference
   agree with each other (14–42°) and sit in loud directions (16–44% of state variance), but they are as far from the
   writing direction as the probe is, and pushing along them buys ~0.01 nats. Three neighbourhoods: the class *shows*
   in loud directions, the probe *reads* it in a near-silent one (~0.1% of variance), the model *listens* in a
   quiet-but-not-silent one (1–5%).
3. **Both axes separate the classes** (probe d′ ≈ 1.7; writing axis d′ 0.5 → 1.2 with depth), and the clouds are
   diagonal: two views of one structure.
4. **What each axis is about.** The probe ranks the same adjectives top at every layer (small, little, loud, new, old,
   big): a *category*. The writing axis ranks *what comes next*: early, the most predictable continuations whatever
   their class (time, girl, boy, little); late, words starting with the adjectives' frequent first bytes — little,
   girl, boy, old, loud, bird, man, small, big. Its cosine with the frequency-weighted unembedding of adjective first
   bytes rises 0.01 → 0.33 (mlp4); the probe's stays ≈ 0. Agreement between the two coordinates across contexts rises
   0.04 → 0.70 with depth while the angle stays random-like.

**Reading (open, H-BYTE-GRAIN):** the model writes at its output grain — the next byte. "Adjective" is a reader's
category; the model can only push "the next word begins with l / b / s …", which carries every word that begins that
way. That is the parallax, and it accounts for KS-READOUT-GRAIN and for the fine-tune's collateral loss on nouns.
Staked as PARALLAX-1 on CONFIRM.

## Locked results

**PARALLAX-1** (CONFIRM, its second read; `parallax1_result.json`): pushing the adjective writing direction at mlp3 and
mlp4 raised adjectives (+0.11, +0.10; plant fired). **P2 holds** — nouns starting l/b/s did far better than other
nouns (+0.21 [0.14, 0.29], +0.52 [0.37, 0.65]): the pure-category alternative is refuted. **P1 fails by a hair** (the
in-set nouns' own gain has CI lower bounds −1e-6 and −0.004), so H-BYTE-GRAIN *as staked* is not supported; the
per-letter split shows why: 'b' nouns gain, 's' nouns lose — the forecast is concentrated on 'l' and 'b', and the
staked set wrongly bundled 's'.

**Exploration that followed (CAL; `explore_clouds.*`, `explore_play.*`, `figs/explore_clouds.png`).** The state cloud
before a word is organised by position (η² 0.55 → 0.36) and the determiner just read (0.48 → 0.28), turning with depth
into the next word's first byte (0.15 → 0.35); POS is a thin slice (0.08 → 0.14). The adjective writing direction is a
first-byte forecast to R² 0.60 / 0.76 / 0.95 at mlp3–5 (random null ≈ 0.1) but only 0.36–0.45 at mlp0–2. Pushing the
early **residual** lifted adjectives and lowered nouns in every first-letter group; pushing the late pure 'l' direction
lifted every l-word whatever its class.

**PARALLAX-2 — locked on FRESH-1** (unseen TinyStories-train slice; `parallax2_result.json`,
`figs/parallax2_locked.png`): **both hold.** H-CATEGORY-EARLY 9/9 cells: adjectives over nouns within each first-letter
group at mlp0–2, +0.11 to +0.29, every CI excluding 0, zero-push floor exactly 0. H-BYTE-LATE: the 'l' push at mlp5
lifts l-nouns +1.76 [1.70, 1.83], drops b-adjectives −0.82 [−0.85, −0.80].

**What the parallax is (measured, this checkpoint):** the model writes the category early, in a small direction that
is not a byte forecast; downstream turns it into a next-byte forecast, which is all the late layers carry; the probe
reads the category from a third, near-silent direction neither writer uses. Where to change the source to effect the
destination therefore depends on the grain of the effect: category edits early (in the non-byte residual), byte edits
late. Open residuals: mass nouns (grass, water, sun) rose with the adjectives under the early residual push; and the
early residual's other parts are unexplained.
