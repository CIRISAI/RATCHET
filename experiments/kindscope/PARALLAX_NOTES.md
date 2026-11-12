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
