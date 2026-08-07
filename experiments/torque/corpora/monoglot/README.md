# Historical monoglot sources

The polyglot corpora (`accord.polyglot_full`, `accord.polyglot_compressed`,
`polyglot.pdma_framing`) are woven from 15 languages by semantic weight. Building
polyglot-neutral and polyglot-Anthropic counterparts is a research programme in
its own right, not a build step — so **both arms run monolingual English** and
that is carried as a stated domain limit (POLYGLOT_PROBLEM.md).

The English originals are not invented for this campaign. They are **recovered
from git history**, from before the polyglot extraction.

| file | source | bytes |
|---|---|---|
| `pdma_framing_ciris_en.txt` | `b7897b9b4^:ciris_engine/logic/dma/prompts/pdma_ethical.yml` → `system_guidance_header` | 3,675 |

That commit is *"PDMA v3.2: polyglot extraction + 28-locale fan-out"* — so its
parent is the last state before the English framing became polyglot. The text
names all six CIRIS Foundational Principles inline, which is the axiotic content
the polyglot version now carries at 15,046 B.

**Why this matters for the campaign.** Using the real historical artifact rather
than a substitute means the CIRIS arm's monolingual configuration is one CIRIS
actually shipped, not one assembled for the experiment. The domain limit is then
"an earlier, monolingual configuration of this agent" rather than "a
configuration nobody has ever run" — a materially weaker limit, and a true one.

The alt-arm counterpart is authored through the same partition → isolate →
assemble → verify path as the accord.
