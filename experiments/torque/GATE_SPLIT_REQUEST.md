# Upstream request: §11 corpus splitting for eight `mixed` block families

## What happened

Running `compose_dump gate` on TORQUE's built arm manifests refuses:

```
FAIL [6] REFUSE en:aspdma.action_selection_pdma.closing_reminder:
  a mixed block cannot carry 'vary' in Phase 1 — split it in the corpus
  first (§11), then its routed fragments vary cleanly
```

36 refusals across **eight block families**:

| block | why it is `mixed` |
|---|---|
| `optimization_veto_conscience.system_prompt` | axiotic + deontic + epistemic + procedural |
| `epistemic_humility_conscience.system_prompt` | axiotic + epistemic + procedural |
| `coherence_conscience.system_prompt` | axiotic + epistemic + procedural |
| `pdma_ethical.system_guidance_header` | axiotic + deontic + procedural + ontological |
| `action_selection_pdma.closing_reminder` | axiotic + procedural + structural |
| `action_selection_pdma.context_integration.slots` | axiotic + procedural + structural |
| `language_guidance` | axiotic + procedural + pragmatic + ontological |
| `system.head` | axiotic + ontological + structural |

The refusal is correct. "Vary the axiotic class" is ambiguous at a block that is
only partly axiotic: the gate can see the block changed and cannot see that
*only its axiotic lines* changed.

## Why it matters beyond this campaign

These are not obscure blocks. Three of them are conscience system prompts — the
faculties that can veto an action. Any ablation study of H3ERE that wants to vary
values while holding procedure will hit exactly these eight, because that is
where values and procedure are written into the same block.

So the gate currently certifies the *easy* surface and refuses the surface that
matters most. Splitting these eight is what makes the facility usable for the
question it was built for.

## What is being asked

The same shape as work already landed:

- **#974** routed the DEFER policy, the ASPDMA user-message template, the DSDMA
  user message and the CORE IDENTITY blocks out of the residue
- **#991** routed the 57 formatter labels

Route the axiotic fragments of these eight families into their own blocks, so
each emitted block is singly-classed and `vary` is unambiguous.

A useful ordering, cheapest and highest-value first:

1. `action_selection_pdma.closing_reminder` — one line of axiotic content
   (`Recall CIRIS principles override personal preference.`) in an otherwise
   procedural block. Smallest possible split, immediate gate win.
2. `action_selection_pdma.context_integration.slots`
3. the three conscience `system_prompt`s — the highest-value, and the largest
4. `pdma_ethical.system_guidance_header`
5. `language_guidance`, `system.head`

## What TORQUE is doing meanwhile

Not claiming a pass. The campaign has line-level evidence that is finer-grained
than the gate's block-level check but is not a substitute for it:

- every varied file carries a frozen, published line partition
- `partition.py verify` asserts byte-identity on every non-SWAP line and refuses
  to assemble when a declared SWAP has no authored text
- the accord: 1,104 of 1,153 lines byte-identical, difference set exactly the
  declared 49
- `detect_residue.py` enumerates retired vocabulary per unit; ruled exceptions
  carry written reasons

Stated in the regime as: *block-level certification unavailable in Phase 1 for
mixed blocks; line-level partition verification performed and published instead.*

## One thing the gate found that is worth keeping

Before the dispositions were declared, the gate caught that TORQUE's manifests
set four `corpus` keys and nothing else — every conscience prompt was CIRIS in
every arm, and `values_effect` would have been an accord-only contrast wearing a
whole-pipeline label:

```
FAIL [2] en:aspdma.action_selection_pdma.csdma_ambiguity_guidance:
  axiotic is varied by the regime but the block is byte-identical across
  arms (sha256 e3d6cba10755…) — the ablation did not reach it
```

That single assertion is worth the whole facility. It is the check no author
performs on their own work.
