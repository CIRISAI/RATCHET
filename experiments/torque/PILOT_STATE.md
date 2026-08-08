# Pilot — run state, 2026-08-08

## What ran, and what it produced

**`bare` — works end to end.** One full 10-turn arc against the real harness,
transcript carried forward (`prior_turns` 0→18), scored against gold with
`score.py`:

```
scored 10 turns: 6 correct, 0 unknown
  concordance      0.60   on 10 extractable
  instruction fid  1.00
  pre-switch  3/5    post-switch 3/5
```

Both halves of the withdrawal split are measurable and the UNKNOWN floor is zero,
which is what the design needs from the instrument.

`values-ciris` composes and holds 21 accord blocks by source hash against the
h3ere reference; it uses the same runner as `bare`.

**These numbers are discarded.** Per PILOT.md the pilot reports instrument
health, not effects, and its items are excluded from the main draw. The 0.60 is
reported here as "the scorer produced a number from a real arc", nothing else.

## What does not run

**The four h3ere arms cannot start.** The agent runtime fails during adapter
startup, before any question is asked:

```
✗ Start Adapters failed: node fold failed to start (node-fails ⇒ agent-fails)
  RuntimeError: TWO FEDERATION IDENTITIES IN ONE NODE — refusing to start

  The persist Engine and this process sign as DIFFERENT keys:
    engine  ed25519 pubkey  pQXT1fp4h2qYIa5RwY5r0MiGP6w0p0Ik/S2SGq6IP40=
    compose ed25519 pubkey  0/KFIAL52seDulATs20DpcgYlrjgr6860ZdT6fdIljA=
```

This is an environment fault in the scratch checkout at `/tmp/a2911`, not a
TORQUE defect. The runner wipes data on every run and mints a fresh bootstrap
identity each time; `identity/` has accumulated dozens, and the Engine and the
compose process no longer agree on which one is the node's.

**One repair attempted, and it did not work.** The error prescribes deleting the
locally-minted pair (`data/local_signing.seed`, `data/local_pqc_signing.seed`),
so I did. The failure moved rather than cleared:

```
✗ Start Adapters failed: node fold failed to start
  RuntimeError: open-or-mint sealed
```

Recorded because it is the wrong kind of progress: one blind change traded one
identity failure for another, and a second guess would be worse than none. The
two seed files are gone and cannot be restored from here.

## What this blocks and what it does not

| | |
|---|---|
| corpora, manifests, partitions, digests | unaffected — all verified |
| `preflight.py` | 31 pass, 0 fail, 0 unrunnable |
| the 600-item draw | locked and pinned |
| `bare`, `values-ciris` | runnable now |
| `h3ere-ciris/alt/neutral/blank` | blocked on agent startup |
| `scaffold_floor` (blank − bare) | blocked — needs one h3ere arm |
| `pipeline_effect`, `values_effect`, `form_vs_content`, `reversion` | blocked |

Five of the six arms' *configuration* is verified; four of them cannot be
executed until the runtime starts.

## What would clear it

A clean CIRISAgent checkout with one coherent federation identity, or guidance
from the agent team on repairing this one. The error text names the fix
(`Engine(local_key_path=…/identity/ed25519.seed, …)`) but the mapping from that
to the qa_runner's startup path is theirs, not a thing to guess at with a live
budget attached.
