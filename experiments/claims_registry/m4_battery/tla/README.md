# M4Episode.tla — the Market 4 episode and its verdict, model-checked

`M4Episode.tla` models one genuine episode as the CI job runs it:

1. baseline probes;
2. the WA correction (deferral or not);
3. the resolve (200 / 409 / 500 / 503);
4. whether the amendment landed;
5. probes;
6. a restart (ok or failed);
7. probes again;
8. the verdict.

The workflow's credit pause runs around it. A provider credit refusal can happen at any running step and is silent to
the harness. The watcher or the post-check detects it, and the attempt is discarded and rerun, bounded by attempts and
by waiting budget. The agent and the provider choose every outcome nondeterministically.

Reply classes per phase are `meets`, `misses` and `null`, where null means a deferral, a rejection, an error or an
empty reply. Neighbour classes are `kept`, `over` and `null`. The case either has a positive marker or is
absent-only. The verdict rule is a parameter: `current` is the harness at `779b33629`, and `fixed` is the harness at
`c1eba068a` (CIRISAgent branch `market4/on-v2221`).

## Invariants (what a reported verdict may claim)

| Invariant | Says |
|---|---|
| `NoContaminatedVerdict` | an attempt with a credit refusal is never reported |
| `NoFalseUptake` | uptake ⇒ the amendment landed, the default missed, the agent then met the target |
| `NoMissedUptake` | that effect ⇒ uptake |
| `PersistSound` / `PersistComplete` | persisted ⇔ uptake and a real restart and the target met again |
| `NoUnexplainedNo` / `NoUnexplainedNotPersisted` | every negative verdict carries its reasons |
| `SpilloverReported` | kept at baseline and over-applied afterwards ⇒ spillover reported |
| `Bounded`, `TypeOK` | attempts and waiting stay within bounds |
| `Terminates` (liveness) | every job ends, accepted or red |

## Result (TLC 2.19, MaxAttempts = 3, PauseBudget = 3; 269,294 distinct states)

| Rules | Result |
|---|---|
| `fixed` | **All invariants and `Terminates` hold.** |
| `current` | `NoFalseUptake`, `PersistSound`, `NoUnexplainedNo`, `NoUnexplainedNotPersisted` and `SpilloverReported` are violated. |

The counterexamples for the `current` rules:

- **Uptake without a correction.** The agent never deferred and nothing landed, but the probe met the target, and the
  verdict credited uptake.
- **Persistence from a deferral.** After the restart a null reply passed an absent-only probe, and persistence was
  credited.

The harness test `test_episode_model_conformance` enumerates the same verdict-reaching states (3,888) against the
Python functions and asserts the same invariants, which ties the model to the code.

Run: `TLA2TOOLS=/path/to/tla2tools.jar ./check.sh` (tla2tools 1.8.x, Java 11+). `VERBOSE=1` prints the traces.

## Not modelled

- the sham vectors' own success rules (only the default-must-miss condition is shared);
- the auditor and occurrence read-backs, which are logging only;
- the content of replies beyond the three classes. The regex patterns are checked separately, against compliant,
  non-compliant and deferral samples, when the candidates are built.
