# The kindscope loop

A closed loop for principled exploration: a deterministic proposer stakes predictions in a typed language, an auditor
refuses any stake whose verb cannot meet the obligations of the misfits that bind it, the engine runs each stake behind
a plant and a power check, and a miss that no registered misfit explains goes to a diagnoser (an LLM — this session —
writing in the same language). A diagnosis becomes a registered misfit only if the discriminating test stake it names
hits on the CONFIRM split; if the test misses, the candidate is recanted and its row stays (CC 2.4 row acts).

- Language: `ks/lang.py` (docstring is the spec). Registry: `registry.ks` (source of truth; `../MISFITS.md` is rendered).
- Ledger: `ledger.jsonl`, append-only — stakes (with forecast author, what was known at freeze, batch hash), results,
  misfit instances, diagnoses, status acts, determinism gates, run headers (load, clock, HEAD, driver).
- Frozen batches: `frozen/Bxxxx.ks`; `ks run` refuses a stake whose batch changed after freeze, refuses without a
  passing determinism gate for the current engine hash, and refuses while instrument files are uncommitted.
- Lineage of the rules: CIRISHolon misfit registry and engine (`8477c00`), the Gemma hand-over (cirisagent-fd),
  CIRISConstitution CC 2.4 / CC 3.1.5.5 (cirisconstitution-40), all 2026-11-11.

```
ks status | ks import-vocab1 <json> | ks build-cache <model> <corpus> | ks gate | ks propose -n N | ks run -n N
ks miss <stake> | ks diagnose <file.ks> | ks render
```
