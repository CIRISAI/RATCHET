# TORQUE — external reproducibility review

Reviewed 2026-08-07 against `master` @ `40e7a31`. Paths are relative to the repo
root unless absolute.

**The test applied.** A competent outsider clones `CIRISAI/RATCHET` six months
from now. No access to this machine, no one to ask. Can they (a) rebuild every
artifact bit-for-bit, (b) verify the intervention is what it claims, (c) re-run
the experiment, (d) check reported numbers against raw output?

Everything below was executed, not read. Where a claim in the repo turned out to
be true, it is credited in the last section rather than omitted — that section is
substantial, and it is the honest half of the answer.

---

## Ranked findings

### 1. The CI workflow cannot check out the agent — the pinned ref does not exist
**blocks-external-replication**

`.github/workflows/torque_pilot.yml:24` defaults `agent_ref: "2.9.11-stable"`, and
`TORQUE_REGIME.yaml:890` pins `harness: {agent: "2.9.11-stable"}`. The tag in
`CIRISAI/CIRISAgent` is **`v2.9.11-stable`**:

```
$ git -C /home/emoore/CIRISAgent show-ref | grep 2\.9\.11
7e71d038102fa220ebdefad003ac87d8f8330aca refs/tags/v2.9.11-stable
$ git -C /home/emoore/CIRISAgent rev-parse --verify 2.9.11-stable
fatal: Needed a single revision
```

`actions/checkout` fails on a ref that does not resolve, so **both** the
`preflight` job (line 56) and the `compose` job (line 84) die at their agent
checkout. The commit hash in the trailing comment (`# 7e71d0381`) is correct; the
ref string is not. Nothing in the repo dereferences the hash.

**Remedy:** `agent_ref` default `v2.9.11-stable`; same in the regime pin; and pin
the 40-char commit alongside so a retagged release is caught.

---

### 2. `gate_projection.py` crashes against the current regime — the compose job fails on every machine
**blocks-external-replication**

```
$ python3 gate_projection.py --regime TORQUE_REGIME.yaml --out /tmp/rg.yaml
  File "gate_projection.py", line 50, in project
    holds["corpus"] = r["corpus"]["primary"]
KeyError: 'primary'
```

The `corpus:` block has no `primary` key — it carries `method_reference`,
`method_reference_n`, `seed`, `source`, `strata`, `n_total`, `sampling`. Commit
`4e7afc5` disambiguated that block and `gate_projection.py` was not updated.

This is the **first command** in the workflow's "Compose dumps and ablation gate"
step (`torque_pilot.yml:128`), under `set -euo pipefail`. So the compose job fails
even on this machine, with every external dependency present. The workflow has
not been green against the current regime, and `preflight.py` never calls
`gate_projection.py`, so nothing catches it.

**Remedy:** fix the projection, and add a `gate_projection.py` smoke run to
`preflight.py` so a regime edit that breaks the projection fails before spend.

---

### 3. `preflight.py` prints PASS and exits 0 with its two provenance gates silently skipped
**blocks-external-replication**

`preflight.py:75` (`/tmp/a2911/accord_1.2b.txt`) and `preflight.py:134`
(`Path("/home/emoore") / c["method_reference"]`) are hard-coded absolute paths. Both
are `if exists()` guards whose `else` branch appends to `skip`, and `skip` never
affects the exit code (`return 1 if bad else 0`, line 171). The word `PASS` is
printed unconditionally at line 160.

Run in a clean clone with those two paths pointed at `/nonexistent`:

```
PASS
  ... 15 gates ...
NOT CHECKABLE BEFORE THE RUN
  PROV: agent checkout absent — cannot reverify corpora from the shipped accord
  CORPUS-3/4: method reference not found at /nonexistent/CIRISBench/...
15 pass, 0 fail, 9 deferred
EXIT=0
```

The gates that vanish are exactly the ones that matter: corpus provenance and the
gold-label check. An outsider sees `PASS`, exit 0, and a green CI badge.

Worse, **this is the CI's permanent state, not an edge case.** The workflow checks
the agent out to `$GITHUB_WORKSPACE/agent` (`torque_pilot.yml:58`), never to
`/tmp/a2911`, and `CIRISBench` is never cloned at all. So on a GitHub runner PROV
and CORPUS-3/4 always skip — while the step's own comment
(`torque_pilot.yml:63-66`) advertises "corpora reproducing from the shipped
accord" as one of the gates being run. It is not run there and never has been.

Note the preflight job installs the full agent requirements (`torque_pilot.yml:61`)
for a script that reads nothing from the agent checkout.

**Remedy:** take the agent root and the bench root from env/argv with the CI
values wired in; make a missing input a **FAIL**, not a skip, unless an explicit
`--allow-missing-externals` is passed; print `PASS`/`FAIL` from the actual result.

---

### 4. The provenance gate does not read the file it guards on
**blocks-external-replication**

`preflight.py:75-86`:

```python
src = Path("/tmp/a2911/accord_1.2b.txt")
if src.exists():
    sub = HERE / "corpora" / "values-alt" / "accord-substituted.txt"
    ... partition.py verify  str(sub)  accord-meanings.tsv  <out>  --swaps ...
```

`src` is tested for existence and then **never used**. The verification runs
against `accord-substituted.txt`, a committed intermediate. So `PROV/alt` and
`PROV/neutral` assert that alt and neutral derive from a file already in this
repo — they assert nothing about the shipped accord. The chain's first hop
(`accord_1.2b.txt --substitute(terms.tsv)--> intermediate`), which is the whole
provenance claim at `TORQUE_REGIME.yaml:877-882`, is unchecked by any gate.

The claim itself is **true** — I verified it end to end (see the last section) —
which makes this cheap to fix and worth fixing, because right now the strongest
thing this campaign has is asserted by prose and not by the script that says it
is checking it.

**Remedy:** run `substitute_terms.py` on the shipped accord inside the gate and
assemble from *that*, not from the committed intermediate. Assert
`sha256(accord_1.2b.txt) == pins.accord_ciris_sha256` first.

---

### 5. `accord-substituted.txt` no longer reproduces from `terms.tsv`
**weakens-it**

```
$ python3 substitute_terms.py --in /tmp/a2911/accord_1.2b.txt \
      --table corpora/terms.tsv --out /tmp/resub.txt
$ diff /tmp/resub.txt corpora/values-alt/accord-substituted.txt
928c928
<     * Compile "Sunset PDMA" focusing on harm avoidance vectors (...)
---
>     * Compile "Sunset PDMA" focusing on non-maleficence vectors (...)
```

The committed intermediate was produced by an earlier `terms.tsv` that lacked the
lowercase `non-maleficence` mapping. It is **inert** — line 928 is `SWAP`, so the
assembled artifacts are unaffected (verified: assembling from the regenerated
intermediate gives byte-identical alt and neutral). But an outsider who checks
step 1 of the published chain gets a mismatch on a line that still carries a
retired CIRIS principle name, and has no way to tell that it does not matter.

**Remedy:** regenerate `accord-substituted.txt`, or delete it and make the chain
regenerate it — it is a build artifact, not a source.

---

### 6. Two resolvers disagree about which file is unit F's alt corpus, in opposite precedence, and the files differ
**blocks-external-replication**

`values-alt` is the only directory where both filename variants exist for a unit,
and they are not the same text:

| file | content |
|---|---|
| `corpora/values-alt/F-lg-axiotic.txt` | authored alt line ("Someone in the user's own life …") |
| `corpora/values-alt/F-lg-axiotic-mechanical.txt` | **the unswapped CIRIS line** ("validating 'talk to someone you trust' as a real first step matters") |

`unit_keys.py:133` resolves `["{}.txt", "{}-mechanical.txt"]` — picks the authored
one, and it is what `arms/h3ere-alt.json` actually ships (confirmed by grepping
the manifest).

`preflight.py:94-98` resolves `-mechanical.txt` **first** — so the residue sweep
audits the stale file. The `RESIDUE: 14 unit artifacts sweep clean` line is a
statement about a file the experiment does not use.

`build_neutral_units.sh`'s cross-arm assertion reports `F-lg-axiotic 1/1 ok`
because it counts diff lines, not correctness: alt-carrying-CIRIS and
alt-carrying-alt both differ from neutral by exactly one line.

No gate compares a shipped unit corpus against `assemble(src, partition, swaps)`,
which is why a stale artifact survived. (6 of 7 alt units and 7 of 7 neutral units
do reproduce; F is the one that does not.)

**Remedy:** delete `F-lg-axiotic-mechanical.txt`; make ambiguous resolution a
refusal in both resolvers; add a preflight gate that re-assembles every unit and
asserts byte-identity against the shipped file.

---

### 7. Gate B3 is hard-coded to pass and compares against nothing
**blocks-external-replication**

`preflight.py:66`:

```python
check(f"B3/{name}", True, f"sha256:{hashlib.sha256(body.encode()).hexdigest()[:16]}… ...")
```

The literal `True` is the verdict. B3 cannot fail. It also hashes the *whole TSV
body* (line numbers, per-line digests, full text), which is a different function
from `partition.py freeze`'s digest (`n\ttag` only) — so the value it prints
(`43e4936d…`) is not the `partition_digest` PILOT.md B3 says it is matching, and
it is matched against no pinned value anywhere.

PILOT.md:68 states B3's job as "`partition_digest` matches the frozen partition."
That check does not exist in any form.

**Remedy:** pin `partition_digest` for `accord-meanings.tsv` and
`G-framing-meanings.tsv` in `TORQUE_REGIME.yaml.pins`, and have B3 shell out to
`partition.py freeze` and compare.

---

### 8. `partition/FROZEN.md` pins the wrong partition
**blocks-external-replication**

The file named FROZEN.md pins, twice (lines 4 and 123):

```
partition_digest: sha256:10327bc7...  lines: 1153  SWAP: 32  HOLD: 1121
```

That is `partition/accord.tsv`. The partition the corpora were actually built
against is `partition/accord-meanings.tsv`:

```
$ python3 partition.py freeze partition/accord-meanings.tsv
partition_digest: sha256:4135fffff971ee58e48d2e198306b340a6ecdd494eb6f5e125486f98c5e0e75e
  lines: 1153  SWAP: 49  HOLD: 1104
```

`4135ffff` appears in exactly one place — `corpora/values-alt/ACCORD_DONE.md:4` —
and **not** in `TORQUE_REGIME.yaml`'s `pins:` block, which pins seven other
hashes. An outsider following the obvious trail (`partition/FROZEN.md`) gets a
digest for a superseded 32-SWAP partition and concludes the corpus was built
against something else.

Related: `partition/G-pdma-framing.tsv` is not frozen at all (`freeze` refuses, 10
`SWAP?` rows); the operative file is `G-framing-meanings.tsv` (12 SWAP,
`fd5a8adf`, recorded only in `partition/G_RULING.md`). `STATUS.md:26` says
"partition frozen (6 SWAP)", which matches neither.

**Remedy:** move every operative partition digest into `pins:`; make FROZEN.md
point at the operative partition or delete it; correct STATUS.md.

---

### 9. The alt corpus's source document is named but not pinned, obtainable, or committed
**weakens-it**

`TORQUE_REGIME.yaml:134`:

```
source: "Anthropic, Claude's Constitution, January 2026, artifact 26-02.02a, adapted"
```

No URL, no hash, no committed copy, no excerpt file. `corpora/ALT_NAME_TABLE.md`
cites it by section ("tier 4", "§Avoiding harm", "§Avoiding problematic
concentrations of power"), and the 49 authored accord lines plus 9 authored unit
lines are all derived from it.

An outsider can verify that the alt corpus is a 49-line perturbation of the CIRIS
accord. They **cannot** verify that those 49 lines say what the alt value system
says. The single most contestable step in the whole build — the one that decides
whether `values_effect` is a values contrast or an authoring artifact — is the
one step taken entirely on trust.

The published diff (`partition/accord-meanings_swaps.tsv`) is the mitigation and
it is a real one: a reader can judge the authored text on its face. That is not
the same as checking it against a source.

**Remedy:** pin the source by URL + retrieval date + sha256 of the retrieved
bytes. If it cannot be redistributed, commit the hash and a per-swap line
citation (`swap 230 ← §Being honest, ¶3`) so the mapping is auditable even when
the source must be fetched separately.

---

### 10. CIRISBench is unpinned, and three different load-bearing things come from it
**weakens-it**

`build_he300_arcs.py:48` hard-codes `/home/emoore/CIRISBench/engine/datasets/ethics`.
`TORQUE_REGIME.yaml:509,524` reference `CIRISBench/...` as bare relative strings.
The scorer discussion (`TORQUE_REGIME.yaml:1288-1400`) analyses
`he300_runner.heuristic_classify`, `core/simple_llm.classify_ethical_response` and
`utils/response_normalizer.get_label_from_response` in detail.

Nowhere is a CIRISBench commit, tag, or file hash recorded. The repo is public
(HTTP 200 unauthenticated), so it is *reachable* — but "the scorer" six months
from now is whatever `main` says, and the whole `ground_truth_instrument` section
is an analysis of code that can change underneath it. The ETHICS CSVs are
likewise unhashed (`cm_test.csv` here is `aa521e2a…`); the draw digest is a
function of their exact bytes.

**Remedy:** pin a CIRISBench commit in `pins:`; hash the four CSVs the strata draw
from; state the upstream ETHICS release the CSVs came from.

---

### 11. The regime's `pins:` are recorded but no gate asserts any of them
**weakens-it**

`TORQUE_REGIME.yaml:883-889` pins `accord_ciris_sha256`, `accord_alt_sha256`,
`accord_neutral_sha256`, `framing_ciris_sha256`, `framing_alt_sha256`,
`terms_table_sha256`, `template_sha256`. `preflight.py` reads exactly one field
from `pins` — `residue_digest`, line 71 — and asserts none of the hashes.

I checked them by hand: all seven match. `framing_neutral_sha256` **does not
exist** — the neutral framing (`85f242ec…`) is the one varied artifact with no
pin, in the arm whose entire job is to be a control.

The `pins:` comment (line 866-869) makes the right argument — "an unchanged digest
and an unchecked one are indistinguishable in the file and opposite in meaning" —
and then the checker does not check them.

**Remedy:** a `PINS` gate that hashes every pinned path and fails on mismatch; add
`framing_neutral_sha256`.

---

### 12. The residue sweep excludes the two largest varied artifacts
**weakens-it**

`preflight.py:91` sweeps seven units × two arms. It does **not** sweep
`corpora/values-alt/A-accord-FINAL.txt`, `corpora/values-neutral/A-accord-NEUTRAL.txt`,
or any of the three `corpora/monoglot/pdma_framing_*.txt`. The accord is 1,153
lines and, per `ADAPTATION_MAP.md`, 76% of the varied surface.

I ran the detector on the alt accord manually: it is clean (0 undeclared, 4
adjudicated). So this is a coverage gap in the gate, not a defect in the corpus —
but the gate's headline (`14 unit artifacts sweep clean`) reads as whole-corpus
coverage and is not.

**Remedy:** add the accord and framing artifacts to the sweep list.

---

### 13. `manifest_digest` is not recorded anywhere; B2 checks a different hash
**weakens-it**

PILOT.md:67 requires "`manifest_digest` recorded, and **different** per arm."
`preflight.py:44-45` computes `sha256(json.dumps(m, sort_keys=True))` — its own
function, not the agent's. `build_arm_manifest.py:248-262` obtains the real
content-addressed digest from `research_overrides manifest-digest` and **prints it
to stdout**; nothing writes it to a file. `TORQUE_REGIME.yaml:897` says
`manifest_digest: computed-at-load`.

So the identity a trace would carry is not in the repo, and the identity in the
repo is not the one a trace would carry. When traces exist there will be no
committed value to join them against.

**Remedy:** write the agent's digest into `arms/<arm>.digest` at build time and
have B2 compare traces against it.

---

### 14. CI verifies the committed manifests, then rebuilds them, and never re-verifies
**weakens-it**

The `preflight` job runs against the committed `arms/*.json`. The `compose` job
(`torque_pilot.yml:96-116`) then rebuilds all four from the agent checkout and
overwrites them, with no second preflight and no diff against what was verified.
An agent-side change that alters the baseline key space, or a corpus edit
post-dating the last manifest build, ships silently.

**Remedy:** rebuild first, then preflight the rebuilt manifests; and assert the
rebuild is byte-identical to what is committed, or fail.

---

### 15. The pilot draw is recoverable, but nothing in CI builds it and the arcs live only in a scratch checkout
**weakens-it**

`build_he300_arcs.py` writes ten arc manifests into `--safety-dir`, i.e. into the
agent checkout. On this machine they are **untracked files** in
`/tmp/a2911/tests/safety/english_he300_a00..a09/`. They are in neither repo.
`torque_pilot.yml` has no step that runs the builder, so even with `run-arms`
enabled the corpus would not exist on the runner.

The good news is that the draw *is* recoverable — see the last section, the digest
reproduces exactly. This is a plumbing gap, not a provenance gap.

**Remedy:** add a build step to the workflow before `run-arms`, and assert the
printed `draw sha` equals `arc_construction.pilot_draw.draw_sha256`
(`TORQUE_REGIME.yaml:1520`) — the regime already tells you to do this in prose
(line 1522) and nothing does it.

---

### 16. `cm_test` label polarity is documented in the regime and carried on no artifact
**weakens-it**

`TORQUE_REGIME.yaml:576,793` record that `cm_test.csv` label 1 = UNETHICAL, the
reverse of the other three subsets, and warn that "a pooled scorer that misses
this inverts one stratum silently." PILOT.md gate C4 is `cm_test polarity handled`.

The generated arc manifests carry `he300.gold_label` raw with no polarity field,
no `label_semantics`, and a rubric that says only "the standard is the
`gold_label`". Every one of the 100 pilot items is `commonsense:cm_test:*`, so the
entire pilot runs on the inverted stratum. The knowledge lives in a YAML comment
and travels with nothing.

**Remedy:** emit `"gold_polarity": "1=unethical"` per question in
`build_he300_arcs.py` and have the scorer read it.

---

### 17. The domain limit is popped out of the manifest before it is written
**worth-noting**

`build_arm_manifest.py:70-78` declares `DOMAIN_LIMIT` "Written into every manifest
so the limit travels with the artifact rather than living only in a document
nobody opens at analysis time." Line 231: `limit = m.pop("_torque_domain_limit")`
— removed to satisfy `extra="forbid"`, then printed to the console. Confirmed
absent from all four committed manifests.

The monolingual-variant caveat is the campaign's sharpest declared limit and it
does not travel with the artifact.

**Remedy:** emit `arms/<arm>.limits.txt` alongside the manifest.

---

### 18. Residue adjudications are keyed by bare filename and leak across arms
**worth-noting**

`detect_residue.py:126` sets `unit = text.name`. `corpora/adjudicated.tsv` rows
are keyed `B-optveto-mechanical.txt`, `A-accord-FINAL.txt`, etc. Both
`values-alt/` and `values-neutral/` contain identically named files, and
`preflight.py:93` sweeps both with the same adjudication table — so a line ruled
"ordinary prose" about the alt corpus silently suppresses the same line number in
the neutral corpus.

Benign today (the ruled lines are HOLD and byte-identical across arms). Not
structurally sound, and it is the same "an allowlist entry with no argument behind
it is a suppression" failure the file's own header warns about.

**Remedy:** key adjudications `<arm>/<unit>`.

---

### 19. `corpora/values-alt/README.md` says the corpus is refuted and unshippable
**worth-noting**

It opens: *"Alt-values drafts — NOT SHIPPABLE … The verification pass refuted the
drafts. They are committed as work in progress, not as a corpus."* It describes
the superseded free-authoring drafts (`A-accord.md`, `B-conscience.md`,
`D-aspdma.md`), which sit in the same directory as the shipped
`A-accord-FINAL.txt` with nothing distinguishing them. `STATUS.md` says the same
corpora are verified.

An outsider reading the directory's own README concludes the alt arm is refuted.

**Remedy:** a one-line header separating superseded drafts from shipped artifacts,
or move the drafts to `values-alt/superseded/`.

---

### 20. Smaller items
**worth-noting**

- `preflight.py:160` prints `PASS` before knowing the result; the uploaded
  `preflight.txt` artifact leads with `PASS` even on a failing run.
- κ table rows in `partition/README.md` for batches 1, 2 and 4 have no committed
  annotator files — only `batch3`, `conflict16`, `bf`, `residue` are in
  `partition/adjudications/`. `kappa.py` refuses that format
  (`parsed zero annotations`), so no committed tool recomputes them; batch3
  (0.432) and conflict16 (1.000) do reproduce by hand in ~12 lines.
- `TORQUE_REGIME.yaml:348-355` still names `h3ere-nonsense` in a `kills` clause;
  the arm is `h3ere-neutral`. A stake references an arm that does not exist.
- `TORQUE_REGIME.yaml:36-45`: the correction comment argues for `file:` + the
  monoglot accord; the field it corrected reads
  `inject: {axiotic: "corpus:accord.polyglot_compressed"}`. That resolves to
  monoglot text only because `CIRIS_RESEARCH_PROMPT_OVERRIDES` is exported in a
  shell line (`torque_pilot.yml:136`). The polyglot confound this note exists to
  prevent is gated on an env var, not on a check.
- `torque_pilot.yml:135-136` passes `h3ere-ciris.json` as overrides for the `bare`
  arm too, the arm defined as "no system content at all". Probably harmless for
  `compose-dump`; worth an assertion before `run-arms`.
- `corpora/values-alt/measure_{D,E,F}.py` hard-code both `/tmp/a2911` and
  `/home/emoore/RATCHET/...`; they are documentation-support scripts, not gates,
  but they are dead on any other machine.

---

## What an outsider CAN verify today

This is not a short list, and several items are stronger than the corresponding
claims in the README.

**1. The headline provenance claim is true, and I confirmed it end to end.**
Clone `CIRISAgent` at `v2.9.11-stable`, then from `experiments/torque/`:

```bash
python3 substitute_terms.py --in <agent>/accord_1.2b.txt \
    --table corpora/terms.tsv --out /tmp/sub.txt
python3 partition.py assemble /tmp/sub.txt partition/accord-meanings.tsv \
    partition/accord-meanings_swaps.tsv --out /tmp/alt.txt
python3 partition.py assemble /tmp/sub.txt partition/accord-meanings.tsv \
    partition/accord-neutral_swaps.tsv --out /tmp/neu.txt
```

Both outputs are **byte-identical** to `corpora/values-alt/A-accord-FINAL.txt` and
`corpora/values-neutral/A-accord-NEUTRAL.txt`, and hash to the pinned
`accord_alt_sha256` / `accord_neutral_sha256`. `sha256(accord_1.2b.txt)` equals
the pinned `accord_ciris_sha256`. The docs page's "the whole corpus regenerates
from the shipped original plus that published diff" is accurate. (The committed
intermediate is stale by one inert line — finding 5 — but the *chain* is sound.)

**2. Both framing corpora regenerate with no external dependency at all.**
`partition.py verify` on `pdma_framing_alt_en.txt` and `pdma_framing_neutral_en.txt`
against `pdma_framing_ciris_en.txt` + `G-framing-meanings.tsv` + swaps: 43
byte-identical + 12 replaced = 55 of 55, both arms, SWAP text matched. Entirely
inside this repo.

**3. Six of seven alt units and seven of seven neutral units regenerate** from
`partition/src/` + the committed swaps. `bash build_neutral_units.sh` in a fresh
clone leaves the tree clean. Only `F-lg-axiotic` in `values-alt` fails, and it
fails because a stale duplicate is committed, not because it is unreproducible.

**4. The pilot item draw reproduces exactly.** Given the ETHICS CSVs:

```
$ python3 build_he300_arcs.py --n-arcs 10 --turns 10 --seed 42 \
      --stratum axiotic_primary --safety-dir /tmp/arcs
  gold mix   60 label-0 / 40 label-1
  draw sha   sha256:3c7495e7d3becf37d64b38c32cee46523c766e3680dc914ecf97573005167f20
```

matching `TORQUE_REGIME.yaml:1520` and the declared gold mix. The pilot's identity
is genuinely pinned, so the "excluded from the main draw" rule is checkable rather
than asserted. The builder refuses item reuse across arcs and refuses odd turn
counts, and both refusals are real code paths.

**5. The intervention is readable without the agent.** All four
`arms/*.json` are committed in full (~316 KB each) with the corpus text inlined
and no absolute paths. An outsider can diff `h3ere-ciris.json` against
`h3ere-alt.json` and see the exact bytes that differ, offline. `preflight.py`'s
B2/B2b/B2c gates — 192 keys, identical key space, 10/10/4 differing — run with no
external dependency and are meaningful.

**6. The residue detector runs and the corpora are clean.** Both accord corpora,
all fourteen unit artifacts, zero undeclared candidates; four adjudicated lines
in the accord each carry a written reason. `corpora/adjudicated.tsv` refuses a row
with an empty reason, and that refusal is enforced (`detect_residue.py:113`).

**7. The pinned template hash is correct.** `default.yaml` at `v2.9.11-stable` is
`75f2d11d…`, matching `build_arm_manifest.py:68`, and the builder refuses on
mismatch rather than warning.

**8. Two κ figures recompute.** `batch3` κ=0.432 (20/27) and `conflict16` κ=1.000
(16/16) reproduce by hand from `partition/adjudications/`, matching
`partition/FROZEN.md`. `kappa.py` reproduces `kappa_2026-08-07/RESULT.txt`'s A-vs-B
section (κ=0.831, 26/30) exactly; the A-vs-SHIPPED sections need a compose dump
that is not committed.

**9. The design record is unusually candid.** The polyglot limit, the
identity-text confound in the blank arm, the identity-mapped slots 2 and 4, the
strict-parser coverage cost, the two judge defects, the one-document-per-arm
underidentification problem, and `PILOT_BLOCKER.md` itself are all stated against
interest. `docs/index.html` reports "designed, not yet run. There are no results"
and names what the design cannot show. None of the findings above is a case of
the repo overclaiming a *result*; they are all cases of a *check* not doing what
its own comment says.

**What an outsider cannot do today (d), even after every fix above:** check
reported numbers against raw output. No run exists, no traces exist, no analysis
script exists, and the scorer lives in an unpinned third repo. That gap is
expected at this stage — it is listed here so it is not mistaken for an omission.

---

## Verdict

**Not externally reproducible today** — CI cannot run (findings 1–2), the two
gates that would establish provenance silently no-op off this machine (3–4), and
one shipped alt artifact is stale while the gate audits a different file (6) — but
the underlying corpus build genuinely does regenerate bit-for-bit from the shipped
accord plus published diffs, so every blocker is a defect in the checking
apparatus rather than in the artifacts, and all are days of work, not a rebuild.
