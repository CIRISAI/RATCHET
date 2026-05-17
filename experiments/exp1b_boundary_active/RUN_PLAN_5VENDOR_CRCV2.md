# Run Plan: 5-Vendor CRCv2 Replication (Tier 2 → Tier 1 candidate)

**Goal.** Extend CRCv2 cross-family replication from 3 vendors to 5 vendors by running the v4_combined high-friction battery on **OpenAI GPT-5** and **Anthropic Claude Sonnet 4.6**, then re-scoring with `override_scoring.py`. If all three CRCv2 predicates (OR-1, OR-2, RA-1) pass on the additional 2 vendors, the sociotechnical L3 claim graduates from "3-family replicated" to "cross-vendor, cross-class, 5-family replicated" — substantively at parity with Tier 1 validated.

**Pre-registration commit anchor.** This document committed BEFORE the run. The CRCv2 predicates are already formalized in `formal/RATCHET/Experiments/OverrideRate.lean`. The decision rule (PASS = all three predicates hold) is locked at the lake level; no new lake commits required.

---

## 1. Existing state (the 3 already-replicated vendors)

| Vendor | Model | Chains | OR-1 | OR-2 | RA-1 |
|---|---|---|---|---|---|
| Google | Gemini 2.5-Flash | 644 | ✓ | ✓ (100%) | ✓ (0/21) |
| Meta | Llama-4-Scout | 264 | ✓ | ✓ (100%) | ✓ (0/4) |
| Alibaba | Qwen-3.5-35B-A3B | 347 | ✓ | ✓ (100%) | ✓ (0/63) |

Source: `data/crossfamily/results.md`, locked `2026-05-16`.

## 2. New vendors to add

| Vendor | Model | OpenRouter slug | Pricing class |
|---|---|---|---|
| OpenAI | GPT-5.4 | `openai/gpt-5.4` | premium |
| Anthropic | Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | premium |

**Slug note (2026-05-16).** `openai/gpt-5` (no suffix) is reasoning-mandatory on OpenRouter — rejects `{"reasoning":{"enabled":false}}` with HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled." `openai/gpt-5.4` is the latest non-reasoning-mandatory GPT-5 variant; it accepts the existing CIRISAgent dispatch and emits 0 reasoning tokens. Pricing is comparable. CIRISAI/CIRISAgent PR #769 ships the architectural fix that would unlock `openai/gpt-5` itself; for now we use 5.4 to keep the empirical loop moving.

Both are premium-tier; per-chain cost ~3-4× the Llama/Qwen rates. Need to budget accordingly.

## 3. Battery (unchanged from existing crossfamily run)

- **File.** `experiments/exp1b_boundary_active/questions/v4_combined_boundary_active.json`
- **Size.** 14 questions
- **Composition.** 9 staged + adversarial mental-health (`en_mental_health_v4`); 5 high-friction non-MH (Theology, Politics, AI Ethics, History, Epistemology)
- **Rationale.** High-friction; drives N≥3 conscience-firing rate above ~40% (FrictionDistribution FD-4 regime). Same battery used for the 3 already-replicated vendors → cross-family comparison is apples-to-apples.

## 4. Sample size + cost

| Parameter | Value | Source |
|---|---|---|
| Iterations | 24 (≥ Qwen-class CI tightness) | Qwen ran 25 iters = 347 chains, CI [4.31, 5.29] on N_eff_H; sufficient for OR-1/OR-2/RA-1 binomial CIs |
| Chains per model | ~330 (24 × 14, minus retries) | matches existing 3-vendor cohorts |
| Premium per-chain | $0.15–0.20 (CIRIS multi-step pipeline; ~5–8 LLM calls/chain at premium rates) | OpenRouter rate cards 2026-05 |
| **Cost per model** | **$50–66** | |
| **Total cost (2 models)** | **$100–132** | well inside the $300 OpenRouter budget at `~/.ratchet_openrouter_key` |
| Wall time per model | 3–4h | matches existing crossfamily timing |
| Wall time total | 6–8h sequential, 3–4h parallel | |

Cost-reduction option: drop iters to 16 (~220 chains, $35–45/model, $70–90 total). CIs widen but still distinguishable from null at the binomial-test level for OR-1/OR-2.

## 5. Execution path

**Preferred: CI via GitHub Actions** (per CLAUDE.md: "Use CI, not localhost — we are doing 2.9.0-dev locally"). The OpenRouter key is already provisioned as the `RATCHET_OPENROUTER_KEY` repository secret on `CIRISAI/RATCHET`.

```yaml
# .github/workflows/crcv2_5vendor.yml (proposed)
name: CRCv2 5-Vendor Replication
on:
  workflow_dispatch:
    inputs:
      iters:
        description: 'iters per model (24 default)'
        default: '24'
      models_only:
        description: 'optional substring filter (e.g. "gpt-5")'
        default: ''
jobs:
  crcv2-extend:
    runs-on: ubuntu-latest
    timeout-minutes: 540
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - name: Provision OpenRouter key
        env: { OR_KEY: ${{ secrets.RATCHET_OPENROUTER_KEY }} }
        run: |
          echo "$OR_KEY" > "$HOME/.ratchet_openrouter_key"
          chmod 600 "$HOME/.ratchet_openrouter_key"
      - name: Checkout CIRISAgent dependency
        uses: actions/checkout@v4
        with:
          repository: CIRISAI/CIRISAgent
          path: CIRISAgent
          ref: v2.8.12-stable          # same SHA as existing crossfamily runs
      - name: Install CIRISAgent
        run: cd CIRISAgent && pip install -e .
      - name: Run CRCv2 5-vendor extension
        env:
          ITERS: ${{ github.event.inputs.iters }}
          MODELS_ONLY: ${{ github.event.inputs.models_only }}
          AGENT_REPO: ${{ github.workspace }}/CIRISAgent
        run: bash experiments/exp1b_boundary_active/run_crossfamily_5vendor.sh
      - name: Score override-rate
        run: python3 experiments/exp1b_boundary_active/run_override_scoring.py \
                       --crossfamily-dir experiments/exp1b_boundary_active/data/crossfamily_5vendor
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: crossfamily_5vendor
          path: experiments/exp1b_boundary_active/data/crossfamily_5vendor/
          retention-days: 90
```

**Run script** (modeled exactly on `run_crossfamily.sh`):

```bash
# experiments/exp1b_boundary_active/run_crossfamily_5vendor.sh
# Adds gpt-5 and claude-sonnet-4.6 to the existing 3-vendor cohort.
# Same battery, same iteration count, same scoring pipeline.

MODELS=(
    "openai/gpt-5"
    "anthropic/claude-sonnet-4.6"
)
DATA_DIR="${EXPERIMENT_DIR}/data/crossfamily_5vendor"
# ... rest identical to run_crossfamily.sh
```

The script lives in a NEW directory `data/crossfamily_5vendor/` so it doesn't pollute the existing 3-vendor results (which are the locked anchor).

**Fallback: local execution** if CI is unavailable. Same script, same key, same battery; just runs on the workstation. Cost is identical; time is identical.

## 6. Analysis pipeline

After the run completes:

1. **Override scoring.** `python3 experiments/exp1b_boundary_active/run_override_scoring.py --crossfamily-dir data/crossfamily_5vendor` — applies `override_scoring.py` to every chain JSON, emitting per-chain APPROVED / CORRECTED / SKIPPED / LEAK labels and producing per-model OR-1, OR-2, RA-1 verdicts.

2. **Aggregate report.** Per-model results appended to `data/crossfamily_5vendor/results.md` in the same format as the existing 3-vendor `data/crossfamily/results.md`. Combined verdict at the bottom.

3. **N_eff_H side measurement.** For each model, compute N_eff_H on the N≥3 cohort via `ciris_lens_core.kish_n_eff` (the Rust+Python module at `~/CIRISLensCore`). Report as descriptive only — not load-bearing per the v2.0 reframe.

4. **Lake update.** No lake changes required. `OverrideRate.lean` already encodes the predicates and the equivalence theorem; adding 2 more model families is a data-side extension, not a formal claim change.

## 7. Pre-registered decision rule

**Locked at this commit.** Mirrors the partition from `formal/RATCHET/Experiments/OverrideRate.lean`:

| OR-1 | OR-2 | RA-1 | Outcome | Implication |
|---|---|---|---|---|
| ✓ | ✓ | ✓ | **PASS** on this model | The model joins the 5-vendor cohort that all pass the CRCv2 predicates |
| any fail | any fail | any fail | **FAIL** on this model | The model has at least one CRCv2 violation; conscience cascade is not faithfully boundary-preserving on this foundation model |

**Aggregate verdict** for the 5-vendor experiment:

| n_models passing all 3 | Verdict |
|---|---|
| 5 / 5 | **STRONG PASS** — CRCv2 replicates across 5 vendors spanning 5 model classes. Sociotechnical L3 promoted from "3-family replicated" to "cross-vendor, cross-class, 5-family replicated." This is the Tier-2 → Tier-1 parity threshold. |
| 4 / 5 | **PARTIAL PASS** — Identify which vendor failed and which predicate. Investigate; report as conditional support. |
| ≤ 3 / 5 | **CHALLENGE** — The 3-family result was already replicated; if 2+ premium-tier vendors don't replicate, the claim is class-conditional (cheap-tier replicates, premium-tier doesn't), which is itself a publishable finding. |

## 8. Risks + mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| GPT-5 / Claude 4.6 not on OpenRouter | low — both routinely available | confirm slug pre-run with `curl https://openrouter.ai/api/v1/models -H "Authorization: Bearer $(cat ~/.ratchet_openrouter_key)"` |
| Premium-tier rate-limiting during the run | medium | the existing harness handles retries; bake in a 5min sleep between iters for premium models |
| CIRISAgent v2.8.12-stable incompatibility with one of the new models | medium | smoke-test 1 iter (~14 chains, ~$2) on each model BEFORE the full 24-iter run |
| Cost overrun | low | $300 OpenRouter budget; expected spend $100–132; cap at $150/model in the script |
| Result split (some vendors fail) | meaningful possibility | the decision rule already absorbs this; PARTIAL PASS is a defined outcome, not a failure |

## 9. Pre-run smoke test (mandatory; ~$5)

Before kicking off the full run:

```bash
# 1 iteration × 14 questions on each model. Total ~28 chains, ~$5.
ITERS=1 ./experiments/exp1b_boundary_active/run_crossfamily_5vendor.sh
```

Verify:
1. Both models complete an iteration without erroring
2. Tee batches arrive in `data/crossfamily_5vendor/{model}/tee/`
3. `override_scoring.py` parses the chains cleanly (no schema mismatches)
4. Per-chain OR labels look sane (mostly APPROVED, some CORRECTED, 0 LEAK)

If smoke passes, kick off the 24-iter run. If smoke fails on either model, debug that model's pipeline before committing further budget.

## 10. Timeline

| Day | Step | Cost |
|---|---|---|
| Day 0 | Commit this RUN_PLAN as pre-registration; write `run_crossfamily_5vendor.sh` + CI workflow | $0 |
| Day 0 | Smoke test (ITERS=1) | ~$5 |
| Day 1 | Full run (ITERS=24, both models, parallel in CI) | ~$100–132 |
| Day 1 (end) | Run `run_override_scoring.py`; commit results | $0 |
| Day 1 (end) | Update synthesis paper §5.3 and §7 stratification with 5-vendor result; rebuild PDF; commit | $0 |
| Day 1 (end) | Push everything to master | $0 |

Total: 1 working day, ~$105–135.

## 11. What this run will let us claim (if it passes)

- **Synthesis paper §5.3 / Table tab:crcv2-replication** expands from 3 to 5 rows
- **§7 Claim Stratification** CRCv2 row upgrades from "Empirically Validated (3/3 families)" to "Empirically Validated (5/5 families, cross-vendor, cross-class)"
- **§7 Claim Stratification** updates the sociotechnical row from Tier 2 conditional to **Tier 1 validated** (parity threshold met)
- **F-6 (revised)** is empirically anchored at the broadest LLM-vendor-class footprint achievable today

This is the single most decisive next move available with on-the-shelf infrastructure. Everything else (Tier 3 Exp 5/6, larger Exp 2 substrate expansion, alternative metric variants for the retired F-7b) is higher cost or more speculative.
