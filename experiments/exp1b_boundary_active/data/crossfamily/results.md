# Cross-Family Replication — OR-1, OR-2, RA-1

## `llama-4-scout`

- Chains scored: **264**
- APPROVED / CORRECTED / SKIPPED / LEAK: 260 / 4 / 0 / **0**
- **OR-1 (zero leak):** ✓ PASS
- **OR-2 (full alignment):** 100.00% ✓ PASS
- **RA-1 (ratchet asymmetry):** violations=0 / corrected=4 ✓ PASS
- N_eff_H (N≥3 subset, n=71): point 6.401, 95% CI [5.122, 6.474]
- Firing distribution: N=0:70 N=1:2 N=2:5 N=3:0 N=4:71

## `qwen-3.5-35b-a3b`

- Chains scored: **347**
- APPROVED / CORRECTED / SKIPPED / LEAK: 284 / 63 / 0 / **0**
- **OR-1 (zero leak):** ✓ PASS
- **OR-2 (full alignment):** 100.00% ✓ PASS
- **RA-1 (ratchet asymmetry):** violations=0 / corrected=63 ✓ PASS
- N_eff_H (N≥3 subset, n=70): point 5.007, 95% CI [4.307, 5.289]
- Firing distribution: N=0:56 N=1:12 N=2:6 N=3:16 N=4:54

---

## CRCv2 Replication Verdict

- OR-1 across all models: ✓ REPLICATED
- OR-2 across all models: ✓ REPLICATED
- RA-1 across all models: ✓ REPLICATED

With the Gemini v4_combined cohort already validating all three, **three** model families all support the CRCv2 L3 predicates.
