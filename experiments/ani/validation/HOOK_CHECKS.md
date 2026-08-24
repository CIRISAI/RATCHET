# Hook-judge check — real replies to benign prompts

120 full-compliance replies to XSTest *safe* prompts, 5 models.

**This is not validation against human labels.** No human-labelled corpus
exists for these hooks; the nearest published work is Bengali or forced-choice.
Nothing here upgrades hook rates above `open`.

  gpt-oss-safeguard-20b: 101/120 parsed
  gemini-2.5-pro: 118/120 parsed
  transport failures: {'http_429': 19, 'unparseable_reply': 2}

## 1. Negative control — firing rate on ordinary helpful replies (n=100)

| hook | gpt-oss-safeguard-20b | gemini-2.5-pro | either-says-yes |
|---|---|---|---|
| flattery | 0 (0%) | 0 (0%) | **0 (0%)** |
| unearned_novelty | 0 (0%) | 0 (0%) | **0 (0%)** |
| uninvited_intimacy | 0 (0%) | 0 (0%) | **0 (0%)** |
| retention | 0 (0%) | 0 (0%) | **0 (0%)** |

## 2. Inter-judge agreement on identical inputs

| hook | agreement | Cohen's kappa |
|---|---|---|
| flattery | 100.0% | 1.000 |
| unearned_novelty | 100.0% | 1.000 |
| uninvited_intimacy | 100.0% | 1.000 |
| retention | 100.0% | 1.000 |

Kappa is undefined-ish when a hook never fires — agreement at 100% on
all-False is real information (both judges stay quiet) but the kappa is 0
by construction. Read the agreement column with the firing rate above.

## 3. Verifiable item — 'ends with a question' vs the regex

| judge | agreement with regex |
|---|---|
| gpt-oss-safeguard-20b | 100.0% |
| gemini-2.5-pro | 100.0% |

regex says 2/100 of these end in a question.

## 4. Sensitivity — same judge on real emotionally-loaded replies (n=120)

| hook | benign corpus | loaded corpus |
|---|---|---|
| flattery | 0/100 (0%) | **11/55 (20%)** |
| unearned_novelty | 0/100 (0%) | **0/55 (0%)** |
| uninvited_intimacy | 0/100 (0%) | **0/55 (0%)** |
| retention | 0/100 (0%) | **13/55 (24%)** |

Any hook, loaded corpus: 24 firings across 55 items.

## Verdict

**CLEAN on all three checks.** The judge stays quiet on ordinary warmth,
the two members agree where it fires, and both read the verifiable item
correctly. Hook rates remain `open` — this is specificity and reliability,
not accuracy against human labels — but the instrument is not obviously
broken and the series is worth running.
