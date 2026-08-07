"""Refuse a cohort that is ABSENT but reads as behaviour.

`load_chains_from_tee_dir` already refuses an EMPTY cohort, because a silent
empty return would compute every downstream statistic over nothing and report a
clean run. This is the case that guard does not catch: a cohort that is the
right SHAPE and the wrong CONTENT.

The EN battery run of 2026-08-05 (CIRISAgent run 31183628588) is the worked
example. Three individually-reasonable links composed into it:

1. the runtime starts without an LLM, silently — no API key means
   ``logger.info(...)`` and ``return``, not a warning and not a refusal, so the
   agent boots, accepts messages, and has nothing to think with;
2. every interact rides to a fixed 180 s ceiling, because a `setdefault` wins
   over the battery's own 1800 whenever it is set first;
3. the harness scores ``success=bool(response_text)``, and the timeout literal
   "Still processing. Check back later…" is non-empty text.

Result: nine green checkmarks, nine timeout strings recorded as the agent's
answers, and `task_id` never assigned because no task was ever created. Scored,
that judges the agent against a canned string — the same shape as the AM q06
failure, where a timeout string failed on script ratio for containing no
Ethiopic and was read as the agent writing in the wrong script.

For a pre-registered campaign this is worse than missing data. Missing data is
visible. This is absent data wearing the shape of behaviour, and it reads as a
result.

The three signatures, any one of which is disqualifying:

* **fixed duration** — nine questions at 180.3 s is a ceiling, not nine agents
  each deliberating for exactly three minutes;
* **no task id** — no task created means the failure is upstream of the LLM
  call entirely;
* **still-processing literal** — matched PER LOCALE. An English-only check
  reproduces the AM q06 defect on any non-English cell.

Wire into the loader before scoring; do not use as an after-the-fact review.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

#: Localized still-processing strings. English-only matching is the AM q06
#: defect: the literal is emitted in the cell's own language, so a non-English
#: cell sails past an English check and is then judged for writing in the wrong
#: script. Extend from the agent's localization bundle rather than by hand —
#: `prompts.*still_processing*` — and treat an unlisted locale as UNKNOWN, not
#: as clean.
STILL_PROCESSING: Dict[str, Sequence[str]] = {
    "en": ("Still processing. Check back later",),
}

#: Durations within this fraction of each other are a ceiling, not deliberation.
DURATION_TOLERANCE = 0.02

#: Absent task identifiers as the harness writes them.
ABSENT_TASK_IDS = {"", "-", "—", "None", "null"}


class AbsentCohort(RuntimeError):
    """The cohort is the right shape and the wrong content. Do not score it."""


def _fixed_duration(durations: Sequence[float]) -> Optional[str]:
    usable = [d for d in durations if d and d > 0]
    if len(usable) < 3:
        return None
    spread = (max(usable) - min(usable)) / max(usable)
    if spread <= DURATION_TOLERANCE:
        return (
            f"{len(usable)} responses within {spread:.1%} of {statistics.median(usable):.1f}s "
            f"— a fixed ceiling, not {len(usable)} independent deliberations"
        )
    return None


def _absent_tasks(task_ids: Sequence[Any]) -> Optional[str]:
    absent = [t for t in task_ids if str(t).strip() in ABSENT_TASK_IDS]
    if absent and len(absent) == len(task_ids):
        return f"all {len(task_ids)} rows carry no task id — no task was created, so this is upstream of the LLM call"
    if absent:
        return f"{len(absent)}/{len(task_ids)} rows carry no task id"
    return None


def _still_processing(responses: Sequence[str], locale: str) -> Optional[str]:
    needles = STILL_PROCESSING.get(locale)
    if needles is None:
        return (
            f"locale {locale!r} has no registered still-processing literal — cannot rule out "
            f"timeout text scored as an answer. Register it before scoring; an unchecked locale "
            f"is not a clean one"
        )
    hits = [r for r in responses if any(n in (r or "") for n in needles)]
    if hits:
        return f"{len(hits)}/{len(responses)} responses are the still-processing literal, recorded as the agent's answer"
    return None


def assert_cohort_present(
    rows: Sequence[Dict[str, Any]],
    *,
    locale: str,
    duration_key: str = "duration_s",
    task_key: str = "task_id",
    response_key: str = "agent_response",
) -> None:
    """Raise :class:`AbsentCohort` if the cohort is absent-but-shaped.

    Checks all three signatures and reports EVERY one that fires, so a caller
    fixing the run sees the whole picture rather than the first item.
    """
    if not rows:
        raise AbsentCohort("zero rows — refusing to score an empty cohort")

    problems = [
        p
        for p in (
            _fixed_duration([r.get(duration_key) for r in rows]),  # type: ignore[arg-type]
            _absent_tasks([r.get(task_key) for r in rows]),
            _still_processing([r.get(response_key, "") for r in rows], locale),
        )
        if p
    ]
    if problems:
        raise AbsentCohort(
            f"cohort ({locale}, n={len(rows)}) is absent, not weak — it has the shape of behaviour "
            f"and none of the content:\n  - " + "\n  - ".join(problems)
        )
