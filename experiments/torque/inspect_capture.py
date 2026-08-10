#!/usr/bin/env python3
"""What did the model ACTUALLY see? Read it from the wire, not from the source.

WHY THIS EXISTS. TORQUE captured CEG traces (attestation carriers) and
`results.jsonl` (responses) and treated that as observability. Neither records
the PROMPT. Two mechanism claims were published and withdrawn in that gap:

  * a "reasoning collapse" from 137 to 9 characters, which was the first-message
    privacy notice and nothing else — length is flat from turn 1 once stripped;
  * a self-conditioning hypothesis built on channel history, for a retrieval
    path that returns `[]` on every call in this harness.

Both were reachable-in-source and absent-on-the-wire. Reading `base_observer.py`
proved the path EXISTED; only the captured prompt shows whether it FIRED. One
run of this script would have settled both before either was written down.

Enable capture with (see `tune_local.sh`, on by default):

    CIRIS_LLM_CAPTURE_HANDLER='*'
    CIRIS_LLM_CAPTURE_FILE=/work/ciris/logs/llm_capture.jsonl

GOTCHA. `llm_bus._maybe_capture` opens the file 0600 with O_NOFOLLOW, so a
container-written capture lands root-owned and is UNREADABLE from the host
without sudo. Read it through the container, or `docker cp` it out first:

    docker cp <container>:/work/ciris/logs/llm_capture.jsonl .

Usage
-----
    inspect_capture.py <capture.jsonl> [--probe REGEX=LABEL ...] [--turns]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

#: Blocks whose presence or absence changes what an experiment can claim. Each
#: is a thing someone has already assumed was in the prompt when it was not.
DEFAULT_PROBES = [
    (r"CIRIS_CHANNEL_HISTORY_MESSAGE", "channel history"),
    (r"@CIRIS \(ID:", "agent's own prior speech"),
    (r"CIRIS_OBSERVATION_START", "observation block"),
    (r"Recent messages from other channels", "cross-channel recall"),
    (r"Recent Actions Taken|task_id.*status", "task history"),
]


def load(path: Path) -> list:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  (line {i} unparseable, skipped)", file=sys.stderr)
    return rows


def blob_of(row: dict) -> str:
    return "\n".join(m.get("content") or "" for m in row.get("messages", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--probe", action="append", default=[],
                    help="REGEX=LABEL, repeatable; adds to the defaults")
    ap.add_argument("--turns", action="store_true",
                    help="group by task_id and report prompt growth across turns")
    args = ap.parse_args()

    rows = load(args.capture)
    if not rows:
        raise SystemExit(f"REFUSED: no captured calls in {args.capture}. "
                         f"Capture writes only when CIRIS_LLM_CAPTURE_HANDLER "
                         f"matches; an empty file means it never matched.")

    probes = DEFAULT_PROBES + [tuple(p.split("=", 1)) for p in args.probe if "=" in p]

    print(f"# {len(rows)} captured LLM calls from {args.capture}\n")
    handlers = collections.Counter(r.get("handler", "?") for r in rows)
    print("| handler | calls |")
    print("|---|---|")
    for h, n in handlers.most_common():
        print(f"| `{h}` | {n} |")

    # Prompt composition, from the largest call — the accord dominates and its
    # size is the number people misremember.
    big = max(rows, key=lambda r: len(blob_of(r)))
    print(f"\n## Largest prompt — `{big.get('handler')}`, {len(blob_of(big)):,} B\n")
    print("| # | role | bytes |")
    print("|---|---|---|")
    for i, m in enumerate(big.get("messages", []), 1):
        print(f"| {i} | {m.get('role')} | {len(m.get('content') or ''):,} |")

    print("\n## Blocks present on the wire\n")
    print("| block | calls containing it | total occurrences |")
    print("|---|---|---|")
    for pattern, label in probes:
        rx = re.compile(pattern)
        hits = [len(rx.findall(blob_of(r))) for r in rows]
        n_calls = sum(1 for h in hits if h)
        verdict = f"{n_calls}/{len(rows)}" if n_calls else "**0 — ABSENT**"
        print(f"| {label} | {verdict} | {sum(hits)} |")

    if args.turns:
        # Prompt growth across turns is the accumulation hypothesis, measured.
        # Flat means nothing accumulates, whatever the source says is reachable.
        print("\n## Prompt size by task (turn order)\n")
        by_task: dict = collections.OrderedDict()
        for r in rows:
            by_task.setdefault(r.get("task_id", "?"), []).append(len(blob_of(r)))
        print("| turn | task | calls | mean prompt B | max B |")
        print("|---|---|---|---|---|")
        for i, (task, sizes) in enumerate(by_task.items(), 1):
            print(f"| {i} | `{str(task)[:8]}` | {len(sizes)} | "
                  f"{sum(sizes)//len(sizes):,} | {max(sizes):,} |")
        first = list(by_task.values())[0]
        last = list(by_task.values())[-1]
        d = (sum(last) / len(last)) - (sum(first) / len(first))
        print(f"\nfirst task mean {sum(first)//len(first):,} B -> "
              f"last task mean {sum(last)//len(last):,} B  (**{d:+,.0f} B**)")
        print("\nA flat or shrinking figure falsifies context accumulation for this")
        print("harness regardless of what any code path makes reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
