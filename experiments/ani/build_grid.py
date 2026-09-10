#!/usr/bin/env python3
"""Lay out every ANI cell measured so far, on the corrected criteria.

One table, so the confounds are visible rather than argued about. Each row is a
(agent version, model, arm) cell; every number is scored by the SAME scorer, so
differences are the cell and not the measure.

Read down a column to compare arms; read across matched rows to compare agent
versions or models. The pairs that isolate one thing are named at the bottom.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torque"))
from ani_score import (DEFERRAL, ENDS_QUESTION, HOOK_PROMPT, JUDGE_MODELS,  # noqa: E402
                       _warm_offset, load, unprompted_recall)
from judge_pool import JudgePool  # noqa: E402

#: (label, captures dir, agent ref, model, retention-discipline present)
CELLS = [
    ("Aug baseline", "ani",  "v2.9.37", "scout", False),
    ("scout, after", "ani2", "v2.11.1", "scout", True),
    ("qwen, after",  "ani3", "v2.11.1", "qwen",  True),
    ("qwen, before", "ani4", "v2.9.48", "qwen",  False),
    ("qwen, after",  "ani5", "v2.11.1", "qwen",  True),
]
SP = Path("/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/scratchpad")


def main() -> int:
    key = Path.home().joinpath(".openrouter_key").read_text().strip()
    rows = []
    for label, d, ref, model, disc in CELLS:
        root = SP / d / "all"
        if not root.exists():
            continue
        pool = JudgePool(key, cache=Path(f"judge_cache_{d}.jsonl"), verbose=False)
        data = load(root)
        for arm in sorted(data):
            turns = data[arm]
            n_def = sum(1 for r in turns if DEFERRAL.search(r["reply"] or ""))
            live = [r for r in turns
                    if r["probe"] != "history" and not DEFERRAL.search(r["reply"] or "")]
            if not live:
                continue
            # counted
            q = sum(1 for r in live if ENDS_QUESTION.search((r["reply"] or "").strip()))
            rec_h = rec_o = 0
            for h in ("cold", "warm"):
                sub = [r for r in live if r["arc"] == "a0_hooks" and r["history"] == h]
                off = _warm_offset() if h == "warm" else 0
                sub = [r for r in sub if r["index"] - off > 2]
                a, b = unprompted_recall(sub)
                rec_h += a; rec_o += b
            # judged
            outs = pool.map([(m, HOOK_PROMPT.format(user=r["user"][:800],
                                                    reply=r["reply"][:2000]))
                             for r in live for m in JUDGE_MODELS])
            k = len(JUDGE_MODELS)
            hooks = {x: 0 for x in ("flattery", "unearned_novelty",
                                    "uninvited_intimacy", "retention")}
            scored = 0
            for i in range(len(live)):
                vs = [o for o in outs[i * k:(i + 1) * k] if o]
                if not vs:
                    continue
                scored += 1
                for x in hooks:
                    hooks[x] += any(o.get(x) for o in vs)
            rows.append(dict(label=label, ref=ref, model=model, disc=disc, arm=arm,
                             n=len(live), defer=n_def, total=len(turns),
                             q=q, rec_h=rec_h, rec_o=rec_o, scored=scored, **hooks))
            print(f"  scored {label:14} {arm:14} n={len(live):3}", flush=True)

    print("\n# ANI grid — every cell on the corrected criteria\n")
    print("| agent | model | retention text | arm | deferrals | flattery | intimacy | "
          "retention | unprompted recall | q-density |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pc = lambda v, d: f"{v} ({v/d:.0%})" if d else "—"    # noqa: E731
        print(f"| {r['ref']} | {r['model']} | {'yes' if r['disc'] else 'no'} | "
              f"`{r['arm']}` | {pc(r['defer'], r['total'])} | "
              f"{pc(r['flattery'], r['scored'])} | {pc(r['uninvited_intimacy'], r['scored'])} | "
              f"**{pc(r['retention'], r['scored'])}** | "
              f"**{r['rec_h']}/{r['rec_o']}** | {pc(r['q'], r['n'])} |")
    Path("grid.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
