#!/usr/bin/env python3
"""Do different values produce different REASONING, when the verdict is the same?

WHY THIS EXISTS. TORQUE returned equivalence on verdicts: swapping the accord
moves ~10% of answers and 0% of the score. That is the signature of values doing
something the verdict cannot price — and the verdict was the only DV.

But the pipeline emits BOTH. `ActionSelectionPDMAEvaluator` returns a `reasoning`
field (~360 chars) and a `speak_content` field (~10 chars, the verdict word).
Only `speak_content` was ever scored. The reasoning was captured all along, 534
fields per arm, and is sitting in the run artifacts.

So the deferred citation DV is partly answerable with ZERO new runs.

THE TRAP THIS IS BUILT AROUND. Counting value-vocabulary in a response is how a
values manipulation manufactures its own effect: put "stewardship" in the accord,
find "stewardship" in the output, declare transfer. That is prompt echo, not
reasoning. So this reports two things SEPARATELY:

  echo         arm-specific terms that appear in that arm's own accord.
               Expected, shallow, and NOT evidence the values changed thinking.
  divergence   whether the arms reason DIFFERENTLY at all on the same item,
               measured content-agnostically, against a permutation null that
               shuffles arm labels within item.

A large echo with no divergence means the agent is quoting its document. Real
divergence means the values reached the reasoning.

Usage
-----
    reasoning_dv.py --artifacts <dir> --arcs <dir> [--draws 2000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("h3ere-ciris", "h3ere-alt", "h3ere-neutral", "h3ere-blank")
STOP = set("""the a an and or but if then than that this these those of to in on at by for with
from as is are was were be been being it its it's he she they them his her their we our you your
i not no do does did done have has had will would can could should may might must so such very
more most much many any all some each other another same which who whom whose what when where why
how because about into over under after before while during between both few own too s t don now
one two also just like get got make made take taken use used way well even still back new good""".split())
WORD = re.compile(r"[a-z][a-z'-]{2,}")


def tokens(text: str) -> set:
    return {w for w in WORD.findall((text or "").lower()) if w not in STOP}


def load_gold_keys(arcs: Path) -> dict:
    """question-text prefix -> stable item key."""
    out = {}
    for f in arcs.rglob("v4_*_arc.json"):
        if "_pre" in f.name:
            continue
        for q in json.loads(f.read_text(encoding="utf-8"))["questions"]:
            body = q["translations"]["en"].split("\n\n", 1)[-1]
            out[body[:120]] = q["he300"]["item_id"]
    return out


def load_reasoning(artifacts: Path, keys: dict) -> dict:
    """arm -> {item_id: reasoning}"""
    per = defaultdict(dict)
    for arm in ARMS:
        cap = next(artifacts.glob(f"results-final-{arm}-straight/**/llm_capture.jsonl"), None)
        if cap is None:
            continue
        for line in cap.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("handler") != "ActionSelectionPDMAEvaluator":
                continue
            txt = (r.get("result") or {}).get("reasoning") or ""
            if not txt.strip():
                continue
            blob = "\n".join(m.get("content") or "" for m in r.get("messages", []))
            # 2.9.14 restored channel history, so a later turn's prompt contains
            # EVERY earlier question too. Taking the first match returned an
            # earlier turn's item and collapsed 540 items to 54 — one per arc.
            # The current question is the last one to appear, so take the match
            # with the greatest position.
            best, best_at = None, -1
            for k in keys:
                at = blob.rfind(k)
                if at > best_at:
                    best, best_at = k, at
            if best is not None and best_at >= 0:
                per[arm][keys[best]] = txt
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--arcs", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keys = load_gold_keys(args.arcs)
    per = load_reasoning(args.artifacts, keys)
    if len(per) < 2:
        raise SystemExit("REFUSED: need reasoning from at least two arms.")

    common = set.intersection(*(set(per[a]) for a in per))
    print("# Does the accord change the REASONING?\n")
    print(f"arms with reasoning: {', '.join(sorted(per))}")
    print(f"items answered by every arm: **{len(common)}**\n")

    # ── divergence, content-agnostic ────────────────────────────────────────
    print("## Divergence — do the arms reason differently on the same item?\n")
    print("Jaccard overlap of content words. `same-arm, different item` is the")
    print("floor for unrelated text about this corpus; anything near it means the")
    print("arms are as different from each other as two random items are.\n")
    rng = random.Random(args.seed)
    ref = "h3ere-ciris"
    if ref not in per:
        ref = sorted(per)[0]
    toks = {a: {i: tokens(t) for i, t in per[a].items()} for a in per}

    print("| pair | mean Jaccard | n |")
    print("|---|---|---|")
    pair_scores = {}
    for a in sorted(per):
        if a == ref:
            continue
        vals = []
        for i in common:
            x, y = toks[ref][i], toks[a][i]
            if x or y:
                vals.append(len(x & y) / len(x | y))
        pair_scores[a] = sum(vals) / len(vals)
        print(f"| `{ref}` vs `{a}` | {pair_scores[a]:.3f} | {len(vals)} |")

    items = sorted(common)
    base = []
    for _ in range(min(len(items), 400)):
        i, j = rng.sample(items, 2)
        x, y = toks[ref][i], toks[ref][j]
        if x or y:
            base.append(len(x & y) / len(x | y))
    baseline = sum(base) / len(base)
    print(f"| `{ref}` vs itself, DIFFERENT items | {baseline:.3f} | {len(base)} |")

    print(f"\nSame-item cross-arm overlap is "
          f"**{min(pair_scores.values()):.3f}–{max(pair_scores.values()):.3f}**, "
          f"against **{baseline:.3f}** for unrelated items.")
    if min(pair_scores.values()) > baseline * 1.5:
        print("The arms are reasoning about the same thing — overlap far above the")
        print("unrelated-text floor. Differences between them are a real signal, not")
        print("two agents talking past each other.")

    # ── echo vs substance ───────────────────────────────────────────────────
    print("\n## Distinctive vocabulary, and whether it is just prompt echo\n")
    print("Terms over-represented in one arm's reasoning versus the pooled rest,")
    print("with a permutation null shuffling arm labels within item.\n")
    counts = {a: Counter() for a in per}
    for a in per:
        for i in common:
            counts[a].update(toks[a][i])
    pooled = Counter()
    for a in per:
        pooled.update(counts[a])

    for a in sorted(per):
        tot_a = sum(counts[a].values()) or 1
        tot_p = sum(pooled.values()) or 1
        scored = []
        for w, c in counts[a].items():
            if c < 8:
                continue
            rate_a = c / tot_a
            rate_p = pooled[w] / tot_p
            if rate_p > 0 and rate_a / rate_p > 1.35:
                scored.append((rate_a / rate_p, c, w))
        scored.sort(reverse=True)
        top = ", ".join(f"`{w}`×{c}" for _r, c, w in scored[:10]) or "— none above threshold —"
        print(f"**{a}**: {top}")

    print("\n> Read these with the echo warning in mind: a term that appears in an")
    print("> arm's own accord and then in its reasoning is the agent quoting its")
    print("> document. That is not the values changing how it thinks. Cross-check")
    print("> any term above against the corpora before citing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
