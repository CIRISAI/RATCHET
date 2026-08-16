#!/usr/bin/env python3
"""Measure our judge against human labels before trusting a number it produces.

`do-not-answer` ships 3,600 responses across six models with HUMAN action
labels. That is a validation set, and it is the reason to borrow their taxonomy
rather than invent one: an invented schema has nothing to be checked against.

This runs OUR judge — our prompt, our model choice — over a stratified sample of
those responses and reports agreement. The agreement figure is then part of the
instrument's description, not a footnote: any downstream claim inherits it.

WHY THIS STEP IS NOT OPTIONAL HERE. Four hand-written classifiers in this
campaign were wrong in ways that inverted a conclusion — a 400-character window
that read a reasoning chain as a refusal, a marker list that scored the run's
best behaviour as its worst, a path glob that scored a Bengali report from May.
Each was caught by an audit that happened to be run. A judge validated against
human labels turns "I checked" into a number someone else can check.

Usage
-----
    validate_judge.py --dna <dna.json> --n 180 [--model ...]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refusal_taxonomy import ACTION, REFUSAL_ACTIONS, build_prompt  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"


def judge(request: str, response: str, model: str, key: str) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(request, response)}],
        "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            m = re.search(r"\{[^{}]*\}", txt, re.S)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            if attempt == 2:
                return {}
            time.sleep(3)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dna", type=Path, required=True)
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--model", default="anthropic/claude-opus-4-5")
    ap.add_argument("--key-file", type=Path,
                    default=Path.home() / ".openrouter_key")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    key = args.key_file.read_text().strip()
    rows = json.loads(args.dna.read_text())
    models = [k[:-7] for k in rows[0] if k.endswith("_action")]

    # STRATIFY BY LABEL. The natural distribution is 40% level-4 and 1.5%
    # level-2; a random sample would report agreement dominated by the easy
    # majority class and say nothing about the levels the study turns on
    # (0 = bare refusal vs 3 = refusal with care).
    pool = defaultdict(list)
    for r in rows:
        for m in models:
            a, t = r.get(f"{m}_action"), r.get(f"{m}_response")
            if a is not None and t:
                pool[a].append((r["question"], t, a))
    rng = random.Random(42)
    per = max(1, args.n // len(pool))
    sample = []
    for a in sorted(pool):
        rng.shuffle(pool[a])
        sample += pool[a][:per]
    rng.shuffle(sample)

    print(f"# Judge validation — {args.model}\n")
    print(f"stratified sample: {len(sample)} responses, "
          f"{per} per action level, from {len(rows)} items x {len(models)} models\n")

    exact = 0
    binary = 0            # does it agree on refusal-vs-not, the axis we use
    conf = Counter()
    scored = 0
    for i, (q, resp, gold) in enumerate(sample, 1):
        out = judge(q, resp, args.model, key)
        pred = out.get("action")
        if pred is None:
            continue
        scored += 1
        exact += (pred == gold)
        binary += ((pred in REFUSAL_ACTIONS) == (gold in REFUSAL_ACTIONS))
        conf[(gold, pred)] += 1
        if i % 30 == 0:
            print(f"  … {i}/{len(sample)}", flush=True)

    if not scored:
        raise SystemExit("REFUSED: judge returned nothing parseable — do not "
                         "report agreement from an empty run.")

    print(f"\n| metric | value |")
    print(f"|---|---|")
    print(f"| responses scored | {scored} / {len(sample)} |")
    print(f"| **exact agreement** (6-way) | **{exact/scored:.1%}** |")
    print(f"| **refusal-vs-not agreement** | **{binary/scored:.1%}** |")

    print("\n## Confusion — human label (row) vs ours (col)\n")
    ks = sorted(ACTION)
    print("| human \\ ours | " + " | ".join(str(k) for k in ks) + " |")
    print("|" + "---|" * (len(ks) + 1))
    for g in ks:
        row = [str(conf.get((g, p), 0)) for p in ks]
        print(f"| **{g}** | " + " | ".join(row) + " |")

    print("\n## Reading\n")
    print("The BINARY figure is the one downstream claims rest on: this campaign")
    print("uses refusal-vs-not plus the care triple, not the 6-way label. Exact")
    print("6-way agreement is reported anyway because a judge that collapses")
    print("levels 0 and 3 would look fine on the binary axis while destroying")
    print("the bare-versus-with-care distinction the study is about.")
    if args.out:
        args.out.write_text(json.dumps(
            {"model": args.model, "n": scored, "exact": exact / scored,
             "binary": binary / scored,
             "confusion": {f"{g}->{p}": c for (g, p), c in conf.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
