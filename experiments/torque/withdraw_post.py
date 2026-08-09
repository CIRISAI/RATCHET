#!/usr/bin/env python3
"""The POST half of the withdrawal: bare provider, carrying the agent's transcript.

WHY THIS SHAPE. Six attempts at a runner that boots the agent itself and drives
both halves failed on six different pieces of qa_runner's startup sequence —
token path, missing user, fixed sleep, an AgentState string, a PID check, and a
`set -e` exit. Every one is something qa_runner already handles, and the straight
arms have been green throughout precisely because they let it.

So nothing here starts an agent. The withdrawal is assembled from two paths that
both already work:

    turns 1-5   an ORDINARY battery cell (`..._pre`) run by qa_runner
    turns 6-10  a plain provider call, made here, seeded with that cell's
                transcript

The switch is the boundary between two files. `intact` carries the agent's turns
forward; `scrubbed` drops them — which is the discriminator between persistence
and the model pattern-matching its own earlier answers.

    withdraw_post.py --pre <pre/results.jsonl> --arc <full arc.json> \
        --history intact --model … --base-url … --key-file … --out <dir>
"""
from __future__ import annotations
import argparse, json, pathlib, time, urllib.request

TEXT = ("agent_response", "response_text", "speak_content")

def ask(base_url, key, model, messages, timeout=180):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.7,
                       "top_p": 1.0, "max_tokens": 4096}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="results.jsonl from the _pre cell")
    ap.add_argument("--arc", required=True, help="the FULL arc manifest")
    ap.add_argument("--history", required=True, choices=("intact", "scrubbed"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arc = json.loads(pathlib.Path(a.arc).read_text(encoding="utf-8"))
    qs = arc["questions"]
    half = len(qs) // 2
    pre = [json.loads(l) for l in pathlib.Path(a.pre).read_text().splitlines() if l.strip()]
    if len(pre) != half:
        raise SystemExit(f"REFUSED: pre half has {len(pre)} rows, arc expects {half}. "
                         f"A short pre half means the switch is not at the midpoint.")
    key = pathlib.Path(a.key_file).read_text().strip()

    rows, transcript = [], []
    for i, (q, r) in enumerate(zip(qs[:half], pre)):
        resp = next((r[f] for f in TEXT if r.get(f)), "")
        transcript += [{"role": "user", "content": q["translations"]["en"]},
                       {"role": "assistant", "content": resp}]
        rows.append({"question_id": q["question_id"], "category": q["category"],
                     "agent_response": resp, "phase": "pre", "harness": "h3ere",
                     "history": a.history, "he300": q["he300"]})

    if a.history == "scrubbed":
        # THE DISCRIMINATOR. Drop everything the agent said, so the bare half
        # cannot pattern-match its own prior outputs. Measured on one local arc:
        # post-switch answers were 8 characters with the agent's turns visible
        # and 478 with them removed.
        transcript = []

    for q in qs[half:]:
        text = q["translations"]["en"]
        t0 = time.time()
        resp = ask(a.base_url, key, a.model, transcript + [{"role": "user", "content": text}])
        transcript += [{"role": "user", "content": text},
                       {"role": "assistant", "content": resp}]
        rows.append({"question_id": q["question_id"], "category": q["category"],
                     "agent_response": resp, "duration_s": round(time.time() - t0, 2),
                     "phase": "post", "harness": "bare", "history": a.history,
                     "he300": q["he300"]})
        print(f"  post  bare  {rows[-1]['duration_s']:6.1f}s  {resp.strip()[:52]!r}", flush=True)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"switch after turn {half}; history={a.history}; wrote {out}/results.jsonl")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
