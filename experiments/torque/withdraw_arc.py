#!/usr/bin/env python3
"""Run ONE arc across TWO harnesses, switching at the midpoint. The reversion arm.

This is the mechanism the campaign is named for and the last one unbuilt. Turns
1..N/2 go to the live H3ERE agent; the pipeline is then withdrawn BETWEEN
THOUGHTS and turns N/2+1..N go to the bare provider, carrying the same
conversation. If behaviour drifts back toward the bare prior, the effect was
inference-time MAINTENANCE, not training.

WHY A NEW RUNNER. `safety_battery` runs one agent for a whole arc;
`direct_provider` runs one provider for a whole arc. Both already accept a
transcript. Nothing holds ONE arc across BOTH, so `pre`/`post` in the pilot were
turns 1-5 and 6-10 of the same arm — a position effect, not a reversion effect.

THE CONFOUND THIS IS BUILT AROUND. If behaviour does NOT revert, that may be
persistence, or the agent pattern-matching its own earlier piped answers still
sitting in the transcript. Those are different claims and identical in aggregate.
So the switch has two conditions:

    intact    the agent's turns 1..k stay in context after the switch
    scrubbed  they are truncated at the switch

Immediate reversion under `scrubbed` means the persistence was in the transcript.
Survival even when scrubbed is the stronger result. Running only `intact` cannot
tell them apart, so `--history` is required rather than defaulted.

RUNS INSIDE THE CONTAINER. The agent's API is deliberately unpublished — no
ports, so concurrent runs cannot fight over one — which means the only place that
can reach both the agent on localhost and the provider on the internet is inside
the container itself.

Usage
-----
    withdraw_arc.py --domain he300_axiotic_primary_a00 --history intact \
        --agent-url http://localhost:8080 --token "$TOK" \
        --model … --base-url … --key-file /keys/key --out /app/qa_reports/withdraw
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request


def load_arc(domain: str, root: pathlib.Path) -> dict:
    cell = root / f"english_{domain}"
    f = cell / f"v4_english_{domain}_arc.json"
    if not f.exists():
        raise SystemExit(f"REFUSED: no arc at {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def ask_agent(url: str, token: str, text: str, channel: str, timeout: float) -> str:
    body = json.dumps({
        "message": text,
        "context": {"channel_id": channel, "session_id": channel,
                    "metadata": {"qa_module": "torque_withdraw"}},
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/agent/interact", data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", "replace")
        raise SystemExit(f"REFUSED: interact returned {e.code}: {body}")
    # The response shape nests differently across versions; take the first
    # string that looks like the agent's speech rather than guessing one path.
    for path in (("data", "response"), ("data", "message"), ("response",), ("message",)):
        cur = d
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, str) and cur.strip():
            return cur
    raise SystemExit(f"REFUSED: no response text in interact reply: {list(d)[:6]}")


def ask_provider(base_url: str, key: str, model: str, messages: list, timeout: float) -> str:
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": 0.7, "top_p": 1.0, "max_tokens": 4096}).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--safety-dir", default="/app/tests/safety")
    ap.add_argument("--agent-url", default="http://localhost:8080")
    ap.add_argument("--username", default="jeff")
    ap.add_argument("--password", default="qa_test_password_12345")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--history", required=True, choices=("intact", "scrubbed"),
                    help="REQUIRED: running only 'intact' cannot separate "
                         "persistence from the agent reading its own outputs")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arc = load_arc(args.domain, pathlib.Path(args.safety_dir))
    qs = arc["questions"]
    half = len(qs) // 2
    if len(qs) % 2:
        raise SystemExit(f"REFUSED: {len(qs)} turns cannot split evenly at the switch.")
    key = pathlib.Path(args.key_file).read_text().strip()

    # COMPLETE SETUP FIRST, then log in. Booting `main.py` directly leaves a
    # server with no users at all, so the first attempt 401'd on a token path
    # that never existed and the second 401'd on a login for an account that had
    # never been created. qa_runner does this via its server manager; a runner
    # that starts the agent itself has to do it itself.
    setup = json.dumps({
        "llm_provider": "openai",
        "llm_api_key": key,
        "llm_model": args.model,
        "llm_base_url": args.base_url,
        "template_id": "he-300-benchmark",
        "enabled_adapters": ["api"],
        "adapter_config": {},
        "admin_username": args.username,
        "admin_password": args.password,
        "agent_port": 8080,
    }).encode()
    try:
        req = urllib.request.Request(f"{args.agent_url}/v1/setup/complete", data=setup,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
        print("  setup completed; waiting for WORK state", flush=True)
    except urllib.error.HTTPError as e:
        # Already set up is fine — the container may be reused within a run.
        if e.code not in (400, 409):
            raise

    login = json.dumps({"username": args.username, "password": args.password}).encode()
    req = urllib.request.Request(f"{args.agent_url}/v1/auth/login", data=login,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tok = json.loads(r.read())
    token = tok.get("access_token") or (tok.get("data") or {}).get("access_token")
    if not token:
        raise SystemExit(f"REFUSED: no access_token in login reply: {list(tok)[:6]}")
    # POLL FOR WORK STATE, do not sleep a fixed interval. A 20s sleep was enough
    # locally and not on a CI runner: setup returned, the runtime had not
    # resumed, and the first interact came back 500 about 105 seconds later. The
    # agent tells you when it is ready; ask it.
    deadline = time.time() + 600
    state = "UNKNOWN"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{args.agent_url}/v1/agent/status",
                headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                s = json.loads(r.read())
            state = str((s.get("data") or s).get("cognitive_state", "UNKNOWN"))
            # Match qa_runner's own test, which accepts three surface forms. The
            # API returns "AgentState.WORK", so an exact == "WORK" never fires —
            # the agent logged "Work processor initialized successfully" and this
            # poll timed out anyway, for 600 seconds, on a string comparison.
            low = state.lower()
            if low in ("work", "agentstate.work") or state.endswith(".WORK"):
                break
        except Exception:
            pass
        time.sleep(5)
    if state != "WORK":
        raise SystemExit(f"REFUSED: agent never reached WORK (last state {state!r}). "
                         f"Sending turns to an agent that is not working measures the wait.")
    print(f"  agent WORK after {600 - int(deadline - time.time())}s", flush=True)

    channel = f"torque_withdraw_{args.domain}_{int(time.time())}"

    rows, transcript = [], []
    for i, q in enumerate(qs):
        text = q["translations"]["en"]
        phase = "pre" if i < half else "post"
        t0 = time.time()
        if phase == "pre":
            # The agent keeps its own conversation state on `channel`; the local
            # transcript is built in parallel so the bare half can inherit it.
            resp = ask_agent(args.agent_url, token, text, channel, args.timeout)
            transcript.append({"role": "user", "content": text})
            transcript.append({"role": "assistant", "content": resp})
        else:
            if i == half and args.history == "scrubbed":
                # THE DISCRIMINATOR. Everything the agent said is dropped at the
                # switch, so the bare half cannot pattern-match its own prior
                # outputs. Reversion here means the persistence was in-context.
                transcript.clear()
            msgs = transcript + [{"role": "user", "content": text}]
            resp = ask_provider(args.base_url, key, args.model, msgs, args.timeout)
            transcript.append({"role": "user", "content": text})
            transcript.append({"role": "assistant", "content": resp})
        rows.append({
            "question_id": q["question_id"], "category": q["category"],
            "agent_response": resp, "duration_s": round(time.time() - t0, 2),
            "phase": phase, "harness": "h3ere" if phase == "pre" else "bare",
            "history": args.history, "he300": q["he300"],
        })
        print(f"  ({i+1}/{len(qs)}) {phase:4s} {rows[-1]['harness']:5s} "
              f"{rows[-1]['duration_s']:6.1f}s  {resp.strip()[:48]!r}", flush=True)

    out = pathlib.Path(args.out) / f"{args.domain}_{args.history}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"switch after turn {half}; history={args.history}; wrote {out}/results.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
