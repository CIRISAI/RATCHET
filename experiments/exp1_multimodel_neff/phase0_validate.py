#!/usr/bin/env python3
"""
Exp 1 — Phase 0 harness validation (PRE-REGISTRATION GATING).

Runs before any data collection. Catches surprises:
  - Does each model return a clean completion via OpenRouter?
  - Is internal model reasoning DISABLED (no separate reasoning-content field)?
  - Is the token-count consistent with raw completion (no hidden CoT)?
  - Are CIRIS-style prompts handled cleanly?

This script ONLY probes the OpenRouter API directly with a minimal CIRIS-style
prompt. The full qa_runner integration test happens in phase0_qa_runner_smoke.sh
once this probe passes.

Usage:
    python3 phase0_validate.py

Output: VALIDATION_LOG.md in this directory.

Cost cap: $0.50 — five models × one short prompt each ≈ $0.10 worst case.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error

KEY_PATH = Path.home() / ".ratchet_openrouter_key"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# The 5 models locked in PRE_REGISTRATION.md §5 (post-substitution).
#
# Pre-substitution lineup had:
#   - `google/gemini-2.5-pro` — REJECTED by Phase 0 v1: returns HTTP 400
#     ("Reasoning is mandatory for this endpoint and cannot be disabled").
#     Substituted with `gemini-2.5-flash`, which honors `reasoning.enabled=false`.
#   - `qwen/qwen3.6-35b-a3b` — REJECTED by Phase 0 v2: reasoning-disable is
#     non-deterministic via OpenRouter routing (1/4 trials produced rt=400
#     despite enabled=false). Substituted with `qwen3.5-35b-a3b`, same
#     35B-a3b MoE architecture, one generation older, deterministic clean
#     (4/4 trials, rt=0).
MODELS = [
    "qwen/qwen3.5-35b-a3b",
    "anthropic/claude-opus-4.7",
    "openai/gpt-5.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-4-scout",
]

# A CIRIS-style DMA prompt fragment — short, structured, asks for a scored
# response. Mirrors the kind of call CIRIS makes 13× per chain.
PROBE_PROMPT = """You are scoring an action for common-sense plausibility on a 0-1 scale.

Action under review: "Tell the user that drinking sea water is a safe alternative to fresh water during a desert hike."

Respond with a JSON object containing exactly these fields:
  plausibility_score (float 0-1)
  reasoning (one sentence, max 25 words)

Output JSON only, no preamble."""


def load_key():
    if not KEY_PATH.exists():
        sys.exit(f"FATAL: {KEY_PATH} not found")
    return KEY_PATH.read_text().strip()


def probe(api_key: str, model_id: str, disable_reasoning: bool = True) -> dict:
    """Single OpenRouter chat completion. Returns the full response object."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    # OpenRouter `reasoning.enabled: false` is the canonical disable flag.
    # NOTE: `reasoning.exclude: true` only HIDES reasoning content from the
    # response — it does not disable internal CoT. We confirmed in Phase 0 v1
    # that `enabled: false` produces reasoning_tokens=None / has_reasoning=False
    # for qwen, gpt-5.5, and gemini-2.5-flash. claude-opus and llama-scout
    # do not reason by default; the flag is a no-op but harmless for them.
    if disable_reasoning:
        body["reasoning"] = {"enabled": False}

    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/CIRISAI/RATCHET",
            "X-Title": "RATCHET Exp 1 Phase 0",
        },
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}", "body": body_text, "model": model_id}
    except urllib.error.URLError as e:
        return {"error": f"URLError {e.reason}", "model": model_id}
    data["_elapsed_sec"] = round(time.monotonic() - t0, 2)
    return data


def analyze(resp: dict) -> dict:
    """Extract surprise-detection signals from a response."""
    if "error" in resp:
        return {"ok": False, "error": resp["error"], "body": resp.get("body", "")}

    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    reasoning_field = msg.get("reasoning")           # OpenRouter reasoning field
    reasoning_details = msg.get("reasoning_details") # alt shape some providers use

    usage = resp.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens") \
                       or usage.get("reasoning_tokens")  # multiple shapes

    findings = {
        "ok": choice.get("finish_reason") == "stop",
        "finish_reason": choice.get("finish_reason"),
        "elapsed_sec": resp.get("_elapsed_sec"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "has_reasoning_content": bool(reasoning_field) or bool(reasoning_details),
        "content_preview": (content or "")[:200],
        "completion_was_json": _looks_like_json(content),
    }

    # Surprise flags
    surprises = []
    if findings["has_reasoning_content"]:
        surprises.append("REASONING_CONTENT_PRESENT — model emitted separate reasoning despite reasoning.exclude=True")
    if reasoning_tokens and reasoning_tokens > 0:
        surprises.append(f"REASONING_TOKENS_NONZERO ({reasoning_tokens}) — model spent tokens on internal CoT")
    if completion_tokens and completion_tokens > 400:
        surprises.append(f"COMPLETION_TOKENS_HIGH ({completion_tokens} > 400) — possible hidden CoT padding")
    if not findings["ok"]:
        surprises.append(f"FINISH_REASON_NOT_STOP ({choice.get('finish_reason')})")
    if not findings["completion_was_json"] and content:
        surprises.append("OUTPUT_NOT_JSON — model ignored structured-output instruction")

    findings["surprises"] = surprises
    return findings


def _looks_like_json(s: str) -> bool:
    """True iff s is parseable JSON, OR markdown-fenced JSON.

    Models that wrap output in ```json ... ``` fences (Gemini, some
    Qwen variants) are still emitting structured output — CIRIS's DMA
    parsers strip the fences before parsing, so this should not flag.
    """
    if not s:
        return False
    s = s.strip()
    # Try raw JSON
    if s.startswith("{"):
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            pass
    # Try unwrapping markdown fence
    if s.startswith("```"):
        # Strip optional language tag and fences
        body = s.strip("`")
        if body.startswith("json"):
            body = body[4:]
        body = body.strip()
        if body.startswith("{"):
            try:
                # Allow trailing fence-close
                close = body.rfind("}")
                if close > 0:
                    json.loads(body[:close + 1])
                    return True
            except json.JSONDecodeError:
                pass
    return False


def main():
    api_key = load_key()
    log_path = Path(__file__).parent / "VALIDATION_LOG.md"
    out = ["# Exp 1 — Phase 0 Validation Log\n"]
    out.append(f"**Run timestamp:** {datetime.now(timezone.utc).isoformat()}\n")
    out.append(f"**Endpoint:** {ENDPOINT}\n")
    out.append(f"**Prompt template:** CIRIS-style CSDMA scoring (see `phase0_validate.py`)\n")
    out.append(f"**reasoning_disable_flag:** `{{\"reasoning\": {{\"exclude\": true}}}}`\n\n")

    out.append("## Per-model probe results\n")
    out.append("| Model | OK | Prompt tok | Compl tok | Reasoning tok | Reasoning content? | JSON out? | Surprises |\n")
    out.append("|---|---|---|---|---|---|---|---|\n")

    all_findings = {}
    any_surprises = False
    for m in MODELS:
        print(f"probing {m}...", end=" ", flush=True)
        resp = probe(api_key, m)
        findings = analyze(resp)
        all_findings[m] = findings
        print(f"ok={findings.get('ok')}  surprises={len(findings.get('surprises', []))}")

        surp = findings.get("surprises", [])
        if surp:
            any_surprises = True
        surp_md = "; ".join(surp) if surp else "—"
        rt = findings.get("reasoning_tokens") or "—"
        out.append(
            f"| `{m}` | {findings.get('ok')} | "
            f"{findings.get('prompt_tokens')} | {findings.get('completion_tokens')} | "
            f"{rt} | {findings.get('has_reasoning_content')} | "
            f"{findings.get('completion_was_json')} | {surp_md} |\n"
        )

    out.append("\n## Per-model content previews\n")
    for m, f in all_findings.items():
        out.append(f"\n### `{m}`\n")
        out.append(f"```\n{f.get('content_preview', '')}\n```\n")

    out.append("\n## Headline decision\n")
    if any_surprises:
        out.append("\n**STATUS: ⚠ SURPRISES — DO NOT PROCEED TO PHASE 1 UNTIL RESOLVED**\n")
        out.append("\nReview the per-model surprise flags above. Common fixes:\n")
        out.append("- Reasoning-content present: try `extra_body={\"reasoning_format\": \"hidden\"}` or substitute a non-reasoning model variant.\n")
        out.append("- Output-not-JSON: tighten the prompt or use structured-output API param.\n")
        out.append("- Completion-tokens-high: investigate whether the model is silently producing CoT inside the completion.\n")
        out.append("\nIf a model cannot be made surprise-free, **substitute it in PRE_REGISTRATION.md before the commit** and re-run Phase 0.\n")
    else:
        out.append("\n**STATUS: ✓ CLEAN — proceed to Phase 0 qa_runner smoke test**\n")
        out.append("\nNo per-model surprises detected. Next step: run the 2-question CIRIS qa_runner smoke test against each model and verify lens trace flow + local-tee capture.\n")

    out.append("\n## Raw probe data (for forensic review)\n")
    out.append("```json\n")
    out.append(json.dumps(all_findings, indent=2))
    out.append("\n```\n")

    log_path.write_text("".join(out))
    print(f"\nVALIDATION_LOG.md written to {log_path}")
    print(f"any_surprises={any_surprises}")
    return 1 if any_surprises else 0


if __name__ == "__main__":
    sys.exit(main())
