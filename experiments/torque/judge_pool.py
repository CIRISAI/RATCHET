#!/usr/bin/env python3
"""Parallel, resumable judging against OpenRouter. Shared by every scorer.

WHAT WENT WRONG WITHOUT THIS, all of it measured on this project:

  * SEQUENTIAL PER-MODEL PASSES. Each model was scored in its own loop, so the
    slowest model set the wall clock and a rate-limited one stalled everything
    behind it. Models are independent; nothing required them to be serial.

  * AN EMPTY 200 WAS TREATED AS AN ANSWER. Under concurrency the API returns
    HTTP 200 with empty content. That fell through to "unparseable" and returned
    immediately WITHOUT RETRY, while genuine 429s got full backoff — so the one
    transient failure that looked like a model problem was the only one not
    retried. 108 of 120 items lost to it; serial calls parsed 3/3.

  * FIXED CONCURRENCY. Guessing 8 produced empty 200s; guessing 4 produced a
    40-minute run. The right number differs per model and per moment, so it is
    measured rather than guessed: each model's slot count falls on a 429 and
    recovers on sustained success.

  * NOTHING WAS CACHED. A killed run lost everything it had paid for, and
    re-scoring the same responses under a changed rule cost the same again. The
    cache is keyed on (model, prompt) so a resume is free and a scoring-rule
    change costs nothing.

  * SILENT SHRINKAGE. Losses reduced denominators without saying so. A sample
    cut by transport failure is not a smaller sample, it is a biased one, so the
    pool reports loss by cause and callers can refuse on it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://openrouter.ai/api/v1/chat/completions"


class _Slots:
    """Adaptive concurrency for one model. Falls fast on 429, recovers slowly.

    Multiplicative decrease / additive increase, the same shape TCP uses, for
    the same reason: the ceiling is unknown, changes, and is only discoverable
    by touching it.
    """

    def __init__(self, lo: int, hi: int, start: int):
        self.lo, self.hi = lo, hi
        self.n = start
        self._sem = threading.Semaphore(start)
        self._lock = threading.Lock()
        self._ok = 0

    def acquire(self):
        self._sem.acquire()

    def release(self):
        self._sem.release()

    def penalise(self):
        with self._lock:
            if self.n > self.lo:
                self.n -= 1          # hold one slot back; do not release it
                self._ok = 0
                return
        self._sem.release()

    def reward(self):
        with self._lock:
            self._ok += 1
            if self._ok >= 20 and self.n < self.hi:
                self.n += 1
                self._ok = 0
                self._sem.release()
                return
        self._sem.release()


class JudgePool:
    """Score many (model, prompt) pairs concurrently, with a resumable cache.

    Models run INTERLEAVED, not in per-model passes: every task goes into one
    queue and each model's own slot count throttles it independently, so a
    rate-limited model slows itself and nothing else.
    """

    #: Per-model bounds. Measured, not guessed: gpt-oss-safeguard-20b returned
    #: 19 genuine 429s at 4 workers while gemini-2.5-pro parsed 118/120, so they
    #: do not get the same ceiling.
    DEFAULT_LIMITS = {
        "openai/gpt-oss-safeguard-20b": (1, 4, 2),
        "google/gemini-2.5-pro": (2, 12, 8),
    }
    FALLBACK = (1, 8, 4)

    def __init__(self, key: str, cache: Path | None = None, limits: dict | None = None,
                 max_attempts: int = 6, verbose: bool = True):
        self.key = key
        self.max_attempts = max_attempts
        self.verbose = verbose
        self.errors: Counter = Counter()
        self._limits = {**self.DEFAULT_LIMITS, **(limits or {})}
        self._slots: dict = {}
        self._slots_lock = threading.Lock()
        self.cache_path = cache
        self._cache: dict = {}
        self._cache_lock = threading.Lock()
        self._hits = 0
        if cache and cache.exists():
            for line in cache.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        self._cache[r["k"]] = r["v"]
                    except Exception:
                        continue   # a truncated final line is not a reason to die

    # ── internals ────────────────────────────────────────────────────────────

    def _slot(self, model: str) -> _Slots:
        with self._slots_lock:
            if model not in self._slots:
                lo, hi, start = self._limits.get(model, self.FALLBACK)
                self._slots[model] = _Slots(lo, hi, start)
            return self._slots[model]

    @staticmethod
    def _key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()

    def _remember(self, k: str, v: dict) -> None:
        with self._cache_lock:
            self._cache[k] = v
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"k": k, "v": v}) + "\n")

    def _once(self, model: str, prompt: str, max_tokens: int) -> tuple:
        """-> (result_or_None, retryable, cause)"""
        body = json.dumps({"model": model,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(
            API, data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            return (None, e.code == 429 or e.code >= 500, f"http_{e.code}")
        except Exception as e:
            return (None, True, type(e).__name__)

        if "choices" not in payload:
            # a 200 carrying an error object, which upstreams do under pressure
            return (None, True, "no_choices")
        txt = (payload["choices"][0].get("message") or {}).get("content") or ""
        if not txt.strip():
            return (None, True, "empty_200")
        for cand in reversed(re.findall(r"\{[^{}]*\}", txt, re.S)):
            try:
                out = json.loads(cand)
                if out:
                    return (out, False, "")
            except Exception:
                continue
        # real output that is not JSON: retry once in case it was a bad draw,
        # but this one is usually the prompt's fault, not the transport's
        return (None, True, "unparseable_reply")

    def _run(self, model: str, prompt: str, max_tokens: int) -> dict:
        k = self._key(model, prompt)
        with self._cache_lock:
            if k in self._cache:
                self._hits += 1
                return self._cache[k]
        slot = self._slot(model)
        cause = "unknown"
        for attempt in range(self.max_attempts):
            slot.acquire()
            try:
                out, retryable, cause = self._once(model, prompt, max_tokens)
            finally:
                if cause.startswith("http_429"):
                    slot.penalise()
                else:
                    slot.reward()
            if out is not None:
                self._remember(k, out)
                return out
            if not retryable or attempt == self.max_attempts - 1:
                break
            # jitter: without it, every worker that hit the same 429 retries in
            # lockstep and reproduces the burst that caused it
            time.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))
        self.errors[cause] += 1
        return {}

    # ── public ───────────────────────────────────────────────────────────────

    def map(self, tasks: list, max_tokens: int = 1200) -> list:
        """tasks: [(model, prompt)] -> [dict], same order. {} where it failed."""
        if not tasks:
            return []
        total = sum(hi for _, hi, _ in
                    (self._limits.get(m, self.FALLBACK) for m, _ in tasks))
        workers = max(2, min(32, total))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(lambda t: self._run(t[0], t[1], max_tokens), tasks))
        if self.verbose:
            ok = sum(1 for o in out if o)
            print(f"  judged {ok}/{len(tasks)} in {time.time()-t0:.0f}s "
                  f"({self._hits} cached)"
                  + (f" · losses {dict(self.errors)}" if self.errors else ""),
                  flush=True)
        return out

    def loss(self, out: list) -> float:
        return 1.0 - (sum(1 for o in out if o) / len(out)) if out else 1.0

    def refuse_if_lossy(self, out: list, ceiling: float = 0.10) -> None:
        """A sample cut by transport failure is biased, not merely smaller."""
        l = self.loss(out)
        if l > ceiling:
            raise SystemExit(
                f"REFUSED: {l:.0%} of judgements failed (ceiling {ceiling:.0%}). "
                f"Causes: {dict(self.errors)}. Re-run — the cache makes "
                f"everything already judged free.")
