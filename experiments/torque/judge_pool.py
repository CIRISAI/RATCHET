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


def extract_json(txt: str) -> dict:
    r"""Pull the OUTERMOST JSON object out of a model reply.

    The previous extractor was `re.findall(r"\{[^{}]*\}")`, which by
    construction cannot match a brace-nested object: `[^{}]*` stops at the first
    inner `{`. On flat judge output — four booleans — that is fine, and it ran
    for weeks. Given a nested schema it silently returned the LAST INNER object
    instead of the wrapper, so a generation request for
    `{"pairs": [{...}, {...}]}` came back as the final pair and the caller saw
    zero results from a call that had succeeded.

    Scanning for balanced braces costs nothing and cannot fail that way. String
    contents are tracked so a `{` inside a quoted value does not shift depth.
    """
    best: dict = {}
    n = len(txt)
    for start in range(n):
        if txt[start] != "{":
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, n):
            c = txt[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        out = json.loads(txt[start:i + 1])
                        # prefer the LARGEST object that parses: the wrapper,
                        # not one of the things it wraps
                        if isinstance(out, dict) and out and len(txt[start:i + 1]) > len(json.dumps(best)):
                            best = out
                    except Exception:
                        pass
                    break
    return best


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

    def acquire(self, timeout: float | None = None) -> bool:
        """Bounded wait. An unbounded acquire made the run's deadline
        unreachable: threads queued behind a floored slot count never returned
        to the retry loop where the deadline was checked, so a 900s limit did
        nothing and the run sat for 28 minutes at 0s CPU."""
        return self._sem.acquire(timeout=timeout) if timeout else self._sem.acquire()

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
        """Return the slot just used, and occasionally add one.

        The release count has to match the intent exactly. An earlier version
        released ONCE on the growth path — which returns the slot the caller
        used and adds nothing, while `n` was incremented as though it had. So
        every fifth success recorded growth and delivered none, `n` drifted
        above real capacity, and capacity ratcheted DOWN to zero: 27 of 120
        tasks then timed out waiting for a slot that no longer existed.
        Growing by one means releasing twice.
        """
        with self._lock:
            self._ok += 1
            grow = self._ok >= 5 and self.n < self.hi
            if grow:
                self.n += 1
                self._ok = 0
        self._sem.release()              # the slot this caller used
        if grow:
            self._sem.release()          # ...plus the one just added


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
        out = extract_json(txt)
        if out:
            return (out, False, "")
        # real output that is not JSON: retry once in case it was a bad draw,
        # but this one is usually the prompt's fault, not the transport's
        return (None, True, "unparseable_reply")

    _deadline = float("inf")

    def _run(self, model: str, prompt: str, max_tokens: int) -> dict:
        k = self._key(model, prompt)
        with self._cache_lock:
            if k in self._cache:
                self._hits += 1
                return self._cache[k]
        slot = self._slot(model)
        cause = "unknown"
        for attempt in range(self.max_attempts):
            # CHECK BEFORE QUEUEING, not only after failing. A task that never
            # gets a slot must still be able to give up.
            left = self._deadline - time.time()
            if left <= 0:
                self.errors["deadline"] += 1
                return {}
            if not slot.acquire(timeout=max(1.0, min(left, 60.0))):
                self.errors["slot_timeout"] += 1
                return {}
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
            if time.time() > self._deadline:
                self.errors["deadline"] += 1
                return {}
            # jitter: without it, every worker that hit the same 429 retries in
            # lockstep and reproduces the burst that caused it
            time.sleep(min(2 ** attempt, 12) * (0.5 + random.random()))
        self.errors[cause] += 1
        return {}

    # ── public ───────────────────────────────────────────────────────────────

    def map(self, tasks: list, max_tokens: int = 1200,
            deadline_s: float = 900) -> list:
        """tasks: [(model, prompt)] -> [dict], same order. {} where it failed.

        DEADLINE, because an adaptive throttle can converge to near-serial and
        then simply keep going. A first version of this ran three hours on 240
        tasks: sustained 429s drove one model's slots to its floor of 1, and six
        retries with 45s backoff each did the rest. Work already done is in the
        cache, so hitting the deadline costs nothing but the wait — re-running
        resumes free.
        """
        if not tasks:
            return []
        self._deadline = time.time() + deadline_s
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


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST. Run `python3 judge_pool.py` to exercise the concurrency invariants
# with no network and no cost.
#
# Every bug this module has had was a concurrency-accounting bug that looked
# like a model problem from the outside, and each one cost a multi-hour run to
# notice. They are cheap to catch here and expensive to catch in CI:
#
#   * release-count mismatch on growth      capacity ratcheted to zero
#   * unbounded semaphore wait              made the run deadline unreachable
#   * deadline checked only after a failure a queued task could never give up
#   * nested JSON dropped by the extractor  a successful call returned nothing
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> int:
    import threading as _t
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    # capacity must never drain under sustained success
    s = _Slots(1, 8, 4)
    for _ in range(40):
        if not s.acquire(timeout=2):
            break
        s.reward()
    free = 0
    while s.acquire(timeout=0.05):
        free += 1
    check("capacity grows under success", free >= 4, f"ended with {free}")

    # capacity must not fall below the floor under sustained failure
    s = _Slots(2, 8, 6)
    for _ in range(40):
        if not s.acquire(timeout=2):
            break
        s.penalise()
    free = 0
    while s.acquire(timeout=0.05):
        free += 1
    check("capacity floors, never zero", free >= 2, f"ended with {free}")

    # bounded acquire must actually time out rather than block forever
    s = _Slots(1, 1, 1)
    s.acquire()
    check("bounded acquire times out", s.acquire(timeout=0.3) is False)
    s.release()

    # concurrent hammering must not leak or over-release slots
    s = _Slots(1, 6, 3)
    errs = []

    def worker():
        for _ in range(30):
            if not s.acquire(timeout=3):
                errs.append("starved")
                return
            (s.penalise if _ % 4 == 3 else s.reward)()
    ts = [_t.Thread(target=worker) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check("no starvation under contention", not errs, f"{len(errs)} starved")

    # the extractor must survive every shape a model actually emits
    for txt, want in [
        ('{"pairs": [{"a": 1}, {"b": 2}]}', "pairs"),
        ('```json\n{"flat": true}\n```', "flat"),
        ('prose { with "a { brace }" inside } then {"real": 1}', "real"),
        ('{"deep": {"deeper": {"deepest": 1}}}', "deep"),
        ('no json at all', None),
    ]:
        got = extract_json(txt)
        check(f"extract_json {txt[:28]!r}",
              (want is None and not got) or (want and want in got), str(got)[:40])

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
