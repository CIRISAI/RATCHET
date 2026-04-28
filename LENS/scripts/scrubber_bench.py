#!/usr/bin/env python3
"""
Scrubber throughput benchmark using the production HF corpus as
realistic load. Measures end-to-end `cirislens_core.scrub_trace` perf
on a configurable sample of real traces.

Usage:
    CIRISLENS_NER_MODEL_DIR=/tmp/xlmr_ner \\
        python3 scripts/scrubber_bench.py [-n 50] [-l full_traces|detailed]

Reports throughput (traces/sec), per-trace latency p50/p95/p99, and total
elapsed wall-clock. Designed to be cheap enough to re-run after each
optimization (caching, model swap, quantization, …).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cirislens_core


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=50, help="trace count (default 50)")
    p.add_argument(
        "-l",
        "--level",
        default="full_traces",
        choices=["generic", "detailed", "full_traces"],
    )
    p.add_argument(
        "--src",
        default=os.path.expanduser("~/RATCHET/release/data/accord_traces.jsonl"),
        help="JSONL source (default: HF release raw corpus)",
    )
    p.add_argument("--warmup", type=int, default=2, help="warmup traces (excluded from stats)")
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="batch size for scrub_traces_batch (1 = use scrub_trace per call)",
    )
    args = p.parse_args()

    # Load N+warmup lines from the corpus.
    src = Path(args.src)
    if not src.exists():
        print(f"corpus not found: {src}", file=sys.stderr)
        return 1
    lines: list[str] = []
    with src.open() as f:
        for line in f:
            lines.append(line.strip())
            if len(lines) >= args.n + args.warmup:
                break

    if len(lines) < args.warmup + 1:
        print("not enough traces in corpus", file=sys.stderr)
        return 1

    print(f"benchmark: {args.n} traces at level={args.level}, warmup={args.warmup}")
    print(f"NER configured: {cirislens_core.ner_is_configured()}")
    print(f"source: {src}")

    # Warmup (first call eats backend init + cache cold).
    for line in lines[: args.warmup]:
        try:
            cirislens_core.scrub_trace(line, args.level)
        except Exception as e:
            print(f"warmup error: {e}", file=sys.stderr)
            return 1

    # Timed run. When --batch > 1, dispatch via scrub_traces_batch.
    durations_ms: list[float] = []
    failures = 0
    cum_hits = cum_misses = 0
    t_total = time.perf_counter()
    work = lines[args.warmup : args.warmup + args.n]
    bs = max(1, args.batch)
    for i in range(0, len(work), bs):
        chunk = work[i : i + bs]
        t0 = time.perf_counter()
        try:
            if bs == 1:
                results = [cirislens_core.scrub_trace(chunk[0], args.level)]
            else:
                results = cirislens_core.scrub_traces_batch(chunk, args.level)
        except Exception as e:
            failures += len(chunk)
            print(f"  scrub error: {e}", file=sys.stderr)
            continue
        elapsed_chunk = (time.perf_counter() - t0) * 1000
        per = elapsed_chunk / len(chunk)
        durations_ms.extend([per] * len(chunk))
        for r in results:
            s = r.get("stats", {})
            cum_hits += s.get("ner_cache_hits", 0)
            cum_misses += s.get("ner_cache_misses", 0)
    elapsed = time.perf_counter() - t_total

    n = len(durations_ms)
    if n == 0:
        print("all traces failed", file=sys.stderr)
        return 1

    print()
    print(f"  total wall-clock:  {elapsed:.2f}s")
    print(f"  successful:        {n}/{args.n} ({failures} failed)")
    print(f"  throughput:        {n/elapsed:.2f} traces/sec")
    print(f"  latency p50:       {statistics.median(durations_ms):.1f} ms")
    print(f"  latency p95:       {sorted(durations_ms)[int(n * 0.95)]:.1f} ms")
    print(f"  latency p99:       {sorted(durations_ms)[min(n - 1, int(n * 0.99))]:.1f} ms")
    print(f"  latency mean:      {statistics.mean(durations_ms):.1f} ms")
    print(f"  latency stdev:     {statistics.stdev(durations_ms) if n > 1 else 0:.1f} ms")
    if cum_hits + cum_misses > 0:
        print(
            f"  ner cache:         {cum_hits} hits / {cum_misses} misses "
            f"({cum_hits/(cum_hits+cum_misses)*100:.1f}% hit rate)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
