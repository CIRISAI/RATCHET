#!/usr/bin/env python3
"""Sum LLM_CALL.cost_usd across a glob of accord-batch JSON files.

Usage: iter_cost.py <glob-pattern>

Used by the Phase 1 workflow to roll up per-iter and cumulative-cell
cost from local-tee batches. Defensive — never raises; missing/bad
files just don't contribute. Prints 4-decimal USD to stdout.
"""
import glob, json, os, sys


def main():
    if len(sys.argv) < 2:
        print("0.0000")
        return 0
    total = 0.0
    for f in sorted(glob.glob(os.path.expanduser(sys.argv[1]))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for ev in d.get("events", []) or []:
            if ev.get("event_type") != "complete_trace":
                continue
            comps = (ev.get("trace") or {}).get("components", []) or []
            for c in comps:
                if c.get("event_type") != "LLM_CALL":
                    continue
                data = c.get("data") or {}
                cu = data.get("cost_usd") or 0.0
                try:
                    total += float(cu)
                except (TypeError, ValueError):
                    pass
    print(f"{total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
