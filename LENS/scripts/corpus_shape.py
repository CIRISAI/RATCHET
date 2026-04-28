#!/usr/bin/env python3
"""
corpus_shape — print the faceted shape of the CIRISLens trace corpus.

Run this BEFORE any analysis so the shape (task class, language, region,
agent, model) is explicit. Prevents the common failure mode of inferring
patterns from a mixed or non-stationary sample.

Usage:
    corpus_shape.py                    # last 24h
    corpus_shape.py --window 10m
    corpus_shape.py --window 7d
    corpus_shape.py --since 2026-04-24T03:52:00
    corpus_shape.py --since 2026-04-24T03:52 --until 2026-04-24T05:00
    corpus_shape.py --task-class qa_eval
    corpus_shape.py --agent Ally

Connection:
    Reads CIRISLENS_PSQL env var as the full command prefix. Default invokes
    psql against the production container over SSH. Override for dev:

    export CIRISLENS_PSQL='docker exec -i cirislens-db psql -U cirislens -d cirislens'
    export CIRISLENS_PSQL='psql "$DATABASE_URL"'
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass


DEFAULT_PSQL = (
    "ssh -i ~/Desktop/ciris_transfer/.ciris_bridge_keys/cirisbridge_ed25519 "
    "root@108.61.242.236 "
    '"docker exec -i cirislens-db psql -U cirislens -d cirislens -A -F \'|\' -t"'
)


@dataclass
class Args:
    window: str | None
    since: str | None
    until: str | None
    task_class: str | None
    agent: str | None


def parse_window(window: str) -> str:
    """Translate `10m`, `24h`, `7d` to a SQL INTERVAL literal."""
    m = re.fullmatch(r"(\d+)\s*([smhd])", window.strip().lower())
    if not m:
        raise ValueError(f"window must match Nm/Nh/Nd (got {window!r})")
    n, unit = int(m.group(1)), m.group(2)
    unit_full = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]
    return f"{n} {unit_full}"


def build_filter(args: Args) -> str:
    """Return WHERE clause to apply to trace_context queries."""
    clauses: list[str] = []
    if args.since:
        clauses.append(f"timestamp >= '{args.since}'")
    elif args.window:
        clauses.append(f"timestamp > NOW() - INTERVAL '{parse_window(args.window)}'")
    else:
        clauses.append("timestamp > NOW() - INTERVAL '24 hours'")
    if args.until:
        clauses.append(f"timestamp <= '{args.until}'")
    if args.task_class:
        clauses.append(f"task_class = '{args.task_class}'")
    if args.agent:
        clauses.append(f"agent_name = '{args.agent}'")
    return " AND ".join(clauses)


def run_psql(sql: str) -> str:
    """Invoke psql (local or via SSH), pipe SQL via stdin, return stdout.

    CIRISLENS_PSQL should be a command that reads SQL from stdin and writes
    unaligned, tuples-only, pipe-separated rows to stdout. See DEFAULT_PSQL.
    """
    cmd = os.environ.get("CIRISLENS_PSQL", DEFAULT_PSQL)
    # ON_ERROR_STOP makes psql exit non-zero on query errors (default is to
    # keep going and just print to stderr). Without this, a typo in an
    # analysis query silently returns an empty result.
    wrapped = "\\set ON_ERROR_STOP on\n" + sql
    r = subprocess.run(
        cmd,
        shell=True,
        input=wrapped,
        check=False,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or r.stderr.strip():
        sys.stderr.write(r.stderr)
        r.check_returncode()
    return r.stdout.strip()


def rows(out: str) -> list[list[str]]:
    return [line.split("|") for line in out.splitlines() if line]


def fmt_count_table(title: str, pairs: list[tuple[str, int]], total: int) -> str:
    if not pairs:
        return f"  {title}: (none)\n"
    width = max(len(p[0]) for p in pairs)
    lines = [f"  {title}:"]
    for label, n in pairs:
        pct = (100.0 * n / total) if total else 0.0
        lines.append(f"    {label:<{width}}  {n:>5}  ({pct:4.1f}%)")
    return "\n".join(lines) + "\n"


def section(title: str) -> str:
    return f"\n{'─' * 60}\n{title}\n{'─' * 60}"


def main() -> int:
    p = argparse.ArgumentParser(description="Print corpus shape for analysis.")
    p.add_argument("--window", help="Nm/Nh/Nd (default: 24h)")
    p.add_argument("--since", help="ISO timestamp (overrides --window)")
    p.add_argument("--until", help="ISO timestamp")
    p.add_argument("--task-class", help="filter: qa_eval, real_user_web, wakeup_ritual, other")
    p.add_argument("--agent", help="filter: agent_name")
    raw = p.parse_args()
    args = Args(
        window=raw.window,
        since=raw.since,
        until=raw.until,
        task_class=raw.task_class,
        agent=raw.agent,
    )

    where = build_filter(args)

    # ── Header / total counts ──────────────────────────────────────────────
    q_totals = f"""
        SELECT
            COUNT(*) AS traces,
            COUNT(DISTINCT task_id) AS tasks,
            COUNT(DISTINCT thought_id) AS thoughts,
            COUNT(DISTINCT agent_name) AS agents,
            MIN(timestamp) AS first_ts,
            MAX(timestamp) AS last_ts
        FROM cirislens.trace_context
        WHERE {where}
    """
    r = rows(run_psql(q_totals))
    if not r or not r[0][0]:
        print(f"(no traces in window: {where})")
        return 0
    traces, tasks, thoughts, agents, first_ts, last_ts = r[0]
    traces_n = int(traces)
    tasks_n = int(tasks)

    print(section("CORPUS SHAPE"))
    print(f"  window:   {where}")
    print(f"  traces:   {traces_n}")
    print(f"  tasks:    {tasks_n}")
    print(f"  thoughts: {thoughts}")
    print(f"  agents:   {agents}")
    print(f"  span:     {first_ts}  →  {last_ts}")

    # ── Faceted breakdowns ─────────────────────────────────────────────────
    def facet(title: str, col: str, order: str = "n DESC") -> str:
        q = f"""
            SELECT COALESCE({col}::text, '(null)') AS label, COUNT(*) AS n
            FROM cirislens.trace_context
            WHERE {where}
            GROUP BY 1 ORDER BY {order}
        """
        pairs = [(r[0], int(r[1])) for r in rows(run_psql(q))]
        return fmt_count_table(title, pairs, traces_n)

    print(section("BY TASK CLASS"))
    print(facet("task_class", "task_class"), end="")

    print(section("BY AGENT"))
    print(facet("agent_name", "agent_name"), end="")

    print(section("BY COGNITIVE STATE"))
    print(facet("cognitive_state", "cognitive_state"), end="")

    print(section("BY TRACE LEVEL"))
    print(facet("trace_level", "trace_level"), end="")

    print(section("BY MODEL"))
    print(facet("primary_model", "primary_model"), end="")

    # QA breakdown only if QA traffic present
    q_qa = f"SELECT COUNT(*) FROM cirislens.trace_context WHERE {where} AND task_class = 'qa_eval'"
    qa_count = int(run_psql(q_qa).strip() or "0")
    if qa_count > 0:
        print(section("QA EVAL — BY LANGUAGE"))
        print(facet("qa_language", "qa_language", "label"), end="")
        print(section("QA EVAL — BY QUESTION NUMBER"))
        print(facet("qa_question_num", "qa_question_num", "label"), end="")

    # Region / deployment only if populated
    q_region = f"""
        SELECT COUNT(*) FROM cirislens.trace_context
        WHERE {where} AND (deployment_region IS NOT NULL OR user_timezone IS NOT NULL)
    """
    region_count = int(run_psql(q_region).strip() or "0")
    if region_count > 0:
        print(section("BY DEPLOYMENT REGION"))
        print(facet("deployment_region", "deployment_region"), end="")
        print(section("BY USER TIMEZONE"))
        print(facet("user_timezone", "user_timezone"), end="")
        print(section("BY COARSENED LOCATION (~55km grid)"))
        q_loc = f"""
            SELECT COALESCE(user_latitude_cell::text || ',' || user_longitude_cell::text, '(null)') AS cell,
                   COUNT(*) AS n
            FROM cirislens.trace_context
            WHERE {where}
            GROUP BY 1 ORDER BY n DESC LIMIT 20
        """
        pairs = [(r[0], int(r[1])) for r in rows(run_psql(q_loc))]
        print(fmt_count_table("lat,lon_cell", pairs, traces_n), end="")

    # ── Signal health / flags ──────────────────────────────────────────────
    print(section("SIGNAL HEALTH"))
    q_signals = f"""
        SELECT
            COUNT(*) FILTER (WHERE csdma_plausibility_score = 0) AS csdma_zero,
            COUNT(*) FILTER (WHERE csdma_plausibility_score >= 0.9) AS csdma_high,
            COUNT(*) FILTER (WHERE coherence_passed = false) AS coherence_vetoed,
            COUNT(*) FILTER (WHERE entropy_passed = false) AS entropy_vetoed,
            COUNT(*) FILTER (WHERE optimization_veto_passed = false) AS opt_vetoed,
            COUNT(*) FILTER (WHERE epistemic_humility_passed = false) AS eph_vetoed,
            COUNT(*) FILTER (WHERE idma_fragility_flag = true) AS fragile,
            COUNT(*) FILTER (WHERE signature_verified = true) AS sig_ok,
            COUNT(*) FILTER (WHERE action_was_overridden = true) AS overridden,
            COUNT(*) AS total
        FROM cirislens.trace_context
        WHERE {where}
    """
    r = rows(run_psql(q_signals))[0]
    csdma_zero, csdma_high, coh_v, ent_v, opt_v, eph_v, frag, sig_ok, ovr, total = map(int, r)
    flag_pairs = [
        ("csdma = 0.00", csdma_zero),
        ("csdma ≥ 0.90", csdma_high),
        ("coherence vetoed", coh_v),
        ("entropy vetoed", ent_v),
        ("optimization vetoed", opt_v),
        ("epistemic vetoed", eph_v),
        ("idma fragile", frag),
        ("signature verified", sig_ok),
        ("action overridden", ovr),
    ]
    print(fmt_count_table("flags", flag_pairs, total), end="")

    # ── Non-stationarity check ─────────────────────────────────────────────
    # If CSDMA mean varies meaningfully across first/second half of the window,
    # flag that the corpus isn't stationary (important for any correlation work).
    q_drift = f"""
        WITH bounded AS (
            SELECT timestamp, csdma_plausibility_score
            FROM cirislens.trace_context
            WHERE {where} AND csdma_plausibility_score IS NOT NULL
        ),
        halves AS (
            SELECT csdma_plausibility_score,
                   CASE WHEN timestamp < (SELECT (MIN(timestamp) + (MAX(timestamp) - MIN(timestamp))/2) FROM bounded)
                        THEN 'first' ELSE 'second' END AS half
            FROM bounded
        )
        SELECT half,
               ROUND(AVG(csdma_plausibility_score)::numeric, 3) AS csdma_avg,
               COUNT(*) AS n
        FROM halves GROUP BY half ORDER BY half
    """
    halves = rows(run_psql(q_drift))
    if len(halves) == 2:
        print(section("STATIONARITY CHECK (first vs second half of window)"))
        for h in halves:
            print(f"  {h[0]:<7} half:  csdma_avg={h[1]}  n={h[2]}")
        try:
            delta = abs(float(halves[0][1]) - float(halves[1][1]))
            if delta >= 0.10:
                print(f"\n  ⚠ non-stationary: CSDMA mean shifted by {delta:.2f} across window.")
                print("    Run correlations on halves separately, not on the whole window.")
        except ValueError:
            pass

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
