#!/usr/bin/env python3
"""Certify what the compose gate can actually see.

`compose_dump gate` verifies that blocks it enumerates are held or varied as the
regime declares. It says nothing about override keys that reach no enumerated
block — and a key the gate cannot see is a key whose ablation cannot be
certified. A regime varying only such keys gets `GATE: PASS` over ground never
inspected. That is the same failure shape as the `full_traces` 3.0x inflation:
a clean-looking validation with the defect outside its field of view.

This probe measures coverage directly rather than reasoning about it. For each
override namespace it replaces every key with a unique marker, dumps, and diffs
`sha256` per block against an unmarked arm. A namespace that moves zero blocks is
invisible to the gate.

History, because the trend is the point:

* v2.9.9-stable (b684b7f56): **32 of 101 keys dark** — `conscience_prompt`
  (12 keys, 48,463 B of conscience criteria), the `string:conscience.*` retry
  envelope (17), and `template` (3, agent identity). CIRISAgent#986.
* v2.9.10-stable (14fc414ef): **0 of 191 keys dark**, 178 blocks in the
  reference arm; the enumerated surface went 8 steps/35 blocks to 21/542,
  including the bounce and retry states. COVERAGE_2_9_10.json.

A caution this probe learned about itself: the first run against 2.9.10 reported
0 dark keys while having probed 61 of 191, because it marked only scalar values
and CIRISAgent#994 made localized keys `{locale: text}` mappings. An instrument
that under-probes reports a false all-clear -- the exact failure it exists to
catch. `_mark()` is now shape-aware and unmarkable shapes warn loudly.

Usage
-----
    # certify a worktree, write the record
    python3 gate_coverage.py --agent-root /tmp/a299 --out COVERAGE.json

    # fail if coverage regressed against a pinned record (CI)
    python3 gate_coverage.py --agent-root /path --expect COVERAGE_2_9_9.json

    # per-key attribution instead of per-namespace (101 dumps, slow)
    python3 gate_coverage.py --agent-root /path --mode key --out KEYMAP.json

Exit codes: 0 clean, 1 coverage regressed against --expect, 2 probe failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

MARKER = "ZZPROBE"

# `research_overrides baseline` deliberately leaves the value-bearing keys unset,
# carrying a REPLACE:: marker, so an unfilled arm cannot silently reuse CIRIS
# values. The probe must fill them or every composition refuses.
REPLACE_SENTINEL = "REPLACE::"

# baseline emits a key its own validator rejects (extra_forbidden). Worked around
# here rather than patched; see CIRISAgent#986.
BASELINE_EXTRA_KEYS = ("_baseline_note",)


def _run(cmd: List[str], cwd: Path, timeout: int = 900) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def baseline_manifest(agent_root: Path) -> Dict[str, Any]:
    """Fetch the live baseline manifest and make it validator-clean."""
    rc, out = _run(
        [sys.executable, "-m", "ciris_engine.logic.utils.research_overrides", "baseline"],
        agent_root,
    )
    if rc != 0:
        raise SystemExit(f"research_overrides baseline failed ({rc}):\n{out[-2000:]}")
    # The command prints diagnostics before and after the JSON on some paths, so
    # take the first complete object rather than assuming it is the whole stream.
    start = out.find("{")
    if start < 0:
        raise SystemExit(f"no JSON in baseline output:\n{out[-2000:]}")
    manifest, _ = json.JSONDecoder().raw_decode(out[start:])
    for k in BASELINE_EXTRA_KEYS:
        manifest.pop(k, None)
    return manifest


def fill_sentinels(manifest: Dict[str, Any], text: str) -> int:
    """Fill every REPLACE:: key. Returns how many were filled."""
    n = 0
    for ns, block in manifest.get("overrides", {}).items():
        for k, v in block.items():
            if isinstance(v, str) and v.startswith(REPLACE_SENTINEL):
                block[k] = text
                n += 1
    return n


def dump(agent_root: Path, manifest: Dict[str, Any], arm: str, locales: str, work: Path) -> Dict[str, str]:
    """Compose under `manifest`; return {block_id: sha256}."""
    mpath = work / f"m_{arm}.json"
    opath = work / f"d_{arm}.jsonl"
    mpath.write_text(json.dumps(manifest))
    rc, out = _run(
        [
            sys.executable, "-m", "ciris_engine.logic.utils.compose_dump", "dump",
            "--arm", arm, "--locales", locales,
            "--manifest", str(mpath), "--out", str(opath),
        ],
        agent_root,
    )
    if rc != 0 or not opath.exists():
        raise SystemExit(f"compose_dump failed for arm={arm} ({rc}):\n{out[-3000:]}")
    shas: Dict[str, str] = {}
    for line in opath.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "compose_dump_meta":
            continue
        shas[row["block_id"]] = row["sha256"]
    if not shas:
        raise SystemExit(f"arm={arm} produced zero blocks — refusing to report coverage over nothing")
    return shas


def moved(base: Dict[str, str], probe: Dict[str, str]) -> List[str]:
    return sorted(k for k in base if k in probe and base[k] != probe[k])


def _mark(value: Any, tag: str) -> Any:
    """Marker of the same SHAPE as the value.

    A ``string`` override may be a scalar or a ``{locale: text}`` mapping
    (CIRISAgent#994). Marking only scalars silently skips every localized key —
    which is how an earlier run of this probe reported 0 dark keys while having
    probed 61 of 191. A coverage instrument that under-probes reports a false
    all-clear, which is the exact failure it exists to catch.
    """
    if isinstance(value, str):
        return tag
    if isinstance(value, dict):
        return {loc: f"{tag}-{loc}" for loc in value}
    return None


def key_bytes(manifest: Dict[str, Any], ns: str, keys: Set[str]) -> int:
    tot = 0
    for k in keys:
        v = manifest["overrides"].get(ns, {}).get(k)
        if isinstance(v, str):
            tot += len(v.encode("utf-8"))
        elif isinstance(v, dict):
            # Bytes of the base-locale value, so a localized key is counted once
            # rather than summed across every locale it ships in.
            base = v.get("en") or next(iter(v.values()), "")
            tot += len(str(base).encode("utf-8"))
    return tot


def probe_namespaces(
    agent_root: Path, manifest: Dict[str, Any], base: Dict[str, str], locales: str,
    work: Path, subsets: Dict[str, str],
) -> Dict[str, Any]:
    """Mark each namespace wholesale; record which blocks move."""
    result: Dict[str, Any] = {}
    targets: List[Tuple[str, str, Optional[str]]] = [(ns, ns, None) for ns in manifest["overrides"]]
    for label, prefix in subsets.items():
        ns = label.split(":", 1)[0]
        if ns in manifest["overrides"]:
            targets.append((label, ns, prefix))

    for label, ns, prefix in targets:
        probe_manifest = json.loads(json.dumps(manifest))
        block = probe_manifest["overrides"].get(ns, {})
        marked: Set[str] = set()
        for k in list(block):
            if prefix is not None and not k.startswith(prefix):
                continue
            marker = _mark(block[k], f"{MARKER}-{ns}-{k}")
            if marker is not None:
                block[k] = marker
                marked.add(k)
        skipped = len(block) - len(marked) if prefix is None else 0
        if skipped:
            print(f"  {label:26s} WARNING: {skipped} key(s) of an unmarkable shape — NOT probed", flush=True)
        if not marked:
            continue
        arm = "p_" + label.replace(":", "_").replace(".", "_").replace("*", "")
        shas = dump(agent_root, probe_manifest, arm, locales, work)
        mv = moved(base, shas)
        result[label] = {
            "keys": len(marked),
            "bytes": key_bytes(manifest, ns, marked),
            "blocks_moved": len(mv),
            "gated": len(mv) > 0,
            "blocks": mv,
            "key_names": sorted(marked) if not mv else [],
        }
        print(
            f"  {label:26s} keys={len(marked):>3}  bytes={result[label]['bytes']:>7,}  "
            f"moved={len(mv):>2}  {'GATED' if mv else 'INVISIBLE'}",
            flush=True,
        )
    return result


def probe_keys(
    agent_root: Path, manifest: Dict[str, Any], base: Dict[str, str], locales: str, work: Path
) -> Dict[str, Any]:
    """One dump per key. Slow, but attributes each key to the blocks it drives."""
    out: Dict[str, Any] = {}
    allk = [(ns, k) for ns, b in manifest["overrides"].items() for k in b]
    for i, (ns, k) in enumerate(allk, 1):
        pm = json.loads(json.dumps(manifest))
        if not isinstance(pm["overrides"][ns].get(k), str):
            continue
        pm["overrides"][ns][k] = _mark(pm["overrides"][ns][k], f"{MARKER}-{ns}-{k}")
        shas = dump(agent_root, pm, f"k{i}", locales, work)
        mv = moved(base, shas)
        out[f"{ns}:{k}"] = {"blocks_moved": len(mv), "blocks": mv, "gated": bool(mv)}
        print(f"  [{i}/{len(allk)}] {ns}:{k:52s} moved={len(mv)}", flush=True)
    return out


def compare(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """Regression check. New blind spots and lost blocks are failures; gains are not."""
    problems: List[str] = []
    exp_ns = expected.get("namespaces", {})
    act_ns = actual.get("namespaces", {})

    for label, e in exp_ns.items():
        a = act_ns.get(label)
        if a is None:
            problems.append(f"{label}: present in pinned record, absent now — namespace vanished")
            continue
        if e["gated"] and not a["gated"]:
            problems.append(f"{label}: REGRESSION — was gated, now moves zero blocks")
        elif e["gated"] and a["blocks_moved"] < e["blocks_moved"]:
            problems.append(
                f"{label}: coverage shrank — {e['blocks_moved']} -> {a['blocks_moved']} blocks"
            )
    for label, a in act_ns.items():
        if label not in exp_ns:
            problems.append(f"{label}: new namespace not in pinned record — re-pin before staking numbers")

    e_blocks, a_blocks = expected.get("total_blocks"), actual.get("total_blocks")
    if e_blocks and a_blocks and a_blocks < e_blocks:
        problems.append(f"total blocks fell {e_blocks} -> {a_blocks} — the dump lost coverage")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-root", required=True, type=Path, help="agent checkout/worktree to probe")
    ap.add_argument("--locales", default="en")
    ap.add_argument("--mode", choices=("namespace", "key"), default="namespace")
    ap.add_argument("--out", type=Path, help="write the coverage record here")
    ap.add_argument("--expect", type=Path, help="pinned record; exit 1 if coverage regressed")
    ap.add_argument("--keep", action="store_true", help="keep intermediate dumps")
    args = ap.parse_args()

    root = args.agent_root.resolve()
    if not (root / "ciris_engine").is_dir():
        print(f"error: {root} has no ciris_engine/ — not an agent checkout", file=sys.stderr)
        return 2

    rc, sha = _run(["git", "rev-parse", "HEAD"], root)
    sha = sha.strip() if rc == 0 else "unknown"
    rc, desc = _run(["git", "describe", "--tags", "--always"], root)
    desc = desc.strip() if rc == 0 else "unknown"

    print(f"probing {root}  ({desc} / {sha[:12]})")

    manifest = baseline_manifest(root)
    n_filled = fill_sentinels(manifest, "Baseline placeholder for the coverage probe.")
    n_keys = sum(len(b) for b in manifest["overrides"].values())
    print(f"manifest: {n_keys} keys across {len(manifest['overrides'])} namespaces; "
          f"filled {n_filled} REPLACE:: sentinel(s)")

    tmp = tempfile.TemporaryDirectory(prefix="gatecov-")
    work = Path(tmp.name)
    try:
        base = dump(root, manifest, "base", args.locales, work)
        print(f"reference arm: {len(base)} blocks\n")

        record: Dict[str, Any] = {
            "agent_git_sha": sha,
            "agent_describe": desc,
            "locales": args.locales,
            "total_keys": n_keys,
            "total_blocks": len(base),
            "mode": args.mode,
        }

        if args.mode == "key":
            record["keys"] = probe_keys(root, manifest, base, args.locales, work)
            blind = [k for k, v in record["keys"].items() if not v["gated"]]
            record["invisible_keys"] = sorted(blind)
            print(f"\ninvisible to the gate: {len(blind)}/{len(record['keys'])} keys")
        else:
            subsets = {"string:conscience.*": "conscience."}
            ns = probe_namespaces(root, manifest, base, args.locales, work, subsets)
            record["namespaces"] = ns
            blind_keys = sum(v["keys"] for v in ns.values() if not v["gated"])
            blind_bytes = sum(v["bytes"] for v in ns.values() if not v["gated"])
            record["invisible_key_count"] = blind_keys
            record["invisible_bytes"] = blind_bytes
            print(
                f"\ninvisible to the gate: {blind_keys} keys / {blind_bytes:,} B "
                f"(of {n_keys} keys, {len(base)} blocks)"
            )
            for label, v in ns.items():
                if not v["gated"]:
                    print(f"  {label}: {', '.join(v['key_names'][:4])}"
                          f"{' …' if len(v['key_names']) > 4 else ''}")

        if args.out:
            args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(f"\nwrote {args.out}")

        if args.expect:
            expected = json.loads(args.expect.read_text())
            problems = compare(record, expected)
            if problems:
                print(f"\nCOVERAGE REGRESSED against {args.expect.name}:")
                for p in problems:
                    print(f"  - {p}")
                return 1
            print(f"\ncoverage holds against {args.expect.name} "
                  f"(pinned {expected.get('agent_describe','?')})")
        return 0
    finally:
        if args.keep:
            keep = Path.cwd() / "gate_coverage_work"
            subprocess.run(["cp", "-r", str(work), str(keep)], check=False)
            print(f"kept intermediates in {keep}")
        tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
