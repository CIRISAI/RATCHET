#!/usr/bin/env python3
"""ks — the kindscope loop CLI.

  ks status                         map, queue, open misses, misfits by status
  ks import-vocab1 <result.json>    seed the map with VOCAB-1's 39 measured cells (recorded as imported, with its
                                    uncited contacts named)
  ks build-cache <model> <corpus>   collect the residual states once (GPU); cache in $KS_CACHE
  ks propose [-n N]                 deterministic proposer -> audit -> freeze a batch (loop/frozen/Bxxxx.ks)
  ks gate                           determinism gate for the current engine (required before run)
  ks run [-n N]                     run pending frozen stakes; record, classify (hit | miss | void), settle diagnoses
  ks miss <stake>                   the context packet for diagnosing one open miss
  ks diagnose <file.ks>             submit a diagnosis (new misfit -> candidate; its test stake is frozen)
  ks render                         write MISFITS.md from registry.ks + ledger
"""

import json
import os
import pathlib
import subprocess
import sys

# determinism (holon: fixed-order reductions; pinned cuBLAS workspace) — must precede CUDA initialisation
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ks import loop as L  # noqa: E402
from ks.lang import SITES, STATES, LangError, check_stake  # noqa: E402

CACHE = os.environ.get("KS_CACHE", "/tmp/claude-1000/-home-emoore-RATCHET/4fdbd195-6bf1-45c9-8ffc-931540da4e4d/"
                                   "scratchpad/ks_cache")
def CAPS(stake):
    from ks import engine
    return engine.capabilities(stake)


def cmd_status(_):
    led = L.ledger()
    meas = L.measured()
    print(f"ledger {len(led)} entries; measured {len(meas)}; pending {len(L.pending())}; "
          f"open misses {len(L.open_misses())}; engine {L.engine_sha()}")
    agg = {b["at"]: r for b, r in meas.values() if b["noun"] == "pos10"}
    mark = {True: "C", False: ".", None: "?"}
    print("\npos10 carried-beyond-surface (C yes, . no, ? void, blank unmeasured)")
    print("        " + " ".join(f"{s[:5]:>5}" for s in STATES))
    for site in SITES:
        print(f"{site:>6}  " + " ".join(f"{mark[agg[f'{site}/{s}'].get('carried')] if f'{site}/{s}' in agg else ' ':>5}"
                                        for s in STATES))
    res = [e for e in led if e["type"] == "result"]
    if res:
        v = {k: sum(1 for e in res if e["verdict"] == k) for k in ("hit", "miss", "undecided", "void")}
        print(f"\nstakes run: {len(res)}  hit {v['hit']}  miss {v['miss']}  undecided {v['undecided']}  void {v['void']}")
        reads = {sp: sum(1 for e in res if e["record"].get("split") == sp) for sp in ("explore", "confirm")}
        print(f"reads per split (G-SECOND-READ): explore {reads['explore']}  confirm {reads['confirm']} "
              f"(+ VOCAB-1 read both halves once, pos10 only, before the loop)")
    print("\nmisfits:")
    for mid, m in L.registry().items():
        n = sum(1 for e in led if e["type"] == "instance" and e["misfit"] == mid)
        print(f"  {mid:<24} {m.get('status', '?'):<10} instances {n}")
    om = L.open_misses()
    if om:
        print("\nopen misses (need a diagnosis):")
        for e in om[:20]:
            b = check_stake([x for x in led if x["type"] == "stake" and x["id"] == e["id"]][0]["block"])
            print(f"  {e['id']}  {b['noun']:<16} {b['at']:<12} predicted {b['predict']:<16} "
                  f"carried={e['record'].get('carried')} V1={e['record'].get('V1')} V2={e['record'].get('V2')} "
                  f"V3={e['record'].get('V3')}")


def cmd_import_vocab1(args):
    res = json.load(open(args[0]))
    have = {e.get("id") for e in L.ledger()}
    n = 0
    for cell, c in res["cells"].items():
        sid = f"S-V1-{cell}"
        if sid in have:
            continue
        block = {"kind": "stake", "id": sid, "goal": "G1", "verb": "probe", "noun": "pos10", "at": cell,
                 "predict": "carried = true", "from": "prior-art:PREREG_VOCAB1"}
        rec = {"acc": c["acc_all"], "acc_novel": c["acc_novel"], "V1": c["V1"]["pass"], "V2": c["V2"]["pass"],
               "V3": c["V3"]["pass"], "carried": c["carried_beyond_surface"], "diff_novel": c["V3"]["diff_novel"],
               "ci95": c["V3"]["ci95"], "null_p95": c["V1"]["null_p95"], "ctrl_max": c["V2"]["ctrl_max"],
               "base_rate": res["majority_acc_test"], "plant_ok": None,
               "provenance": "VOCAB-1 (frozen run, no plant; uncited contacts M-PLANT-OBS, M-JOINT-PASS-REGION, "
                             "M-EXIT-DISCRIMINATOR, M-DEVICE-CLASS, M-FLOOR-UNSTAKED, M-ONE-MODEL-DELTA)"}
        L.append({"type": "imported", "id": sid, "block": block, "record": rec})
        n += 1
    print(f"imported {n} VOCAB-1 cells")


def cmd_build_cache(args):
    from ks import engine
    engine.build_cache(args[0], args[1], CACHE)
    print(f"cache built at {CACHE}")


def _n(args, default):
    return int(args[args.index("-n") + 1]) if "-n" in args else default


def cmd_propose(args):
    blocks = L.propose(_n(args, 10))
    if not blocks:
        print("proposer: nothing to propose")
        return
    batch, refused = L.freeze(blocks, CAPS)
    for sid, why in refused:
        print(f"REFUSED {sid}: {'; '.join(why)}")
    if batch:
        print(f"froze {batch}: {len(blocks) - len(refused)} stakes -> loop/frozen/{batch}.ks")


def run_conditions():
    """Launch header (holon run_conditions.sh): load, clock, git state of the instrument, driver."""
    here = pathlib.Path(__file__).resolve().parent
    git = lambda *a: subprocess.run(["git", *a], cwd=here, capture_output=True, text=True).stdout.strip()  # noqa
    dirty = git("status", "--porcelain", "--", "ks", "ks.py", "vocab1.py", "cal1.py")
    try:
        cur = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read())
        mx = int(open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read())
        clock = round(cur / mx, 3)
    except OSError:
        clock = None
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,temperature.gpu,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"loadavg": os.getloadavg(), "cpu0_clock_frac": clock, "head": git("rev-parse", "HEAD"),
            "instrument_dirty": bool(dirty), "engine_sha": L.engine_sha(), "gpu": drv}


def cmd_gate(_):
    """Determinism gate for the current engine: one fixed stake, two runs, every field compared exactly."""
    from ks import engine
    import torch
    torch.use_deterministic_algorithms(True)
    cache = engine.Cache(CACHE)
    st = check_stake({"id": "GATE", "verb": "probe", "noun": "pos-ovr:DET", "at": "pre/attn0",
                      "predict": "carried = true", "from": "plant:gate", "split": "explore"})
    a, b = engine.run_probe(st, cache), engine.run_probe(st, cache)
    strip = lambda r: json.dumps({k: v for k, v in r.items() if k not in ("wall_sec", "env")}, sort_keys=True)  # noqa
    ok = strip(a) == strip(b)
    L.append({"type": "gate", "engine_sha": L.engine_sha(), "deterministic": ok, "env": a["env"],
              "conditions": run_conditions()})
    print(f"determinism gate on engine {L.engine_sha()}: {'PASS' if ok else 'FAIL'} (2 runs, {len(strip(a))} bytes compared)"
          " — the gate stake's own result is not recorded as a reading")
    sys.exit(0 if ok else 1)


def cmd_run(args):
    from ks import engine
    import torch
    torch.use_deterministic_algorithms(True)
    sha = L.engine_sha()
    if not any(e["type"] == "gate" and e["engine_sha"] == sha and e["deterministic"] for e in L.ledger()):
        sys.exit(f"ks run: no passing determinism gate for engine {sha} — run `ks gate` first")
    cond = run_conditions()
    if cond["instrument_dirty"]:
        sys.exit("ks run: instrument files are uncommitted — commit them first (a result must cite a committed "
                 "instrument; holon M-STALE-INSTRUMENT / M-PROVENANCE-OVERREACH)")
    cache = engine.Cache(CACHE)
    todo = L.pending()[:_n(args, 10)]
    L.append({"type": "run", "phase": "start", "n": len(todo), "conditions": cond})
    if not todo:
        L.append({"type": "run", "phase": "end", "done": 0, "conditions": run_conditions()})
        sys.exit("ks run: zero pending stakes — nothing was run (a zero count is a failure, M-VACUOUS-SUCCESS)")
    done = 0
    for st in todo:
        rec = engine.run_probe(st, cache)
        rec["engine_sha"] = sha
        verdict, matched = L.classify(st, rec)
        L.append({"type": "result", "id": st["id"], "record": rec, "verdict": verdict})
        for mid in matched:
            L.append({"type": "instance", "stake": st["id"], "misfit": mid})
        print(f"{st['id']} {st['noun']:<16} {st['at']:<12} predicted {st['predict']:<16} -> {verdict}"
              f"{' [' + ','.join(matched) + ']' if matched else ''}  plant_ok={rec['plant_ok']} "
              f"power_ok={rec.get('power_ok')} carried={rec.get('carried')}{'  VOID: ' + rec['void'] if rec.get('void') else ''}  ({rec['wall_sec']:.0f}s)", flush=True)
        done += 1
    L.append({"type": "run", "phase": "end", "done": done, "conditions": run_conditions()})
    print(f"ran {done}/{len(todo)} stakes")
    for m in L.settle_diagnoses():
        print(m)


def cmd_miss(args):
    sid = args[0]
    led = L.ledger()
    st = check_stake([e for e in led if e["type"] == "stake" and e["id"] == sid][0]["block"])
    rec = [e for e in led if e["type"] == "result" and e["id"] == sid][0]["record"]
    meas = L.measured()
    src = meas.get(st["from"])
    print(json.dumps({"stake": st, "record": rec, "bar_source": src[1] if src else st["from"],
                      "same_cell_other_nouns": {b["noun"]: {k: r.get(k) for k in ("carried", "V1", "V2", "V3", "acc",
                                                                                  "diff_novel")}
                                                for b, r in meas.values() if b["at"] == st["at"]},
                      "same_noun_other_cells": {b["at"]: r.get("carried") for b, r in meas.values()
                                                if b["noun"] == st["noun"]}}, indent=1, default=str))


def cmd_diagnose(args):
    try:
        print(L.diagnose(open(args[0]).read(), CAPS))
    except LangError as e:
        print(f"REFUSED: {e}")
        sys.exit(1)


def cmd_render(_):
    reg = L.registry()
    led = L.ledger()
    rows = ["# Kindscope misfit registry (rendered by `ks render` from loop/registry.ks + loop/ledger.jsonl)\n",
            "| id | status | one line | rule | instances | born | rests on |", "|---|---|---|---|---|---|---|"]
    for mid, m in reg.items():
        n = [e["stake"] for e in led if e["type"] == "instance" and e["misfit"] == mid]
        rows.append(f"| **{mid}** | {m.get('status')} | {m.get('line', '')} | {m.get('rule', '')} | "
                    f"{len(n)}{' (' + ', '.join(n[:6]) + ('…' if len(n) > 6 else '') + ')' if n else ''} | "
                    f"{m.get('born', '')} | {m.get('rests_on', '')} |")
    (pathlib.Path(__file__).resolve().parent / "MISFITS.md").write_text("\n".join(rows) + "\n")
    print(f"rendered {len(reg)} misfits")


if __name__ == "__main__":
    cmds = {"status": cmd_status, "import-vocab1": cmd_import_vocab1, "build-cache": cmd_build_cache,
            "propose": cmd_propose, "run": cmd_run, "gate": cmd_gate, "miss": cmd_miss, "diagnose": cmd_diagnose, "render": cmd_render}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]](sys.argv[2:])
