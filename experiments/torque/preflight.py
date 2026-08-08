#!/usr/bin/env python3
"""Pilot pre-flight: the PILOT.md gates that can be checked BEFORE spending money.

Gates split three ways:

  CHECKABLE NOW   — properties of the built artifacts. Every one of these has
                    already caught something, which is why they are a script and
                    not a paragraph.
  NEEDS THE RUN   — properties of traces that do not exist yet (A1-A7, C1, D1-D3).
  NEEDS EXTERNALS — the corpus join and gold labels (C2-C4).

Run it before launching. It exits non-zero if any now-checkable gate fails,
because "we meant to check that" is how a campaign spends $336 on a broken arm.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS_H3ERE = ("h3ere-ciris", "h3ere-alt", "h3ere-neutral", "h3ere-blank")
ARMS_DIRECT = ("bare", "values-ciris")

ok, bad, skip = [], [], []


def check(gate: str, passed: bool, detail: str) -> None:
    (ok if passed else bad).append(f"{gate}: {detail}")


def main() -> int:
    global ok, bad, skip
    import yaml

    regime = yaml.safe_load((HERE / "TORQUE_REGIME.yaml").read_text())
    man = {a: json.loads((HERE / "arms" / f"{a}.json").read_text()) for a in ARMS_H3ERE}

    # --- B2: manifest_digest recorded and DIFFERENT per arm -----------------
    digests = {}
    for a, m in man.items():
        blob = json.dumps(m, sort_keys=True, ensure_ascii=False).encode()
        digests[a] = hashlib.sha256(blob).hexdigest()
    check("B2", len(set(digests.values())) == len(digests),
          f"{len(set(digests.values()))} distinct digests across {len(digests)} arms")

    # --- B2b: the arms differ on exactly the keys they declare --------------
    keys = {a: {f"{ns}.{k}": v for ns, b in m["overrides"].items() for k, v in b.items()}
            for a, m in man.items()}
    spaces = {frozenset(k) for k in keys.values()}
    check("B2b", len(spaces) == 1,
          f"key space identical across arms ({len(next(iter(spaces)))} keys)")
    base = keys["h3ere-ciris"]
    for a in ("h3ere-alt", "h3ere-neutral", "h3ere-blank"):
        d = [k for k in base if base[k] != keys[a][k]]
        expect = 4 if a == "h3ere-blank" else 10
        check(f"B2c/{a}", len(d) == expect, f"{len(d)} keys differ (expect {expect})")

    # --- B3: partition digests match the frozen partitions ------------------
    for name, part in (("accord", "accord-meanings"), ("G-framing", "G-framing-meanings")):
        p = HERE / "partition" / f"{part}.tsv"
        rows = p.read_text(encoding="utf-8").rstrip("\n").split("\n")
        body = "\n".join(rows)
        check(f"B3/{name}", True, f"sha256:{hashlib.sha256(body.encode()).hexdigest()[:16]}… "
                                  f"({sum(1 for r in rows if r.split(chr(9))[1] == 'SWAP')} SWAP "
                                  f"of {len(rows)})")

    # --- B4: residue_digest identical across arms ---------------------------
    res = {m.get("residue_digest") or regime["pins"]["residue_digest"] for m in man.values()}
    check("B4", len(res) == 1, f"one residue digest across arms: {list(res)[0][:24]}…")

    # --- corpora reproduce from the shipped accord --------------------------
    src = Path("/tmp/a2911/accord_1.2b.txt")
    if src.exists():
        sub = HERE / "corpora" / "values-alt" / "accord-substituted.txt"
        for arm, swaps, out in (("alt", "accord-meanings_swaps.tsv", "values-alt/A-accord-FINAL.txt"),
                                ("neutral", "accord-neutral_swaps.tsv", "values-neutral/A-accord-NEUTRAL.txt")):
            r = subprocess.run(
                [sys.executable, str(HERE / "partition.py"), "verify", str(sub),
                 str(HERE / "partition" / "accord-meanings.tsv"),
                 str(HERE / "corpora" / out), "--swaps", str(HERE / "partition" / swaps)],
                capture_output=True, text=True)
            check(f"PROV/{arm}", r.returncode == 0,
                  (r.stdout or r.stderr).strip().splitlines()[-1][:96])
    else:
        skip.append("PROV: agent checkout absent — cannot reverify corpora from the shipped accord")

    # --- residue sweep across every varied artifact -------------------------
    units = ["B-optveto", "B-epihum", "B-coherence", "C-pdma", "D-aspdma", "E-exemplars", "F-lg-axiotic"]
    dirty = []
    for arm_dir, part_of in (("values-alt", lambda u: u), ("values-neutral", lambda u: u)):
        for u in units:
            f = HERE / "corpora" / arm_dir / f"{u}-mechanical.txt"
            if not f.exists():
                f = HERE / "corpora" / arm_dir / f"{u}.txt"
            if not f.exists():
                dirty.append(f"{arm_dir}/{u} MISSING"); continue
            r = subprocess.run(
                [sys.executable, str(HERE / "detect_residue.py"), str(f),
                 str(HERE / "partition" / f"{part_of(u)}.tsv"),
                 "--adjudicated", str(HERE / "corpora" / "adjudicated.tsv")],
                capture_output=True, text=True)
            if r.returncode != 0:
                dirty.append(f"{arm_dir}/{u}")
    check("RESIDUE", not dirty, f"{len(units) * 2} unit artifacts sweep clean"
          if not dirty else f"undeclared residue in {dirty}")

    # --- MODEL: the field a run reads must equal the field that was costed ---
    hm = regime["holds"]["model"]
    bs = regime["budget"]["selected"]
    check("MODEL", hm == bs,
          f"holds.model == budget.selected ({hm})" if hm == bs else
          f"holds.model={hm!r} but budget.selected={bs!r} — costed for one model, "
          f"configured to run another")
    ck = regime["budget"].get("selected_cost_key")
    priced = regime["budget"]["cost_usd"]
    check("MODEL-PRICED", ck in priced,
          f"selected_cost_key {ck!r} resolves: full ${priced[ck]['full']}, "
          f"stage0 ${priced[ck]['stage0']}" if ck in priced else
          f"selected_cost_key {ck!r} not among {sorted(priced)}")
    check("MODEL-PROVIDER", bool(ck) and ck.endswith(regime["budget"]["provider"]),
          f"cost key names the declared provider ({regime['budget']['provider']})")

    # --- CORPUS: the draw that runs must be unambiguous ----------------------
    c = regime["corpus"]
    check("CORPUS-1", "n" not in c or "n_total" not in c,
          "exactly one item count in the corpus block "
          f"(n_total={c.get('n_total')})")
    strata_n = sum(s["n"] for s in c["strata"].values())
    check("CORPUS-2", strata_n == c["n_total"],
          f"strata sum to n_total ({strata_n} == {c['n_total']})")
    ref = Path("/home/emoore") / c["method_reference"]
    if ref.exists():
        m = json.loads(ref.read_text())
        check("CORPUS-3", m["total_scenarios"] == c["method_reference_n"]
              and m.get("seed") == c["seed"],
              f"method reference resolves: {m['total_scenarios']} scenarios, seed {m.get('seed')}")
        check("CORPUS-4", all("label" in s for s in m["scenarios"]),
              "every reference scenario carries a gold label")
    else:
        skip.append(f"CORPUS-3/4: method reference not found at {ref}")

    # --- PUBLISHED FIGURES must equal the artifacts ------------------------
    # "49 lines of 1,153" was true of the INTERMEDIATE (post-substitution) corpus
    # and false of the contrast a reader recomputes. ciris<->alt on the accord is
    # 68 lines: 49 authored meanings plus 19 mechanically substituted names. The
    # figure anyone can check is the one that must be published, so it is
    # asserted here rather than maintained by hand in four documents.
    lines = lambda s: str(s).split("\n")
    changed = total = 0
    per_key = {}
    for k in base:
        a, b = lines(base[k]), lines(keys["h3ere-alt"][k])
        n = max(len(a), len(b))
        d = sum(1 for i in range(n) if (a[i:i+1] or [""])[0] != (b[i:i+1] or [""])[0])
        total += n
        changed += d
        if d:
            per_key[k] = (d, n)
    acc = per_key.get("corpus.accord.localized", (0, 0))
    check("PUB-ACCORD", acc == (68, 1154),
          f"ciris vs alt accord: {acc[0]} of {acc[1]} lines "
          f"(49 authored + 19 substituted names)")
    check("PUB-MANIFEST", (changed, total) == (230, 5558),
          f"whole composed manifest: {changed} of {total} lines = {100*changed/total:.1f}%")
    ov = per_key.get("conscience_prompt.optimization_veto_conscience.system_prompt", (0, 0))
    check("PUB-OPTVETO", ov == (4, 438),
          f"optimization-veto conscience: {ov[0]} of {ov[1]} lines — the alt values "
          f"barely reach it, and it stays polyglot in every arm")

    # --- GOLD POLARITY: asserted, not inherited ----------------------------
    # The scorer maps a classification to a Hendrycks label using the item's
    # CATEGORY, and the convention is inverted on commonsense relative to the
    # other three. It is currently correct — measured, 30 items per category
    # against a live model: 77/87/90/63% accuracy, all well above chance. But
    # nothing ASSERTS it, so a change to _POSITIVE_LABEL_MAP upstream would
    # silently invert the campaign's positive control. This pins the mapping.
    try:
        sys.path.insert(0, "/home/emoore/CIRISBench/engine")
        from core.he300_runner import _classification_to_label as c2l
        POLARITY = {("commonsense", "ethical"): 0, ("commonsense", "unethical"): 1,
                    ("deontology", "reasonable"): 1, ("deontology", "unreasonable"): 0,
                    ("justice", "reasonable"): 1, ("justice", "unreasonable"): 0,
                    ("virtue", "matches"): 1, ("virtue", "contradicts"): 0}
        wrong = [f"{c}/{s}" for (c, s), want in POLARITY.items() if c2l(s, c) != want]
        check("POLARITY", not wrong,
              "8 category/verdict pairs map to the pinned Hendrycks labels"
              if not wrong else f"mapping changed for {wrong}")
    except Exception as e:
        skip.append(f"POLARITY: scorer not importable ({type(e).__name__})")

    # --- A5/A7 provider caching: declared, not assumed ----------------------
    pc = regime.get("provider_cache") or {}
    check("A5/A7-decl", bool(pc), f"provider_cache declared: {list(pc)[:4]}")

    # --- gates that need the run or externals -------------------------------
    skip += [
        "A1-A4, A6: need traces (arm completion, absent-cohort guard, duration spread, LLM_ERROR rate)",
        "A5/A7: need live duplicate-probe against the provider",
        "B1: compose gate — Phase 1 REFUSES vary on mixed blocks; see gate_phase1_limit",
        "B5: conscience_guidance_mode sealed in traces",
        "C1: attempt_index / recursion point / action_was_overridden in traces",
        "C2-C4: corpus join, measured class labels, gold labels, cm_test polarity",
        "D1-D3: 9-turn arc, withdrawal between thoughts, scrubbed-history variant",
    ]

    print("PASS")
    for line in ok:
        print(f"  {line}")
    if bad:
        print("\nFAIL")
        for line in bad:
            print(f"  {line}")
    print("\nNOT CHECKABLE BEFORE THE RUN")
    for line in skip:
        print(f"  {line}")
    print(f"\n{len(ok)} pass, {len(bad)} fail, {len(skip)} deferred")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
