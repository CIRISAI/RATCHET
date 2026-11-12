"""Ledger, registry, auditor, proposer and classifier — the deterministic half of the loop.

Ledger: loop/ledger.jsonl, append-only. Entry types:
  stake     {id, block, batch, batch_sha256}            frozen stake (block = the .ks fields incl. cites)
  result    {id, record, verdict: hit|miss|void}        engine output for a stake
  imported  {id, block, record}                         a measured cell from an earlier frozen run (e.g. VOCAB-1)
  instance  {stake, misfit}                             a miss explained by a registered/candidate misfit signature
  diagnosis {id, block, test_stake}                     a diagnosis written in the language
  status    {misfit, status, by}                        misfit status change, CC 2.4 acts: candidate -> registered
                                                        (authorised by its test) | recanted (test missed; row kept);
                                                        replaced (a refined id supersedes it; row kept)
"""

import datetime
import hashlib
import json
import pathlib
import re

from ks.lang import (POS, SITES, STATES, LangError, check_stake, emit, parse, predicate, stake_line)

ROOT = pathlib.Path(__file__).resolve().parent.parent / "loop"
LEDGER = ROOT / "ledger.jsonl"
REGISTRY = ROOT / "registry.ks"
FROZEN = ROOT / "frozen"


def today():
    return datetime.date.today().isoformat()


def ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def append(entry):
    entry = {"at": datetime.datetime.now().isoformat(timespec="seconds"), **entry}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def registry():
    """id -> misfit block, with status updated by ledger status entries."""
    reg = {b["id"]: b for b in parse(REGISTRY.read_text()) if b["kind"] == "misfit"}
    for e in ledger():
        if e["type"] == "status" and e["misfit"] in reg:
            reg[e["misfit"]]["status"] = e["status"]
    return reg


# ------------------------------------------------------------------ audit

def audit(stake, capabilities):
    """Return (cites, refusals). A binding misfit's every requirement must be a capability of the stake
    (capabilities: function stake -> set, from the engine)."""
    caps = capabilities(stake)
    cites, refusals = [], []
    for mid, m in registry().items():
        if m.get("status") in ("recanted", "withdrawn", "replaced"):
            continue
        if m.get("applies") and predicate(m["applies"])(stake):
            cites.append(mid)
            for req in [r.strip() for r in m.get("requires", "cite").split(",") if r.strip()]:
                if req != "cite" and req not in caps:
                    refusals.append(f"{mid} requires `{req}`; not provided for {stake['verb']} at {stake['at']}")
    return sorted(cites), refusals


# ------------------------------------------------------------------ map

def measured():
    """stake id -> (block, record) for every stake with a non-void result, plus imported cells."""
    out, blocks = {}, {}
    for e in ledger():
        if e["type"] in ("stake", "imported"):
            blocks[e["id"]] = check_stake(e["block"])
        if e["type"] == "imported":
            out[e["id"]] = (blocks[e["id"]], e["record"])
        if e["type"] == "result" and e["verdict"] != "void":
            out[e["id"]] = (blocks[e["id"]], e["record"])
    return out


def key(b):
    return (b["verb"], b["noun"], b["at"])


def outcome(pred, rec):
    m = re.match(r"^(\w+)\s*(=|!=|<=|>=|<|>)\s*(\S+)$", pred)
    f, op, v = m.groups()
    real = rec.get(f)
    if real is None:
        return None, real
    return predicate(f"{f}{op}{v}")(rec), real


# ------------------------------------------------------------------ proposer

def propose(n):
    """Deterministic acquisition. Order: (0) diagnosis tests already queued are frozen by `diagnose`, not here;
    (1) per-class expansion of every measured pos10 cell, transition cells first, predicting the aggregate verdict."""
    done = {key(b) for b, _ in measured().values()}
    queued = {key(check_stake(e["block"])) for e in ledger() if e["type"] == "stake"}
    agg = {b["at"]: (sid, rec) for sid, (b, rec) in measured().items() if b["noun"] == "pos10" and b["verb"] == "probe"}

    def transition(at):
        site, state = at.split("/")
        j = STATES.index(state)
        here = agg[at][1].get("carried")
        nb = [agg.get(f"{site}/{STATES[k]}") for k in (j - 1, j + 1) if 0 <= k < len(STATES)]
        return any(x is not None and x[1].get("carried") != here for x in nb)

    order = sorted(agg, key=lambda at: (not transition(at), SITES.index(at.split("/")[0]),
                                        STATES.index(at.split("/")[1])))
    out = []
    for at in order:
        sid, rec = agg[at]
        if rec.get("carried") is None:
            continue
        for c in POS:
            b = {"verb": "probe", "noun": f"pos-ovr:{c}", "at": at, "split": "explore",
                 "predict": f"carried = {'true' if rec['carried'] else 'false'}", "from": sid,
                 "author": "proposer:rule-A (per-class expansion of the aggregate verdict)"}
            if (b["verb"], b["noun"], b["at"]) in done | queued:
                continue
            out.append(b)
            if len(out) >= n:
                return out
    return out


def next_id(prefix="S"):
    ids = [e["id"] for e in ledger() if e["type"] in ("stake", "diagnosis") and re.match(fr"^{prefix}\d+$", e["id"])]
    return f"{prefix}{max([int(i[1:]) for i in ids], default=0) + 1:04d}"


def known_at_freeze():
    """What was readable when a batch froze: result count and the ledger's hash (holon: record what was KNOWN)."""
    raw = LEDGER.read_bytes() if LEDGER.exists() else b""
    return f"{sum(1 for e in ledger() if e['type'] in ('result', 'imported'))} results; ledger sha256 " \
           f"{hashlib.sha256(raw).hexdigest()[:16]}"


def freeze(blocks, capabilities, goal="G1"):
    """Audit and freeze a batch: write loop/frozen/<batch>.ks, hash it, append stake entries. Returns (batch, refused)."""
    FROZEN.mkdir(parents=True, exist_ok=True)
    n0 = int(next_id()[1:])
    known = known_at_freeze()
    accepted, refused, text = [], [], [f"# frozen {today()} — generated by ks; audited against registry.ks\n"]
    for i, b in enumerate(blocks):
        blk = {"kind": "stake", "id": b.get("id") or f"S{n0 + i:04d}", "goal": goal, **{k: v for k, v in b.items()
                                                                                          if k not in ("id",)}}
        rec = check_stake(blk)
        cites, refusals = audit(rec, capabilities)
        if refusals:
            refused.append((blk["id"], refusals))
            continue
        blk["cites"] = ", ".join(cites)
        blk["known"] = known
        accepted.append(blk)
        text.append(emit(blk))
    if not accepted:
        return None, refused
    batch = f"B{len(list(FROZEN.glob('B*.ks'))) + 1:04d}"
    path = FROZEN / f"{batch}.ks"
    path.write_text("\n".join(text))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for blk in accepted:
        append({"type": "stake", "id": blk["id"], "block": {k: v for k, v in blk.items()}, "batch": batch,
                "batch_sha256": sha})
    return batch, refused


def pending():
    done = {e["id"] for e in ledger() if e["type"] == "result"}
    out = []
    for e in ledger():
        if e["type"] == "stake" and e["id"] not in done:
            path = FROZEN / f"{e['batch']}.ks"
            if hashlib.sha256(path.read_bytes()).hexdigest() != e["batch_sha256"]:
                raise RuntimeError(f"{e['batch']} changed after freeze — refusing to run {e['id']}")
            out.append(check_stake(e["block"]))
    # diagnosis tests first (they settle candidate misfits), then proposer stakes, each in ledger order
    return sorted(out, key=lambda st: 0 if st.get("from", "").startswith("D") else 1)


# ------------------------------------------------------------------ classify

def classify(stake, rec):
    """Return (verdict, matched misfit ids)."""
    if rec.get("void"):
        return "void", []
    hit, _ = outcome(stake["predict"], rec)
    if hit is None:
        return "undecided", []  # neither met nor killed: a first-class outcome (holon)
    if hit:
        return "hit", []
    full = {**stake, **{k: v for k, v in rec.items() if not isinstance(v, (dict, list))}, "hit": False}
    matched = [mid for mid, m in registry().items()
               if m.get("signature") and m.get("status") in ("registered", "candidate") and predicate(m["signature"])(full)]
    return "miss", matched


def open_misses():
    explained = {e["stake"] for e in ledger() if e["type"] == "instance"}
    for e in ledger():
        if e["type"] == "diagnosis":
            explained |= {s.strip() for s in e["block"]["for"].split(",")}
    return [e for e in ledger() if e["type"] == "result" and e["verdict"] == "miss" and e["id"] not in explained]


# ------------------------------------------------------------------ diagnose

def engine_sha():
    """The instrument a result or misfit rests on: hash of the engine, language and frozen VOCAB-1 pipeline."""
    h = hashlib.sha256()
    for p in ("ks/engine.py", "ks/lang.py", "vocab1.py", "cal1.py"):
        h.update((ROOT.parent / p).read_bytes())
    return h.hexdigest()[:16]


def diagnose(text, capabilities):
    """Accept a diagnosis block. A new misfit id becomes a candidate whose signature must match every miss it claims;
    its test stake is frozen and queued. Returns a message; raises LangError on refusal."""
    blocks = [b for b in parse(text) if b["kind"] == "diagnosis"]
    if len(blocks) != 1:
        raise LangError("exactly one diagnosis block per file")
    d = blocks[0]
    for k in ("for", "misfit", "test"):
        if not d.get(k):
            raise LangError(f"diagnosis needs `{k}`")
    claimed = [s.strip() for s in d["for"].split(",")]
    misses = {e["id"]: e for e in open_misses()}
    stakes = {e["id"]: check_stake(e["block"]) for e in ledger() if e["type"] == "stake"}
    for s in claimed:
        if s not in misses:
            raise LangError(f"{s} is not an open miss")
    reg = registry()
    mid = d["misfit"]
    if mid not in reg:
        for k in ("line", "rule", "signature"):
            if not d.get(k):
                raise LangError(f"new misfit {mid} needs `{k}`")
        sig = predicate(d["signature"])
        for s in claimed:
            r = misses[s]["record"]
            full = {**stakes[s], **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}, "hit": False}
            if not sig(full):
                raise LangError(f"signature does not match the miss it claims: {s}")
    test = stake_line(d["test"])
    test["split"] = "confirm"
    test["author"] = d.get("author", "ratchet-b9 (diagnoser)")
    if not test["from"]:
        test["from"] = d["id"]
    if test["from"] != d["id"]:
        raise LangError("a diagnosis test takes its bar from the diagnosis itself (`from` = the diagnosis id)")
    if (test["verb"], test["noun"], test["at"]) in {key(b) for b, _ in measured().values()}:
        raise LangError("the test stake is already measured — a diagnosis must predict something not yet read")
    batch, refused = freeze([test], capabilities)
    if refused:
        raise LangError(f"test stake refused by audit: {refused}")
    tid = [e for e in ledger() if e["type"] == "stake"][-1]["id"]
    if mid not in reg:
        mb = {"kind": "misfit", "id": mid, **{k: d[k] for k in ("line", "rule", "applies", "requires", "signature",
                                                                 "replaces") if d.get(k)},
              "born": f"{d['id']} {today()}", "status": "candidate",
              "rests_on": ", ".join(claimed) + f"; engine {engine_sha()}"}
        with open(REGISTRY, "a") as f:
            f.write("\n" + emit(mb))
    append({"type": "diagnosis", "id": d["id"], "block": d, "test_stake": tid, "misfit": mid})
    return f"{d['id']}: {mid} ({'candidate' if mid not in reg else reg[mid].get('status')}); test {tid} frozen in {batch}"


def settle_diagnoses():
    """After results: a candidate whose test hit is registered; one whose test missed is refuted (kept visible)."""
    res = {e["id"]: e for e in ledger() if e["type"] == "result"}
    settled = {e["by"] for e in ledger() if e["type"] == "status"}
    msgs = []
    for e in ledger():
        if e["type"] == "diagnosis" and e["id"] not in settled and e["test_stake"] in res:
            v = res[e["test_stake"]]["verdict"]
            if v in ("void", "undecided"):
                if not any(x["type"] == "status" and x["by"] == e["id"] for x in ledger()):
                    append({"type": "status", "misfit": e["misfit"], "status": registry()[e["misfit"]]["status"],
                            "by": e["id"], "note": f"test {e['test_stake']} {v}: candidate stays a candidate"})
                    msgs.append(f"{e['misfit']}: test {e['test_stake']} {v} — stays candidate")
                continue
            st = "registered" if v == "hit" else "recanted"  # CC 2.4.1.1: never right -> recant, row kept
            if registry().get(e["misfit"], {}).get("status") == "candidate":
                append({"type": "status", "misfit": e["misfit"], "status": st, "by": e["id"]})
                msgs.append(f"{e['misfit']}: {st} by {e['id']} (test {e['test_stake']} {v})")
            else:
                append({"type": "status", "misfit": e["misfit"], "status": registry()[e["misfit"]]["status"],
                        "by": e["id"]})
                msgs.append(f"{e['misfit']}: existing id; test {e['test_stake']} {v} — recorded")
    return msgs
