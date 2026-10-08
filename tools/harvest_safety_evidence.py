#!/usr/bin/env python3
"""Harvest battery evidence out of expiring Actions artifacts into the repo.

WHY. Artifacts expire at 90 days. Enumerating all 53 RATCHET battery runs, 16 had
already lost theirs — exp1_phase1 (8), crcv2_5vendor (2) and exp1b_crossfamily (1) have
no surviving evidence at all. A results table saying a cell passed, without the answer
that passed it, is an assertion rather than evidence. See RATCHET#26.

DESTINATION, chosen to match CIRISAgent's store so one provenance join works on both:

    qa_reports/safety_battery/<battery_id>_<run_id>/
        results.jsonl          the questions and the agent's answers
        summary.json           per-cell roll-up
        manifest_signed.json   signed digests of the two above, plus agent_version,
                               model, CI run id, and the agent_audit_anchors that
                               join each question to the agent's own task_id

Our artifact layout is NOT the destination layout (CIRISAgent's is, so theirs restores
with `unzip -d qa_reports/`). Ours is
`captures-<arm>-<locale>/<arm>__<locale>__<model>/results.jsonl`, so the key is read from
`manifest_signed.json` — the file that already carries battery_id and run_id — falling
back to `results.jsonl`. Nothing parses a directory name, so there is no mapping table to
drift out of date.

WHAT IS LEFT BEHIND. Everything else: logs, console output, incident reports, server
logs. `summary.json` is taken VERBATIM even though `delivery.probe_lines` is 96.5% of all
summary bytes (52.3 MB of 54.2 MB) — raw agent log scraped from latest.log. It cannot be
stripped, because the manifest signs `summary_json_sha256` and the attestation is the
point of keeping the file.

IDEMPOTENT. Re-running is safe: a cell already present is skipped unless the incoming
copy is more complete. A missed sync is picked up by the next run with no operator action.

VERIFIES WHAT IT WRITES. Every harvested cell has its results.jsonl and summary.json
hashed against the manifest's declared digests. Mismatches are reported and the cell is
still written (a mismatch is evidence about the pipeline, not a reason to drop data), but
the exit summary names them.
"""
import argparse, hashlib, json, os, shutil, sys, collections

EVIDENCE = ("results.jsonl", "summary.json", "manifest_signed.json")


def cell_key(d):
    """(battery_id, run_id) for a capture directory, from the manifest then results."""
    mf = os.path.join(d, "manifest_signed.json")
    if os.path.exists(mf):
        try:
            m = json.load(open(mf))
            if m.get("battery_id") and m.get("run_id"):
                return "%s_%s" % (m["battery_id"], m["run_id"]), m
        except Exception:
            pass
    rf = os.path.join(d, "results.jsonl")
    if os.path.exists(rf):
        for line in open(rf):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                break
            if r.get("battery_id") and r.get("run_id"):
                return "%s_%s" % (r["battery_id"], r["run_id"]), None
            break
    return None, None


def alt_key(d, root):
    """Key for a capture whose harness writes no battery_id/run_id."""
    rel = os.path.relpath(d, root).split(os.sep)
    ci = next((p.split("_", 1)[1] for p in rel
               if p.startswith(("RATCHET_", "CIRISAgent_")) and "_" in p), None)
    harness = None
    rf = os.path.join(d, "results.jsonl")
    for line in open(rf):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            break
        harness = r.get("harness")
        break
    if not (ci and harness):
        return None
    return os.path.join("alt_harness", "%s_%s_%s" % (harness, ci, rel[-1]))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", help="repo-relative destination, normally qa_reports")
    ap.add_argument("--from-dir", required=True,
                    help="directory of downloaded artifacts to harvest from")
    ap.add_argument("--only-repo", default=None,
                    help="harvest only cells whose manifest ci_provenance names this "
                         "repo (e.g. CIRISAI/RATCHET). Another repo's evidence belongs "
                         "in that repo's own store, not duplicated here.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # every directory holding a results.jsonl is a candidate cell
    cands = collections.defaultdict(list)
    unkeyed = []
    stat_alt = []
    for root, _dirs, files in os.walk(a.from_dir):
        if "results.jsonl" not in files:
            continue
        key, man = cell_key(root)
        size = sum(os.path.getsize(os.path.join(root, n))
                   for n in EVIDENCE if os.path.exists(os.path.join(root, n)))
        have = sum(1 for n in EVIDENCE if os.path.exists(os.path.join(root, n)))
        if key is None:
            # A different harness (withdraw_arc.py) writes no battery_id/run_id. Do NOT
            # drop it: key it by harness + the CI run that produced it + the cell dir,
            # under an `alt_harness/` prefix so the schema difference stays visible.
            key = alt_key(root, a.from_dir)
            if key is None:
                unkeyed.append(root)
                continue
            stat_alt.append(key)
        cands[key].append((have, size, root, man))

    stat = collections.Counter()
    bad_digest = []
    skipped_repo = 0
    for key, lst in sorted(cands.items()):
        have, size, src, man = max(lst)           # most complete copy wins
        if a.only_repo:
            repo = ((man or {}).get("ci_provenance") or {}).get("github_repository")
            if repo and repo != a.only_repo:
                skipped_repo += 1
                continue
        if len(lst) > 1:
            stat["duplicate_cells_collapsed"] += len(lst) - 1
        out = (os.path.join(a.dest, key) if key.startswith("alt_harness" + os.sep)
               else os.path.join(a.dest, "safety_battery", key))
        present = [n for n in EVIDENCE if os.path.exists(os.path.join(out, n))]
        if len(present) >= have:
            stat["already_present"] += 1
            continue
        stat["written" if not present else "completed"] += 1
        if a.dry_run:
            continue
        os.makedirs(out, exist_ok=True)
        for n in EVIDENCE:
            s = os.path.join(src, n)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(out, n))
        # verify against the signed digests
        mf = os.path.join(out, "manifest_signed.json")
        if os.path.exists(mf):
            try:
                m = json.load(open(mf))
            except Exception:
                m = {}
            for fname, field in (("results.jsonl", "results_jsonl_sha256"),
                                 ("summary_json", "summary_json_sha256")):
                real = "summary.json" if fname == "summary_json" else fname
                want = (m.get("bundle") or {}).get(field)
                p = os.path.join(out, real)
                if want and os.path.exists(p):
                    got = sha256(p)
                    stat["digest_ok" if got == want else "digest_MISMATCH"] += 1
                    if got != want:
                        bad_digest.append((key, real, want[:12], got[:12]))

    for u in unkeyed:
        stat["unkeyed_skipped"] += 1
    if skipped_repo:
        stat["skipped_other_repo"] = skipped_repo
    if stat_alt:
        stat["alt_harness_cells"] = len(stat_alt)

    print("harvest summary")
    for k, v in sorted(stat.items()):
        print(f"  {k}: {v}")
    print(f"  distinct cells seen: {len(cands)}")
    if unkeyed:
        print(f"\n{len(unkeyed)} cells had no battery_id/run_id and were SKIPPED "
              f"(different harness schema, e.g. withdraw-intact):")
        for u in unkeyed[:5]:
            print("   ", os.path.relpath(u, a.from_dir))
    if bad_digest:
        print(f"\n{len(bad_digest)} DIGEST MISMATCHES (written anyway, reported here):")
        for k, n, w, g in bad_digest[:20]:
            print(f"    {k}/{n}: manifest {w}… actual {g}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
