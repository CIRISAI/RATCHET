Harness smoke cases for the Market 4 correction-episode workflow. Not frozen cases, not evidence:
they exist so the harness can be exercised end to end without running (and so seeing results on)
the frozen pilot cases. The objective and probe differ from every frozen case.

`smoke-cases-v2.jsonl` replaces the v1 genuine case for storage-path checks: the v1 paragraph said
"harness test only", and on RATCHET run 39785601766 the agent read that as an adversarial embedded
directive and REJECTed the approved change, so the run never reached the storage path under test.

`smoke-cases-v4.jsonl` (2026-10-22): live check of CIRISAgent#1314 (WA approval rendered into ASPDMA, MSASPDMA
and the optimization veto). Two objectives of the same weight as the pilot's ('Verify:' closing line, 'Plan:'
opening line) but not in any frozen battery, so the frozen cases stay unseen, plus the OBSERVER-resolve sham,
which exercises the harness fix that builds the WA request from the frozen paragraph.
