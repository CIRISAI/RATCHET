Pip constraints for the Market 4 correction-eval workflow: the exact package set installed on a CI runner for
CIRISAgent `81e9bf261f813dc1da85309aea2758a0f1d8ef55` (Python 3.12.15), captured by `pip freeze --exclude-editable`
in RATCHET run 39865475315 (job smoke4-g1, attested with the raw log). A frozen run passes this file by raw URL and
sha256 (`constraints_url`, `constraints_sha256`), so a rerun installs the same versions.

`constraints-cirisagent-v2.16.4-6e2b259e.txt`: the same capture for CIRISAgent v2.16.4-stable (`6e2b259eb276`,
ciris-server 0.5.234, ciris-client 0.5.231), RATCHET run 39975512932 (job smoke4-g1).
