Pip constraints for the Market 4 correction-eval workflow: the exact package set installed on a CI runner for
CIRISAgent `81e9bf261f813dc1da85309aea2758a0f1d8ef55` (Python 3.12.15), captured by `pip freeze --exclude-editable`
in RATCHET run 39865475315 (job smoke4-g1, attested with the raw log). A frozen run passes this file by raw URL and
sha256 (`constraints_url`, `constraints_sha256`), so a rerun installs the same versions.

`constraints-cirisagent-v2.16.4-6e2b259e.txt`: the same capture for CIRISAgent v2.16.4-stable (`6e2b259eb276`,
ciris-server 0.5.234, ciris-client 0.5.231), RATCHET run 39975512932 (job smoke4-g1).

`constraints-cirisagent-v2.20.0-f0a07995.txt`: the same capture for CIRISAgent v2.20.0-stable (`f0a079953da5`,
ciris-server 0.5.245, ciris-client 0.5.240; Python 3.12.16), 131 packages, RATCHET run 42889080852 (job smoke5-obj,
harness `f8c8da03`). The file's digest is attested by all seven jobs of that run (identical installs).

`constraints-cirisagent-v2.21.0-d4b2313b.txt`: the same capture for CIRISAgent v2.21.0-stable (`d4b2313bb77f`,
ciris-server 0.5.246, ciris-client 0.5.240; Python 3.12.16), 131 packages, RATCHET run 43450895582 (job smoke6-obj,
harness `market4/on-v2210` `79d47b56`). sha256 `a391505ea7c864024c98a4422dae42df721a9f15c0bb56778783a6339bce27cc`. One job only, so the digest is attested once.
