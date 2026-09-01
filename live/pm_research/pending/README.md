# Pending collector changes — apply at the NEXT ruled boundary, not before

Anything here modifies `collect_pm.py`, which the live unit executes with
**`Restart=always`**. Applying a patch to the working tree does **not** deploy
it — and that is exactly the trap: the unit reloads from disk on any crash,
OOM or kill, so an "undeployed" edit sitting in the live path is **armed**,
not parked. That happened on 2026-08-31 (fixed at `27a7c33`): the running
process held `4d15d2dd` while disk held `8c3df881` for roughly an hour, and
any restart would have swapped the collector to code the era row does not
describe.

**The rule: `collect_pm.py` on disk must always hash to the bytes the running
process started from.** `--verify-health` enforces it and will refuse
otherwise.

## To apply, at a ruled boundary only

    git apply --check live/pm_research/pending/F3_connection_local_telemetry.patch
    git apply live/pm_research/pending/F3_connection_local_telemetry.patch
    python3 live/pm_research/collect_pm.py --selftest
    # then re-pin CAND_SHA/CAND_COMMIT in v41_boundary_preflight.py,
    # run v5_deploy_gates.py, and follow the deploy runbook.

## F3_connection_local_telemetry.patch

Codex V41-F3. The disconnect diagnostics were **coin-global**: overlapping
current/next market tasks meant a healthy sibling socket credited the failing
one (`conn_msgs=7` for a socket that received zero), `conn_lifetime_s` used a
task-scoped start, and `silence_before_close_s` used a `last_recv_ns` retained
across reconnects. `qmax` and `ever_paused` were task-scoped for the same
reason.

Now attempt-local, reset once per reconnect iteration, computed by one
`connection_diagnostics()` helper that **production and the tests both call** —
because the first fix's test built its own dict and a mutant restoring the
broken expression still passed it. Verified: three mutants (coin-global leak,
task-scoped lifetime, silence falling back to 0) each **kill** the suite.

It also fixes **`consec_fail`**, which drives the reconnect **backoff** and
read the same coin-global counter: a healthy sibling's traffic made a
never-delivering connection look like one that "worked, then died", resetting
the backoff to 1. **The coin most likely to have a sibling — the busiest one —
was the least likely to back off correctly.** Same defect as `conn_msgs`, but
in a field that decides retry PACING rather than only diagnosis.

The patch is generated with `git diff` (both headers repo-relative) and
`git apply --check` is verified before committing it. An earlier version was
produced with `diff -u` against a scratch copy, so its target header pointed
into `/tmp` and the documented command could not apply it.

**Until this ships, `conn_msgs`, `conn_lifetime_s` and `silence_before_close_s`
in the gap ledger are UNTRUSTED for causal attribution.**
