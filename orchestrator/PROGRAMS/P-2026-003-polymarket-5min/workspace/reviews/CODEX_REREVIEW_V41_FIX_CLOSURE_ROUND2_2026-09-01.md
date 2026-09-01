# Codex re-review — v4_1 fix closure round 2 — 2026-09-01

**Exact review tip:** `fc58d4db906efe53a4815a8230c9bf4b3785880d`

**Scope:** re-review of V41-RR1, V41-RR2, and V41-RR3 from
`CODEX_REREVIEW_V41_FIX_CLOSURE_2026-09-01.md`. Production was inspected
read-only. No collector, service, production ledger, or tape was modified.

## Verdict

### HOLD NOT RELEASED — RR2 closed; RR1 and RR3 remain open

The same-day late, cross-midnight late, and early reconstruction emitter now
uses the observed start and produces a transition/rollback pair that the day
consumer reads as impure. Inline and sidecar provenance now share semantic
Git/blob verification. The pending F3 patch also has a useful real `_market()`
probe.

The two remaining failures are again path failures: the claimed RR1
“entry-point” test does not call the CLI entry point, and the RR3 “sibling” is
preloaded rather than active during the attempt. Executing the real path and
the exact old global-delta mutation reproduces both residuals under the green
161-check suite.

## V41-RR1 — HIGH — emitter fixed, but early/abort operator paths remain false

Three distinct residuals remain.

1. `--post-recovery` scans target starts from `ep`, while discovery scans from
   `ep - EARLY_SCAN_LOOKBACK_S`. Discovery therefore returns an early PID, but
   the emitting CLI cannot retrieve that PID's start and reaches
   `target_start=None`. Executed T-60s fixture: discovery returned PID `777`,
   `recovery_kind=early`; the exact target scan used by `main()` returned an
   empty list.
2. Discovery prefers `in_window` over `early`. Executed starts at T-60s and
   T+5s: it selected the T+5 PID and kind `in_window`, erasing the real
   arm-window span that began first. Selection must follow the earliest actual
   target start, with kind derived from that start.
3. `make_abort_row()` still accepts real early and late starts and emits
   `aborted:true`, even though the updated runbook says abort is only for “never
   ran” and says the command refuses any target start. Executed early-only and
   late-only fixtures both emitted abort rows. Both consumers skip those rows,
   so a direct or mistaken abort still erases the span.

The runbook is also internally contradictory: its new kind table says all
three shapes recover, while the immediately preceding explanation still says
recovery accepts only `[T,T+120s]`, labels `later_starts` non-candidates, and
the failure table says the gate refuses an outside-window PID.

The new selftest section labelled “ENTRY POINT” calls
`recovery_pid_candidates()`, `make_recovery_bundle()`, and
`day_era_admission()` directly. It never calls `main()` and therefore cannot
see the scan-start mismatch above.

Required closure:

- Make `--post-recovery` search the same early-inclusive population as
  discovery and bind the selected PID **and recv_ns**, not merely the PID.
- Choose the earliest actual target start across early/in-window/late; test
  early+in-window together.
- Restore a fail-closed abort invariant: any observed v4.1 start refuses abort.
- Remove the stale contradictory runbook text.
- Add a real entry-point fixture through `main()` for early-only,
  early+in-window, same-day late, cross-midnight late, and no-start abort.

## V41-RR2 — CLOSED

`_verify_code_identity()` is now shared by inline and sidecar paths, resolves
the named commit, hashes `live/pm_research/collect_pm.py`, and compares the
digest. The exact all-zero inline known-bad now returns `IDENTITY_INVALID`; the
actual historical sidecar still returns `SUPERSEDED` with commit
`2b1ea0d...` and SHA `4d15d2dd...`.

## V41-RR3 — HIGH diagnostic/reconnect correctness — the exact old sibling leak still passes the new real-loop probe

The probe preloads `msg_by_coin['btc']=7` before `_market()` starts, but it
does not increment the sibling counter while an attempt is running.
`attempt_msgs` snapshots the same `7`; therefore the exact old expression
`msg_by_coin - attempt_msgs` sees only the target connection's messages:

- target delivers 3: global delta is 3, so the probe's expected 3 passes;
- target delivers 0: global delta is 0, so the expected 0 and escalating
  backoff both pass.

Executed mutation in an isolated patched clone: production diagnostics and
backoff were rewired to the exact old coin-global delta/condition. Collector
selftest exited `0`, and all three `V41-RR3 WIRING` checks printed `PASS`.
Thus the live concurrency defect is still not represented by the fixture.

Required closure: make a sibling increment `msg_by_coin['btc']` **after the
attempt snapshot and before the target disconnects**. Test both a silent target
plus active sibling and a target delivering 3 plus active sibling. The exact
old delta must report the sibling traffic and kill the suite; the attempt-local
implementation must continue to report 0 and 3 respectively. Exercise the
backoff decision in the silent-target/active-sibling case as well.

## Executed evidence

| check | result |
|---|---|
| HEAD vs origin before filing | exact at `fc58d4d...` |
| integrated gates | **14/14 pass** |
| v4_1 seam selftest | **161 checks pass** |
| early-only discovery | PID `777`, kind `early` |
| actual CLI target population for PID `777` | **empty** |
| early + in-window discovery | incorrectly selects in-window PID |
| abort with an early real start | **accepted** |
| abort with a late real start | **accepted** |
| all-zero inline provenance | correctly `IDENTITY_INVALID` |
| exact old coin-global delta mutant | **selftest 0; all wiring checks pass** |
| pending patch `git apply --check` | pass |
| collector tree/deployed/pinned SHA | exact `4d15d2dd...` |
| live unit | active/running, PID `1108125`, `NRestarts=0` |
| live health | fixed PID; all seven coins advance |
| `git diff --check` | pass |

## Disposition

The healthy deployed v4.1 collector remains untouched and should not be
restarted. RR2 is released. RR1 and RR3 remain held for the operator/consumer
and concurrent-sibling paths above.

