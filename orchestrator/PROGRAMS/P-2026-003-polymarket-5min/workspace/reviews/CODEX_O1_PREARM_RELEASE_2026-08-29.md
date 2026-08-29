# Codex O1 pre-arm release — 2026-08-29

**Reviewed closure:** `a980c63d86c4c7d47e339a5a86e4d7f9206cdde0`  
**Reviewing tip:** `8b79b32` (the immediately preceding maintained-hold
filing; no collector or checker change after `a980c63`)  
**Target boundary:** `2026-08-30T00:00:00Z`

## Decision

**HOLD RELEASED — CLEAR TO ARM THE ONE O1 DEPLOYMENT AT THE RULED
2026-08-30T00:00:00Z BOUNDARY.**

The release is only for the exact restart-first, verifier-emitted-stamp-last
sequence in `live/pm_research/plans/O1_DEPLOY_RUNBOOK_2026-08-30.md`. Any
preflight or postflight refusal still postpones/aborts under that runbook; this
filing does not convert a future refusal into permission to improvise.

## Final executed evidence

Executed at 2026-08-29T23:01:29Z:

| Requirement | Evidence | Verdict |
|---|---|---|
| ruled boundary | source constant and installed wake both name `2026-08-30T00:00:00Z`; derived epoch control passes | **PASS** |
| old operating instruction removed | old `co-prep-wake` units absent | **PASS** |
| installed operating instruction correct | `co-prep-wake30` installed for 23:55:00/05Z; payload names the Aug-30 runbook, selftest/pre-arm, restart first, postflight, emitted stamp last | **PASS** |
| full preflight instrument | **21/21** controls pass | **PASS** |
| exact event identity | heartbeat-with-note and missing-event rows refuse | **PASS** |
| exact timestamp identity | finite float and bool refuse; exact integer positive row passes and remains integer in the stamp | **PASS** |
| boundary time in checker | pre-boundary declaration refuses independently of observer filtering | **PASS** |
| live pre-arm | v3_1 hold intact, reviewed v4 in HEAD, unit active PID 1048, no conflicting era row, 3511 seconds to boundary | **PASS** |
| live held bytes | `c0a52d3337022db3ad6686ae95a242b0f4800d067c919c6aadf74d1735d62203` | **PASS** |
| committed/pinned v4 bytes | HEAD and `6786a02` both `5b718a15501549c5c39c1a11d7dc9f8c22f755eef64ffc866d0a285831953409` | **PASS** |
| current era ledger | absent; therefore no conflicting live v4 row | **PASS** |
| O1 behavioral package | 10/10 at the unchanged collector bytes | **PASS** |
| real producer -> DAY_BAR_V2 seam | 7/7 at the unchanged collector bytes | **PASS** |

The final narrow defect at `9ac0bd1` is closed: `recv_ns` now requires
`type(value) is int` before the boundary comparison. The previously accepted
`1.788048005e+18` float and a boolean both refuse by name, while the exact
integer positive control is accepted.

## Operating bounds that remain mandatory

1. Re-run `--selftest` and real `--pre-arm` at the scheduled prep. Any refusal
   means postpone and record it.
2. At the boundary, restore the reviewed v4 bytes and restart the collector
   before asking postflight to identify the new process.
3. Append only the JSON emitted by successful `--post-restart OLD_PID`; verify
   the appended boundary, new PID, collector version, and ordering fields.
4. Follow the runbook's visible abort/supersession path on any restart,
   declaration, or stamp failure.
5. Do not touch the prices collector.
6. Treat 2026-08-29 as entirely v3_1 and **not** a post-O1 forward day. The
   earliest possible complete post-O1 day is 2026-08-30, conditional on this
   deployment succeeding and the close-time day gate passing.

This release changes no model, threshold, strategy arm, multiplicity, or
research conclusion. `QR_CANCEL_HOLD_X_SKEW` remains the queue-realistic
baseline and `QR_SKEW_ONLY` remains the required comparator. The five-day
forward-validation requirement remains untouched.
