# Codex O1 05:30 pre-arm re-review — 2026-08-30

**Reviewed closure:** `2150ebab99007d01738e0a406e2604e3ae73b481`  
**Prior amended tips:** `877b1cb` (02:30 ruling), `50fd3f9` (05:30 ruling)  
**Target boundary:** `2026-08-30T05:30:00Z` (`1788067800`)

## Decision

**HOLD RELEASED — CLEAR TO ARM THE ONE O1 DEPLOYMENT AT THE RULED
2026-08-30T05:30:00Z BOUNDARY.**

This release supersedes the stale 02:30 review request because the repository's
authoritative R-334 ruling postponed the unreviewed instant to 05:30. It is only
for the exact restart-first, verifier-emitted-stamp-last sequence in
`live/pm_research/plans/O1_DEPLOY_RUNBOOK_2026-08-30.md`. Any preflight or
postflight refusal still postpones or aborts the deployment; it does not permit
a late or improvised transition.

## Finding and closure

The first pass at `50fd3f9` maintained the hold. Its amendment banner named
05:30 and mixed-era 08-30, but the operating body still named the old 00:00
prep/deploy/verification times, emitted 00:00 in the abort row, and called
08-30 day one. That was a real single-authority contradiction.

`2150eba` closes it at the artifact an operator executes:

- prep is 05:25, transition is 05:30, and verification is 05:31–05:35;
- successful stamp verification and the visible abort row both bind
  `2026-08-30T05:30:00Z`;
- 08-29 and mixed-era 08-30 are explicitly never post-O1 days; 08-31 is day
  one;
- `check_runbook_consistency()` is consumed by real `--pre-arm`, checks the
  live runbook, and has positive plus four known-bad controls. The exact
  banner-right/body-stale abort-row shape now refuses.

## Executed evidence

Executed at the exact reviewed tip on 2026-08-30T05:09Z:

| Requirement | Evidence | Verdict |
|---|---|---|
| authoritative instant | source and runbook bind `05:30:00Z`; epoch derives to `1788067800` | **PASS** |
| superseded identities | direct calls with matching 00:00 and 02:30 UTC/epoch pairs both refuse as stale | **PASS** |
| runbook body | no operative old deploy, stamp, day-one, or prep strings remain; live-file consistency check accepts | **PASS** |
| preflight instrument | **26/26** controls pass, including four runbook-consistency controls | **PASS** |
| live pre-arm | **PASS**: held v3_1, reviewed v4 in HEAD, active PID 1048, no conflicting era row, 1,224 seconds to boundary | **PASS** |
| no missed-instant transition | PID 1048 has run since 2026-08-26T04:38:13Z; no post-midnight `collector_start`; era ledger absent | **PASS** |
| live held bytes | `c0a52d3337022db3ad6686ae95a242b0f4800d067c919c6aadf74d1735d62203` | **PASS** |
| committed v4 bytes | HEAD and `6786a02` both `5b718a15501549c5c39c1a11d7dc9f8c22f755eef64ffc866d0a285831953409` | **PASS** |
| abort restore bytes | `6786a02^` is the exact held v3_1 SHA | **PASS** |
| O1 behavioral package | **10/10** at `O1_REF=6786a02` | **PASS** |
| real producer -> DAY_BAR_V2 seam | **7/7** at `O1_REF=6786a02` | **PASS** |
| installed fallback delivery | one transient service runs one script at 05:29 with `AccuracySec=1s`; that script sends the 05:30 instruction, waits two seconds, then sends Enter in program order | **PASS** |

The installed fallback is no longer the failed two-independent-timer shape.
The coordinator is also already active with the release monitor, so the timer
is a re-invocation backstop rather than the only delivery path.

## Mandatory operating bounds

1. Re-run `--selftest` and real `--pre-arm` at the 05:25 prep. Any refusal
   postpones the transition.
2. At 05:30:00Z, restore reviewed v4 and restart before asking postflight to
   identify the new process.
3. Append only JSON emitted by successful `--post-restart OLD_PID`, then
   re-read and verify its boundary, new PID, version, integer collector-start
   timestamp, and restart-first/stamp-last ordering.
4. Follow the visible abort/supersession path on any checkout, restart,
   declaration, or stamp failure. Do not improvise late.
5. Do not touch the prices collector.
6. **Neither 2026-08-29 nor 2026-08-30 is admissible as a post-O1 forward
   day under any generic eligibility field.** The earliest complete post-O1
   day is 2026-08-31, subject to the ordinary close-time day gate.

This operational release changes no model, threshold, strategy arm,
multiplicity, or research conclusion. `QR_CANCEL_HOLD_X_SKEW` remains the
queue-realistic baseline and `QR_SKEW_ONLY` remains the required comparator;
the independent five-complete-day forward requirement remains unchanged.
