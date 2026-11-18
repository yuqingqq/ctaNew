# Manual fix review — 2026-11-18

Pinned target: **084d5a1921460a9a74ba47bef8bb41ff38921676**.
Reviewed range: **64290a228aedef845093ce2b2217b0bbc07c36cb..084d5a1921460a9a74ba47bef8bb41ff38921676**,
including 1a9258bb, 291b8f40, 2d025c35 and 084d5a19. Source was reviewed in
the detached checkout `/tmp/predictfun-review-084d5a19`.

**The previous two P2 findings are fixed, and all 20 saved scenario results
reproduce. One new P1 blocks the claim that timing is cleared: K8 converts
the feasibility of a common timestamp offset into a bound on every event's
creation delay. That inference fails under its stated nonnegative-lag premise.**
The settlement identifier is useful new evidence; it does not yet establish
the claimed match windows or unconditional measured action delays.

## P1 — K8's common-offset fit does not bound event-dependent creation lag

Pinned source evidence:

- `predictfun/research/pf_keep_pull_exec_windows.py:1088–1109` computes
  `d_lo = max(t_set - upper)` and `d_hi = min(t_set - place)`, declares the
  common fit consistent, then emits `lag_bound_ms = d_hi - K7.lo` and match
  bounds `[t_set - d_hi, t_set - d_lo]`.
- `:372–390` uses this as the ST window's maximum lag and imports it into
  the public-day ledger/replay. `:1205–1216` constructs the known-case ST
  windows from this same fit and reports the gate as PASS.
- `:1062–1074` propagates the fitted match bounds into activation and cancel
  bounds; `:1198–1204` reports the notice/cancel conclusions.
- `predictfun/HANDOFF.md:1–7`, `STATUS.yml:2–10` and `FINDINGS.md:2062,2206–2207`
  promote this to cleared timing, measured slow cancels and a match-time
  source. These claims need qualification/superseding receipts.

Let the venue-minus-bot clock offset be `delta` and each event's delay from
matching to UUID creation be `lag_i`. Even granting creation at/after matching,

```
t_set_i = match_i_bot + delta + lag_i,  lag_i >= 0.
```

A constant clock does not imply that `d_i = delta + lag_i` is constant. The
intersection computed by K8 says that **one constant-d explanation is
possible**. It neither identifies that explanation as true nor excludes
other explanations with different delays on different events. In particular,
the smallest per-event upper bound on `d_i` cannot bound every event's `d_i`.

The independent synthetic falsifier uses milliseconds within one second,
with the clock offset known exactly to be zero:

| Event | Placement | Last possible match | UUID creation | Actual match in compatible counterexample | Creation lag |
|---|---:|---:|---:|---:|---:|
| A | 0 | 40 | 30 | 20 | 10 |
| B | 0 | 900 | 800 | 100 | 700 |

Both matches obey their lifecycle, precede UUID creation, and are in the same
Matched second as their UUID. K8 returns **CONSISTENT**, common `d ∈ [-10,30]`,
and **maximum lag 30 ms**. Its event-B match window is **[770,810] ms**, which
excludes the admissible 100 ms match and 700 ms lag. A constant-lag positive
control (both lags 20 ms) fits the emitted windows; an incompatible-common-fit
control is rejected. Thus the issue is a demonstrated inference error, not
an assertion that the real venue actually has a 700 ms UUID creation delay.

The error reaches the action diagnostics. For event B, add a cancel request
at 200 ms and acknowledgment at 900 ms. K5S reports cancellation latency
**[570,700] ms**, contradicting its [0,100] range. The same observations also
permit a match at 100 ms and cancellation effect at 250 ms: an actual
**50 ms** cancellation. Likewise, the emitted notice-delay interval [190,230]
ms excludes a compatible 900 ms delay. These are synthetic controls, not
estimates of live latency. See [counterexample](receipt.json) and
[propagation evidence](effects.json).

There is also no independent semantic confirmation in the fitted ST gate.
On the same calibration rows, a feasible common fit guarantees
`t_set_i - d_hi <= upper_i`. At K7's lowest offset, the constructed ST lower
endpoint is exactly `t_set_i - d_hi`; for cancelled legs, `upper_i <= ack_i`.
Consequently this construction already prevents that lower endpoint from
certifying a known fill as cancelled first. Passing this in-sample check is
useful consistency, but cannot establish the lag premise. Nor can a bound
fit on 99 selected Nov 8 fills automatically become an evidence-backed
maximum for Oct 23 public trades.

The external specifications do not supply the missing premise. UUIDv7 stores
a millisecond timestamp; its format does not define the application's matching
event. [RFC 9562, §§5.7 and 6.1](https://www.rfc-editor.org/rfc/rfc9562.html#section-5.7).
Predict documents matching and settlement as separate lifecycle stages; the
consulted lifecycle page does not promise that settlement UUID creation is
simultaneous with matching. [Predict order lifecycle](https://dev.predict.fun/trading/order-lifecycle).
The observed agreement with the Matched **second** is useful but does not
identify a subsecond lag. Similarly, a small book-minus-UUID timestamp
difference alone mixes inter-event delay with clock offset; it does not
independently establish clock alignment.

The current **259.034 ms maximum lag**, **162–655 ms notice range**, and
**10/77 slow-cancel count** must therefore remain conditional on the stronger
timestamp premise. This review does not establish their real values are wrong;
it establishes the cited checks do not justify presenting them as measured
facts under only nonnegative creation lag. A constant-d/zero-lag scenario can
remain explicitly labelled as an assumption.

## What is fixed, and what the results establish

Compared with the [previous review](../20261118_1fd77004/report.md):

- **Resolved — K5's interval and sampling-unit error.** Base K5 checks both
  endpoints once per order and separates contradicted, compatible-unidentified
  and certified. On the saved consumed Nov 8 sample (99 legs, 86 orders,
  as of 2026-11-18T10:56:10.677848Z), it correctly retains one activation
  contradiction from the 21.165 ms acknowledgment upper bound. Among 77 orders
  with cancel confirmations, 65 are certified within [0,100] ms and 12 remain
  compatible-unidentified under its stated acknowledgment premises. The new
  settlement-derived tightening is the separate P1 above.
- **Resolved — contradiction attribution.** Event/fill status causes now
  reconcile. On consumed Oct 23 (17,670 baseline fill rows, ledger as of
  2026-11-18T10:56:46.998284Z), the 748 S1 all-candidate contradictions comprise
  708 leg-intersection, 37 evidence-cap and three mixed cases. ST's 12 are all
  leg-intersection cases. Those are classifications under the chosen windows,
  not 736 independently verified repairs of match times.
- **Implemented — prior next priority.** The retained known-case ledger traces
  all 99 legs, 93 with anchors and six without, through 131 candidate rows;
  identity, timestamp semantics and clock assumptions are recorded. The gate
  now tests the ledger's windows. Held contradictory-event denominators and
  economic weights are reported in the replay.
- **Useful new observation.** The saved settlement extraction (as of
  2026-11-18T10:54:17.185120Z) maps 682,499 of 802,157 Oct 23 legs; 119,658
  ambiguous-content legs remain explicit statuses. Its row-aligned output and
  source hashes match the receipts. This review did not reread/re-extract the
  raw REST pages.

All **20** scenario rows reproduce exactly from the saved order/fill tables.
Consumed Oct 23, report as of **2026-11-18T11:00:51.886418Z**: 30,466 counted
states, including 28,799 eligible, 1,289 post-only refusals, 354 missing endpoints
and 24 stale future books; the replay covers 30,088 states with paths. These
are summed local **END +5 s values in cents**, not settlement or portfolio P&L.
`E` is the score's advantage over the matched STATES null, not total earnings.

| Scenario | E | KEEP + score |
|---|---:|---:|
| M50 baseline, activation 58 / cancel 37 / notice 100 ms | +34,948 | -174,961 |
| SThi, trades at UUID creation time | +40,692 | -135,060 |
| STlo, trades 259.034 ms before UUID creation | -8,757 | -294,899 |
| M50 with assumed 13% slow-cancel mixture at 4 s | +30,475 | -258,508 |

These results support the accounting under the specified scenarios. They
still do not identify the score advantage's sign across timing assumptions,
and even SThi has negative combined local value. The evaluated score is the
existing **A+0 fair-edge rule**, not a newly trained predictor. The latency
variants were selected after seeing consumed own-order evidence and are
sensitivity scenarios, not an empirical latency distribution or untouched
validation. Neither these results nor passing self-tests justify model tuning
on the assumption that the execution timing is now validated.

## Verification and limitations

Read AGENTS, CLAUDE reliability rules, HANDOFF/STATUS/FINDINGS and the prior
review; inspected changed source, drivers, mutation definitions, receipts and
result labels. Independent checks ran only through the absolute `run_light.sh`
with **PF_SLOT=codex** and the pinned detached source:

- [verify.py](verify.py): three complete pinned synthetic self-tests; 35
  source/output/receipt hashes; saved contradiction reconciliation; all 20
  scenario rows and readings; K0–K8 readings from the saved known-case tables;
  corrupted scenario-output hash refusal; constant-lag positive control,
  variable-lag falsifier and incompatible-fit refusal. Passed in **27.24 s**,
  peak **715 MiB**. [Run](run.txt), [receipt](receipt.json),
  [self-tests](selftests.txt), [recomputed table](scenario_recompute.txt).
- [check_effects.py](check_effects.py): the falsifier's propagation into K5S
  cancellation and notice intervals, with two observation-compatible worlds
  and the calibration-gate identity. Passed in **0.94 s**, peak **196 MiB**.
  [Run](effects_run.txt), [evidence](effects.json).

Committed mutation logs report 102/102, 31/31 and 10/10 planted errors detected;
those suites were inspected, not rerun. No full tape replay, joint-bound solver
rerun, training, live action or untouched validation day was used. The current
review has not independently established real venue creation lag, clock
alignment, queue-rule validity or portfolio/repost/inventory behavior. The
existing old-anchor mismatches and S1/S2/S3 failures remain in the readings;
they are not resolved by an assumption-dependent ST fit.

## One next priority

**Repair the K8 timestamp model and its downstream evidence gate before more
scenario days or model changes.** Separate a shared clock offset from each
event's creation lag; retain the UUID observations, but label constant-d or
zero-lag results as conditional. Carry the admissible per-event match ranges
through ST, notice and order-path checks without inventing a finite universal
lag bound. Add the variable-lag falsifier above, and supersede the receipts
and “timing cleared” claims. Promote narrower bounds only when independent
event semantics or identity-linked evidence justifies them; an in-sample fit
and a selected-sample maximum are not that gate. Re-evaluate the existing
consumed known cases first. This is an evidence/label correction, not a request
to discard the new timestamps or to retune the predictor.

Review only: implementation and other-session edits preserved. No scheduler
was enabled or acknowledged; the user removed it.
