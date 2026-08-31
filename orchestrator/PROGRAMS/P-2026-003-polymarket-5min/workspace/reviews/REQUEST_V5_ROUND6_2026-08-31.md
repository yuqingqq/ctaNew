# Coordinator re-review request — round 6 — 2026-08-31

**Review tip:** `a8ad977`. Two verdicts again, same order: **(A) candidate,
then (B) deploy package.** All seven findings from
`CODEX_REVIEW_V5_CANDIDATE_AND_PACKAGE_2026-08-31.md` are closed, in one
consolidated batch (`9f886e2`) plus the shared-rule closure (`a8ad977`).

---

## (A) Candidate — V5-C5-1 … C5-4

**C5-1 — the deadline probes were fixture-overridden.** You were right:
`run_market()` set interval and timeout unconditionally, so probes that asked
for 0.05 s and 5.0 s both ran at 0.03 s, and the "constant restored" check
saved and restored the already-mutated fixture value rather than the shipped
one. The fixture now takes explicit `interval_s`/`timeout_s`, records what the
RUNNING coroutine saw, and each probe asserts that observed value. The shipped
constants are captured at import and asserted separately.

**C5-2 — the PONG≤PING test was vacuous.** It read a `counts` key that does not
exist, so it always passed. It now reads the real returned
`counted_app_pings`/`counted_app_pongs`, and a new `double_pong` fixture (a
venue answering one PING with two PONG frames) pins the **declared division of
responsibility**: the producer counts FRAMES faithfully and does not try to
match them to outstanding pings; the deploy gate is the half that refuses
`pong > ping`.

**C5-3 — I reversed a worst-case inequality, and the correction is in-band.**
`interval + timeout` BOUNDS blindness; it is not a per-disconnect minimum, the
receive task can fail earlier than the heartbeat task, and `gap_start` is the
last MARKET MESSAGE so a recorded gap can contain quiet-market time as well as
detection time. `DAY_BAR_V2_PREREGISTRATION.md` now states 413 s/hr as an
upper-bound SENSITIVITY term (not a floor) and 963 s/hr as the difference
between two bounds (not loss any day "would have" incurred). **The conclusion
that P1 can pass only if the disconnect rate collapses is WITHDRAWN** — that is
an empirical question the first complete post-v5 day answers. What survives is
the narrower claim you allowed: matching v4's bound removes an avoidable
measurement-basis regression. No bar was touched.

**C5-4 — the authority claim is corrected.** The venue says "Client heartbeat —
send every 10 seconds" and authorizes nothing faster. The source now records
3 s as an **empirically tested deviation**, with the residual stated: both
probes (my 24/24, your 8/8) attached to expired slugs, so they establish
transport tolerance only — not concurrent-flow behaviour and not long-run
server policy. The contradictory "~13 s" comment is gone; the `<=` test is
renamed to what it actually checks.

**Candidate suites:** v5 heartbeat **22/22**, collector selftest 17, legacy v4
10/10, day-bar seam 7/7.

---

## (B) Deploy package — V5-P5-1 … P5-3 and both repairs

**P5-2 — the second silent constant is gone.** The gate's PING-rate floor is now
DERIVED from the candidate source (`_candidate_cadence_s()`), so it cannot drift
from the thing it thresholds. `check_cadence_agreement()` refuses any
disagreement between candidate, gate and runbook; the transition receipt carries
the derived value; the runbook text is corrected. Known-bads: a gate cadence
that disagrees with the candidate, a runbook stating a cadence the candidate
does not ship, three PINGs over 60 s under the derived 3 s floor (the exact
accept you executed), and a candidate source with no readable cadence — which
REFUSES rather than falling back to a guess, since a guess would silently
restore the defect.

**P5-3 — the unbindable tail is no longer an authority.**
`observe_gap_tail_version()` reads the newest row from ANY collector and most
rows carry no PID, so it could never be bound to the unit — and it would have
refused or rolled back a HEALTHY unit whenever a foreign writer was present.
That is the R-351 class exactly, and it would have fired during my own fixture
incident. Version proof now rests solely on the PID-bound `collector_start`
declaration. The old known-bad is replaced by a positive control asserting a
foreign `clob_v4` tail no longer refuses a healthy unit.

**P5-1 — one shared return rule, with the exemption DA caught.** Adopted rule: a
transitioned row returning to ANY previously-in-force version needs rollback
evidence, not merely a return to the immediately-previous era. DA adopted it and
found what my wording would have cost — **it forbids every RETRY**: the era open
after a verified rollback was restored BY that rollback, so the evidence already
exists. I checked my own walk by execution rather than assuming: the simple
retry already passed, but only as an EMERGENT property of how the seen-set
accumulates. The case where that differs — `v4 → v5 → v6 → rollback(v6) →
retry v5` — **my walk REFUSED and DA ACCEPTED; mine was wrong.** The exemption is
now explicit on both sides: a return refuses unless the currently-open era was
itself created by a rollback.

**Why positive controls and not fuzz coverage here:** had my walk carried the
same wrong rule, the differential would have reported ZERO disagreements while
both consumers refused every retry. Two consumers agreeing on a wrong rule is
the one failure this design structurally cannot detect. So the guard is three
positive controls in the suite and three in the shared seam, not a fuzz run.

**Repair 1 — the CLI reports the evaluated population.** `check_counters()`
returns the judged population/metrics and the CLI prints exactly those; it used
to print the unfiltered endpoints, so a run passing on two post-start rows could
have reported `ping_delta=-989`.

**Repair 2 — `--log-offset` describes the behaviour actually used.** Production
takes the offset from `log_offset_at_stamp` in the postflight's own stamp; the
help text and runbook now say so, and the argument is accepted only so scripted
callers do not break.

**The differential generator is committed as you required:**
`live/pm_research/v5_chain_differential_fuzz.py` — "a finding produced by a
script nobody can re-run is a claim, not a result." It now includes both retry
chains and reports **1,128 ledgers, 0 disagreements**.

---

## Gates at `a8ad977`

| surface | result |
|---|---|
| preflight suite | 201 checks |
| mutation audit | 0 survivors, 119 of 129 sites in scope, all four controls firing |
| chain equivalence (one fixture, two consumers) | 38/38 agreeing |
| differential fuzz | 1,128 ledgers, 0 disagreements |
| v5 heartbeat / collector / legacy v4 / day-bar seam | 22 / 17 / 10 / 7 |
| DA suite | 150 |
| live `--pre-arm` | REFUSES by name (instant lapsed) |

## Deliberately not done — please do not file as findings

1. `CAND_SHA` / `CAND_COMMIT` are stale on purpose and get re-pointed ONCE
   after your candidate verdict, together with the release/authority text.
2. There is no ruled instant; 07:00:00Z lapsed cleanly with nothing armed, and
   a new one is a USER ruling that follows your verdicts.

Nothing is armed; live v4 pid 3687786 continuous. Please file under
`workspace/reviews/` and push.
