# Check of the settlement-lag fix 9a114cc1 — 2026-11-18

USER: **"Check latest commit, I have fixed the issues"**. Checked: review 52c45d57 (P1, K8's common-offset fit
read as a per-event lag bound) and the USER's fix 9a114cc1. Session aea5bfb9; every job through
`run_light.sh` with PF_SLOT=claude. No tape was read. Nothing under `data/` was written. The corrected
readings below come from the saved known v4 case tables.

**The review's P1 is fixed.** K8 now separates one shared clock offset from a nonnegative creation lag per leg.
No universal lag bound remains, ST is upper-only and STlo is gone. Two gaps remained and are now fixed in code:
(1) the K6 gate silently changed its offset basis, so S3's FAIL became a PASS only under an unestablished clock
premise; (2) the default (upper-only) production path was never rehearsed by the module selftests. My own stale
claims from 084d5a19 are now superseded in place (rule 13).

## The fix, verified

Model (bot clock; match m, placement p, last possible match u, first notice carrying the UUID n; UUID time t):
`t = m + delta + lag`, `lag >= 0`, `p <= m <= u`, `t <= n + delta`. The bounds the code emits follow from these
constraints:

| Quantity | Code (pf_keep_pull_exec_windows.py `settlement_checks`) | Follows from |
|---|---|---|
| delta | [max(K7.lo, max(t - n)), min(K7.hi, min(t - p))] | creation before its notice; m >= p and lag >= 0 |
| match | [p, min(u, t - delta_lo)] | lag unbounded above, so no lower bound beyond p |
| lag | [max(0, t - delta_hi - u), t - delta_lo - p] | the same constraints |

- The review's 10/700 ms world is accepted with valid intervals (module selftest `_settlement_selftest`).
- K5S no longer infers slow cancels: the match's lower bound is the placement.
- At 9a114cc1 all three selftests pass, and so do the full mutation suites: settlement 10/10, exec-windows
  104/104, scenarios 31/31 ([logs](checks_9a114cc1/)). The fix report had rerun only 10 targeted mutants.
- Known v4's `source_sha256` (951354dd…) equals the HEAD source.
- On the 99 legs, no leg's creation-lag lower bound exceeds 0. The upper bounds run from 180.5 ms (min) through
  318.2 ms (median) to 5,895.3 ms (max): allowed limits, not delays.

## Gap 1 — the K6 gate moved to the UUID-derived clock under the K7 key

9a114cc1 evaluated `K6 GATE {k}` at K8's shared clock (`off_eval = k8['clock_offset']`). Known v3 had declared
this gate at K7's offsets. As a result the receipt reads `K6 GATE S3: PASS`, under the key that read FAIL in v3,
and no reading carries the K7 result any more.

Known fills called CANCEL_FIRST at some admissible offset. Saved known v4 tables (consumed Nov 8, 99 legs; v4 as of
2026-11-18T11:43:28.451761Z), through the module's own `window_checks`:

| Offset basis | Interval (ms) | S1 | S2 | S3 | EV | M50 |
|---|---|---:|---:|---:|---:|---:|
| K7 (the declared gate) | [-226.4, +49.3] | 9 | 5 | 3 | 0 | 65 |
| K8 shared clock (conditional) | [-147.8, +32.6] | 5 | 5 | 0 | 0 | 55 |
| K7 lower end .. K8 lower end only | [-226.4, -147.8] | 9 | 5 | 3 | 0 | 65 |
| Zero-lag settlement premise (conditional) | [+14.5, +32.6] | 4 | 5 | 0 | 0 | 5 |

- **Where it comes from.** The some-offset test runs at the lowest offset. The whole difference comes from the
  lower end, `max(t_set - first notice) = -147.8 ms`.
- **The three S3 legs.** All are REST-range windows (E2 unsupported). Each is called avoided only below one
  offset: -210.3, -218.8 or -224.8 ms.
- **The premise S3's PASS needs.** The UUID bound reaches these windows only if the UUID clock is the REST status
  clock. The review states that premise is not established; K7's own premises do not need it. S3's PASS is that
  premise, not a repair.

Fix:
- `K6 GATE {k}` is back at K7's offsets, as declared.
- `K6 GATE {k} (K8 shared clock)` (per-leg columns `_clk`) and `K6 GATE {k} (settlement offsets)` (`_set`) are
  separate, conditional readings.
- KNOWN VERDICT carries all three; LIMITS and the column docs say which basis is which.
- Known ledger v5 (`OUT_K`); the driver uses the new path. v4 stays as provenance.
- Falsifier in `_k6_selftest`: a planted notice 20 ms after its UUID lifts the shared clock's lower end to -20 ms
  (K7: -100), and a planted S3 window is called avoided only below -30 ms. The checks: FAIL at K7, PASS at the K8
  clock and at the settlement offsets, each under its own key.
- Three mutants: gate moved to the K8 clock, K8 reading never evaluated, K8 reading taken from the K7 gate. All
  three are killed by this check, not by a crash.

## Gap 2 — the production default was not rehearsed

The module selftests ran `run()` (ledger) and `day_run()` (scenarios) only with an explicit `st_lag=150`. The
driver's `day` phase runs only the scenario selftest before reading data. So the default path had never run end
to end inside the gate: the known receipt read, ST lower ends -inf, 19 scenarios, no STlo. The ledger side had
been rehearsed once, in 9a114cc1's `verify.py`; the scenario day never.

Both selftests now rehearse it. The known receipt is a temporary file in the selftest's own directory.

| Module | Checks |
|---|---|
| Ledger | ST lower end -inf on mapped legs, S3 fallback on unmapped legs, `unknown` lower-bound types, ST bounds and sign reading present, role UNIDENTIFIED |
| Scenarios | lag None and role UNIDENTIFIED in the receipt, the 19 KSCEN names, no STlo, SThi simulated, the receipt's hash recorded |

The scenario selftest's direct `timelines(..., None)` call is now inside try/except, so a refusal there is
reported by the check instead of crashing the selftest.

Mutants:
- Ledger: the upper-only run labelled an explicit lag assumption. Only the rehearsal can catch it.
- Ledger: the corrected known receipt yields a transferable lag (caught by the existing receipt-reader check).
- Scenarios: the production day ignores the corrected known receipt.
- Scenarios: an upper-only window replayed as STlo.

## In-band corrections (my rows from 084d5a19; the review asked for them, and 9a114cc1 added only a correction section at the top)

| Place | Change |
|---|---|
| FINDINGS C23 (checklist) | rewritten: a UUID time is a creation time, an upper bound on the match; no per-event bound from a common fit; no transfer of a sample maximum |
| FINDINGS P60 / P61 | P60's "constant-offset test" advice replaced; P61 added (a feasible common fit is not a per-event bound; keep a gate's declared basis) |
| FINDINGS rows | settlement row CORRECTED; known-order row CORRECTED (K7 vs K8 gate counts above); scenario row and the ledger's ST figures SUPERSEDED (STlo, 259 ms, assumed cancel mixture) |
| STATUS `clear_issues_settlement_20261023` | status `superseded_timing_not_cleared`, `superseded_by`, `withdrawn` list; old values kept as provenance |
| HANDOFF 11:16Z entry | SUPERSEDED line under its heading |

## Verification after the fix

| Check | Result |
|---|---|
| Exec-windows selftest | pass ([selftest_after.log](checks_after/xw_selftest.log)) |
| Scenarios selftest | pass ([selftest_after.log](checks_after/sc_selftest.log)) |
| Exec-windows mutation suite | 109/109 (new: 105-109) ([log](checks_after/xw_mutants.log)) |
| Scenarios mutation suite | 33/33 (new: 32-33) ([log](checks_after/sc_mutants.log)) |
| Corrected known readings from the saved v4 tables | [known_readings_corrected.txt](known_readings_corrected.txt): K8 fields identical to v4; only the K6 gate keys and KNOWN VERDICT change |

## What remains

- **The score's sign is not identified.** The corrected Oct 23 ledger (v5) and scenario replay (v4) have not run.
  Next, on the USER's go: `run_keep_pull_scenarios.sh day 20261023`. It first rebuilds known v5 from Nov 8's
  tables (consumed), then runs ledger v5 and scenarios v4 under upper-only ST. Oct 24–27 follow.
- **K8 readings stay conditional** on creation at/after the match and aligned UUID / status / book clocks (the
  shared clock interval, the notice 87 / 99).
- **Narrower timing needs independent evidence** of the creation delay or the clock offset, not a fit (9a114cc1's
  next gate).
- Unchanged: 14.9% of Oct 23 legs have ambiguous settlement mapping; own-order evidence is one partial day of
  filled orders; the latency ranges were chosen after seeing that evidence.
