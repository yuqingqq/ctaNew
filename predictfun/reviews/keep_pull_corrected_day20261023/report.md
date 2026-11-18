# Corrected one-day test, Oct 23 — 2026-11-18

USER: **"Ok, go run one day test right?"** (the go for `run_keep_pull_scenarios.sh day 20261023` after the check
of 9a114cc1).

- **Code:** 0e20fa8d (the USER's settlement-lag fix plus the check's K6-basis and rehearsal fixes).
- **Run:** 12:54:53–12:59:49Z, slot claude. Every step rc 0; peak 1.1 GiB.
- **Data:** consumed days only — Oct 23, and Nov 8 for the known ledger.

| Step | Wall | Output |
|---|---:|---|
| scenarios selftest (incl. the upper-only production rehearsal) | 20 s | pass |
| Oct 23 settlement times | — | reused (exists) |
| known ledger v5 (the module's pipeline from Nov 8's tables) | 5 s | `keep_pull_exec_windows_known_v5_20261118` |
| ledger v5 on Oct 23 | 37 s | `keep_pull_exec_windows_v5_20261118_20261023` |
| scenario replay v4, Oct 23, and its report | 231 s + 4 s | `keep_pull_scenarios_v4_20261118` |

## Reproduction checks (declared before the outputs were read: [compare.py](compare.py))

| Predicate | Result |
|---|---|
| Known v5 reproduces the readings recomputed from the saved v4 tables | identical |
| Ledger v5: every non-ST part equals v4 (42 parts) | identical; only ST parts changed; V0 CONTROLS YES |
| Scenarios: every scenario except the ST timeline equals v3; STlo absent | 18 identical; STlo dropped; SThi changed |

## Results under the corrected timing

ST is now upper-only: a mapped leg's match lies at or before its settlement-id creation time. That holds only
under creation at/after the match and aligned clocks, and no lower bound is identified.

**Known orders (Nov 8, 99 legs).**
- `K6 GATE S3` at K7's offsets: FAIL (3 known fills called avoided).
- At the UUID-derived shared clock, a conditional reading: PASS.
- ST: UNIDENTIFIED-UPPER-BOUND-ONLY.
- Everything else as in known v4.

**Ledger (Oct 23).**

| ST under the cancel window CP | v4 (259 ms lower end, withdrawn) | v5 (upper-only) |
|---|---:|---:|
| removed gross that is CANCEL_FIRST | 55.8% (6,072 fills) | 1.5% (214 fills) |
| UNKNOWN / AMBIGUOUS share of removed gross | 0% / 42.6% | 71.5% / 25.4% |
| fills contradictory on every candidate | 12 | 5 |
| exact joint bounds on the score's excess at 37 ms (c) | [-81,591, +105,739] | [-145,038, +168,157] |
| replay trade times at or before the creation time | — | 89.9% (10.1% after it; p95 +352 ms) |

- **Where ST's 214 CANCEL_FIRST fills come from.** All lie on events with a finite lower end from the S3 fallback
  of legs without a unique settlement id ([st_cancel_first.txt](st_cancel_first.txt), computed). No mapped leg can
  certify one.
- **Unknown lower ends.** 38,581 of the day's 50,553 events have an unknown (-inf) ST lower end.
- **Sign.** V1 SIGN is SIGN-NOT-IDENTIFIED under every scenario (S1, S2, S3, ST, EV).
- **Unchanged.** S1 / S2 / S3 / EV are identical to v4: CANCEL_FIRST 66.1% / 52.5% / 33.2% / 0% of the removed
  gross.

**Scenario replay (19 scenarios, local END +5 s values in cents; E is the score's advantage over the matched
STATES null, not earnings).**
- SThi (every trade at its settlement id's creation time, the latest admissible instant):
  - E +40,687, against +40,692 under v3.
  - Held contradictory fills: 3 (v3: 14).
  - KEEP -353,878; KEEP + score -135,043.
- No earliest-admissible counterpart exists (STlo dropped).
- Across the 19 scenarios E spans [-13,342, +40,687] (M250 .. SThi), and Z1 / Z2 / Z3 read NOT-STABLE-IN-TESTED-SCENARIOS.
- TIMING EVIDENCE reads LOWER-MATCH-BOUND-UNIDENTIFIED.

## Reading

- **The corrected timing changes the evidence, not the economics of the other scenarios.**
- **ST no longer certifies the score's removals.** Under the withdrawn 259 ms bound it certified 55.8% of the
  removed gross. Upper-only, it certifies 1.5%, and only through legs without a settlement id.
- **The sign is still not identified.** The ST bounds on the score's excess widened to [-145k, +168k] c at 37 ms.
- **The replay's own trade times sit after the latest admissible match on 10.1% of legs** (under the premises).
- **What remains open.**
  - An independent creation-delay or clock bound: none exists, so no lower ST scenario can be stated.
  - Oct 24–27, not run.
  - 14.9% ambiguous settlement mapping.
  - Own-order evidence: one partial day of filled orders.
  - Latency ranges chosen after seeing that evidence.
