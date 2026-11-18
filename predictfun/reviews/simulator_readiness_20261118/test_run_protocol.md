# Test-run protocol — P-2026-005 keep/pull simulator (DRAFT, finalized by the readiness loop)

This is what a test run of the keep/pull simulator must produce, and what its results may and may not claim. It applies to every run
through `run_keep_pull_scenarios.sh` (the ledger, the known-order ledger and the queue-aware scenario replay). Check every item against
FINDINGS.md section 1 (C1-C24) before quoting a number.

## 1. What the simulator is

- **Re-simulates.** It replays the declared keep/pull rule (A+0, the fair-edge rule) through a validated queue / fill kernel, on
  consumed days, under declared execution scenarios.
- **Scenarios are assumption ranges, not measurements:** timelines S1 / S2 / S3 / ST / EV, trade-clock shifts M0 / M250, activation,
  cancel latency, notice delay.
- **Values are local END + 5 s marks** in cents. They are not settlement P&L, and not a self-financing portfolio.

## 2. Gates every run must pass, or its numbers are not quoted

| Gate | Source | Must read |
|---|---|---|
| Synthetic selftests of every module | the driver's first step | pass |
| Mutation suites (every mutant caught by a check) | `*_mutants.py` logs in the receipt | n / n |
| Z0 controls (the baseline reproduces the committed replay; KEEP = the sum of its fills) | scenario report | YES |
| V0 controls (the ledger's collapsed E / score = the committed per-day values) | ledger | YES |
| Known-order gate of record: `K6 GATE S3 (per clock domain)` on the known legs | known ledger v7+ | PASS for any S3-based claim |
| Clock census N0 (reproduces the case audit) | census v4+ | YES |

## 3. What a run must report (FINDINGS checklist)

- **Age (C1).** Every value is split by the order's age at the fill, and the headline is also shown without fills on orders < 100 ms old.
- **Fill clocks (C2).** Report M0 / M50 / M250, and state whether the claim holds at M250.
- **Cancel-pending share (C3)** of fills and their value.
- **Populations and denominators (C19).**
- **Exclusions as statuses** with their counts.
- **Days (C8).** Name the days used; consumed days only.
- **Strategy (C11).** Name the strategy: lane A+0 is not the frozen pm_ladder v3.

## 4. What a run MAY claim

- **Scenario-conditional values**, for example "under the S3 windows and a 37 ms cancel, E = ... c, without the < 100 ms fills ... c".
  Stability across the tested scenarios is reported (Z1-Z3, Z1-NY / Z3-NY): "stable in tested scenarios" is the strongest claim.
- **Instrument facts:** the gates above.
- **On the known legs (one consumed partial day):** clock-domain bounds, the gate of record per scenario, and match-before-request
  evidence (K9), each with its stated premises.

## 5. What a run may NOT claim

- **Profitability or P&L.** Values are local marks of a replay whose fills are not validated at the order level (C10, C4: the replay
  credits 1.5-2x live's shares; young fills were 8-12% real).
- **That a pull avoids a fill.** The match-versus-cancel order of public fills is not identified. Every certified CANCEL_FIRST rests on
  an assumed lower match bound (S3's WS_DELAY, REST_LAG, S1 / S2 ranges); a venue Canceled answer does not order the match and the cancel.
- **A cancel-failure probability.** None may be trained from notice-after-answer or from the cancel census.
- **Validation.** Consumed days are development evidence. Validation needs a freeze and at least 5 untouched days (C8 / C16), and an
  order-level check needs a live run with capture (C10). Both are the USER's call.

## 6. Data rules

- Consumed days only: Oct 23-27, Nov 8 (read to 23:02Z).
- Never the forward blocks 11-04..06 and 11-09..13, nor any untouched day.
- Jobs only via `run_light.sh` with PF_SLOT=claude. No orders; live stays off.
