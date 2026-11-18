# Review 8e63f178 fixed — 2026-11-18

USER:
- **"We have committed one more review, check the issues, together with the issues you have found, proceed to fix. Then check if the
  simulator is reliable to run now."**
- Then: **"My goal is to make our simulator ready for test runs and generate correct results, can you launch a loop to achieve this"**.

This is the readiness loop's first completed step (plan: [../simulator_readiness_20261118/plan.md](../simulator_readiness_20261118/plan.md)).

## The review's three P2 findings: all hold, all fixed

**P2-1, constraints dropped before the clocks were converted (pf_keep_pull_exec_windows.py, known v7).**
- Each event window is now kept as its constituents per clock:
  - book clock: WS hulls and the E2 / WS class windows;
  - status clock: REST ranges, imputed windows and the REST cap.
- A known fill is called avoided only at admissible offsets where the converted window is non-empty: book offset >= the leg's bound B,
  status offset in K7. This is solved exactly by two linear programs per row.
- Falsifiers:
  - the review's counterexample: book 90,050 at offset 0, REST 90,000 at -200, answer 90,100. It now FAILs; with the REST constraint
    dropped it passes.
  - an event with no common instant is not called avoided;
  - an unidentified book bound leaves only the cross-clock limits.

**P2-2, permissive comparisons (pf_timing_case_audit.py).**
- Identity fields are compared exactly; wei-parsed quantities within 1e-9; the recomputed endpoint within 1e-6 ms (rtol 0).
- Negative controls are refused at the operational resolution: 1 ms on the endpoint, 1 s on exec_s, 0.001 sh on shares.

**P2-3, cancel labels that disagreed with their evidence.**
- The bound "match <= c_hi" takes its type from the first answer's premise:
  - EVIDENCE only for "already canceled or matched";
  - CONDITIONAL for Canceled;
  - none for a refusal.
- The flag name and the census's N3 no longer claim "then filled".
- N4 compares each snapshot with the fills the bot had ALREADY observed at its receipt.
- The dependent inferences are evidence-only:
  - the known ledger's lifecycle upper bound is the first fill notice (K1-K8; K0 keeps the 11-09 definition);
  - K5's cancel effect has no upper bound from the acknowledgement.
- Known v6 holds the readings under the acknowledgement premise.

## Issues found along the way (fixed)

- **Scenario replay (pf_keep_pull_scenarios.py v5), FINDINGS checklist.** The replay reported neither C1 (the order-age split) nor C3
  (the cancel-pending share). Both are now computed per scenario, with control (i): KEEP equals the sum of its fills' values.
- **The case audit read its population from a moving path** (the ledger's current known receipt). It is now pinned to known v5, its
  declared population. The rerun had failed on it once.
- **Two mutants were killed by crashes, not checks.**
  - Exec-windows mutant 45 was pre-existing: it now plants the wrong landing its name states.
  - Scenario mutant 15: the selftest now records a refused report instead of crashing on its result.
- **One K9 test lost its distinguishing case** when strengthened; it was restored.

## Verification

| Check | Result |
|---|---|
| Mutation suites (every kill from a check; multi-line messages verified one by one) | case audit 40/40, census 17/17, exec-windows 127/127, scenarios 37/37 |
| Reruns on consumed data (case audit v3; the driver's `day 20261023`: census v4, known v7, scenario replay v5) | rc 0 |
| [compare.py](compare.py), declared before the reruns | **ALL true**: the audit's three legs identical (c_hi now CONDITIONAL); census per leg identical; K0 / K7 / K8 identical; all 19 scenario values identical to v4 |

## Results under the fixed instruments (consumed Nov 8 and Oct 23)

**Gate of record, per clock domain with the constituents kept:**

| Scenario | Gate |
|---|---|
| S3 | PASS |
| S1 | FAIL |
| S2 | FAIL |
| EV | PASS |
| ST | upper bound only |

Unchanged from v6: S3's pass survives the P2-1 fix.

**K9, new: match before the cancel request (book evidence).**
- 25 of 88 cancelled known legs.
- Of the 72 legs of orders answered Canceled and filled in full, **24 matched before their cancel request was even sent**. For those
  orders the venue answered Canceled for an order that had already matched. The rest are unidentified.
- K9 is conditional on the value link and on the book-clock premise.

**Cancel census v4:**
- 63 orders answered Canceled ended filled in full. The answer is uninformative about order.
- On all 63, the first fill notice came after the answer: a delayed observation, not the match order.
- On 62 orders the venue's post-cancel snapshot was below fills the bot had already been notified of.

**Scenario replay v5, Oct 23 (C1 / C3).**

| Scenario | E (c) | E from fills < 100 ms (c) | Cancel-pending share of fill gross |
|---|---:|---:|---:|
| M50 baseline | +34,948 | -769 | 2.3% |
| 100 ms cancel (L100) | — | — | 7.0% |
| Slow-cancel mixture | — | — | 12.6% |

- Young fills are 3.6-8% of fills, so the score's value does not sit on them.
- Z1-NY / Z3-NY read NOT-STABLE, like Z1-Z3.

## Still open (the loop's next items)

- C2: an explicit reading at M250. C12: the age split by trade-anchor source.
- The per-cancel race study, as the review specified.
- Oct 24-27 through the corrected chain.
- The test-run protocol (drafted).
- The USER's items: order-level fill validation (live data) and untouched validation days.
