# Simulator readiness plan — P-2026-005 keep/pull simulator

USER goal (2026-11-18): **make the simulator ready for test runs that generate correct results**. A self-paced loop (session aea5bfb9)
works through this list. Each item records its status, the evidence that closes it, and its receipt. New reviews come first.

Statuses: OPEN, IN-PROGRESS, DONE (with receipt), USER (needs a decision or data only the USER can supply).

## Items

| # | Item | Status | Evidence that closes it |
|---|---|---|---|
| 1 | Review 8e63f178's P2s fixed in code: P2-1 window constituents per clock with feasibility (known v7); P2-2 exact identity / endpoint comparisons (case audit v4); P2-3 the c_hi bound typed by the answer's premise, N3 / N4 restated, the dependent lifecycle and K5 inferences evidence-only, K9 declared | DONE | suites case audit 40/40, census 17/17, exec-windows 127/127, scenarios 37/37 (every kill from a check); [receipt](../review_8e63f178_fixes_20261118/report.md) |
| 2 | Reruns on consumed data with the fixed code: case audit v3 receipt (3 legs), census v4, known v7, scenario replay v5 (Oct 23, the driver's `day` path), and compare.py R1-R4 (everything outside the fixes reproduces) | DONE | compare.py ALL true (R1-R4); readings in the receipt |
| 3 | Scenario replay meets the FINDINGS checklist items code can meet: C1 age split and C3 cancel-pending (v5, coded); C2 an explicit reading at M250; C12 the age split by trade-anchor source (WS / E2); C15 / C19 / C20 / C24 re-checked against the outputs | IN-PROGRESS | C1 / C3 DONE (v5, Oct 23); C15 / C19 / C20 / C24 met by the outputs; next C2 (an explicit M250 reading) and C12 (needs each re-simulated fill's event class) |
| 4 | Per-cancel race study as review 8e63f178 specified: every cancel on Nov 8 (no-fill orders and retries included), request / response / match bounds / notice times kept apart, censored and unknown outcomes kept, the match-before-request evidence (K9-type) per cancel; never a cancel-failure model from notice-after-answer | OPEN | instrument with selftest, falsifiers and mutants; one run; receipt |
| 5 | Oct 24-27 through the corrected chain (the driver's `all` phase: ledger, census-gated known ledger, scenario replay per day, pooled report) | OPEN | every step rc 0; Z0 controls YES per day; readings pooled with by-day values |
| 6 | Test-run protocol: what a run must output (checklist), what it may claim (scenario-conditional values, stated premises) and what it may not (P&L, order-level fill validity, match-vs-cancel order), the data rules and the gates | OPEN | test_run_protocol.md committed |
| 7 | Order-level fill validation (FINDINGS C10: no replay number is validated at the order level; only a live run with capture can validate one) | USER | live run with capture is the USER's call |
| 8 | Untouched validation (C8 / C16: >= 5 untouched days after a freeze) | USER | the USER's choice of days and freeze |

## Rules every item follows

- Readings are declared before data is read; consumed days only (Oct 23-27, Nov 8 to 23:02Z); never 11-04..06, 11-09..13 or an
  untouched day; no orders.
- Jobs only via `run_light.sh` with PF_SLOT=claude; memory caps never raised; no edits to a source a running job reads.
- Every checker ships a falsifier; mutants must be caught by a check, not a crash.
- One commit per step, then fetch and an ahead-only push.

## Log

- 2026-11-18 17:16Z — plan created (tick 1). Items 1 and 3 in progress (code done, suites running); item 2 next.
- 2026-11-18 17:29Z — scenarios 37/37; exec-windows 127/128: the K9 test lost its book-bound case when strengthened - fixed (leg 4 now separates
  hi_book from hi_book - B); both K9 mutants caught; full suite rerunning.
- 2026-11-18 18:06Z — items 1-2 DONE: every suite passes with kills from checks; compare.py ALL true; the C1 / C3 splits computed on Oct 23
  (young fills 3.6-8%, E -769 c on them; cancel-pending 2.3% of gross). Committing this step; next item 3 (C2, C12).
