# Per-cancel race study — 2026-11-18

**USER:** "My goal is to make our simulator ready for test runs and generate correct results, can you launch a loop to achieve this"
(readiness loop, [plan item 4](../simulator_readiness_20261118/plan.md)).

**Specification** (review 8e63f178's next priority): separate request, response, actual match bounds and notice times; include no-fill
orders and retries; preserve censoring and unknown outcomes; never train a cancel-failure model from "notice after answer".

- **Instrument:** `predictfun/research/pf_cancel_race_study.py` v2, with the mutation harness `pf_cancel_race_study_mutants.py`.
  Readings were declared in the module before data was read.
- **Population:** every order with a cancel request in the two Nov 8 kernel journals (runs v6 and basis), read to 23:02Z — 30,859 cancelled
  orders, of which 85 had fills.
- **Clocks:** the bot's own clock for request, response and notices. Match bounds come from the first fill notice (evidence) and the
  venue's Matched second converted with K7 = [-226.4, +49.3] ms (conditional on K7's premises).

## Gates

| Gate | Result |
|---|---|
| Selftest (synthetic: every class, priority rule, retries / re-sends after an HTTP 429, censoring, sibling fills, cutoff, refusals) | pass |
| Mutation suite (every mutant caught by a check) | 25 / 25 |
| C5: reproduces census v4 (class, answer receipt, first notice, full fill) | **YES**: 86 / 86 census orders. 85 have a cancel request in the case table; the 86th has none and is correctly absent |
| C4: book evidence (K9) against the status-clock classes | CONSISTENT on 98 known fills, 0 contradictory |

## v1 → v2 (declared after v1's output was read; v1 kept as provenance in [v1/](v1/))

1. **Retries.** The bot never re-requests a cancel: there is one Control/Cancel per order. pf-exec's re-sends appear as the kernel's
   outbox entries `{"One": {"cid", "market"}}` after a refusal (e.g. `{"Http": 429}`). v2 counts every send.
2. **C5's population.** v1 compared all 86 census orders. One of them was never cancelled, so v1 read NO on 85 / 86. v2 compares against
   the case table's cancel requests, in both directions.
3. **C3b added.** C3 dates a match against the first answer, which may be a refusal; the order then stays live. C3b dates it against the
   first Canceled confirmation.

## Results (Nov 8 to 23:02Z; [v2/study.txt](v2/study.txt))

### Population (C0)

| Run | Placed | Signed | Cancelled orders | Cancel sends | Re-sent | Cancel-all sends |
|---|---:|---:|---:|---:|---:|---:|
| v6 (ended 14:14Z) | 21,661 | 16,585 | 16,575 | 16,608 | 29 | 2 |
| basis (to the cutoff) | 20,966 | 14,150 | 14,284 | 14,225 | 45 | 1 |

- The 275 orders with no answer were all **never signed**, i.e. never sent to the venue (statuses, not races). 280 cancelled orders
  were never signed; 5 of those still received an answer.
- Censored (horizon < 60 s after the request): 16 orders, none with a fill.

### Response latency (C1)

First confirmation minus the request, bot clock. **A response is the venue's answer, not the cancel's effect.**

| First confirmation | Orders | p50 | p90 | p99 | max | > 1 s | > 4 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CANCELED | 30,473 | 19 ms | 40 ms | 131 ms | 6,992 ms | 13 | 7 |
| MATCHED-OR-GONE | 24 | 17 ms | 98 ms | 6,163 ms | 7,971 ms | 1 | 1 |
| NO-EFFECT (refusal, e.g. HTTP 429) | 87 | 12 ms | 54 ms | 318 ms | 328 ms | 0 | 0 |
| NO-ANSWER (never sent) | 275 | — | — | — | — | — | — |

### Race (C2, C3, C3b)

Order class by first confirmation:

| First confirmation | AFTER-ANSWER | AFTER-REQUEST | UNORDERED | BEFORE-REQUEST | NO-FILL | CENSORED |
|---|---:|---:|---:|---:|---:|---:|
| CANCELED | 0 | 2 | 67 | 1 | 30,387 | 16 |
| MATCHED-OR-GONE | 0 | 1 | 0 | 0 | 23 | 0 |
| NO-EFFECT | 14 | 0 | 1 | 0 | 72 | 0 |
| NO-ANSWER | 0 | 0 | 0 | 0 | 275 | 0 |

- **C3b MATCH-AFTER-A-CANCELED-CONFIRMATION: NO** (conditional on K7). None of 30,563 orders with a Canceled confirmation matched
  provably after it.
- **C3 MATCH-AFTER-ANSWER: YES** (conditional on K7). 14 orders matched after their first answer, and **every one of them followed a
  refusal** (NO-EFFECT); the order stayed live. None followed a first Canceled answer (0 of 30,473).

## What this means for the simulator

1. **The landing rule (no fill after the cancel lands) is not contradicted by any Canceled confirmation on Nov 8.** The bound is
   one-sided: a match within about 1.3 s of the confirmation (the status clock's floor second plus K7's width) cannot be ordered against
   it. C3b NO therefore means "none provably after", not "none after".
2. **Refused cancels are not modelled.** 87 first answers (0.28% of the 30,584 answered cancels) were refusals. 14 of those orders matched while still
   live. The simulator lands every pull, so it credits such cases as avoided fills. On Nov 8 the size of that bias is 14 orders in 30,859
   cancels.
3. **The slow-cancel mixture is a stress case far beyond the observed responses.** The mixture (13% of cancels at 4 s) is an explicit
   sensitivity choice (`CANCEL_MIX`). Over all 30,584 answered cancels, 8 responses took more than 4 s (0.03%) and 14 more than 1 s.
   The baseline's 37 ms is near the response p90 (40 ms), but a response is not a landing, and no landing distribution is identified here.
4. **Fills on cancelled orders are rare and mostly unorderable.** 85 of 30,859 cancelled orders filled (0.28%). Of the 70 answered
   Canceled, 67 cannot be ordered against the request at the status clock's resolution. K9's book evidence separately places 25 of 88
   cancelled known legs before their request.

## What this is not

- **Not a cancel-failure probability, and not a training target.** No class here dates the cancel's effect.
- **Not the keep/pull rule.** It covers one consumed partial day of the live strategy's cancels (runs v6 and basis); the keep/pull rule
  itself was not run live.
- **Fills after the horizon are unseen.** Censoring is a status.

## Files

- **Outputs:**
  - [v2/study.txt](v2/study.txt) and [v2/study.json](v2/study.json): readings, R, inputs and hashes;
  - [v2/fills.csv](v2/fills.csv): all 99 fills on cancelled orders;
  - [v2/orders_selected.csv](v2/orders_selected.csv): 484 orders with fills, a non-Canceled first answer, a response over 1 s, a re-send,
    or censoring;
  - [v2/class_by_first_confirmation.csv](v2/class_by_first_confirmation.csv).
- **Logs:** [logs/](logs/), with the selftests, mutation suites and runs of v1 and v2. Sources: [sources.sha256](sources.sha256).
- **Full tables:** `data/predict_fun/research/cancel_race_study_v2_20261118/orders.parquet` and `fills.parquet`. Their hashes are in
  study.json.
