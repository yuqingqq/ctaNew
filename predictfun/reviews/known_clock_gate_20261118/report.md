# Clock gate per domain and cancel-answer census — 2026-11-18

USER: **"Go"**, approving the two next steps of the [timing case audit](../timing_case_audit_20261118/report.md):
1. evaluate the known ledger's gate per clock domain;
2. count the cancel race on every known leg.

## Code

| File | Change | Mutation check |
|---|---|---|
| [pf_known_clock_census.py](../../research/pf_known_clock_census.py) | new: the book-clock bound and the cancel answer per known leg | 14 / 14 |
| [pf_timing_case_audit.py](../../research/pf_timing_case_audit.py) | lifecycle / book_pairs split out for reuse; reader speed-ups; three lifecycle fixes (below) | 36 / 36 |
| [pf_keep_pull_exec_windows.py](../../research/pf_keep_pull_exec_windows.py) | known v6: the per-domain gate | 120 / 120 |
| [run_keep_pull_scenarios.sh](../../research/run_keep_pull_scenarios.sh) | the census runs before the known ledger | — |

- Every mutant is caught by a check.
- The scenario selftest passes.

## Runs

All runs used slot claude and returned rc 0. Data: consumed Nov 8 only, every reader stopping at 23:02Z.

| Run | Window (UTC) | Notes |
|---|---|---|
| Census v1 | 16:04:38–16:06:09Z | |
| Census v2 | 16:10:58–16:12:29Z | |
| Census v3 | 16:14:38–16:16:08Z | 90 s, peak 400 MiB |
| Known v6 | 16:16:20–16:16:26Z | |

## 1. Every known leg's book clock is bounded (census v3)

| Reading | Result |
|---|---|
| N0, reproduces the case audit's three bounds | YES (B, n and clock step equal exactly) |
| N1, records reproduce the case table | 99 of 99 legs |
| N2, book bound identified | 88 of 88 legs with a cancel confirmation; B from -65.1 to -44.7 ms, median -55.3 |

- The bound is the venue book clock minus the bot clock, from the bot's own receipts of identity-matched frames within ±10 s of the
  cancel confirmation. It holds if one upper-half pair is a correct identity.
- K7's lowest status-clock offset, -226.4 ms, is more than 160 ms below every leg's book bound.

## 2. The gate of record per clock domain (known v6)

Each window's lower end is split by the clock of the legs that set it:
- WS and E2 legs are on the book clock; imputed legs and REST ranges are on the status clock.
- Each end is then mapped with the lowest admissible offset of its own clock: the leg's book bound for book ends (fail-closed if
  unidentified; none was), K7 for status ends.

| Scenario | Known fills called avoided at offset 0 | Per clock domain | K7 on every end (v5 basis) | Gate of record |
|---|---:|---:|---:|---|
| S1 | 4 | 5 | 9 | FAIL |
| S2 | 5 | 5 | 5 | FAIL |
| **S3** | 0 | **0** | 3 | **PASS** |
| ST | 0 | 0 | 0 | UNIDENTIFIED-UPPER-BOUND-ONLY |
| EV | 0 | 0 | 0 | PASS |

The replay's point M50 is not a gate: it calls 5 fills avoided at offset 0 and 15 per domain, and stays CONTRADICTED.

- **S3 passes under the right clock.**
- **S1 and S2 fail on their own windows.** Their declared ranges call 4 and 5 known fills avoided at offset 0, before any clock
  question.
- Every v5 reading and every K0–K8 quantity is identical in v6. `K6 GATE {k}` keeps its v5 meaning, K7 on every end. The new key is
  `K6 GATE {k} (per clock domain)`, and KNOWN VERDICT names it as the gate of record.

## 3. Cancel-answer census: a venue "Canceled" answer does not stop a fill

The census covers 86 orders, all of which filled (an outcome-selected population).

| First answer to the cancel | Filled in full | Filled in part |
|---|---:|---:|
| Canceled (pf-exec's CancelPendingAudit) | **63** | 6 |
| "can't be found - already canceled or matched" | 1 | 0 |
| Refused (Http / not in the answer) | 14 | 1 |
| No answer | 1 | 0 |

- **N3 reads NO:** 63 orders were acknowledged Canceled and then filled in full.
  - Descriptive, added after v1 was read: counting any acknowledgement, retries included, the figure is 71 full and 6 partial.
- **N4:** for 74 orders the venue's post-cancel terminal order snapshot understated the final fill. For 62 of them the snapshot
  arrived after the bot had already received the fill notice.
- **N5:** pf-exec alerted live on 73 orders (61 `cancel_quantity_conflict`, 12 `late_fill_beyond_cancel_bound`).

Among filled orders, a Canceled answer is the norm (69 of 86) and "already matched" the exception (1 of 86).
- The venue acknowledges cancels of orders that fill, and its order state lags its own fill notices.
- So K5's premise ("the cancel takes effect before its confirmation") and the ledger's "CANCEL_FIRST means the fill is avoided" are
  not evidenced for orders whose cancel races the match.
- This counts races among filled orders. It is not a rate per cancel.

## Corrections and provenance (all receipts kept)

**Census v1** read N1 76 of 99 because of three defects in the lifecycle code reused from the case audit:
- Fills were matched by trade id alone, so a settlement that filled two own orders lent the sibling's notices (12 legs).
- The confirmed cancel was compared with a refused first answer whose cancel was confirmed on a retry (10 legs).
- The one-bot-clock tolerance was 1 µs, while the two journals' stamps are separate reads of the host clock (4.4 µs on 1 leg, whose
  bound v1 left UNIDENTIFIED).

**Census v2** read 89 of 99. My cancel comparison counted "can't be found" answers, but the fill-clock audit defines the confirmed
cancel as the order's first CancelPendingAudit only.

**Census v3** reads 99 of 99. Each fix has a falsifier and a mutant. Across versions:
- the declared readings and thresholds are unchanged;
- N0, B and N3–N5 are unchanged;
- N2 went from 87 of 88 to 88 of 88.

**The case audit's run windows** (d7591cb9) were shifted back by the wall time: run_light stamps a job's start. They are corrected
in place.

## Next (each on the USER's go)

1. **The cancel race per cancel.** Nov 8's kernel journals hold every cancel (16,575 cancel decisions in v6 alone) and every fill.
   Tabulate the answer class × a later fill × the time from the answer to the match evidence. That turns N3 into the rate the
   replay's cancel model needs.
2. **S1 and S2.** Their declared ranges fail the known gate at offset 0, so keep them as stated findings, not gate candidates.
3. **Unchanged:**
   - S3's lower ends remain an assumption (WS_DELAY): S3 is compatible with the known legs, not identified.
   - The score's sign is not identified.
   - Oct 24–27 not run.
