# Timing case audit of the three S3 false-avoidance legs — 2026-11-18

USER: **"Check review and next priority, proceed next"**.

Review a50bc529 (of 3bf8f3a9) found no new blocking defect. It had one reporting note (exact denominators, now corrected in place
in [the one-day report](../keep_pull_corrected_day20261023/report.md)) and one next priority: "an independent clock/identity case
audit of the three S3 false-avoidance legs ... justify the clock constraint that removes the contradiction, or retain it and identify
the missing observation ... distinguish evidence-backed match windows from assumed fallback windows".

- **Code:** [pf_timing_case_audit.py](../../research/pf_timing_case_audit.py). Readings A0–A6 were declared before any data was read.
  Mutation check 29 / 29, each mutant caught by a check rather than a crash ([log](logs/mutants.log)).
- **Runs:** slot claude, rc 0.
  - v1, 15:11:01–15:11:59Z ([log](logs/run_v1.log)).
  - v2, 15:15:19–15:16:17Z ([log](logs/run_v2.log)). Peak 437 MiB.
  - Run windows corrected in place (d7591cb9 printed them shifted back by the wall time): run_light's stamp is the job's start, not
    its end.
- **Data:** consumed only.
  - Nov 8, read to 23:02Z. The basis readers stopped at the cutoff; no Nov 9 record was parsed.
  - The saved Oct 23 ledger v5 tables, for A6.
  - The one look before declaration was an identity-rule feasibility count, counts only:
    [feasibility_identity_count.txt](feasibility_identity_count.txt).

## Verdict (v2)

**At an independently bounded book clock, none of the three known fills is called avoided.** The contradiction is removed by a
constraint from other evidence, not by a fit to the gate.

- **Where the contradiction comes from.** Each leg is bound by the event window's lower end, `u* - 250 ms`. That end is a **WS book
  stamp** (updateTimestampMs) of other legs of **the same transaction and settlement id**.
- **Why K7 does not apply.** K7's interval [-226.4, +49.3] ms bounds the **REST status clock**, built from floor-second stamps and
  notices. The ledger applied it to this book stamp as well.
- **The independent bound.** The bot itself received the identity-matched book frames, with `u - receipt` between -150 and
  -21 ms on legs 20 and 73. Leg 45 has one pair at +36 ms. The declared statistic B bounds the book clock (venue minus bot) near each
  leg. B is the ceil(n/2)-th largest `u - receipt`; it holds if any pair in the upper half is a correct identity.

| Leg | Run, market | Critical offset (ms) | B / B_max / B_min (ms), pairs | Bot received the binding frame, after c_hi | A5 |
|---|---|---:|---|---|---|
| 20 | v6, 4335911 | -210.3 | -62.0 / -21.9 / -134.4, 121 | +103.8 ms (its own pair: -64.2) | REMOVED |
| 45 | v6, 4337531 | -224.8 | -59.3 / +36.1 / -150.2, 172 | +74.2 ms (-49.1) | REMOVED |
| 73 | basis, 4365071 | -218.8 | -56.0 / -21.0 / -130.7, 214 | not paired (status) | REMOVED |

The A5 preconditions held on all three legs:
- **A0:** every lifecycle field reproduces the known case table, and both bot journals show the same clock.
- **A2:** the binding legs are in the leg's own settlement.
- **A3:** the event's lower end is recomputed exactly, on V_book.
- **Bot clock:** the wall clock did not step (at - mono spread ≤ 0.003 ms over 560–891 kernel steps).

Statuses beside the result:
- Leg 45's B_max of +36.1 ms needs every pair to be correct and is not used. It is most likely a repeated book state.
- Leg 73's binding frame had no unique twin.

**The gate of record is unchanged.** K6 at K7's offsets still reads S3 FAIL, because the ledger applies one status-clock interval to
every endpoint. Changing it means evaluating each endpoint on its own clock domain, which is a known-ledger code change and run (see
"Next").

## Clock domains

| Domain | Fields | Used for |
|---|---|---|
| B | kernel step `at`; pfmm input `known_ms` (A0: the same instant in both journals) | lifecycle, c_hi, the bot's receipt of book frames |
| B_mono | kernel `mono` | the step check of B only |
| V_book | predictOrderbook `updateTimestampMs` | S3's WS windows (pend_inc t_ms), u* |
| V_status | pf-exec Fill `match_time` (rounded to the second), REST `executedAt` (Confirmed second) | K7; S3's REST ranges |
| V_uuid | settlementId UUIDv7 ms | ST; K8 |
| R | research collector `recv_ns` | file selection only |

No two venue domains are assumed aligned. The book-clock bound carries one premise: a frame's updateTimestampMs is stamped before
the bot receives it, and the research tape carries the same stamp for the same book update.

## The bounds of one leg (leg 20; all three in [audit_v2.txt](audit_v2.txt))

| Bound | Value | Clock | Type |
|---|---|---|---|
| match ≥ placement decision | 42.151 | B | evidence |
| match ≤ c_hi (the cancel answer's receipt) | 42.214 | B | a receipt time; the answer does not order cancel vs match (A1) |
| match ≤ the bot's receipt of the binding frame | 42.318 | B | evidence (same settlement, content identity) |
| match ≤ first fill notice | 42.703 | B | evidence |
| match ≤ settlement id creation | 42.213 | V_uuid | conditional (creation at/after the match) |
| match ≤ u* | 42.254 | V_book | evidence |
| match ≥ u* - 250 ms | 42.004 | V_book | **assumption** (S3's WS_DELAY) |
| match within the Matched second | [42, 43) | V_status | assumption (floor) |

(seconds after 2026-11-08 07:55:00Z, each on its own clock)

## A1 — the cancel premise: the venue's answer does not order the cancel and the match

Every cancel's first confirmation was the venue's HTTP **Canceled** answer. pf-exec journals it as CancelPendingAudit at exactly c_hi,
and pf-exec's own semantics note "whether it filled before cancelling is not known until the trade audit".

- **All three orders then filled in full** (5 of 5 shares).
- **The venue's post-cancel terminal order snapshot reported 0 matched.**
  - On legs 45 and 73 that snapshot arrived after the bot had already received the Matched fill notice (+442 / +338 ms after
    c_hi).
  - pf-exec flagged all three live: `late_fill_beyond_cancel_bound` (20) and `cancel_quantity_conflict` (45, 73), in
    [raw_records_v2.jsonl](raw_records_v2.jsonl).
- **The timing is a race.** Each settlement id was created within ±2.2 ms of the cancel answer's receipt (-1.35 / +2.18 / +0.25 ms,
  nominal clocks).

So on these legs the venue's answer and order state do not tell whether the match preceded the cancel's effect.

- **For A5:** this changes nothing. A known fill was not avoided, whatever the order of events.
- **For two premises:** it matters.
  - K5's "the cancel takes effect before its confirmation" is not evidenced here.
  - The ledger's "CANCEL_FIRST means the fill is avoided" did not hold for these three cancels: each was acknowledged and the
    fill came anyway.

## A6 — evidence-backed vs assumed windows (Oct 23 ledger v5, removed fills, cancel window CP)

Removed: 8,862 fills, 528,493 c gross, which reproduces the ledger's cross-tab (YES). Tiers:
- **EVIDENCE:** the evidence-only window EV certifies the same class.
- **BEYOND:** certified only under the scenario's ranges, listed with the basis of the deciding endpoint.

| Scenario | CANCEL_FIRST, evidence | CANCEL_FIRST, beyond evidence | MATCH_FIRST, evidence | MATCH_FIRST, beyond | Uncertified |
|---|---:|---|---:|---:|---:|
| S1 | 0 | 6,769 fills / 349,250 c (WS -250 ms 156,146; E2 -250 ms 113,733; mixed candidates 58,515; WS+E2 16,788; REST -500 ms 4,069) | 3 / 175 c | 5 / 322 c | 2,085 / 178,746 c |
| S2 | 0 | 5,792 / 277,490 c | 15 / 335 c | 7 / 1,344 c | 3,048 / 249,324 c |
| S3 | 0 | 3,947 / 175,442 c, **all WS pending - WS_DELAY (V_book)** | 18 / 388 c | 6 / 476 c (WS pending) | 4,891 / 352,188 c |
| ST | 0 | 214 / 7,934 c, all the S3 fallback's WS_DELAY | 18 / 388 c | 99 / 7,949 c (t_set, conditional) | 8,531 / 512,222 c |

The CR rows are in [evidence_tiers.csv](evidence_tiers.csv).

**No CANCEL_FIRST is evidence-backed in any scenario.**
- Every certified avoidance rests on an assumed lower end. For S3 and ST, that is entirely WS_DELAY: the first frame showing a pending
  increment is at most 250 ms after the match.
- Evidence alone (EV) certifies 18 MATCH_FIRST fills, 388 c, or 0.07% of the removed gross. The other 8,844 removed fills are
  UNKNOWN or AMBIGUOUS under EV.
- So the removed-value economics are scenario-conditional, not evidence-backed targets.

## v1 → v2 (provenance; both receipts kept)

- **v1's declared verdict was RETAINED for all three legs** ([audit_v1.txt](audit_v1.txt)), on one A0 field, `one_bot_clock`. That
  was **a defect of this instrument**:
  - The pfmm engine's order id is unique only within a market; id 146 recurs across many markets.
  - v1 keyed the journal records by the id alone, so the leg's own Accepted was never compared on its own.
  - Keyed by (market, id), each leg's Accepted is identical in both journals (Δ 0.000 ms), and CancelDone equals c_hi.
- **v2 fixed only that**, with a planted same-id record in another market as the falsifier and a mutant for it.
- **v2 also added, without any reading depending on them:**
  - pf-exec's alerts naming the order, kept in the raw records;
  - two descriptive A1 fields.
- **Unchanged:** every declared reading, statistic, threshold and window. A1–A4, A6 and the input hashes are identical between v1 and
  v2.
- **The verdict changed only through that fix, after v1's output was read**, and is reported as such.

## Next

1. **Gate of record by clock domain.** The known ledger evaluates each window endpoint on its own domain:
   - book stamps against a book-clock bound from the bot's own frame receipts, for all 99 legs;
   - status seconds against K7;
   - UUID times against K8.
   This is a code change plus a known-ledger run on consumed Nov 8, on the USER's go.
2. **The cancel race on every known leg.** Tabulate answer class × fill, plus the terminal snapshot, before any cancel-latency or
   "avoided" semantics is trusted. Same kernel journals, consumed Nov 8.
3. **Unchanged:**
   - The score's sign is not identified.
   - S3's lower ends stay assumptions; the known legs can contradict WS_DELAY but never establish it.
   - Oct 24–27 not run.
   - One partial day of own orders.
   - Ranges chosen after seeing that evidence.
