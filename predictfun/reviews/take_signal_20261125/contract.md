# TAKE-SIGNAL v1 — 2026-11-25 (declared before any take outcome is read)

USER (2026-11-25): in 3789484 Maker3 is aggressive at the start while we withhold asks because our signal predicts an uptrend. Is
that signal strong enough to pay the taker fee, and is it good to acquire inventory early? Measure the signal, quantify its strength,
and check whether we could take actively instead of waiting. Consumed Oct 24–27 only, saved Predict book + raw Binance bookTicker,
no replay of our strategy, no model fit, no new days, no live use. Exploratory P-2026-005 mode; descriptive with one declared split.

Already seen before this contract (not outcomes of the test): in 3789484 Maker3's first-40 s fills are all passive. It moved its bid
to 56 c within ~1 s of a +2.6 bps BTC jump at 0.5–0.7 s; its first take is at 82.8 s. Across Oct 24–27 Maker3 has 2,178 taker legs
stamped in the first 30 s (497 markets), mostly position-adding. The raw BTCUSDT bookTicker mid moves in discrete jumps (40 mid changes in
a 300 s window at ~84 updates/s); the maker_trace_v2 100 ms grid matches it, so this is a feed property, not a builder defect.

## Signal (the guard's own feature, recomputed from raw)

`bn_r250(t) = 1e4 · ln(mid(t) / mid(t − 250 ms))`, mid = (bid + ask)/2 of BTCUSDT futures bookTicker. It is computed on the Binance event
clock E, deduplicated by update id, exactly as `pf_step5_features`. Decision time t = E + 5 ms (Tokyo inbound) at every mid-change event.
Secondary: `bn_r1000`. Directions use the frozen pressure-guard edges (pf_ladder_pressure, never recomputed):

- **UP** (the ask guard turns ON: we would withhold asks; candidate = buy YES at the ask) when bn_r250 ≥ 0.285456120967865;
- **DOWN** (the bid guard ON; candidate = sell YES at the bid, i.e. buy NO) when bn_r250 ≤ −0.4666730940341952.

Rows are actions: in each market only the first qualifying event per direction in any 2 s (a new trigger needs 2 s without one) is a
take opportunity. Later events inside the 2 s are counted, not valued. Strength bins of |bn_r250|: [edge, 0.5), [0.5, 1), [1, 2),
≥ 2 bps. Parity check: my bn_r250 at 3789484's 550 ms opening-gate trigger vs the recorded +0.32485 bps (B_IN = 5 ms).

## Execution (measured Tokyo clock, conservative)

- Predict book **known** at decision = last tob frame with upd_ms ≤ t − 63 ms (53 ms publication + 10 ms network). A take needs that frame
  two-sided ok (econ MID rule, ≤ 1,000 ms old). Limit = known best ask (UP) / known best bid (DOWN); no chasing.
- Order reaches the venue at **t_a = t + 42 ms** (2 ms decision + 40 ms order). Fill iff the limit is still executable in BOTH the last frame
  with upd_ms ≤ t_a and the first frame with upd_ms > t_a: the ask at or below the limit for UP, the bid at or above it for DOWN.
  Shares filled = min(Q, the touch size at the executable price in both frames); price = the limit; Q ∈ {10, 50, 100} (size arms;
  primary 50). Taker fee = feeRateBps/1e4 × min(p, 1 − p) per share (200 bps on every market).
- Sensitivity: slow clock with all three latencies doubled (Binance 10, Predict 126, order 84 ms).

## Values per filled share (s = +1 UP, −1 DOWN; YES space; after the entry fee)

- mark_h = s·(mid(t_a + h) − price) − fee, h ∈ {1, 3, 10, 30} s; mid = last two-sided ok frame ≤ t_a + h (≤ 1 s old); censored
  (status, counted) if t_a + h > END = we − 2 s.
- END = s·(m_END − price) − fee; RES = s·(V − price) − fee (V = 1 Up, 0 Down, 0.5 tie).
- Taker round trip at h: s·(exit − price) − fee − exit fee, exit at the opposite touch at t_a + h (lower bound if we must leave by taking).
- Pre-trade staleness = s·(known mid − mid at the frame before t − 1,000 ms), the Predict move already made before we act.

## Strata and readings

Strength bin × time in window (0–30, 30–60, 60–180, 180–240, 240–270, 270–298 s) × moneyness (known mid 10–90 ¢ = NEAR) × spread at
decision (≤ 1 ¢ / > 1 ¢), per day and pooled, both directions and combined. Fill rate and every exclusion status are counted.

Readings, computed in code:

1. Does a filled take earn after the fee at h = 1, 3, 10 s (pooled mean > 0, and the same sign on ≥ 3 of 4 days)?
2. Does the value rise with signal strength (monotone across the 4 bins at h = 3 s)?
3. Is the early window (< 60 s) different from later windows?
4. END / RES values are reported, never used to choose (high variance; one market's outcome dominates its rows).

**Declared selection (one split):** on Oct 24–25 choose the (strength bin ≥ b, time window) cell with the highest pooled mark_3 net per
share among cells with ≥ 30 fills in ≥ 10 markets. Report that one cell's Oct 26–27 value, sign and day split. If no cell qualifies,
report none. Consumed days on both sides; development evidence only, never validation.

## Maker3 alignment

For each Maker3 taker leg (Oct 24–27, status ok; from maker3_book_conditions legs_m3), qualifying event in the window
[stamp − 3.5 s, stamp] (the match lies in [stamp − 2.5 s, stamp], plus 1 s lead):

- agreeing = an event whose direction equals the take's YES direction s;
- opposing = only an opposite-direction event.

Statistic: the agreeing fraction for position-adding (ADD / OPEN / FLIP) and reducing takes. Null: 500 draws that circularly shift each
market's take schedule by a uniform offset within [ws, END] (seeded 20261125), recomputing the fraction with the BTC series fixed. Also
report net-to-resolution, 1 s and END value per share for agreeing vs non-agreeing takes. Association only: no private trigger is
identified.

## Checks

Independent bn_r250 recomputation by a plain loop on sampled events; a causality falsifier (Binance updates after t and Predict frames
after the stated cutoffs perturbed → decision inputs unchanged) with a positive control; synthetic fill cases (stale ask kept → fill;
ask lifted in the next frame → no fill; fee formula); row and status reconciliation; the day rule enforced on hour stems before any file
opens (20261023_23 and every hour of 20261024–27 only; no 20261028 stem is ever opened).
