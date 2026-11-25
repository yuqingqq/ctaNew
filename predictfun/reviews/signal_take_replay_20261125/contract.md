# SIGNAL-TAKE-REPLAY v1 — 2026-11-25 (frozen before any take-arm replay)

USER (2026-11-25): "Yes add the take to test", following TAKE-SIGNAL v1 (take_signal_20261125): taking on our own pressure-guard
trigger looked profitable at Tokyo latency as an isolated opportunity. This test adds the take to our actual strategy, so the taken
inventory is managed by the existing passive rules and caps. Research replay on consumed days only; no new days, fit, threshold
tuning, live execution or data retrieval. Development evidence; no validation or promotion claim.

## Scope (unchanged from the host experiment)

- The same 19 complete consumed Oct 24–27 BOOK markets, in the case list of `inventory_exit_exception_20261125/manifest.json`.
- Full 298 s; opening-only gate; ASYNC_FAST10 passive timing (activation and cancel +10 ms); 100 ms fill notice; the same END marks.
- Maximum 100 shares per passive order, two levels, desired 200 per side; passive fees 0, no rebates.
- Host arms: **BASE** C200 / S100 / R50 and **WIDE** C1000 / S500 / R250, i.e. the saved `BASE` / `WIDE` ledgers of
  `data/predict_fun/research/inventory_exit_exception_20261125` (hash-bound inputs).
- Parity before any reading: my branch with the take disabled must reproduce those saved ledgers exactly (fills, orders, requests,
  value, cash, position, commands, guard log, grid, closed, mode log) on markets 3697902 (Oct 24), 3789484 (Oct 26) and 3899224
  (Oct 27), for both hosts. Any mismatch stops the run.

## Candidate arms: BASE_TAKE, WIDE_TAKE = host + SIGNAL_TAKE

- **Signal:** at each 50 ms decision tick k, after the policy step, side s uses the guard's own signal, ON_s(k) = cache status ok
  and `pf_ladder_pressure.on_value(x_s[k], frozen edges[s])`.
- **Trigger:** fires when ON_s(k) is true and ON_s was false at every tick k−40 … k−1 (2 s quiet).
- **Direction:** an ask-side trigger (BTC up) sends an IOC **buy YES** (engine side 0) at limit = the client-known best ask. A bid-side
  trigger (BTC down) sends an IOC **sell YES** (engine side 1; live it is a BUY NO at 1 − p, so no inventory is needed) at limit =
  the client-known best bid.
- **Size:** q = floor(min(50, room)). Room = C − I − B for a buy and C + I − A for a sell, with client-known inventory and
  reservations including open IOCs.
- **Skip statuses** (counted, never silent): q < 1 (no_room); another IOC open (pending_ioc); g + 40 ms ≥ END (near_end); known book
  invalid or older than 1,000 ms (no_fresh_book); a live own opposite order at or through the limit (own_cross).
- **Execution:** the IOC takes effect at **g + 40 ms** (slower than the host's 10 ms placement, matching the take study's Tokyo
  arrival). It fills at the venue best inside the limit with the pilot's net-depth budget minus intervening public trades, with no
  worse-price sweep. Fill notice comes at effect + 100 ms. Fee = 200 bps × min(p, 100 − p) per share, as a cash-equivalent cost.
- **Afterwards:** inventory is managed only by the unchanged host rules (soft band, release, hard cap, pressure guard, opening gate).
  The soft band does not block a take; only the hard cap does. One size, one delay, one cooldown, frozen edges: nothing is tuned.

## Measurements and readings

Per arm × market:

- END net value (cash + inventory at END − taker fees), completed FIFO turnover and remaining inventory value;
- taker fees; triggers by status, IOCs sent / filled, fill rate, taker shares, passive fill shares;
- mean and peak |inventory|, both-side uptime;
- every filled take's markout at +1 / +3 / +10 s and END (venue mid, ≤ 1 s old).

Paired deltas BASE_TAKE − BASE and WIDE_TAKE − WIDE, pooled, per day, markets improved, top-3 market concentration, and leave-out
3789484. Computed readings:

1. pooled delta > 0, per host;
2. delta > 0 on ≥ 3 of 4 days;
3. mean +3 s markout per filled take share > 0 (consistency with TAKE-SIGNAL);
4. completed-turnover and residual deltas reported separately (a gain only in marked residual is flagged, not credited).

Point estimates only (4 consumed days).

## Checks

- Triggers recomputed by a plain loop from the cache and edges equal the logged triggers.
- Every sent IOC's limit equals the client-known best at its tick, and every IOC result is re-derived from venue frames (price inside
  the limit, quantity ≤ availability).
- Ledger identities: cash, position, fees.
- Synthetic controls (quiet-window rule, no_room, near_end, pending_ioc, own_cross) and real-copy corruption refusals.

## Resources

Run ≤ 450 CPU s: 2 take arms × 19 + 2 hosts × 3 parity markets. Outcome-blind projection after the first market; stop if it exceeds
450. Checks ≤ 60 s, through run_light on the claude slot (core 9), peak RSS < 1.5 GiB, local commits. No other session's file is
edited.
