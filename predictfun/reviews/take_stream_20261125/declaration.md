# TAKE-STREAM v1 — takes decided on the Binance stream, with a book-shrinking trigger, under venue rules (declared before any run)

USER (2026-11-25): "Yes the 50ms snapshots adds latency, we should use stream price in live, it should be faster and probably helps us.
And for the remaining 27%, yea it is Binance Orderbook shrinking, can we also see and test that. We fill in predict fun by immediate or
cancel right? Can we set a limit price, so we only take the best level".

## Why (from the diagnostics, take_exit_venue_20261125)

For the 72 % of the takers who beat our 172 ms takes that react to the Binance futures tick, tick → Predict match is 153 / 181 / 235 ms
(p10 / p50 / p90); our 50 ms grid adds ~24 ms, so our tick →
match is ~196 ms and we get the leftover fills (−0.47 ¢/sh at 1 s vs the winners' +3.00). 27 % act before the tick; for 42 % of them the
Binance futures best level had halved at an unchanged price first.

## Order type

A Predict MARKET order with `pricePerShare` = the last-seen best price and `slippageBps` = 0 fills only at that price or better and
cancels its unfilled part at once ("If the book moves beyond your slippage before the order matches, the order is cancelled without a
fill"); it waits the speed bump like every non-post-only order. This is the replay's take: one level at the limit or better, the rest
cancelled. Live confirmation (zero slippage accepted, partial fills, minimums) is the exec session's.

## Days and population

The five v3 days, consumed (development evidence only). Stage A: the 408 complete markets. Stage B (only if stage A is run and the USER
wants it): every entry-usable market as in the venue comparison.

## Triggers (decision time = Binance event time + 1.5 ms feed + 2 ms decision)

- **GRID (comparator):** the frozen take trigger on the 50 ms grid, take delay 172 ms from the grid decision (venue-comparison VM
  harness at 172 ms).
- **STREAM:** the frozen thresholds (r250 ≥ +0.2855 bps up, ≤ −0.4667 bps down) evaluated on the Binance futures mid stream at every
  breakpoint (change points and change points + 250 ms), OFF→ON after 2 s with no ON on that side, history continuous (no reset at the
  market start).
- **DEPLETION:** on every Binance futures best-level update: the best ask (bid) at an unchanged price at ≤ 50 % of its size 300 ms
  earlier → up (down); 2 s cooldown per side.
- **STREAM+DEPLETION:** the union (time order).
- One take at a time (pending-take rule), the frozen take size (50 within the ±50 cap on client-known tokens), limit = the Predict best
  known at the decision, the venue's 1 USDT buy minimum; the take reaches the book at decision + network + 150 ms.

## Configs (one exit: the current maker exit, BASE)

`GRID-172`, `STREAM-172`, `STREAM-192`, `STREAM+DEPL-172`, `STREAM+DEPLS-172`, `STREAM+DEPLS-192` (network 22 / 42 ms).

DEPLS (added before any stage-A run): DEPLETION with the base level >= 12.609 BTC, the p75 best-level size of 2026-10-24 (a consumed day
outside the five test days; sizes only, no outcome read). Reason, stated openly: the one-market rehearsal (code path; K1 parity exact)
showed the declared DEPLETION rule firing about every 1.5 s in a 5-minute market; DEPLETION stays as declared beside it. Everything else is the
venue-matched harness of the comparison (queue L, exits 58 / 37 / 95 ms, native tokens and fees, ties 50 / 50, settlement, unresolved
orders bounded).

## Readings (predicates evaluated by the code)

Per config: total at settlement (pooled, by day, days positive), the unresolved bounds, takes sent / filled and the limit-moved share, take
value after fee at 1 s and 10 s, round trips and held to settlement. Predicates: `stream_beats_grid` (STREAM-172 − GRID-172 > 0 on ≥ 4 of 5
days), `depletion_adds` / `strict_depletion_adds` (STREAM+DEPL-172 / STREAM+DEPLS-172 − STREAM-172 > 0 on ≥ 4 of 5 days), `profitable_172`
(the best of STREAM-172 / STREAM+DEPL-172 / STREAM+DEPLS-172: lower bound > 0 and ≥ 4 of 5 days positive; the choice is reported as a
multiplicity of 3), `holds_at_192` (the same rule for the 192 ms config of that trigger set).

## Checks

K1 the stream branch fed with the grid trigger times reproduces GRID-172 exactly (every 40th market); K2 stream triggers equal an
independent loop over the breakpoints (synthetic and one real market); K3 depletion triggers on a synthetic book; K4 every take's
arrival = decision + network + 150 ms and decision ≥ its Binance event + 3.5 ms; K5 the venue checks of the comparison (minimum, ties,
bounds, identities).

## Budget

Stage A ≤ 3,600 CPU s across slots (≤ 1,200 per day process), projection after the first day (cost scaled to 408 markets × 1.1);
trigger build ≤ 600 CPU s. No forward or sealed day is read (the book loader reads each day's own 24 hour files only).

Causality: the replay steps on the 50 ms grid; at tick g it sends every take decided in [g, g + 50 ms) on the client state known at g
(never later than the decision) and the Predict book known at the decision time.

## Resource-only amendment R1 (after the first-day projection refusal; before any run of days 2–5; outcomes not read)

Day 1 (20261023, 70 markets) cost 763 CPU s (about 11 s per market; the naive depletion config sends many takes). Its projection,
scaled to 408 markets × 1.1, is 4,891 CPU s against the 3,600 s cap: the gate refused as declared. The USER asked to run the replay
("Ok then we run the replay first, check if we can detect the signals early"). New caps: stage A ≤ 5,400 CPU s, ≤ 1,800 CPU s per day
process (20261031 has 102 complete markets). stream_replay.py changes only these two constants; day 1 ran the original file
(sha256 779db837a09ff5d485ccf3aa0bb468f03a10ee2b63b5e7fad9f4bd7e99bc19ec). No config, day, rule or check changes; the decision used costs only.
