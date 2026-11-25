# TAKE-EXIT-REPLAY v1 — 2026-11-25 (frozen before any replay)

USER (2026-11-25): "can we test on this taking alone, like if we identify some signal, we take, and exit by posting making orders".
A standalone strategy: no market making, only signal takes and passive exits. Research replay on the same 19 complete consumed
Oct 24–27 markets (`inventory_exit_exception_20261125` case list), the same engine, ASYNC_FAST10 passive timing (activation and cancel
+10 ms), 100 ms fill notice, END marks, fees (maker 0, taker 200 bps × min(p, 1 − p)). Consumed days, development evidence only; no
fit, tuning, new days or live use.

## Entry (the SIGNAL-TAKE rule, unchanged from signal_take_replay_20261125)

- **Trigger:** the pressure guard's own signal turns ON after 2 s quiet.
- **Order:** an IOC in its direction: buy YES on an ask-side (BTC up) trigger, sell YES (native BUY NO) on a bid-side trigger.
- **Size:** 50 shares within the hard cap; limit = the client-known touch.
- **Execution:** effective 40 ms after the decision tick, at the venue best inside the limit, with depth not reused.
- **Skip statuses:** no_room / pending_ioc / near_end / no_fresh_book / own_cross, as before.
- **Position limit:** hard cap C = 50, so one ±50 position at a time. An opposite trigger while holding is also a take; it reduces or
  closes the position.

## Exit and arms

- **PASSIVE_EXIT (primary):** while the client-known position is non-zero, post one post-only order on the closing side (ask when long,
  bid when short) at rung 0, i.e. the client-known best price on that side, at the back of its queue. Size = floor(min(100,
  |I| − same-side reservations)). Rung 1 is never used; no order on the adding side. A live exit order is KEPT while the position needs
  it and PULLED when it no longer does. The pressure guard applies to the exit side: it withholds and pulls the exit order while that
  side's signal is ON, so we do not sell into a continuing move. The engine's structural rule re-prices it when the book moves away. A
  position still open at END is marked at the END mark.
- **PASSIVE_EXIT_NOGUARD:** the same without the guard on the exit order.
- **HOLD (control):** takes only, no exit order; a position changes only by an opposite-signal take or END.

The engine's own protections stay authoritative: own-cross blocking, cap, near-END, structural removals and cancel-pending reservations.

## Measurements

Per arm × market and pooled, per day:

- **Value:** END net value (cash + position × END mark − taker fees), completed FIFO turnover, remaining value.
- **Takes:** triggers by status, takes sent / filled, taker shares and fees.
- **Exits:** passive exit fills, exit shares, the share of the position closed passively, time from entry fill to exit fill.
- **Order age at fill:** exit fills on orders younger than 100 ms vs older, value per share in each (the FINDINGS standing split: young
  replay fills are often phantom live).
- **Round trips:** FIFO pairs of take → exit with their gross and net value per share.

Paired comparisons PASSIVE_EXIT vs HOLD and PASSIVE_EXIT vs PASSIVE_EXIT_NOGUARD.

Readings, computed:

1. PASSIVE_EXIT pooled END net > 0;
2. > 0 on ≥ 3 of 4 days;
3. PASSIVE_EXIT − HOLD > 0 pooled;
4. its sign excluding exit fills on orders < 100 ms old (those fills' value removed from the delta, reported as a bound, not a re-run).

## Checks

- A no-trigger falsifier: with takes disabled, every arm must place no order and no fill.
- Every exit order is on the closing side at rung 0 with size ≤ the needed quantity at posting.
- Every IOC result is re-derived from venue frames with its fee.
- Ledger and FIFO identities.
- The trigger set equals SIGNAL-TAKE's for the same markets.
- Synthetic: long → ask only, short → bid only, flat → no order.

## Resources

≤ 400 CPU s run (3 arms × 19), projection after the first market, claude slot, peak RSS < 1.5 GiB. Local commits; no other session's
file is edited.
