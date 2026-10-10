# Forward test F1 v2: superseding v1 before any read (2026-10-10 01:23Z)

Supersedes `docs/FORWARD_FADE_F1.md` (v1, commit 79aede2). **No F1 outcome was ever read.** v1
stays as provenance. Everything in v1 holds unless changed below: population, trade, cost,
H3 and its pass rule, the minimum (≥ 400 complete rows, ≥ 12 complete ISO weeks), the reading
rule, multiplicity. **The commit that adds this file is the F1 freeze**; `research/forward_fade.py`
now reads this declaration.

## Why v1 stopped
The post-restart check at 01:19Z found that Binance's futures REST API (`fapi.binance.com`)
answers HTTP 451 ("restricted location") from this box. v1's funding source
(`/fapi/v1/fundingRate`) and its order-book snapshots (`/fapi/v1/depth`, `/premiumIndex`) could
never have worked. The v1 watcher ran from 01:18Z and wrote no snapshot: no Binance pump
check came due in that time. The websocket streams and Binance Vision are reachable.

## Changes
- **Funding** comes from Binance Vision's monthly `fundingRate` archives, the source the research
  used. A trade's funding is the sum of `last_funding_rate` with `calc_time` in (t4, t24], as in
  `research/btc_hedge.Funding.total`.
  - **New status `FUNDING_PENDING`:** the hold touches a month whose archives are not published
    yet (BTCUSDT's archive for that month is absent; the check reads no rate). Such a row does not
    count toward the minimum. Archives appear early in the following month, so rows become
    complete up to about five weeks after their exit.
  - A missing archive for the coin itself (any month of the hold) gives `NO_FUNDING` at the read,
    never zero and never a partial sum.
- **Cost snapshots** read the first message of two websocket streams:
  - partial depth, 20 levels (`wss://fstream.binance.com/public`, `<s>@depth20@100ms`);
  - markPrice (`/market`, `<s>@markPrice@1s`): mark, index and the current funding rate.

  Twenty levels can be too thin for $50k–$100k on small coins. Those notionals then read
  `book_too_thin`; the record also carries the total USD in the 20 levels.

Falsifiers: 19 tests in `tests/test_forward_fade.py`. Nine deliberate code breaks were each caught,
including partial funding across a month boundary and depth read from the wrong stream.
