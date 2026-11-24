# Maker 3's visible quoting: first minute, market 3789484

2026-11-24. User-directed saved-data reconstruction of the same Oct26 market;
no new day, replay, fitted rule or validation. Population: all17 maker3 hashes
with51 passive fill legs whose settlement upper bounds lie in[0,60). Every
hash is retained, including two refused posting estimates. No taker legs in
this minute. Nine posting windows have inherited conditional bounds; six are
weaker scenarios. This is a filled-order census, not the maker's complete
private quote stream. Later orders with refused posting estimates and
never-filled orders can hide additional early quotes.

## Visible opening chain

| Order | Price | Estimated posting window, seconds | Recorded fills, shares | Evidence |
|---|---:|---:|---:|---|
|First bid56 hash|56c|1.038–1.755|20.10|Weaker scenario|
|Second bid56 hash|56c|1.797–3.999|168.409 over18 records|Conditional bound|
|Ask58|58c|5.239–7.428|90.327 over2 records|Conditional bound|
|Ask61|61c|3.647–8.537|14.590|Conditional bound|
|Ask63|63c|1.601–8.943|160 over2 records|Conditional bound|

Two useful ordering results are computed in maker3_quote_checks.json:

1. The second56c bid's earliest estimated post is after the first56c bid's
   fill upper bound1.755s. This is a same-price new-order/replenishment
   candidate after a fill, not proof the first order was exhausted or pulled.
2. Ask58's earliest estimated post5.239s is after the last56c bid fill's upper
   bound4.201s. This supports an observed buy56→later quote58→sell58 chain,
   conditional on the inherited posting bounds. A fill-triggered motive is
   not established; the market also moved.

Ask61 and ask63 have overlapping, broad posting windows. Their fill order does
not prove a58→61→63 sequence of cancel/reprice commands. They may have been
resting at several prices, or entered sequentially. A simultaneous7c quoted
spread from bid56/ask63 cannot be claimed. The61c whole-level withdrawal from
ask61_report.md remains unassigned to a particular maker or order.

## Subsequent visible quotes

- At roughly18–20s the execution path includes bid58 (118.31sh; post estimate
  refused) and ask60 (202.88sh; weaker posting window11.838–18.765s).
- From22–30s there are distinct bid hashes at54,52,53,54c. The52c hash fills
  three times, with settlement bounds24.792–29.956s; its posting estimate is
  refused. Repeated fills establish same-hash reuse, not uninterrupted rest.
- Ask57 posts within27.849–31.352s and trades47.15sh; ask58 within33.550–36.977s
  trades10.25sh. Another ask51 within37.786–38.039s trades3.17sh (weaker case).
- Later bid37 trades95.24sh and bid38 trades65.07sh. The37c posting window
  0.034–46.536s is too broad to call it a long-resting order. Ask35 is estimated
  within52.645–54.465s and trades171.52sh; this part of the execution path cuts
  through earlier37–38c purchases at a loss, as the prior turnover report shows.

The visible pattern includes repeated fills at fixed prices, same-price new
orders, several price levels across a short period, and substantial downward
price adjustment as the market falls. It is not established that the maker
uses a fixed ladder, a constant quoted spread, a fixed delay after fills, or
holds every order until a profitable exit. These are action observations,
not a recovered condition-to-action rule.

None of these17 hashes has original signed-size evidence in the existing
146-order packet. Cumulative fills give lower bounds, not exact original
sizes: e.g.>=168.409sh for the second56c bid and>=202.880sh for ask60. Do not
replace these with an assumed220-share or100-share order size.

## Market context and uncertainty

maker3_quotes.json includes historical best-bid/ask ranges across every usable
native frame in each enabled posting window (plus the as-of left endpoint).
For the second56c bid the best bid ranges56–61c; for ask63 the best ask ranges
56–63c. Thus the quote may have been posted at the touch or behind it. These
are sampled historical venue states, not identified client information or a
continuous-time guarantee. They cannot establish that an order was deliberately
kept a fixed number of ticks behind the market. No feature uses a selected
midpoint to invent a decision timestamp.

Only one other enabled post window from the displayed market can extend before
60s: bid35 at58.747–65.587s, first filling after60s. Refused later windows remain
unknown; their status counts are retained. Never-filled orders are outside the
observable population. No claim that these17 orders were all the maker posted.

## Checks and delivery

Independent canonical trade-leg reconciliation:51 legs /17 hashes, same signer,
market, normalized side/price/quantity/timestamp; original posting CSV joined
independently. Seven corruptions refused (price, quantity, time, dropped row,
promoted refused interval, invented cancel, invented original size). No quote
or maker strategy was simulated. The two visible ordering statements are
computed, not handwritten verdicts.

The interactive quote map uses separate posting-uncertainty bars and fill-upper
markers, never a continuous lifetime line. All17 selectable rows remain
visible. Desktop/mobile inspection and actual selector tests pass; invalid
estimates stay unknown, no JS errors or horizontal overflow. The data, source
tables and full hashes are retained. UI output is a projection of checked data.

One next useful diagnostic on this same example: compare our initial50/51c ask
commands to the source book and Binance observations actually received before
their0.448s simulated sweep. This isolates whether our first bad cycle started
with stale price information, a posting rule, or a failed timely cancel. The
maker's later profitable fills alone cannot identify which of those differs.

All runs are charged to the existing180CPU-second dashboard allowance, using
run_light/codex. maker3_quote_resources.json records the incremental charge;
no strategy source or another session's state is changed, no push.
