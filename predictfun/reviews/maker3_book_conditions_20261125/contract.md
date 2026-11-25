# MAKER3-BOOK-CONDITIONS v1 — 2026-11-25 (declared before any fill economics were read)

USER (2026-11-25): "volatile" means the **Predict.fun book is thin and the BBO spread is large, especially approaching window end; I can see
quite some fills of maker 3 near the end of the market." Check the earlier analysis first (maker3_spread_risk_20261124 /
maker3_regime_spread_20261125), then a detailed analysis. Saved data only, consumed Oct 24–27, no replay / model / new days / validation.
Exploratory P-2026-005 mode; descriptive attribution, no p-values, no intervals (4 consumed days < 5).

## Check of the earlier analysis (findings that motivated this packet)

- Both earlier packets use the **19 complete BOOK markets** only (17 with Maker3 fills, Oct 24–26 + 2 inactive Oct 27): 1.8 % of Maker3's
  934 active markets. Their population cannot carry a general claim about thin/wide books.
- **Thinness was never measured** — only displayed spread (and, in the regime packet, Predict mid movement, which is not the USER's notion).
- The earlier "wide" result is stated on FIFO pairs with *both* legs wide; the per-fill contribution of wide/thin books was not tabulated.
- Clock sensitivity is large (3789484 late wide-book shares 308.58 pre-window vs 1,551.93 upper): the book changes inside the ~2 s
  execution window.

## Stage 1 (outcome-blind, done before this contract): market-wide book on a 1 s grid

`grid.py` → `data/.../maker3_book_conditions_20261125/grid.parquet`: 1,151 markets × 300 s (1 boundary-censored excluded). Thinner-side touch
(min of best-bid shares, best-ask shares) over all ok seconds: q25 = 43.7, median 139.3, q75 = 306.5 shares. One-sided seconds rise from 0 %
before 120 s to 76 % in 295–300 s; spread > 4 ¢ from 0.2–0.5 % to 35 %. One-sided books are pinned (the present side is at 1 ¢ or 99 ¢).

## Population

`econ/fills_valued.parquet` rows on Oct 24–27 with `status == ok`, market not boundary-censored, `st_res` ok or tie (V defined).
Primary unit = a **maker leg** (passive fill) of an address; Maker3's taker legs are a separate table. Self-trades are kept and flagged.
Maker3 = ex-ante Oct 23 top-10-by-shares rank 3, `0x57FbC891E89184DA071AE9D97E745531cd7F718a`. Comparison groups for maker legs:
**OTHER_TOP10** (the other nine ex-ante top-10-by-shares addresses) and **FIELD** (every other maker address). Every excluded row is counted
by status.

## Clocks and time

- Settlement stamp `settle_ms` (UUIDv7 creation) is an **upper bound** on the match. On lifecycle (BOOK) rows `win_ms = settle_ms − lo` is
  ≤ 2,489 ms on Oct 24 (median 1,607). Primary book clock **P = settle_ms − 2,500 ms** (pre-trade for every observed window; the build counts
  BOOK rows where P > lo, which must be reported). Sensitivity clock **U = settle_ms** (may already show the trade).
- Time coordinate = `t_rel_s` (stamp − window start, s). Buckets: pre (<0), 000–060, 060–120, 120–180, 180–240, 240–270, 270–285,
  285–295, 295–300, post-stamp (≥300: matched in the final ~2.5 s, stamped after the close). The match is 0–2.5 s before the stamp.

## Book state at a clock (the econ MID rule)

Last `tob` frame (ordered upd_ms, recv_ns) with upd_ms ≤ clock: `no_frame`; `frame_status` (bad_frame / mid_mismatch / out_of_order /
off_grid); `one_sided` (one side empty); `crossed`; `stale` (age > 1,000 ms); else ok. Classes (thresholds fixed from stage 1 only):

| Class | Definition |
|---|---|
| CALM | two-sided ok, spread ≤ 1 ¢, thinner-side touch > 43.7 shares |
| THIN | two-sided ok, spread ≤ 1 ¢, thinner-side touch ≤ 43.7 shares |
| WIDE | two-sided ok, spread > 1 ¢ (sub-bands (1,2], (2,4], > 4 ¢ reported) |
| ONE_SIDED | one side empty |
| UNKNOWN | no_frame / frame_status / crossed / stale |

"Volatile book" in the USER's sense = THIN ∪ WIDE (two-sided); ONE_SIDED is reported separately, never merged silently. Moneyness by the
pre-trade mid (for ONE_SIDED by the present side's price): EXTREME = ≤ 10 ¢ or ≥ 90 ¢, NEAR otherwise. The thinner-side touch includes
any Maker3 order resting there; it is a displayed-book state, not other makers' liquidity.

## Economics per leg (YES space, s = +1 buys YES)

From `fills_valued`: q, p, V, `gross_res` = s·q·(V − p), `fee_res`, `net_res`, markouts `v_1s` / `v_10s` / `v_60s` / `v_end` with their
statuses (NaN unless ok). New: `edge_P` = s·q·(mid_P − p) (instantaneous edge vs the pre-trade mid, two-sided ok books only) and the identity
`gross_res = edge_P + s·q·(V − mid_P)` (the second term = outcome/adverse component). Per-share values in ¢ = 100·sum(value)/sum(q over the
rows where the value is defined). Sums of `net_res` are additive across cells and recover each address-market total of
`econ/maker_market.parquet` (checked).

## Inventory (Maker3 only)

Signed YES position before each Maker3 leg within its market, all roles, in the econ FIFO order (settle_ms, sid, px_rank, mk_idx, maker
before taker); self-trades excluded from the position (as in econ lots). Leg class: OPEN (position 0), ADD (same sign), REDUCE (opposite
sign, q ≤ |position|), FLIP (opposite sign, q > |position|). Position at the 240 / 270 / 285 s stamps and at END are reported per market.

## Market level

Per market: share of the 240–299 s grid seconds that are CALM / THIN / WIDE / ONE_SIDED / UNKNOWN; "decided by 270 s" = the 270 s grid state
ONE_SIDED or EXTREME. Maker3's whole-market `net_res`, `rt_pnl`, `resid_res` from `maker_market` by late-book class:
CALM_LATE (≥ 50 % CALM), VOLATILE_LATE (≥ 50 % THIN ∪ WIDE), PINNED_LATE (≥ 50 % ONE_SIDED), MIXED (otherwise).

## Tables and computed readings (all four days and each day; Maker3 vs OTHER_TOP10 vs FIELD)

1. Maker3 maker shares / legs / markets by time bucket × book class (P and U), with Maker3's share of all maker shares in each cell.
2. Economics by book class, by time bucket × book class, by moneyness, spread sub-band; Maker3 taker legs separately.
3. Inventory class × book class × late/early for Maker3.
4. Market-level classes.
Readings evaluated in code: (a) Maker3's share of its maker volume in THIN ∪ WIDE at P, overall and ≥ 270 s; (b) sign and day-consistency
(same sign on every day with ≥ 30 legs) of Maker3's net ¢/share in THIN ∪ WIDE vs CALM; (c) top-3-market fraction of the absolute net
contributions in each reported cell (> 0.5 = concentrated); (d) Maker3 minus FIELD net ¢/share in the same cell; (e) whether the P-clock
reading's sign survives the U clock. A cell with < 3 markets or < 30 legs is THIN-SUPPORT and printed but never read.
