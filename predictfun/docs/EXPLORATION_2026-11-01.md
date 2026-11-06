# Predict.fun exploration: sources, signals, plain baseline

> **2026-11-05 15:30Z - PRE-FIX REPLAY NUMBERS ARE VOID.** Four review rounds on 11-05 found engine defects in pf_step4.py /
> pf_step4_live.py / pf_step4_ladder.py / pf_ladder.py (fill priority among our own orders, ascending-price sweep order, post-only
> rejection, the strategy acting on fills before their notification, request reservation by hidden fills). By user decision every
> replay output produced before the fixed engines (round i: live 742637b9, ladder c682229e) was DELETED; sections are tagged VOID /
> PARTLY VOID below. Market-data, model and probe findings stand. Valid replay numbers start at section 55 (round i).

As of 2026-11-01T03:32Z. Sample: **10-23, 10-24, 10-25, 864 BTC 5m windows** (all with
books, outcome and start price; Up 425 / Down 439). **Exploratory** by user decision: no
held-out split, no freeze; descriptive results on days that are now read. G = 3 days, so
no day-cluster interval is quoted; window-clustered SEs where given.

## 1. Sources and fill model

| Finding | Evidence |
| --- | --- |
| Maker leg `amount` = filled shares | sum(maker amounts) = `amountFilled` on 29,718 / 29,719 tx (10-23 14Z) |
| `settlementsPending` value = USD at the YES price (shares x p_yes), on either YES side | exact-value match 32% of candidates vs ~1% for shares (10-23) |
| `executedAt` = true time **rounded** to the second **(superseded, §12: = settlement block time floored to the second)** | matched offsets median +26..+36 ms, IQR -240..+300 ms, 83% within +-500 ms (3/3 days) |
| ms-timed trades | 52-58% of live maker-leg volume matched to a WS increment (3/3 days) |
| REST first seen | p50 6.1-6.5 s, p95 11-13 s after `executedAt` -> offline labels only |

Fill model used in step 3: REST legs give volumes; timed legs use the WS time; the untimed rest is
placed at S-0.5 s / S / S+0.5 s (early / mid / late). FIFO queue behind the displayed size at our
price, advanced only by trades at that price, capped by the displayed size; a trade at a worse
price on our side sweeps us up to its size. Post-only refusal at activation. No own-impact.

## 2. Book structure (per window, median; p10-p90 in the report)

| Item | Value |
| --- | --- |
| Book updates | 17.6/s; longest gap 0.66 s; 0 windows with a gap > 5 s |
| 1-tick spread | 87% of time; by minute 96/96/95/85/**44%** |
| Touch size | ~240 shares each side |
| Gap behind touch (>= 2 ticks, D1's gap condition) | 3.4% of time per side |
| One-sided / empty | 5.8% (p90 29%, late window) |
| Live volume | ~35k shares, ~120 makers, top maker 11% |

## 3. Signals D1 relies on

Predict YES mid change (cents) after a Binance futures move, receipt-time grid, live t < 240 s:

| Binance move | Time active | +0.25 s | +1 s | +5 s |
| --- | --- | --- | --- | --- |
| 1 s <= -0.5 bps | 3.1-6.6% | -0.71..-0.99 | -1.41..-1.86 | -1.37..-1.90 |
| 1 s flat | 87-94% | ~0 | ~0 | ~0 |
| 0.25 s <= -0.5 bps | 1.0-2.2% | -1.58..-2.26 | -3.34..-4.71 | -3.18..-4.83 |

Up moves symmetric. Half spread ~0.5c. **D1's adverse-side premise holds on Predict, 3/3 days.**
Repricing lag after >= 1 bps / 1 s moves (median share of theory): 0 at 0.1 s, ~0.47 at 0.3 s,
~0.71 at 0.5 s, ~0.80 at 1 s. The tick leg's LEVEL (p_bn with tau = 300 - t, K = start price,
basis-corrected) is worse calibrated than the Predict mid at every checkpoint (t=290: Brier
0.075-0.091 vs 0.049-0.073), and its gap to the mid predicts ~0.1c per 4c. Relay - futures basis
+4.4..+4.6 bps median (p05-p95 3.4-5.6).

## 4. Plain baseline (no D1 rules), 5 shares a side, whole window, no limits
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Gross USD over the 3 days (early ... late timing bound; mid in brackets); 0/3 days positive in every cell.

| Policy | Delay 272 ms (this host) | Delay 50 ms (hypothetical) |
| --- | --- | --- |
| JOIN (touch) | -16.2k ... -10.6k (-13.4k, -1.57 c/sh) | -14.0k ... -7.0k (-10.1k, -0.98 c/sh) |
| MICRO (D1 anchor) | -13.8k ... -8.1k (-10.8k, -2.14 c/sh) | -10.3k ... -4.3k (-7.0k, -1.55 c/sh) |

Delay = 2 x median(receipt - source) on this host (136 ms one way, includes clock offset).
Unverified 25% rebate would add $0.6-1.5k. Liquidity rewards not modelled (5-share quotes are below
the 100-share threshold).

Where it loses (JOIN/mid, 272 ms): swept-through fills 52% of volume at -1.7..-2.1 c/sh at 1 s;
fills at our price 48% at -0.57 (timed) / +0.02 (untimed) c/sh.

Against D1's guard condition (fills joined to the Binance 1 s state at fill time; descriptive, not a
guard replay):

| | Volume share | Settle c/sh | Share of loss |
| --- | --- | --- | --- |
| Against (BUY with r1 <= -0.5 / SELL with >= +0.5) | 13-20% | -3.3..-4.3 | 34-51% |
| Neutral | 74-84% | -0.7..-1.8 | 52-67% |
| Favourable | 4-6% | +0.4..+0.8 | -1..-4% |

## 5. Reading
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


- Data supports a credible replay: complete windows, ms book, REST volumes, half the volume ms-timed.
- Plain quoting loses ~1-2 c/share, mainly to stale quotes swept after others pull; speed helps but
  does not fix it.
- D1's Binance guard addresses a third to a half of the loss; the neutral-fill loss remains, so the
  guard alone is unlikely to make it positive on this evidence.

## Reproduce

```bash
R=predictfun/research/run_light.sh
for d in 20261023 20261024 20261025; do
  $R s1_$d -- python3 -m predictfun.research.pf_step1 $d
  $R t_$d  -- python3 -m predictfun.research.pf_step1_timing $d
  $R s2_$d -- python3 -m predictfun.research.pf_step2 $d
  $R s3_$d -- python3 -m predictfun.research.pf_step3 $d        # add "50" for the 50 ms delay
done
$R rep -- python3 -m predictfun.research.pf_step3_report - 20261023 20261024 20261025
$R att -- python3 -m predictfun.research.pf_step3_guard_attr - 20261023 20261024 20261025
```

Outputs under `data/predict_fun/research/step{1,2,3}/`. Per-run peak memory 0.5-1.3 GiB, disk +442 MB.

## 6. Tokyo latency (user, 2026-11-01: Tokyo host, ~20 ms to Predict)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


This US host: Binance receipt - E 77 ms median (floor 71), Predict receipt - updateTimestampMs 130-136 ms (floor 90);
TCP connect RTT ~148 ms to both `fstream.binance.com` and `ws.predict.fun` (~74 ms one way each). Predict timestamps arrive
~56 ms later than Binance's over the same path: Predict-side publication delay or a clock offset (not separable here).

| Scenario | Predict book known | Binance known | Order/cancel active |
| --- | --- | --- | --- |
| tokyo20 (main, user figure) | updateTimestampMs + 20 ms | E + 5 ms | decision + 20 ms |
| tape (stress) | this host's receipt - 72 ms (~58 ms median) | receipt - 72 ms | decision + 20 ms |

On the Tokyo clock Predict still reprices ~0.3 s behind Binance (median share of a >= 1 bps move: 0 at 0.1 s, ~0.52 at
0.3 s, ~0.71 at 0.5 s). Parity: the step-4 event loop with no rules reproduces step 3 at 40 ms within 0.3-8%.

## 7. D1 rules one at a time and stacked (tokyo20, v1 timing, mid; 3-day gross USD, every delta 3/3 days > 0)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


| Rule alone | JOIN (base -9.2k) | MICRO (base -6.1k) |
| --- | --- | --- |
| G1 Binance 1 s guard | +4.9k | +3.1k |
| G2 Binance 250 ms leg | +4.0k | +2.7k |
| G3 gap guard | +3.1k | +0.5k |
| G4 no off-touch | n/a | +4.2k |
| G5 stop at 240 s | +1.8k | +1.7k |
| G6 inventory 10/25 | +3.7k | +3.2k |

Full stack (G4, G2, G6, G5, G3, G1): MICRO +6$ (early -1,060 / late +747), JOIN -2,377$; tape latency MICRO -453$.

## 8. Narrowing the untimed-trade bracket
> **PARTLY VOID (2026-11-05):** market-data numbers stand; the MICRO full-stack $ per timing model came from a pre-fix replay and is void (outputs deleted).


Book-drop estimator E2 (frame in +-600 ms whose size drop at the trade's side/price best matches its volume), calibrated on
WS-timed trades: median |err| 0-49 ms vs 249-255 ms for "S", within 100 ms 57-62% vs 20%, p90 551-589 ms; adopted by the
declared rule on 3/3 days. E4 (E2 only when the drop matches the volume within 10%): 47-52% coverage, p90 278-340 ms,
failed its declared p90 <= 250 ms rule. MICRO full stack, three timing models: E2 -5$ (-45..+36), v1 +6$, E4 +1$ -> break-even.

## 9. Prediction model (dev days, leave-one-day-out)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; every replay $ of the model guard came from a pre-fix replay and is void (outputs deleted).


LightGBM (pinned, `pf_model_spec.py`) on a 50 ms Tokyo-clock grid, t < 240 s, 32 causal features (Binance returns 25 ms-3 s,
book imbalance, aggressor flow; Predict spread, microprice deviation, L1/L3 imbalance, sizes, frame age/rate, mid changes,
pending-settlement flow; time, sigma, basis, Binance-probability gap). Out-of-day:

| Target | Binance only | Predict only | All |
| --- | --- | --- | --- |
| mid move 250 ms | R2 0.23-0.31 | 0.17-0.27 | 0.38-0.48 (corr 0.64-0.70) |
| mid move 1 s | 0.08-0.11 | 0.08-0.12 | 0.14-0.18 (corr 0.39-0.45) |

At ~9% of time flagged the realised 1 s move is 2.2-2.5c vs 1.0-1.2c under D1's G1/G2 flags. Top gain: Predict L1-3
imbalance 22-25%, Binance 250 ms return 15%.

Model as a guard (withhold the side when predicted 1 s move <= -th / >= +th), MICRO, E2 timing, tokyo20, 3-day gross:

| Arm | early / mid / late | days > 0 (mid) | c/sh |
| --- | --- | --- | --- |
| D1 stack | -45 / -5 / +36 | 1/3 | -0.01 |
| D1 stack + model th 0.5c | +186 / +237 / +290 | 3/3 | +0.24 |
| D1 stack + model th 1.0c | +71 / +118 / +168 | 2/3 | +0.11 |
| model instead of G1/G2, th 0.5c | +145 / +189 / +238 | 3/3 | +0.17 |

th 0.5 was chosen from two values on these days; confirmation C1 (declared in STATUS.yml) runs on 10-26, 10-27, 10-31.

## 10. Confirmation C1 (declared in STATUS.yml before the days were read)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


One model fit on 10-23..25, applied unchanged to **10-26, 10-27, 10-31** (10-28..30 excluded: Binance tape sealed for
P-2026-003). Candidate MICRO + D1 stack + model guard th 0.5c, tokyo20, E2 timing, 5 shares.

| Arm | 10-26 | 10-27 | 10-31 | sum early / mid / late |
| --- | --- | --- | --- | --- |
| D1 stack | +94 | +24 | -202 | -132 / -84 / -61 |
| **D1 stack + model th 0.5c** | +196 | +110 | -115 | +148 / **+191** / +226 |
| model instead of G1/G2, th 0.5c | +130 | +83 | -104 | +76 / +108 / +142 |
| D1 rules without Binance guards | -206 | -378 | -398 | -1,034 / -983 / -929 |

**Pass rule met:** sum +190.67$ > 0 and candidate - D1 positive on 3/3 days (+101.7 / +86.1 / +87.0). Out-of-sample R2 for
the 1 s move 0.189 / 0.191 / 0.121. Rebate (unverified) +115$. Scale is small: ~70k shares, +0.27 c/share, ~0.22$/window.

Data note, 10-31: 29% of trade volume has neither a WS time nor a book-drop time (~1% on other days). Schema, maker-amount
semantics and WS-REST offsets are unchanged; 68% of those trades sit inside the visible top 5 with no size drop within
+-600 ms (likely refilled within one frame). The day's early/late range is only +-30$ per arm.

Limits: modelled fills (FIFO behind displayed size, no own impact, no real-fill calibration); Tokyo latency is the user's
figure, not measured; 5-share size; G = 3 fresh days (no interval quoted); th 0.5 picked from two values on dev days.

## 11. WS pending settlements vs REST trades (10-23, as of 2026-11-01T09:39Z)

The WS book carries no trade list. `settlementsPending` is one USD value per (book side, price): matched trades not yet
settled on-chain (no duplicate prices in 60k frames). A frame-to-frame change at a price = new trades - settlements there.
Frames every 51 ms median (p99 220 ms); 0 out of order. All figures live t 0..300 s, USD at the YES price
(`pf_ws_pending_diag.py`, `pf_ws_pending_offsets.py`).

| Item | Value |
| --- | --- |
| REST maker-leg volume | $6.06M |
| WS increases / decreases | $6.98M (115% of REST) / $6.82M |
| Increase -> next decrease at the same level | p25 145 ms, p50 376 ms, p90 1.8 s; 18% within 100 ms |
| Trading seconds with trades at 2+ prices | 74% |

REST trades seen in WS (per REST cell: WS increases at that price, either side, seconds S-1..S+1 vs REST in S-1..S+1):

| REST USD share | Value |
| --- | --- |
| Fully visible (>= 95%) | 77.5% |
| Partly visible | 22.2% (21.8 pts with the level falling in the same span = netted against settlements) |
| Not visible | 0.27% |

So the ms-timed share (50-58%) is limited by netting and merging in the aggregate, not by missing trades: an exact
one-to-one value match is needed to pin a ms time.

WS increases without a REST trade at that price within +-1 s ("orphans"): $572k = 8.2% of WS increases, 91% on the bid side,
spread over windows (top 5% of windows hold 21%; per-window orphan/REST p50 6.5%, p90 20%). $508k coincide with REST trades
in the same market at another price (nearest-price offset +-1 tick 28%, >= 6 ticks 34%; 29% at prices within 4c of 0/100);
$46k (0.8% of REST volume) have no REST trade in that market within +-1 s. At the taker's limit price: $18k (that
explanation rejected). Orphans rise with REST indexing lag (pooled orphan/REST 6.4% -> 11.2% from the lowest to highest
window-lag quartile, Spearman 0.18; confounded with activity). REST first-seen lag > 50 s for 1.0% of legs, max 615 s.

Reading: REST trades are corroborated (99.7% appear in WS). Whether REST misses trades is open: if every orphan were a
missing trade, the gap would be <= 9.4% of volume; the strict no-trade-nearby part is 0.8%. Not settled here; an on-chain
fill reconciliation (transaction hashes are in REST) would settle it.

Same-share test (09:49Z): only ~20% of orphan USD carries the share count of a REST leg or cell in that market within +-1 s at
another price (+-1 tick 4%, >= 2 ticks 16%); 80% match no REST share count. Worked examples (one trade -> one increase;
merged; netted; an orphan with the same shares booked 1c lower): `pf_ws_pending_examples.py 20261023 14`.

Example D corrected (10:05Z): at 10-23 14:23:36.850 (market 3668233) the REST fill (taker sold Up, 114.334 sh to a maker bid at
0.99) shows at 99c in the book (bid 395.97 -> 281.64 sh) and in pending (bids 0.99 +113.1907 = REST exactly), AND as an extra
pending entry asks 0.98 +112.0473 (same shares at 98c; the book ask side was empty, 98c bid size unchanged); both entries
cleared together at 14:23:38.553. So that orphan is a second booking of a REST trade, not a missing one. Day test
(`pf_ws_pending_taker.py`): 24% of orphan USD equals a REST taker fill's amountFilled within +-1 s (offsets to the
execution price spread -3..+3+ ticks; pending side = taker YES side 63%); REST's taker price equalled the execution price on
every fill in that test, so the taker's own limit is not observable in REST. 76% of orphan USD stays unexplained.

Label note: step 1's `recon.live_ws_inc_shares` is USD (pending is valued at the YES price), not shares; it is not
comparable with `live_rest_shares`.

## 12. On-chain reconciliation (BNB Chain, 10-23, as of 2026-11-01T10:51Z)

Predict settles on BNB Chain: operator EOAs send `0x92895911` calls to 0xdcffeb0c..., which calls the CTF exchange
0x8bc070be... (`OrderFilled` per order, `OrdersMatched` per taker order; collateral BSC-USD, ERC-1155 outcome tokens).
Public RPC (bsc-dataseed.bnbchain.org) serves blocks and receipts but no historical `eth_getLogs`, so every block of a
slice is scanned for txs to the two contracts (`pf_chain_fetch.py`), then reconciled with REST (`pf_chain_recon.py`).

Slices: 14:00-14:15 (3 windows) plus five windows picked as the likeliest to show REST loss: 15:25 and 16:55 (REST poll
retired with backlog), 12:00 (orphan/REST 100%), 08:45 and 13:20 (most orphan USD).

| Slice | On-chain maker fills | USD | In REST | Chain only | REST only | Shares equal | Taker orders = REST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14:00-14:15 | 9,205 | 100,075 | 9,205 | 0 | 0 | 100% | 6,653 / 6,653 |
| 15:25 | 4,684 | 59,244 | 4,684 | 0 | 0 | 100% | 3,162 / 3,162 |
| 16:55 | 3,429 | 24,641 | 3,429 | 0 | 0 | 100% | 2,338 / 2,338 |
| 12:00 | 3,167 | 22,806 | 3,167 | 0 | 0 | 100% | 2,263 / 2,263 |
| 08:45 | 3,426 | 19,490 | 3,426 | 0 | 0 | 100% | 2,547 / 2,547 |
| 13:20 | 1,730 | 26,905 | 1,730 | 0 | 0 | 100% | 1,133 / 1,133 |

(USD includes pre-open trading of listed markets.) **REST is complete and exact on all six slices.** Failed settlement txs
(status 0): 18-85 per slice, none listed by REST.

Timing: `executedAt` = floor(block timestamp) for 100% of fills (= round for 52%), so REST time is the settlement block, not
the match. WS pending appears before the block by p25 321 / p50 469 / p75 613 / p95 887 ms (14:00 slice, n 4,986; other
slices p50 409-497 ms). The earlier "+26..36 ms vs the nearest second" was floor's +0.5 s average minus this lag.

Orphans (`pf_chain_calldata.py`, `pf_chain_failed.py`): the settlement calldata carries the signed orders. 66% of takers
have a limit worse than their fill (slippage room up to 10+ ticks); makers 98% at their fill price. WS pending lists the
taker's order at its signed LIMIT price (filled shares x limit) beside the maker entry; REST reports only the fill price.
Example D: taker limit 112.04733 / 114.334 = 0.98, pending asks 0.98 = 112.0473. 14:00 slice orphans ($2,347 = 5.3% of WS
increases): 55% one taker-at-limit entry, 17% two or more merged, 21% smaller than the takers-at-limit there (netted),
1% larger, 5.5% no taker limit at that price; failed txs 0.3%. Orphans are WS bookkeeping, not missing trades.

Consequences: replay fill volumes stand (REST complete). Timed fills use the WS pending time (match time); untimed ones
the E2 book-drop estimate; the S +-0.5 s bracket was centred correctly by coincidence. Block ms and tx order now give every
fill a settlement anchor (match ~ block - 0.47 s, IQR ~0.3 s), usable for the 10-31 untimed share.
Not covered: other days; whether the taker-at-limit entry is always listed (WS increases are 115% of REST, not more).

## 13. Round O1: order size and strategy/model variants (declared 11:46Z; as of 2026-11-01T13:26Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


tokyo20, E2 timing, MICRO, gross. Selection on 10-23..25; 10-26/27/31 are a second look (seen by C1), not a confirmation.
Multiplicity: 15 variants + 5 sizes. Parity: size-5 arms reproduce C1 exactly. `pf_o1_report.py`, `pf_o1_combo_report.py`.

**Size** (D1 stack + M05, C1 preds; caps scale with size: soft 2x, hard 5x):

| Size | 10-23..25 | 10-26/27/31 | +1 s mark (6 days) | Shares (6 days) | Inventory p95 |
| --- | --- | --- | --- | --- | --- |
| 5 | +237 (3/3) | +191 (2/3) | 0.25 c/sh | 171k | 14 |
| 10 | +396 (3/3) | +262 (2/3) | | | 28 |
| 25 | +1,137 (3/3) | +780 (2/3) | 0.23 c/sh | 736k | 68 |
| 50 | +2,184 (3/3) | +1,819 (2/3) | | | 137 |
| 100 | +5,031 (3/3) | +3,899 (2/3) | 0.26 c/sh | 2.17M | 272 |

Volume grows sub-linearly (x12.7 shares for x20 size). The +1 s edge per share is flat; the higher settlement c/share at
large size comes from carried inventory (first-minute fills settle +0.8..+1.0 c/sh), i.e. directional exposure over 6 days,
with larger losing days (10-31: -2,221 at 100 sh). No own impact is modelled: a 100-share quote is ~40% of the median touch,
so large sizes are optimistic. Declared pick (largest dev gross, >= 2/3 dev days > 0): 100.

**Model:** LODO over 6 days (5 training days) raises 1 s R2 slightly (0.193/0.159/0.146/0.191/0.194/0.124 vs 0.14-0.18 and
0.189/0.191/0.121) but not P&L: D1 + M05 dev +212.5 vs +237.3, second look +188.5 vs +190.7. Peak RSS 2,373 MiB (guard 2,048 is
checked between steps only; within the 4 GiB cap) -> build folds one at a time before any rerun.

**Variants** (preds_lodo6, size 5; delta vs D1 + M05 on 10-23..25, days better):

| Variant | 10-23..25 | Delta | 10-26/27/31 | Adopted |
| --- | --- | --- | --- | --- |
| D1 + M05 (base) | +212.5 | | +188.5 | |
| theta 0.25 | +346.5 | +134.0 (3/3) | +191.6 | yes |
| theta 0.75 / 1.0 | +167.7 / +131.9 | -44.9 / -80.6 | +137.2 / +123.3 | |
| + 250 ms guard 0.25 / 0.5 | +216.4 / +204.3 | +3.9 (1/3) / -8.2 | +163.5 / +182.1 | no |
| 250 ms guard instead of 1 s | +154.2 | -58.4 | +148.4 | |
| stop 270 / none | +85.0 / -1.6 | -127.6 / -214.1 | +237.2 / +176.8 | no |
| drop G1 | +186.8 | -25.7 (1/3) | +147.0 | no |
| drop G2 | +216.9 | +4.4 (2/3) | +180.1 | yes |
| drop G3 | +234.2 | +21.7 (3/3) | +186.9 | yes |
| drop G4 | +232.6 | +20.0 (2/3) | +60.8 | yes |
| drop G6 (inventory caps) | +291.6 | +79.1 (2/3) | +398.7 | yes |

**Declared combination** (MICRO + G1 + G5 + M025; G2/G3/G4/G6 dropped): 10-23..25 **+496.7** (+145/+105/+246, 3/3; early/late
+458/+528), delta +284.2 (3/3); 10-26/27/31 **+679.8** (+522/+306/-148; early/late +620/+678), delta +491.3 (2/3). +1 s mark
0.30 / 0.25 c/sh vs base 0.29 / 0.19. Inventory p95 66 sh vs 14 (no caps). Post-hoc (not declared): D1 + M025 at 25 sh +1,751 /
+864, at 100 sh +5,892 / +4,400.

A first combination run was invalid: the new `custom` armset did not load predictions, so model arms ran as D1 alone (caught:
theta 0.5 and 0.25 identical and equal to D1 alone). Fixed; `pf_step4.py` now refuses model arms without predictions
(falsifier: an empty prediction file is refused, the real one runs). Results above are from the rerun.

## 14. Round O2: adaptive size (declared 14:06Z; as of 2026-11-01T14:45Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


**Loop fix first.** After a full fill the old event loop reset the side's intent even when a replacement was in flight; that
replacement could then stay live while the policy wanted no quote. The fixed loop cancels it. `PF_LEGACY_INTENT=1`
reproduces the old 10-25 numbers exactly. Restated (fixed loop, earlier in brackets): C1 candidate 10-26/27/31 **+191.99**
[+190.67] (early/late +148.2/+227.3), D1 alone -86.15 [-84.17] -> candidate - D1 = +100.5/+89.8/+87.8, C1 still passes;
O1 combo +484.6 / +688.4 [+496.7 / +679.8]; D1+M025 +351.2 / +191.3 [+346.5 / +191.6]. Conclusions unchanged.

Sizes 5..25 per side at each decision; an order is amended down in place or cancel/replaced up (queue reset). preds_lodo6,
tokyo20, E2, mid. 10-23..25 select; 10-26/27/31 second look. +1 s = mark 1 s after the fill (c/share).

| Base | Size rule | 10-23..25 | vs fixed 25 | +1 s c/sh | 10-26/27/31 | +1 s c/sh | Inventory p95 | Adopted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 combo | fixed 5 | +485 | | 0.30 | +688 (+526/+307/-145) | 0.25 | 66 | |
| B1 combo | fixed 25 | +1,884 | | 0.28 | +3,141 (+2,530/+1,314/-704) | 0.25 | 276 | |
| B1 combo | A1 model >= 0 | +1,822 | -61 (1/3) | 0.39 | +3,804 (+2,146/+1,553/+104) | 0.41 | 276 | |
| B1 combo | A1 model >= 0.25 | +1,779 | -105 (2/3) | 0.57 | +3,140 (+1,824/+1,260/+56) | 0.67 | 227 | |
| B1 combo | A3 depth 10% / 25% | +990 / +696 | -894 / -1,188 | 0.11 / 0.17 | +1,312 / +1,797 | 0.02 / 0.11 | 220-250 | |
| B1 combo | A4 inventory | +1,062 | -822 (0/3) | 0.31 | +1,106 | 0.25 | 41 | |
| B1 combo | A6 first 120 s | +1,991 | +107 (2/3) | 0.31 | +3,241 (+2,485/+1,000/-244) | 0.30 | 216 | yes |
| B2 D1+M025 | fixed 5 | +351 | | 0.34 | +191 | 0.27 | 14 | |
| B2 D1+M025 | fixed 25 | +1,764 | | 0.32 | +862 (+831/+532/-500) | 0.25 | 71 | |
| B2 D1+M025 | A1 model >= 0 | +2,020 | +255 (2/3) | 0.38 | +1,297 (+752/+811/-265) | 0.33 | 70 | yes |
| B2 D1+M025 | A1 model >= 0.25 | +1,708 | -56 (1/3) | 0.58 | +1,601 (+882/+879/-160) | 0.60 | 69 | |
| B2 D1+M025 | A3 depth 10% / 25% | +458 / +1,200 | -1,306 / -564 | 0.12 / 0.20 | -10 / -39 | 0.00 / 0.12 | 68 | |
| B2 D1+M025 | A4 inventory | +1,038 | -727 (0/3) | 0.33 | +890 | 0.25 | 41 | |
| B2 D1+M025 | A6 first 120 s | +1,445 | -319 (0/3) | 0.32 | +924 | 0.28 | 57 | |

Reading: sizing by the model (A1) raises the +1 s edge per share from ~0.25-0.32 to 0.38-0.67 c and, on 10-26/27/31, beats
fixed 25 on both bases (B1 3/3 days positive). Sizing up on deep queues (A3) selects worse fills. Inventory sizing (A4) cuts
inventory p95 to ~41 but halves P&L. B1 has no inventory caps (p95 ~220-276 sh at 25). Settlement c/share exceeds the +1 s mark
on B1 (carried inventory). Not a confirmation (seen days, 12 adaptive arms); own impact not modelled (25 sh ~10% of the touch).

## 15. Code review and fill model v2 (as of 2026-11-01T15:58Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Independent reviewer (code only) plus invariant checks (`pf_check_fills.py`: accounting, no fill after stop + 20 ms,
hard cap, fill <= size; each check fires on a corrupted copy, and the stop check sees real late fills on a no-stop arm).

| Finding | Status | Effect |
| --- | --- | --- |
| Queue double-count: the frame stamped at a trade's own ms already shows it (75% of WS-timed legs, 10-23); v1 capped the queue with it, then subtracted the trade again | fixed (frames strictly before the event) | material |
| Predictions end at 239.95 s: S270 / S300 ran without the model guard after 240 s | O1 stop-time comparison **invalid**; needs a model covering 240-300 s | those arms |
| Queue cap applied only at our-side trades | fixed (every frame; frames with our level beyond the 5 shown do not cap) | conservative before |
| Sweep through our level left the queue ahead | fixed (queue -> 0) | small |
| Own fills known instantly | fixed (20 ms) | small |
| Stale / resumed predictions and A1 threshold crossings fired no decision | fixed | small |
| Reduce-only quotes after the stop not capped at inventory | fixed | arms with G6 and no G5 only |
| In-flight replacement left live after a full fill (found in O2) | fixed in O2 | 0.005% of shares, post-stop fills |
| G6 hard cap (5x) never binds given the soft cap (2x) | documented | none |
| Start price used ~0-1.6 s before it is known (features `p_bn`, `gap`) | not fixed | small |
| Legs matched in the last ~0.5 s before 300 s dropped (exec_s >= ws + 300) | not fixed | arms quoting to 300 s only |
| Book crossing our order without a trade gives no fill | not fixed | ~3 per window per side |
| Hour-boundary Binance ordering, basis overlap double count | not fixed | negligible |
| Methodology: dev-day predictions come from models trained partly on later days (not walk-forward) | noted | |

`PF_FILLMODEL=v1` reproduces O2 exactly on 10-25. v2 outputs: 102 arms, 0 invariant violations.

**Key arms under v2** (10-23..25 | 10-26/27/31; v1 in brackets):

| Arm | 10-23..25 | 10-26/27/31 | +1 s c/sh (2nd look) |
| --- | --- | --- | --- |
| C1 candidate D1 + M05 (C1 preds) | +245.1 [+244.1] | **+158.4** [+192.0] (early/late +109/+178) | 0.17 |
| D1 alone | +33.1 [+2.7] | -5.4 [-86.2] | -0.05 |
| O1 combo G1+G5+M025, 5 sh | +576.4 [+484.6] | +729.4 [+688.4] (early/late +673/+726) | 0.22 |
| D1 + M025, 5 sh | +324.0 [+351.2] | +180.2 [+191.3] | 0.24 |
| D1 + M025, 25 sh | +1,660 [+1,764] | +961 [+862] | 0.22 |
| D1 + M025, A1 5..25 (>= 0) | +1,910 [+2,020] | +1,065 [+1,297] (early/late +970/+1,319) | 0.30 |
| D1 + M025, A1 5..25 (>= 0.25) | +1,523 [+1,708] | +1,005 [+1,601] | 0.57 |
| combo, 25 sh | +2,512 [+1,884] | +3,050 [+3,141] | 0.21 |
| combo, A6 first 120 s | +2,308 [+1,991] | +3,283 [+3,241] | 0.27 |
| combo, A1 5..25 (>= 0) | +2,503 [+1,822] | +3,350 [+3,804] (+1,560/+1,486/+305) | 0.38 |

C1's declared rule under v2: candidate +158.36 > 0 and candidate - D1 = +50.2/+50.2/+63.3 (3/3) -> **still passes**, with about
half the v1 margin (D1 alone improves most under v2). O2's rule under v2: B2 + A1 (>= 0) still adopted (+249 vs fixed 25, 2/3);
B1 + A6 no longer (-204); B1 + A1 (>= 0) misses by -9 on 10-23..25 but is the only arm positive on all three later days.

## 16. Live-deviation tests for the execution port (as of 2026-11-02T09:02Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Request from the pmmmrust session (C2-P frozen as tag pfmm-c2p-tokyo-v1). Six read days 10-23..27, 10-31; preds_lodo6
(C2's model spec, out of day); tokyo20, E2, fill v2, mid. Code `research/pf_step4_live.py`, a copy of pf_step4.py with
live rules as rule tokens (pf_step4.py unchanged, C2 hash 5d9bdf87). Parity on 10-25: options off, or engaged but unable
to fire (XMIN0, XCAP1000000, XL20A), equal the v2 run to the cent. Chain `run_live_dev.sh`, report `pf_live_report.py`
-> `step4/live_tokyo20_custom_e2_preds_lodo6_livedev/report_live.json`.

| P = D1 + M025 + A1 5..25 (S = combo + A1) | 6-day gross | vs sim | shares | +1 s c/sh |
|---|---|---|---|---|
| P as simulated (amend in place) | +2,974 | - | 514k | 0.35 |
| P no amend (cancel + new order at the back) | +2,962 | -12 (2/6) | 512k | 0.35 |
| P no amend, removal lock 500 ms, every cancel | -21,943 | -24,918 (0/6) | 2,002k | -1.08 |
| P no amend, lock 1000 ms, every cancel | -42,606 | -45,580 (0/6) | 3,186k | -1.30 |
| P no amend, lock 500 ms, A1 shrink only | -5,199 | -8,173 (0/6) | 839k | -0.47 |
| P no amend, lock 1000 ms, A1 shrink only | -11,769 | -14,743 (0/6) | 1,123k | -0.81 |
| S as simulated / no amend | +5,853 / +5,931 | +78 (5/6) | 746k | 0.38 |
| S no amend, lock 500 / 1000 ms every cancel | -45,422 / -87,827 | | 3,353k / 5,545k | -1.21 / -1.44 |

- No amend costs nothing measurable. A removal lock is the dominant risk: an order still fillable after its cancel is
  filled at the adverse moments the cancel was for (locked fills -1.5..-1.7 c/sh at +1 s, 72% sweeps; lock 500 ms every
  cancel: 1.55M locked shares, -27.4k$), and with ~5 cancels/s several locked orders rest at once (shares x4-6). Locked
  fills also break G6: |net| inside a window reached 135-212 sh (hard 125; 75 without a lock); S reached 3,102.
- Minimum order 1 USD: 1.8% of the simulated fills / 1.4% of shares came from smaller orders, settling -175$; withholding
  them +87$ (amend) / +105$ (no amend), 4/6 days.
- Requests (place + cancel + amend at decision time, quoting minutes only): P as simulated 290/min mean, p50 294, p99 587,
  max 897; 67% of quoting minutes and 95% of windows exceed 240/min; per second mean 4.8, p99 20, max 53. No amend 313/min
  (73% > 240); no amend + min 1 USD 287/min; S no amend 714/min (96%). The request budget binds most of the time.
- Total-loss cap 150$ over the six days in order: live-like P (no amend + min 1 USD) costs -31.67 (23 windows refused on
  10-23, then free). P with amend: first refusal at window 115 (realised -110.3), last fill window 189, realised stuck at
  -149.96 for the remaining 1,540 windows (-3,124$): no order fits the headroom and the halt (realised <= -150) never fires.
  Every start window (1,728, uncapped live-like path): the cap binds within the first day from 30.4% of starts (median
  35 windows after start); 18.3% reach realised -150; six-day max drawdown -328.82$, worst window -87.03$.
- Data: Binance silent > 1 s covers 4-81 s of quoting time per day (2-24 windows); Predict silent > 5 s only on 10-31
  (27.6 s, 2 windows), never 30 s; every market registered >= 253 s before its start (the 5 s rule never binds).
- Not modelled: what the engine does when the request budget is exhausted; whether the replacement waits for the
  cancel (sequential) or rests beside the locked order (modelled here); own-order subtraction (research books never
  contained our orders, so the live core's subtraction reproduces the research view).

## 17. Live-deviation round 2: sequential lock and the exec-port throttle (as of 2026-11-02T09:37Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Live-like C2-P = D1 stack + M025 + A1 5..25, no amend, 1 USD minimum withheld; same six read days and settings as §16.
New tokens in pf_step4_live.py: XS<ms> (sequential: the cancelled order stays fillable and on its side until decision +
max(20, L); the side's then-current intent activates 20 ms after that, so L = 0 means new order at decision + 40 ms) and
XT<r> (the exec-port throttle: places bucket 4 at r/s, cancels bucket 8 at 2/s always sent, place needs cancel tokens >=
open + 1 and < 3 unanswered per side, retry 250 ms doubling to 4 s while the intent stands). Parity with options off equal
to round 1 to the cent (10-25). Every fill ends within stop + lock; G6 holds in every sequential arm (max |net| 75-99,
hard 125); the concurrent 200 ms arm reached 126.7 once. Chain `run_live_dev2.sh`, report `pf_live_report2.py`.

| live-like C2-P, 6 days | gross $ | vs live-like | shares | +1 s c/sh | requests/min | place intents placed now / later / never |
|---|---|---|---|---|---|---|
| no lock, no throttle | +3,067 | - | 504k | 0.36 | 287 (65% > 240) | |
| sequential L = 0 | +1,906 | -1,161 | 439k | 0.23 | 282 | |
| sequential L = 200 / 500 / 1000 ms | -8,218 / -16,047 / -20,856 | | 699k / 974k / 1,182k | -1.05 / -1.57 / -1.64 | 188 / 133 / 95 | |
| concurrent L = 200 ms (500 / 1000 from §16, no 1 USD min) | -6,735 (-21,943 / -42,606) | | 1,107k | -0.57 | 278 | |
| sequential L = 0 + throttle 1 place/s | +958 (5/6 days) | -2,109 | 182k | 0.04 | 106 (0%) | 25% / 15% / 61% |
| sequential L = 0 + throttle 2 places/s | +2,056 (4/6 days) | -1,011 | 293k | 0.15 | 179 (3%) | 51% / 13% / 35% |
| sequential L = 500 ms + throttle 1 / 2 places/s | -12,225 / -15,970 | | 748k / 968k | -1.54 / -1.57 | 100 / 132 | |

- Any removal lock of 200 ms or more makes live-like C2-P lose heavily in either design; sequential beats concurrent at
  500 / 1000 ms but not at 200 ms. Even L = 0 sequential costs 38% (the replacement arrives 20 ms later).
- The throttle at L = 0 keeps the sign: 1 place/s +958 (31% of unthrottled), 2/s +2,056 (67%); delayed places wait
  p50 250-620 ms, p90 0.75-1.75 s; at 1/s most place intents (61%) are never placed.

## 18. Lock-length grid 0-200 ms, sequential design (user: "Test between 0-200ms"; as of 2026-11-02T10:30Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Live-like C2-P, sequential lock (§17 definition), six read days; round 3 adds L = 50/100/150 and the throttled 200 ms
cells (`run_live_dev3.sh`, `pf_live_report3.py` -> `step4/live_..._livedev3/report_lock_grid.json`). Every fill ends
within stop + max(L, 20 ms); max |net| <= 98.9 (hard 125). Cells: 6-day gross $ (days > 0 of 6) [+1 s c/sh].

| L ms | no throttle | throttle 1 place/s | throttle 2 places/s |
|---|---|---|---|
| 0 | +1,906 (5/6) [0.23] | +958 (5/6) [0.04] | +2,056 (4/6) [0.15] |
| 50 | +1,059 (5/6) [0.12] | +667 (4/6) [-0.07] | +966 (4/6) [0.05] |
| 100 | -996 (2/6) [-0.20] | -790 (2/6) [-0.30] | -773 (2/6) [-0.24] |
| 150 | -3,764 (0/6) [-0.55] | -2,531 (0/6) [-0.61] | -2,975 (0/6) [-0.59] |
| 200 | -8,218 (0/6) [-1.05] | -5,246 (0/6) [-1.11] | -6,883 (0/6) [-1.07] |

Break-even lock (linear between 50 and 100 ms, computed): 75.8 ms unthrottled, 72.9 ms at 1/s, 77.8 ms at 2/s. The lock
here counts from the decision; with 20 ms transit a stated venue lock of ~55 ms is already at break-even. Interpolated on
a 50 ms grid, modelled fills, the six days P was chosen on.

## 19. Placement-relative lock grid (exec-port request; as of 2026-11-02T11:12Z) - supersedes §18 for reading the probe
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Predict's lock is per order from placement (OpenAPI: CreateOrderResponseData.removalLockedUntil, optional; removal is
rejected inside the window). New token XP<lambda> in pf_step4_live.py: sequential, the cancelled order stays fillable
until max(decision + 20 ms, activation + lambda), the new order activates 20 ms later. §18 (lock on every cancel from the
decision) is the worst case. Parity control XP0 = round 2's XS0 to the cent (6/6 days). Fills end within stop + 20 ms +
lambda (an order activated just before the stop keeps its lock); max |net| <= 125 everywhere. Chain `run_live_dev4.sh`,
report `pf_live_report4.py` -> `step4/live_..._livedev4/report_plock_grid.json`. Cells: 6-day gross $ (days > 0)
[+1 s c/sh] {share of cancels held by the lock}.

| lambda ms | no throttle | 1 place/s | 2 places/s |
|---|---|---|---|
| 0 | +1,906 (5/6) [0.23] {0} | +958 (5/6) [0.04] {0} | +2,056 (4/6) [0.15] {0} |
| 50 | +1,940 (5/6) [0.22] {0.33} | +592 (5/6) [0.04] {0.23} | +1,854 (5/6) [0.14] {0.27} |
| 100 | +1,471 (4/6) [0.17] {0.51} | +522 (5/6) [0.00] {0.41} | +1,100 (4/6) [0.09] {0.45} |
| 200 | -868 (2/6) [-0.18] {0.65} | +71 (3/6) [-0.20] {0.57} | -284 (3/6) [-0.21] {0.61} |
| 500 | -8,193 (0/6) [-1.08] {0.79} | -5,672 (0/6) [-0.96] {0.76} | -7,735 (0/6) [-1.05] {0.79} |
| 1000 | -15,609 (0/6) [-1.46] {0.89} | -13,608 (0/6) [-1.43] {0.88} | -15,600 (0/6) [-1.46] {0.89} |

Break-even lambda (computed, linear between 100 and 200 / 200 and 500): 163 ms unthrottled, 204 ms at 1 place/s,
180 ms at 2 places/s. Order age at the cancel decision (no-lock schedule): p10 2 ms, p25 15, p50 69, p75 252, p90 719;
a cancel is held iff it lands before activation + lambda: 34% at 50 ms, 53% at 100, 69% at 200, 85% at 500, 93% at 1000.
Read the probe as lambda = removalLockedUntil - order acceptance time.

## 20. Concurrent vs sequential under the placement-relative lock (exec-port request; as of 2026-11-02T11:55Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


The frozen Tokyo version replaces concurrently. New token XQ<lambda> (concurrent: the old order stays fillable beside its
replacement until max(decision + 20 ms, activation + lambda); G6 does not see it; the throttle's open-order count includes
it). Parity XQ0 = live-like to the cent (6/6 days). Every fill ends within stop + 20 ms + lambda. Chain `run_live_dev5.sh`
(two processes per day), report `pf_live_report5.py` -> `step4/live_..._livedev5b/report_conc_vs_seq.json`.
6-day gross $ (days > 0 of 6); concurrent per-window max |net| in brackets (G6 hard cap 125).

| lambda | none: concurrent | none: sequential | 1/s: concurrent | 1/s: sequential | 2/s: concurrent | 2/s: sequential |
|---|---|---|---|---|---|---|
| 0 | **+3,067** (5) [75] | +1,906 (5) | +595 (4) [75] | **+958** (5) | +1,998 (6) [75] | **+2,056** (4) |
| 50 | **+2,770** (5) [97] | +1,940 (5) | +474 (4) [75] | **+592** (5) | +1,711 (6) [95] | **+1,854** (5) |
| 100 | **+1,992** (5) [97] | +1,471 (4) | +435 (4) [77] | **+522** (5) | **+1,226** (5) [95] | +1,100 (4) |
| 200 | **+68** (3) [127] | -868 (2) | -703 (1) [114] | **+71** (3) | **-69** (2) [121] | -284 (3) |
| 500 | -11,166 (0) [163] | **-8,193** (0) | **-4,908** (0) [114] | -5,672 (0) | **-7,293** (0) [120] | -7,735 (0) |

Break-even lambda (computed): concurrent 202 ms unthrottled / 138 ms at 1 place/s / 195 ms at 2/s; sequential 163 / 204 /
180 ms. Concurrent breaks G6's hard cap only unthrottled at 200 ms (0.12% of windows, max 126.7) and 500 ms (2.3%, max
163). With the exec port's 1 place/s throttle, sequential beats concurrent at every lambda <= 200 ms (+87..+774$/6 d);
unthrottled, concurrent is better up to 200 ms; at 2/s the two are within ~150$ up to 100 ms.

## 21. Tokyo probe results read against the research grids (as of 2026-11-03T03:58Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; the §19 / §20 grid $ (the probe facts stand) came from a pre-fix replay and is void (outputs deleted).


Source: pmmmrust fe073d1 `docs/receipts/20261102_pf_tokyo_probe_and_latency_results.json` (+ records), fetched read-only.
- Removal lock: none. removalLockedUntil null on the far order; first removal at 0 ms after the answer succeeded; 12
  immediate trials (remove sent 0-100 ms after the create) all removed on the first attempt. lambda ~ 0 -> §19/§20 row 0:
  sequential +1,906 (none) / +958 (1/s) / +2,056 (2/s); concurrent +3,067 / +595 / +1,998. Tested on 13 far orders only.
- Rate limits (headers): create/remove 40 per s and 500 per min; reads 250 or 500 per min by route. Unthrottled demand
  (requests at decision time, quoting minutes): sequential 282/min mean, 4.3% of minutes > 500, max 850, no second > 40
  except one (46-48 max); concurrent 287/min, 5.2% > 500. The exec-port shadow saw ~264/min (1,386 places + 1,385 cancels
  in 10.5 min), consistent. The 1 place/s throttle (60/min) is far tighter than the venue requires.
- Latency (Tokyo, chrony ~1 us): Predict book delay after updateTimestampMs p50 53 ms (min 13, p90 80, p99 115; it
  publishes ~every 50 ms); Binance E delay p50 1.5 ms; decision ~10 ms; create round trip p50 44 ms (one way ~22).
  tokyo20 assumed Predict +20, Binance +5, no decision time, orders +20: book change -> resting order ~40 ms in the
  replay vs ~85 ms measured. This is now the largest unmodelled deviation (the 'tape' clock, recv - 72 ms on this host,
  keeps Predict's publication delay and is the closer model).

## 22. Latency test L1: live-like C2-P at the measured Tokyo clock (declared 04:21Z; as of 2026-11-03T05:51Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Clock `pf_clock_rl` (Tokyo, exec-port fe073d1): Predict frames per frame at Tokyo's 53.1 ms median delay (this host's
receipt shifted per day, >= update + 13.2 ms) + 10 ms decision; Binance E + 11.5; orders and cancels live 22 ms after
sending; own fills known after 30 ms. Features rebuilt on that clock (`pf_step5_features_rl.py` -> rows_rl); out-of-day
models A (trained on tokyo20 rows = the frozen live spec) and B (refit on Tokyo-clock rows) (`pf_step5_lodo_rl.py`).
Replay `pf_step4_live.py` scenario tokyoreal; throttle XV = Predict's create/remove limits (40 per s, 500 per min).
Chain `run_latency_l1.sh`, report `pf_latency_report.py` -> `step4/live_tokyoreal_..._rl_a_l1/report_latency_l1.json`.
Checks: 156 arm-days, 0 fill-invariant violations; the tokyo20 reruns equal round 2 to the cent.

| live-like C2-P, 6 read days | tokyo20 (so far) | Tokyo clock, model A | Tokyo clock, model B | +1 s c/sh (A) |
|---|---|---|---|---|
| concurrent, no throttle | +3,067 (5/6) | +494 (4/6) | +585 (4/6) | -0.01 |
| concurrent, venue throttle | +2,787 (5/6) | +602 (3/6) | +556 (4/6) | -0.02 |
| concurrent, 1 / 2 places/s | +595 / +1,998 | +941 / +496 | +316 / +812 | -0.28 / -0.16 |
| sequential, no throttle | +1,906 (5/6) | +296 (4/6) | +239 (4/6) | -0.05 |
| sequential, venue throttle | +1,865 (5/6) | +271 (4/6) | +395 (4/6) | -0.06 |
| sequential, 1 / 2 places/s | +958 / +2,056 | +1,126 / +237 | +290 / +397 | -0.30 / -0.18 |
| D1 alone, 5 sh | +260 (5/6) | -596 (0/6) | -596 (0/6) | -0.41 |
| D1 + M025, 5 sh | +602 (5/6) | -58 (4/6) | -22 (3/6) | -0.10 |

- At the measured latency the edge mostly disappears: the unthrottled / venue-throttled arms keep +$240-600 of +$1.9-3.1k
  (-80 to -90%), several days negative; the +1 s mark-out is ~0 or negative in every arm (it was +0.23..+0.36 c/sh), so
  what remains is settlement P&L, not a short-horizon edge. D1 alone loses on 6/6 days; the model guard still turns it
  to about flat.
- The 1 place/s arms look better at the Tokyo clock (+941 / +1,126 with model A) but swing to +316 / +290 with model B and
  carry the worst +1 s mark (-0.28 / -0.30 c/sh): noise on few fills, not an edge.
- Models are not the problem: out-of-day R2 on the Tokyo-clock target A 0.13-0.20, B 0.13-0.20 (+0.005), vs lodo6 on
  tokyo20 0.12-0.19. The loss is execution speed: quotes stale ~45 ms longer (book change -> our order ~85 ms vs ~40 ms)
  are picked off more often.
- Request load at the Tokyo clock: unthrottled 277-285 per quoting minute, 3.6-4.7% of minutes over 500; the venue
  throttle keeps it at 273-280 (1.3-1.5% over 500 by decision-time counting).
- C2 (fresh days 11-01..05) still runs on the tokyo20 clock: a C2 pass would not show live profitability at this latency.

## 23. Latency test L2c: split order / cancel / fill-known latency (as of 2026-11-03T08:05Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Inputs from the exec port's final latency reading, verified here on our clock only (probe 20261102T150745Z, latency run
20261102T162608Z): resting-order remove answered 22-34 ms after send (p50 28); a remove reaching Predict before the order
exists is held and answered 95-135 ms after the create's send; account reads through Cloudflare 10.5 min / 13.6 p50 ms;
Predict's wallet-event stamps lag its HTTP answers, so they are not used. Replay inputs: order live 40 ms after send
(main; 22 ms = half the create round trip, optimistic), resting cancel 22 ms, pending cancel 95 ms after the create's
send, own fills known 100 ms (by analogy with the wallet events; the probe had no fills), decision 10 ms. Model A, six
read days, `run_latency_l2.sh` (L2 / L2b runs superseded and deleted unread), report `pf_latency_report2.py`. Fill
invariants: only fills after the 240 s stop, all within the pending-cancel bound (last 240.055 s <= 240.095).

| live-like C2-P, 6-day gross (days > 0) [+1 s c/sh] | L1 (22/22, fills 30) | main (40/22, fills 100) | order 22 | fills 30 | decision 2 |
|---|---|---|---|---|---|
| concurrent, no throttle | +494 (4) [-0.01] | -514 (2) [-0.10] | +464 (4) | -433 (2) | +227 (3) |
| concurrent, venue throttle | +602 (3) | -288 (2) [-0.11] | +531 (4) | -174 (4) | +216 (3) |
| sequential, no throttle | +296 (4) | -412 (3) [-0.17] | -323 (2) | -374 (3) | +462 (4) |
| sequential, venue throttle | +271 (4) | -420 (2) [-0.17] | -325 (2) | -300 (3) | +522 (3) |
| sequential, 1 place/s (frozen v6) | +1,126 (5) [-0.30] | +929 (4) [-0.36] | +874 (6) | +933 (4) | +795 (4) |

- On the measured path every unthrottled / venue-throttled arm loses (-288 to -514 $ over six days, 2-3 of 6 days up);
  the order-live time is the largest lever (40 -> 22 ms: concurrent back to about +500), our decision time next
  (10 -> 2 ms: sequential about +500, concurrent about +220); fill-known 100 vs 30 ms changes little.
- The frozen v6 configuration (sequential, 1 place/s) stays positive in every column (+795..+1,126), but with the worst
  +1 s mark-out (-0.30..-0.37 c/sh: its fills are adverse) and, in L1, +1,126 with model A vs +290 with model B. Its
  result is settlement P&L on few fills, sensitive to the model; not a measured market-making edge.

## 24. Round O3: optimisation at the measured Tokyo clock (declared 08:47Z; as of 2026-11-03T09:59Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Clock = L2c main with decision 2 ms (order live 40, resting cancel 22, pending cancel 95 after the create, fills known
100 ms). Base = live-like C2-P sequential + venue throttle; preds_o3 = model A pred_1000 + a y250 model refit on
Tokyo-clock rows (out-of-day R2 0.39-0.53). Scripts `pf_step5_o3_preds.py`, `run_o3.sh`, `run_o3_combo.sh`,
`pf_o3_report.py` (adoption predicates computed). Base equals L2c's decision-2 arm to the cent. Fills after a stop all
within the pending-cancel bound (latest 0.083 s after its stop, bound 0.095).

| variant (dev 10-23..25 / second look 10-26/27/31) | dev $ (days) | +1 s c/sh dev | 2nd look $ (days) | 6 days | adopted |
|---|---|---|---|---|---|
| base (sequential, venue) | +402 (2/3) | -0.040 | +119 (1/3) | +522 | - |
| F2 Binance: B2W100 (best) / B1T03 / B2T03 / B2W100T03 | +404 / +240 / +352 / +374 | -0.037 | +100 | +505 | yes (+2 $, noise) |
| F3 model: Q025 (best) / Q05 / Q10 / M015 / M05 | +442 / +420 / +390 / +338 / +271 | -0.039 | +104 | +546 | yes (+40 $) |
| F4 stand back: BKA (best) / BK60 / BKM | +555 / +226 / +264 | +0.032 | +317 | +872 | yes |
| F5 time: S60 (best) / S120 / S180 / A7 | +773 / +375 / +754 / +409 | +0.086 | +896 (3/3) | +1,669 (6/6) | yes |
| F6 dead-band: HY50 / HY100 / HY250 | +158 / -178 / -246 | -0.085 | | | no |
| COMBINED sequential (base + B2W100 + Q025 + BKA + S60) | +1,140 (3/3) | +0.121 | +477 (2/3) | +1,617 (5/6) | |
| COMBINED concurrent | +1,169 (3/3) | +0.204 | +637 (2/3) | +1,806 (5/6) | |

- Quoting only the first 60 s of each window is the dominant change (S60 alone +1,669, 6/6 days); standing back one tick
  is the second. Together they restore a positive short-horizon edge (combined +0.11 / +0.21 c/sh at 1 s, +0.8 at 60 s,
  +2.5..2.6 at settlement) on far fewer shares (64-68k vs 369k).
- The first 2 s of each window (where features see the start price up to ~1.6 s early) contribute only +42..105 $; the
  profit is made at 10-60 s.
- Not a confirmation: variants were chosen on dev days (multiplicity 19), the second-look days were seen before, and
  the fill model is unverified at the touch. Next = fresh-day confirmation C3, frozen before reading.

## 25. Confirmation C3 on the first fresh days, 11-01 and 11-02 (declared 10:32Z; as of 2026-11-03T10:39Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Declared before any O3 arm was run on these days (STATUS.yml 10:32Z). Clock = Tokyo measured (order live 40, resting
cancel 22, pending cancel 95 after the create, fills known 100, decision 2 ms); models fit on the six read days only
(pred_1000 = model_c2_y1000 be6e916e, pred_250 = model_c3_y250 e5cf2fe6), out-of-sample R2 1 s 0.11 / 0.16, 250 ms
0.34 / 0.47. Chain `run_c3.sh`; 14 arm-days, 0 fill-invariant violations. Two days: point estimates, no verdict.

| arm (11-01, 11-02) | gross $ | by day | shares | +1 s c/sh | +60 s c/sh |
|---|---|---|---|---|---|
| P3 full combined, sequential (primary) | -155.55 | +38.77 / -194.32 | 17,770 | -0.517 | -1.096 |
| P3 concurrent | +6.37 | +115.89 / -109.52 | 19,135 | -0.413 | -0.706 |
| simplified (one tick back + first 60 s) | -79.79 | +40.88 / -120.67 | 18,097 | -0.497 | -0.977 |
| first 60 s only | -499.44 | +118.51 / -617.95 | 36,654 | -0.474 | -0.082 |
| base (O3 base) | -1,009.96 | -264.96 / -745.00 | 111,637 | -0.524 | -0.600 |
| P3 at 300 requests/min | +48.90 | -12.18 / +61.08 | 14,126 | -0.531 | -0.795 |
| P3 with the 150 $ loss cap | -149.08 | +38.77 / -187.85 | 17,567 | -0.519 | -1.152 |

- The read-day edge does not appear on these two days: every arm's +1 s mark-out is -0.41..-0.53 c/sh (read days:
  combined +0.12 / +0.20). The O3 changes still cut the loss against the base (-156 vs -1,010) mostly by trading less.
- The capped arm ends at realised -149.08 with 40,145 refused places and no halt: the simulator's XCAP does not have the
  exec port's halt fix (halt when the headroom is below the smallest order's worst case); live would have halted.
- C2's day-1 interim (tokyo20 clock) already showed the +1 s edge near zero on 11-01 for every arm.

## 26. Round O4: sweep protection on all 8 days (declared 10:49Z; as of 2026-11-03T11:20Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


User: optimise on the data we have, no waiting. All 8 days used (10-23..27, 10-31, 11-01, 11-02), no hold-out; days from
11-03 are the later check. Diagnosis first: queue fills earn +0.30..+0.65 c/sh at 1 s on every day; sweep fills lose
-0.87..-1.24 c/sh on 10-31, 11-01, 11-02 vs -0.25..-0.37 on the other days (10-31 and 11-01 are the two quietest days).
Base P3 = O3's full combined (sequential, venue throttle), Tokyo clock, decision 2 ms; P3 here equals O3 / C3 to the cent.
`run_o4.sh`, `pf_o4_report.py`. Fill checks: only fills after a stop, all within the pending-cancel bound (<= 0.075 s).

| variant (8 days) | 8-day $ (days > 0) | 6 read days | 2 fresh days | +1 s c/sh | better than P3 on |
|---|---|---|---|---|---|
| P3 (base) | +1,462 (6/8) | +1,617 | -156 | -0.025 | - |
| stand back 2 ticks (BKD2) | +866 (5/8) | +868 | -2 | +0.221 | 2/8 |
| touch cooldown CD250 / CD500 / CD1000 | +1,382 / +1,425 (7/8) / +1,004 | | -15 / +75 / -37 | -0.014 / -0.007 / -0.155 | 4/8 (CD500) |
| fill cooldown FC500 / FC2000 | +1,418 / +1,287 | | -96 / -149 | -0.002 / -0.039 | 3/8 |
| stop S30 / S45 / S90 | +787 / +1,175 / +1,996 | | +62 / +97 / -222 | 0.000 / -0.027 / -0.104 | 7/8 (S90) |
| size fixed 5 / A1 5..10 | +354 / +599 | | -12 / -101 | -0.193 / -0.137 | 2/8 |
| M015 / Q015 / Q05 | +1,574 / +1,539 / +1,496 | | -123 / -135 / -84 | +0.019 / -0.010 / -0.026 | 5/8 (M015) |

- Nothing is adopted: S90 has the largest sum and beats P3 on 7/8 days but worsens the +1 s mark (-0.104 vs -0.025);
  M015 beats P3 on only 5/8 days; the rest lose in sum. Differences are within day-to-day noise.
- Most consistent: CD500 (positive on 7/8 days, both fresh days positive, +1 s mark -0.007) - not adopted (sum below P3).
- Across the 8 days P3's +1 s mark is about zero (-0.025 c/sh): the strategy family has no measurable short-horizon edge
  once the fresh days are included; its +1,462 rests on settlement P&L of the read days.

## 27. Re-split: validate on the first two days (user; as of 2026-11-03T12:20Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Same O3 / O4 selection rules, training days 10-25, 10-26, 10-27, 10-31, 11-01, 11-02, validation 10-23 and 10-24 (rules
scaled to six days: better on >= 4/6 in stage 1). O3's 20 arms were also run on 11-01/02 (`run_resplit_o3fresh.sh`; base
equals C3 to the cent; 0 invariant violations). Report `pf_resplit_report.py` -> `step4/report_resplit_first2.json`.
Caveat: read-day predictions come from leave-one-day-out models that include the validation days in training.

- Stage 1 adopts only S60 (training +855 vs base -622, better on 5/6 days, +1 s -0.156 vs -0.282). Stand back (BKA),
  the 250 ms guard (Q025) and the Binance window (B2W100) all fail here (each better on only 3/6 training days).
- Validation (10-23, 10-24): base + S60 +314.65 (+150.43 / +164.22), +1 s +0.093 c/sh; base +134.09 (+285.51 / -151.42).
  O3's original pick P3 shows +780.69 on these days but was selected on them (not a validation).
- Stage 2 (O4 families) needs runs on base + S60; not done.
- base + S60 over all 8 days: +150 / +164 / +458 / +598 / +232 / +66 / +119 / -618 = +1,169, positive on 7/8 days.

## 28. Spread capture vs leftover carry; round LC (declared 13:10Z; as of 2026-11-03T13:24Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


User's frame: earn the spread, keep leftovers small (their settlement carry has expected value ~0 but adds risk). New
per-window accounting in pf_step4_live.py: mtm_stop = P&L marked to the venue mid at stop + 100 ms (spread captured minus
adverse selection, no outcome) and carry_stop = net x (outcome - mid); gross = mtm_stop + carry_stop. Base = O3 base + S60
(sequential, venue throttle, Tokyo clock, decision 2 ms), all 8 days; `run_lc.sh`, `pf_lc_report.py` -> `step4/report_lc.json`;
0 fill-invariant violations; base equals O3's S60 arm to the cent.

| variant (8 days) | mark-at-stop $ (days > 0) | t (per window) | carry $ | carry sd per window | mean leftover | gross $ |
|---|---|---|---|---|---|---|
| base + S60 | +236 (6/8) | 0.99 | +934 | $15.9 | 28.3 sh | +1,169 |
| CAP10 | +265 (6/8) | 1.88 | +403 | $8.5 | 15.8 sh | +668 |
| CAP25 | +271 (6/8) | 1.57 | +277 | $11.0 | 20.0 sh | +548 |
| RO45 / RO50 | +228 / +245 (7/8) | 0.99 / 1.05 | +765 / +706 | $12.8 / $13.9 | 20.3 / 23.0 | +993 / +951 |
| RO45 + CAP25 | +264 (6/8) | 1.58 | +326 | $8.8 | 13.8 sh | +590 |
| size 5..10 / fixed 5 | +38 / -3 | 0.39 / -0.07 | +276 / +134 | $6.5 / $3.3 | 11.6 / 5.9 | +315 / +130 |

- 80% of S60's gross (+934 of +1,169) is leftover carry, i.e. most likely luck; its spread capture after adverse
  selection is +236 over 8 days (~$30 / day, +0.14 c/sh, t 0.99).
- Inventory caps halve the leftover risk (carry sd $15.9 -> $8.5 with CAP10) without hurting spread capture (+265,
  t 1.88). Under the declared rule nothing is adopted (best sum CAP25 beats base on 5/8 days); RO45 + CAP25 beats base on
  6/8 days but its sum is below CAP25's.
- Both fresh days are negative on the mark-at-stop in every capped arm (CAP10 -47 / -42, CAP25 -32 / -10).

## 29. Is the leftover carry zero-mean? (as of 2026-11-03T14:30Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Test of the assumption behind "earn the spread, carry leftovers": S60 arm (round LC), 8 days, windows with a leftover
(n 2,123). Carry per leftover share = net x (outcome - venue mid at stop + 100 ms) / |net|, in cents; Binance gap =
p_bn - Predict mid known at 59.9-60 s (what the strategy knew at the stop).
- All windows: +0.51 c/sh +- 0.99 (t +0.5).
- Leftover on the side Binance favours (gap > +2 c): +0.93 +- 1.50; neutral: +0.50 +- 1.85; against (< -2 c): -0.10 +- 1.89.
- Predict mid as a forecast at the stop, outcome - mid: +0.20 / -1.37 / -0.93 c (gap > 2 / |gap| <= 2 / < -2), all |t| < 1.
No detectable bias from a stale mark or an informed leftover; the test excludes biases larger than about 2-4 c/sh. So
expected P&L ~ the mark-at-stop edge, and the carry adds variance only; strategies should be judged on mark-at-stop.

## 30. Where the P&L comes from: spread vs informed vs inventory drift vs carry (as of 2026-11-03T15:05Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


`pf_pnl_decomp.py` (8 days, Tokyo clock, decision 2 ms): per fill, spread = s q (m0 - p), informed = s q (m1s - m0), drift =
s q (m_stop - m1s), carry = s q (y - m_stop), with m0 the venue mid just before the fill; the four sum to gross exactly.

| arm (8 days) | spread | informed (1 s) | drift (1 s -> stop) | carry | gross | per share (c): spread / informed / drift / carry |
|---|---|---|---|---|---|---|
| full window (to 240 s) | +2,869 | -3,872 | +17 | +497 | -488 | 0.60 / -0.81 / 0.00 / 0.10 |
| first 60 s | +984 | -1,124 | +376 | +934 | +1,169 | 0.59 / -0.67 / 0.23 / 0.56 |
| first 60 s + cap 10 | +734 | -870 | +401 | +403 | +668 | 0.59 / -0.70 / 0.32 / 0.32 |

- The spread earned at the fill (~0.6 c/sh, a bit more than half the 1-tick spread) does not cover the 1 s adverse move
  (-0.67..-0.81 c/sh) in any arm. In the first minute part of that adverse move reverts by the stop (drift +0.23..+0.32
  c/sh), which leaves +0.12..+0.21 c/sh before carry; over the full window it does not revert (drift 0.00).
- Carry +0.32..+0.56 c/sh is not significant (§29).

## 31. Why the 60 s cut-off helps: market-wide vs our toxicity by minute (as of 2026-11-03T15:11Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; the 'ours' columns came from a pre-fix replay and is void (outputs deleted).


`pf_toxicity_by_minute.py`: all market maker fills (REST legs in the live window; WS / book-drop time where known, else
executedAt) vs our fills (full-window base arm), 8 days; cents per share, maker view.

| minute | market spread | market informed (1 s) | market net 1 s | ours spread | ours informed (1 s) | ours net 1 s |
|---|---|---|---|---|---|---|
| 0 | 1.08 | -0.95 | +0.13 | 0.59 | -0.67 | -0.08 |
| 1 | 1.03 | -0.91 | +0.12 | 0.59 | -0.83 | -0.24 |
| 2 | 0.99 | -0.92 | +0.07 | 0.59 | -0.91 | -0.31 |
| 3 | 0.95 | -0.93 | +0.02 | 0.63 | -0.90 | -0.27 |

- Market-wide adverse selection is flat over the window (-0.91..-0.95 c/sh): the flow does not become more informed
  later. This corrects my earlier explanation (§ answer of 2026-11-03: "later takers know the outcome").
- The average maker earns ~1.0 c/sh of spread (fills deeper in the book during sweeps); we earn ~0.6 (touch fills).
- Our informed cost is well below the market's in minute 0 (-0.67 vs -0.95) and converges to it later (-0.90): our guard
  / model advantage is concentrated in the first minute; afterwards we pay average toxicity on half the average spread.
  The cut-off works because it keeps us where our filters work, not because the market changes.
- Caveat: ~45% of market legs are timed only to the second (executedAt), so market m0 can be slightly late for them.

## 32. Market maker profitability by depth (as of 2026-11-03T15:18Z)

ms-timed market legs only (WS-exact, so the book just before each fill is right; ~55% of volume, may under-represent
multi-level sweeps), 8 days; depth = ticks behind the best price on the maker's side just before the fill; c/sh, maker view.

| depth | share of volume | spread | informed (1 s) | net 1 s (minutes 0 / 1 / 2 / 3) |
|---|---|---|---|---|
| at best | 68-76% | 0.40-0.50 | -0.21..-0.24 | +0.28 / +0.29 / +0.23 / +0.16 |
| 1 tick back | 13-19% | 1.55-1.61 | -1.12..-1.29 | +0.44 / +0.41 / +0.34 / +0.32 |
| 2+ back | 11-14% | 3.55-4.21 | -2.49..-3.69 | +1.06 / +0.68 / +0.58 / +0.52 |

- Every depth is net positive at 1 s in every minute and on every day (weakest 10-31: +0.05..+0.12).
- The market's at-best fills carry -0.22 c/sh informed cost; ours at the best price carry -0.67..-0.90 (§31): we get the
  fills other at-best makers avoid (back of the queue, ~65% of our fills sweep through our level in every minute, and an
  85-100 ms reaction time). There is no first-minute pattern market-wide.

## 33. Queue position decides fill quality (as of 2026-11-03T15:22Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; the 'Ours' line came from a pre-fix replay and is void (outputs deleted).


Market at-best fills (ms-timed legs, 8 days), split by whether the trade cleared the level (the maker side's best price
moved away at the first frame after the trade; price-time priority means the back of the queue fills only then):
- level not cleared (80% of at-best volume; front / middle of the queue): spread 0.53, informed +0.12, net +0.65 c/sh
- level cleared (20%): spread 0.57, informed -1.80, net -1.23 c/sh
Ours (full-window base, 8 days): 65% of shares are fills swept through our level (+1 s -0.55 c/sh), 35% queue fills at our
level (+0.42 c/sh). Joining at the back (every new order and every reprice queues behind the displayed size) means most
of our fills arrive only when the level is being cleared, i.e. the toxic 20% of the market's at-best flow. Confirms §32.

## 34. Who is at the front of the queue (as of 2026-11-03T15:34Z)

Market makers ranked by live-window volume per day (anonymised signer ids), 8 days; net = spread + 1 s informed cost on
ms-timed fills (c/sh); front of best = share of their at-best volume where the trade did not clear the level; levels =
distinct fill prices per window.

| makers (per day) | number | volume share | net 1 s | volume at best | front of best | price levels per window |
|---|---|---|---|---|---|---|
| top 5 | 5 | 32% | +0.51 | 62% | 84% | 27.3 |
| ranks 6-20 | 15 | 31% | +0.55 | 73% | 82% | 17.6 |
| the rest | ~1,088 | 37% | -0.00 | 75% | 73% | 9.3 |

The profitable makers are the top ~20: they fill at 18-27 different price levels per window, i.e. they keep ladders of
resting orders across the range (the probe's book snapshot showed ~180-share orders at every tick 57-82 c) and let the
price come to them, so their orders are already queued when a level becomes the best. The long tail breaks even.

## 35. Ladder market-making strategy: logic (written before any ladder run; 2026-11-03)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Evidence it rests on (§29-34): queue fills at the front are profitable (+0.65 c/sh market-wide), level-clearing fills
are toxic (-1.23); deeper levels earn more spread and stay net positive (+0.3..+1.1); the profitable makers rest orders
across many levels early and let the price come to them; our current quoting chases the best price and sits at the back.
Profit = spread on fills - adverse selection - inventory drift; leftover carry has expected value ~0 (§29) and is risk.

Rules (per side; bids shown, asks mirror):
1. Ladder band. New orders only BEHIND the best price, at offsets 1..K ticks (b1-1 .. b1-K), size z each. Never join the
   current best price: that is the back of its queue. An order of ours that becomes the best price because the market
   moved to it is kept (it is ahead of everyone who arrives later).
2. Sticky. A resting order is never cancelled to follow the price or to resize; it is cancelled only by rules 3-7.
3. Near-level guards (orders at offsets 0 and 1): pull on a Binance 1 s or 250 ms move >= 0.5 bps against the side, on a
   1 s model prediction against the side >= 0.25 c, and (offset 0 only) when the best level on our side is thin
   (< 30 shares displayed) or its price moved away within the last 500 ms. No new orders at offset 1 while they hold.
4. Deep guard (all levels): pull the whole side on a Binance 1 s move >= 2 bps against it or a model prediction >= 2 c
   against it; no new orders on that side while it holds.
5. Inventory: once the known net position reaches +CAP (long) all bids are pulled and none placed (asks stay to unwind);
   mirror for -CAP. Resting size per side (live + sending) <= BUDGET.
6. Hygiene: cancel an order that is ahead of the market (bid above the known best bid: stale and alone) or more than K+2
   ticks behind it (frees budget); after a fill at a price, no new order at that price for 1 s (re-post deeper instead
   of racing to the back of the level just cleared).
7. Time: everything is cancelled at the stop S; carry to settlement whatever is left (no crossing to flatten).
Live constraints modelled: Tokyo clock (Predict frames at 53.1 ms median + 2 ms decision, Binance E + 3.5 ms), orders live
40 ms after sending, resting cancels 22 ms, a cancel sent before the order is live takes effect 95 ms after its create,
own fills known after 100 ms, post-only, 1 USD minimum, no amend, venue throttle (40 / s, 500 / min shared bucket).
Fill model: price-time priority; our queue position = displayed size at our price in the frame before activation (the
5th level's size when our price is beyond the 5 shown levels), reduced as the displayed size there drops; trades at our
price consume the queue ahead first; a trade through our price fills us (our own orders processed best price first,
then oldest first, sharing the trade's volume).
Checks of the logic: (a) rule 1 + 2 is what places us at the front: tested by the share of our fills that are queue
fills at a level that was not cleared; (b) rules 3-4 bound sweep losses: tested by informed cost per share vs §32;
(c) rule 5 bounds the leftover: |net| <= CAP + one in-flight size; (d) gross = mark-at-stop + carry exactly.
Smoke check of the engine (10-25, 12 windows, 3 arms): matching selftest 6/6; gross = mark-at-stop + carry exactly; no
fill after the stop; max |net| <= CAP + resting budget (by design: the cap acts on known fills, a sweep can fill several
levels at once). Logic issue found: the near-level model guard at 0.25 c fired ~1,100 times in 12 windows (|pred| >= 0.25 c
about 46% of the time), so near orders were pulled and re-placed at the back constantly, defeating rule 2, and the venue
throttle refused ~2,400 places. The near-guard threshold TH is therefore a tested parameter (0.25 / 0.5 / 1.0).

## 36. Ladder test result (declared 15:55Z; as of 2026-11-03T16:36Z) - negative
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


14 arms x 8 days (`run_ladder.sh`, report `pf_ladder_report.py` -> `step4/report_ladder_lad1.json`). Mark-at-stop is
negative for every arm, on 0-1 of 6 read days and on both fresh days; no arm is eligible under the declared rule.
Best arms: K 3, z 5, S 60: read -407, fresh -172; K 6, z 5, S 60: -446 / -124; every S 240 arm -1,505 .. -4,360 (read).
Single-quote reference S60 + CAP10: read +353, fresh -89.
- Spread per share is high (2.0-2.25 c, vs 0.59 for the single quote) but the 1 s informed cost is higher (-2.4..-3.2 c).
  70-78% of ladder fills are swept through our level (net -0.9..-1.6 c/sh); queue fills at our level are positive
  (+0.56..+0.89 c/sh) but a minority.
- Requests 0.6-2.7 million over 8 days with 0.2-4.3 million throttle refusals: re-centering the band (deep cancels) and
  near-level pulls churn far beyond the venue limit, so the ladder is often incomplete.
Correction to §32 (ms-timed sample only): with all market fills (book-drop timing for the untimed ones), market makers'
net at 1 s is about 0 at the best price (-0.06..+0.02 c/sh), +0.18..+0.33 one tick back and +0.33..+0.68 two or more back.
The ms-timed sample over-stated the at-best edge. Deep levels stay positive for the market's makers, but our deep fills
are not: the established ladders rest there first, so our later orders queue behind them and are filled mostly when the
level is swept - the same back-of-queue problem one level deeper.

## 37. Why our ladder orders are at the back of the queue (as of 2026-11-03T21:48Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; everything from our ladder's order log came from a pre-fix replay and is void (outputs deleted).


User: "Why we are still in the back?" Two measurements, both on the 8 days already read.

Market side (`pf_queue_priority.py` -> `step4/queue_priority/`; 2,304 windows, 23,040 level rows, all status ok):
- The recorder first sees each market 283 s before its window opens (p10/p90 297 / 270 s), with 376 orders already
  resting (median; 11.9k shares over the 10 shown levels) and the mid at 50.5 c (p10/p90 49.5 / 50.5). The book stays
  like that until about 30 s before the open; the last 2 s remove most of it (2 s before: 296 orders, 4.9k shares; at the
  open: 144 orders, 1.8k shares).
- Queue at the open, median shares: level 1 101, level 2 194, level 3 205, level 4 139, level 5 105. The shown size at
  those prices never fell below 61% (level 1) and 87-95% (levels 2-5) of it from 10 s after first sight to the open
  (lower bound of size resting continuously, unless replaced within one book update; levels that left the 5 shown are
  excluded: 12-80%). The queue at the open was built at listing.
- A 5-share order joining each level at the open (+40 ms), never moved, held 60 s: first fill from the queue 19-29%,
  swept 36-54%, none 17-44%. The same order at the front of the queue: from the queue 29-59%, swept 28-35%.

Our ladder (best lad1 arm K 3, z 5, S 60, TH 0.5, rerun with an order log, tag lad1q, `pf_ladder_orders_report.py`;
summary identical to lad1 on all 8 days):
- 181 orders per 60 s window; median life 0.76 s (55% < 1 s, 87% < 5 s). 95% end in our own cancel: near-guard pull 43%,
  re-centre (too deep) 24%, big move 14%, best moved 7.5%; 4.7% fill.
- Queue ahead at activation, median: 367 / 280 / 339 shares at 1 / 2 / 3 ticks behind the best, against our 5. 12% of
  orders reach the front. Orders pulled by the near guard had worked from 340 to 187 shares ahead (medians): we leave
  while the level is being worked through. Filled shares: 74% swept.

Why: (1) the incumbents post at listing, when the market sits at 50 and carries no information, and keep their orders
through the open; everything we place later queues behind them. (2) The ladder moves: every re-centre or pull restarts
the order at the back. (3) Even an order that never moves, placed at the open, is behind 100-200 shares and fills mostly
in sweeps. The front of a queue goes to orders that were there first and stayed.

Correction (2026-11-03T22:12Z, superseded by §38): the "since listing + 10 s" figures above are the minimum shown size at
the price over that span. That is an UPPER bound on the size resting continuously (new orders joining can hide cancels),
not a lower bound. They do not show that the queue at the open was posted at listing; §38 measures the other direction.

## 38. Do the other makers hold through the early window? (as of 2026-11-03T22:12Z) - mostly not
> **PARTLY VOID (2026-11-05):** market-data numbers stand; our ladder's swept share came from a pre-fix replay and is void (outputs deleted).


User: "Can you check if others hold through the early window?" `pf_hold_check.py` -> `step4/hold_check*/` (selftest 6/6
incl. a positive control and a known-bad late-stamp input). The book shows size per price, not orders, so only a lower
bound on cancels exists: for the size Q0 resting at a price at the open, cancelled by u >= Q0 - shown(u) - traded at that
price up to u + slack. Holding cannot be proven from these data; pulling can. 23,040 levels at the open, 2,304 windows,
8 days (read); statuses: no book at the open 0. Trade-to-book mapping check (`pf_hold_align_check.py`, 2 days): 1.0% of timed legs
inside the spread; at-best legs see the size at their price drop in the next frame 88.7% vs 43.1% one tick deeper.

Share of the open's queue cancelled (lower bound; slack 1 s, in brackets 30 s, which credits any trade within 30 s):

| level at the open | Q0 (sh) | by 10 s | by 60 s | before it becomes the best | before the first sweep through it |
|---|---|---|---|---|---|
| 1 (best) | 189 | 38% (29%) | 43% (34%) | - | 44% (34%) |
| 2 | 218 | 67% (54%) | 71% (57%) | 66% (50%) | 74% (57%) |
| 3 | 227 | 70% (57%) | 74% (60%) | 74% (56%) | 80% (62%) |
| 4 | 211 | 71% (60%) | 75% (63%) | 78% (61%) | 82% (66%) |
| 5 | 180 | 64% (53%) | 71% (59%) | 77% (59%) | 81% (64%) |

Levels 2-5 become the best in 70-81% of windows (median 5-22 s); just before a level is swept, the shown size there is
26-33% of Q0, and that may be new orders. Most of what rests behind the best at the open is pulled within 10 s, before the
price gets there.

By maker (timed legs, 68.3M shares; level cleared = the frame after the trade shows the best beyond the fill price):

| makers (per day) | volume share | cleared: all | at best | 1 back | 2+ back |
|---|---|---|---|---|---|
| top 5 | 32% | 22% | 20% | 27% | 24% |
| ranks 6-20 | 31% | 23% | 21% | 28% | 27% |
| the rest | 37% | 28% | 26% | 33% | 32% |

Our ladder: 74% of filled shares swept through (a narrower definition than "cleared", so the gap is at least this big).

Reading: the other makers do not sit still either; the queue churns. Two consequences. (1) Our replay credits cancels
ahead of us only when the shown size falls below our place in line (new orders behind us hide them), so for an order that
stays it is pessimistic about queue position; how much cannot be measured from these books. (2) The top makers are swept
less (22% vs our 74%); whether they get there by being fast to new price levels or by pulling before sweeps is not
determined here.

## 39. How the other makers behave (declared 22:22Z; as of 2026-11-03T22:26Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; every comparison with our ladder fills came from a pre-fix replay and is void (outputs deleted).


User: "we check their behavior first". `pf_maker_behavior.py` (selftest 7/7) -> `step4/maker_behavior/`. 8 read days,
2,304 windows, makers ranked per day by live-window volume (ids are per day). Statuses: 53,522 legs with a 1 s stamp only,
59,163 legs unusable (no frame before, side empty, or inside the spread), 1,916 of our fills unusable. Descriptive only.
Assumes price-time priority at Predict (not verified here).

Fills 0-240 s (ours: first 60 s only), share of each group's timed volume:

| group | at best | behind | level < 1 s old | 1-10 s | > 10 s | behind the shown book (age unknown) | resting since listing | at-best fills at a level born as the best | after a >= 2 bp Binance move against them | net 1 s c/sh | net on those |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 5 | 62% | 39% | 20% | 37% | 6% | 36% | 0% | 70% | 2.2% | +0.32 | -2.05 |
| 6-20 | 74% | 25% | 33% | 38% | 6% | 23% | 0% | 79% | 1.3% | +0.29 | -3.79 |
| rest | 78% | 21% | 20% | 36% | 10% | 33% | 1% | 70% | 3.0% | -0.23 | -3.82 |
| ours: ladder | 6% | 94% | 22% | 33% | 7% | 30% | 2% | 48% | 0.1% | -0.48 | (n small) |
| ours: single quote | 94% | 6% | 49% | 36% | 5% | 9% | 1% | 84% | 0.0% | -0.10 | (n small) |

Front of the queue (first trade at a price in each run of resting size, 1.14M front legs): share of front shares vs of
volume: top 5 30.7% vs 32.0%, ranks 6-20 38.1% vs 30.7%, rest 31.2% vs 37.3%. That first trade comes < 0.1 s after the
level formed in 8-11%, 0.1-1 s in 47-61%, 1-10 s in 26-41%; at a level resting since listing in 0-3%.

Per maker-window (all live legs): leftover |net| / gross: top 5 11% (56% of maker-windows end within 10% of flat),
6-20 15% (55%), rest 58% (11%). P&L c/sh marked at the window's last mid / carry: top 5 +0.10 / -0.06, 6-20 +0.24 / -0.07,
rest +0.20 / -0.32. The top 5 keep little of their +0.32 c/sh at 1 s by the window's end.

Reading: nobody's fills come from a queue held since listing; the queue that fills forms in the last 0.1-10 s, and the
front of a new level is first traded within 1 s in 55-72% of cases. At-best fills come mostly at a level that was born as
the best (someone posted a new best price), not from deeper orders reached by depletion. The top 5 are not at the front
more than their volume share; what sets them apart is more fills behind the best and in deep posted levels, smaller losses
when picked off after Binance moves, and ending windows nearly flat. Their trading P&L per share is small; whether
liquidity rewards pay them is not checked (T5).

## 40. Our fills vs the other makers', like for like (as of 2026-11-03T23:05Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


User: "what the main dif of our strategy against theirs". `pf_fill_mix.py` -> `step4/fill_mix/` (8 read days, first 60 s,
timed fills; statuses 16,514 legs 1 s only, 19,663 unusable). Swept through = the trade went beyond the fill's price: for
market legs, its transaction has a leg on the same side at a worse price; for ours, the replay's through flag (same rule).
The transaction key (legs_timed2 has no hash) agrees with the true hash on 98.4% of shares for the through flag
(`pf_match_type.py`, PF_MT_TXCHECK=1, 10-23). c/sh, spread / informed (1 s) / net, share of the group's volume:

| group | all | at best, at its price | at best, swept through | behind, at its price | behind, swept through | swept share |
|---|---|---|---|---|---|---|
| top 5 | 1.28 / -0.86 / +0.42 | 52%: +0.27 | 8%: -0.14 | 30%: +0.69 | 11%: +0.83 | 19% |
| 6-20 | 1.01 / -0.64 / +0.38 | 58%: +0.30 | 12%: +0.06 | 20%: +0.58 | 9%: +0.85 | 21% |
| rest | 1.06 / -1.43 / -0.37 | 62%: -0.46 | 12%: -0.26 | 20%: -0.47 | 6%: +0.77 | 18% |
| ours: ladder | 2.17 / -2.65 / -0.48 | 2%: -0.01 | 4%: -1.20 | 25%: +0.86 | 69%: -0.93 | 73% |
| ours: single quote | 0.60 / -0.71 / -0.10 | 35%: +0.49 | 59%: -0.62 | 1%: +1.93 | 5%: +1.50 | 64% |

Reading: where we are filled at our own price we do as well as the top makers (single quote at best +0.49 vs +0.27;
ladder behind +0.86 vs +0.69). The difference is the mix and the sweeps: 64-73% of our volume is swept through vs 18-21%
for the market, and our swept fills are far worse (behind: -0.93 vs +0.83, informed -3.18 vs -1.36), so the sweeps that
reach the other makers are the milder ones. Both point to speed: at the back of the queue only trades that exhaust our
price reach us, and our cancels land too late to leave the bad sweeps.

Checks. Match kinds (`pf_match_type.py`, 10-23 + 11-01): 86.2% of maker volume is mint (YES buyer + NO buyer), 13.4%
direct, 0.4% merge; the mix by position is the same for each kind. Skip-matches (`pf_skip_match.py`, 8 days): 80.7% of
maker volume is in transactions that took the shown best, 16.5% in transactions whose best price was behind the shown
best (93% multi-maker; the frame before is a median 51 ms older, so most are a best taken just earlier); in 5.3% the
better best stood before and after (skipped, or re-posted within one book update) - an upper bound on price-priority
breaks the replay ignores.
Corrections: (1) §38's comparison of the market's book-rule "cleared" share (22-28%) with our replay through share (74%)
mixed two rules; like for like the market is 18-21% and we are 64-73%, and the book rule under-counts sweeps (makers
re-post within one book update: 52% of our through fills show the level standing again in the next frame). (2) §38 part 2
read the frame strictly after the trade; the frame at or after moves the cleared shares by about 1 point.

## 41. Avoiding bad fills and getting to the front: the speed race (declared 23:14Z; as of 2026-11-03T23:31Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


User: "check how do we avoid bad fills like top makers, and how do we become the first/front in the queue". Scripts
`pf_speed_race.py` (selftest 6/6), `pf_sweep_precursor.py`, `pf_vacate_gap.py`; 8 read days, live window, descriptive.
Lags compare Binance E with Predict venue time (two clocks; < 10 ms not resolved). Our reach: Binance known E + 3.5 ms,
cancel lands E + 25.5, new order live E + 43.5; Predict frames seen a median 53.1 ms late, so a book change is acted on
in time only if visible by t - 75 ms (cancel) or reached by +93 ms (new order).

Bad fills.
- After a 1 bp Binance move within 250 ms (21,347 events), of the size resting at the best and 2 ticks behind on the side
  the move hurts: cancelled >= 12.5% / 29% / 50% / 57% by 25 / 50 / 100 / 250 ms; traded 0.3% / 0.7% / 1.5% / 4.3%.
  Our cancel (25.5 ms) is ahead of most of the market on Binance moves.
- Fills by lag since an adverse 0.5 bp move (first 60 s): 81-91% of every group's volume has none. Our swept fills: no
  trigger 90% (ladder, -0.84 c/sh) / 97% (single, -0.39); trigger too fast for our cancel 2% / 1%; trigger in time 9%
  (-1.51) / 2% (-2.96). Top 5 swept fills: no trigger 95% at +0.44.
- Warnings visible to us in time (by t - 75 ms) before market sweeps over 2+ prices (96,628): at-risk size pulled >= 30%
  (lower bound) before 18.8% of bad sweeps (informed <= -2 c/sh) vs 11.0% of good ones; a 0.2 bp Binance move in time
  11.6% vs 6.0%. Before our own swept fills they do not separate bad from the rest (ladder pulled >= 30%: 31.2% vs 31.1%;
  single 13.3% vs 11.4%).

The front.
- Spread is 1 tick 96.5% of the first 60 s (93.7% of 0-240 s): no room to post inside it; a new best price appears only
  after the other side leaves that price.
- 592,916 new best prices: 74.4% with no 0.5 bp Binance move that way in the second before; of the Binance-led, 94.9%
  appear later than E + 43.5 ms. But someone posts at the freed price in the same book update in 61.8% (of non-wide),
  within 13-53 ms in 16.0%, after 93.1 ms (our median reach) in 13.6%, after 53.2 ms (feed minimum) in 21.3%.
- First fill at a new best price (62.0% get one; median ~300 ms after it appears): top 5 40-46%, ranks 6-20 32-36%, rest
  23-24% (volume shares 32 / 31 / 37%); net 1 s +0.36..+0.37 c/sh, +1.04 when it appeared within 44 ms of a Binance move.

Reading: on Binance-led risk we are fast enough; the bad fills that hurt us mostly come with no Binance move and no warning
we can see in time, so the top makers avoid them with something we do not see in time (a faster Predict path inside our
75 ms blind spot, or other information). The front of a new price goes to whoever moves the price: in most cases the
freed price is taken in the same book update. With a 1-tick spread, a Binance-led post at the freed price arrives while
the old order is still there (post-only refused; crossing makes us the taker). We could be first only in the ~14% of new
prices left open longer than our reach.

## 42. Do the sweepers see flow on other venues first? (declared 23:49Z; as of 2026-11-03T23:58Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


User: "check if others see the flow in other exchanges or perps/spot flows". `pf_cross_venue.py`, `pf_sweep_reversion.py`;
8 read days. Feeds: Binance USD-M BTCUSDT mid and trade flow, Hyperliquid BTC perp mid (bbo) and trade flow (side
convention checked: 'B' prints at or above the HL mid 99%), Binance spot BTCUSDT last price and flow (public Vision
aggTrades, checksums OK, `data/binance_spot_vision/`), Predict's Chainlink relay (receipt time; whole-second publish times).
Hours inside 10-28..30 and after 11-02 not loaded. Exchange clocks vs Predict venue time; under 10 ms not resolved.
Presence = adverse signal over the 250 ms before the event above the 90th percentile at 23,040 control times (mid moves
are discrete, so controls show 2-10% for them; compare with the controls row).

| events | BN perp mid | HL perp mid | BN spot price | Chainlink | BN perp flow | HL flow | BN spot flow | any |
|---|---|---|---|---|---|---|---|---|
| controls | 3.1% | 2.7% | 10.0% | 1.7% | 10.0% | 10.0% | 10.0% | 31.0% |
| market sweeps, bad (36,813) | 8.8% | 3.1% | 14.8% | 1.6% | 18.0% | 9.8% | 16.4% | 38.4% |
| market sweeps, good (29,170) | 2.0% | 2.1% | 8.5% | 1.5% | 9.3% | 9.5% | 9.1% | 30.0% |
| bad, no BN perp 0.5 bp move (26,704) | 2.9% | 1.2% | 10.5% | 1.2% | 12.7% | 8.7% | 11.9% | 33.1% |
| ours single quote, bad (in time) | 6.5% | 2.3% | 15.3% | 2.2% | 17.1% | 9.7% | 17.0% | 40.3% |
| ours single quote, rest (in time) | 1.5% | 1.7% | 9.1% | 2.3% | 10.5% | 9.9% | 9.4% | 31.9% |

- Order of moves where both crossed 0.5 bp before a bad sweep: Binance perp before Hyperliquid in 62% (median 145 ms
  ahead); spot and perp together (spot first 43%, median 2 ms). Chainlink relay crosses last (median 472 ms before).
- Makers' pulls (>= 30% of at-risk size by t - 75 ms): no signal 12.0%; Binance perp only 21.0%; HL / spot / Chainlink
  without Binance perp 14.4%; both 35.3%. The makers react to Binance perp moves, not to the other venues alone.
- Bad sweeps are lasting: mid move against the makers -3.31 c at 1 s, -3.44 at 10 s, -3.19 at 60 s (96% kept). Our
  replay's bad swept fills sit on bigger moves: ladder -7.10 / -7.16, single quote -5.39 / -4.29.

Reading: no feed we record leads the bad sweeps by much. Binance perp and spot flow come closest (16-18% before bad sweeps
vs 10% at random times and 9% before good ones); Hyperliquid and the Chainlink relay show nothing. For 73% of bad sweeps
(no Binance perp move) every feed is near its random-time rate. The moves are informed (they last), but the information is
not on Binance perp / spot, Hyperliquid or Predict's Chainlink relay in the 250 ms before. Not checked: other exchanges
(Coinbase, OKX, Bybit), Chainlink's own sub-second stream, and Polymarket's parallel BTC 5-min markets (P-2026-003 data).
One usable piece: Binance perp trade flow, which our guard does not use, is present before 17% of our single quote's bad
fills vs 10.5% of the rest, in time for our cancel.

## 43. How freed prices are re-posted in the same update (as of 2026-11-04T00:28Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; anything read from a replay run (PF_RUN) came from a pre-fix replay and is void (outputs deleted).


User: "How do they repost so fast? Do they send cancel and repost together". `pf_repost_anatomy.py`; 8 read days (raw
REST order hashes for 10-23 and 11-01). API facts from the exec port (`pmmmrust/docs/PREDICT_FUN_PLAN.md`, Tokyo probe
fe073d1): create is one order per `POST /v1/orders`; remove takes up to 100 orders a call; no amend, no replace, no
combined cancel-and-create; from Tokyo a remove is answered ~28 ms after send, a create ~44 ms (live within ~40 ms).

- Book publication: gaps between frames of one market (10.0M, live): < 20 ms 2.5%, 20-40 13.1%, 40-60 61.4%, 60-100
  18.7%, >= 100 4.3%. Predict publishes about every 50 ms, so "the same update" means within ~50 ms, not one action.
- New best prices taken in the update that freed them (346,605): the old order at that price was pulled (no trade) in
  61.2%, lifted by a taker 25.0%, partly traded 6.9%, crossed book 7.0%. Pulled cases: the new size equals the removed
  size in 1.6% (not one order flipped). In those updates the market's resting order count usually moves by 3 or more
  (+3 or more 44-71%, -3 or less 14-38%): several orders change at once.
- Remainders of aggressive orders (an order first seen as a taker, later filled as a maker): 0.9% (10-23) / 2.2% (11-01)
  of live maker volume; 5.1% / 10.3% of first fills at prices taken right after being lifted.

Reading: the fast re-posts are makers, not leftovers of aggressive orders. Since one call cannot cancel and create, a
maker whose cancel and new orders land in the same ~50 ms publication sends them in parallel, on its own signal, without
waiting to see the freed price on the feed; several orders move together (ladders shifted at once). Our live plan does the
opposite: sequential replace (the cancel, then the new order once the old has ended) with 1 place per second, and we see
the feed ~53 ms late, about one publication behind.

## 44. Predict's request limit, from its own headers (as of 2026-11-04)

User: "Is there real such limit?" Read from the exec port's Tokyo probe records (`pmmmrust/docs/receipts/
20261102_pf_tokyo_results/probe/*/probe.jsonl`, 115 answers with IETF `RateLimit` headers; header names and values only):

| endpoint (probe step) | `ratelimit-policy` |
|---|---|
| create order, remove order (`imm_/far_create`, `imm_/far_remove`) | `"rps";q=40;w=1, "rpm";q=500;w=60` |
| order read, book, `/v1/orders`, `/v1/orders/matches` | `"rpm";q=500;w=60` |
| `/v1/account`, `/v1/positions`, category | `"rpm";q=250;w=60` |

- 40 requests in any second and 500 in any minute for order creates and removes. The `rpm` remaining count falls across
  book reads, creates, removes and order reads together (about 5 a second at the probe's own pace), and the `rps` count
  across a create and the remove after it (39 -> 37): one budget per key, not one per endpoint. The counts are slightly
  inconsistent between answers (several counters behind the front, it seems).
- Not tested: what happens at the limit (no 429 was met), and whether market makers can get a higher limit. The exec
  port's plan still cites the developer FAQ's 240 / minute; the headers say 500 for order calls.
- Research's venue throttle (XV: a bucket of 40 refilling 500 / minute shared by places and cancels) matches the policy on
  average; it allows no more than 40 in a second and 500 in a minute either.
- Sending a cancel and its replacement together costs the same two requests as sending them one after the other; the
  limit caps how often orders can be moved, not whether they move in parallel.

## 45. Why the parallel replace still loses (as of 2026-11-04T01:00Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


User: "cancel and post at the same time is the most efficient one already, why it does not help". `pf_replace_compare.py`
on test L2c's outputs (Tokyo clock: order live 40 ms, cancel 22 ms, fills known 100 ms, decision 10 ms, model A, six read
days), C2-P:

| arm | gross $ | shares | +1 s c/sh | +60 s | swept share | +1 s swept | +1 s at our price | fills on a cancelling order | their +1 s |
|---|---|---|---|---|---|---|---|---|---|
| parallel, no limit | -514 | 407k | -0.10 | -0.16 | 64.8% | -0.41 | +0.47 | 17.6% | -0.57 |
| parallel, venue limit | -288 | 399k | -0.11 | -0.14 | 64.7% | -0.42 | +0.46 | 17.3% | -0.61 |
| sequential, no limit | -412 | 366k | -0.17 | -0.29 | 66.2% | -0.47 | +0.42 | 0 | - |
| sequential, venue limit | -420 | 363k | -0.17 | -0.30 | 66.1% | -0.47 | +0.42 | 0 | - |
| sequential, 1 place/s (v6) | +929 | 164k | -0.36 | +0.02 | 65.6% | -0.75 | +0.38 | 0 | - |

- Parallel does help: the +1 s mark-out improves from -0.17 to -0.10 c/sh (+60 s from -0.29 to -0.16), and both kinds of
  fill improve. Gross settlement P&L ranks the arms differently because it carries the leftover's outcome (noise).
- It cannot change the mix: about 65% of our shares are swept through in every arm, at -0.41..-0.75, against +0.38..+0.47
  at our own price. Break-even needs the swept share under ~47% (0.47 x = 0.41 (1 - x)); the top makers are at 19% (§40).
- In the parallel arms, 17.6% of shares fill on an order whose cancel was already sent (-0.57 c/sh): the cancel was right
  but landed too late (we see Predict ~53 ms late, decide in 10 ms, the cancel lands 22 ms after).

Reading: replace speed is a second-order lever. Each replacement is a new order at the back of its queue, so faster
replacing does not move us forward; we are filled mostly when a trade sweeps through the whole level, and those moves
last (§42). The losses come from queue position and seeing the book late, which the replace method does not touch.

## 46. With an ideal anchor price, does quoting work? (declared 01:10Z; as of 2026-11-04T01:29Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


User: "if we have the anchor price, does it work". Upper bound: the model's predictions replaced by the realised moves
(`pf_step5_oracle_preds.py`: pred_1000 = y1000, pred_250 = y250, the venue mid 1 s / 250 ms after our knowledge time minus
the known mid), everything else unchanged (pf_step4_live, Tokyo clock, decision 2 ms, order live 40 / cancel 22 / pending 95 /
fills known 100 ms, v2 fill model, parallel replace, no request limit). Same arms with the real predictions (preds_o3 read
days, preds_c3 11-01/02), same code (`run_anchor.sh`, `pf_anchor_report.py` -> `step4/report_anchor.json`). The real model's
skill on the same rows: R^2 0.20 (10-23), 0.11 (11-01) for the 1 s move.

| arm | anchor | mark at stop, 6 read days | 11-01/02 | days > 0 | gross | shares | +1 s c/sh | swept share | +1 s swept | +1 s at our price |
|---|---|---|---|---|---|---|---|---|---|---|
| P60 (first 60 s) | real model | +380 | -25 | 6/8 | +1,205 | 184k | -0.03 | 62.7% | -0.37 | +0.55 |
| P60 | ideal | +1,816 | +533 | 8/8 | +3,918 | 125k | +1.61 | 53.8% | +1.75 | +1.44 |
| P60Q (+ 250 ms guard) | real model | +390 | -17 | 6/8 | +1,233 | 184k | -0.02 | 62.8% | -0.37 | +0.55 |
| P60Q | ideal | +1,756 | +467 | 8/8 | +3,520 | 110k | +1.65 | 53.4% | +1.78 | +1.49 |
| P240 (full window) | real model | -77 | -841 | 3/8 | -526 | 533k | -0.16 | 64.1% | -0.52 | +0.49 |
| P240 | ideal | +4,757 | +1,466 | 8/8 | +7,193 | 354k | +1.71 | 54.1% | +1.91 | +1.47 |

Reading: with an ideal 1 s anchor the same quoting, at the same latency and the same back-of-queue position, makes money on
every day (+1.6..+1.7 c/sh at 1 s); even its swept fills turn positive, because it rests only when the next second goes its
way. With the anchor we can build (our model, which already uses Binance; R^2 0.11-0.20) it is about break-even in the
first minute and negative over the full window. So the lever is the anchor's quality, not the order mechanics; the
mechanics decide which fills reach us, the anchor decides whether we are there. The ideal anchor is out of reach (it knows
moves that no feed we record shows in advance, §42); how good an anchor must be to pay is not yet measured.

Bad fills in the same runs (`pf_anchor_report.py`; fill quality at +1 s: bad <= -2 c/sh, small loss -2..0, good >= 0; 8 days):

| arm | anchor | shares | bad: share of shares, $ | small loss $ | good $ |
|---|---|---|---|---|---|
| P60 | real model | 184k | 15.5%, -1,374 | -432 | +1,751 |
| P60 | ideal | 125k | 0.3%, -10 | -26 | +2,042 |
| P240 | real model | 533k | 17.0%, -4,797 | -1,283 | +5,256 |
| P240 | ideal | 354k | 0.4%, -40 | -75 | +6,168 |

With our model the good fills already earn about what the ideal anchor earns; the bad fills (15-17% of shares) give it
back. The ideal anchor removes nearly all of them and keeps the good ones, with a third fewer shares.

## 47. The other makers' bad fills, same definition (as of 2026-11-04T01:52Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; the 'Ours' rows came from a pre-fix replay and is void (outputs deleted).


`pf_maker_bad_fills.py`; timed maker legs, 8 read days; mark-out at +1 s from the fill price; bad <= -2 c/sh. Ours from §46.

| group | window | bad share of shares | bad $ as % of good $ | net c/sh at 1 s | good $ per share filled |
|---|---|---|---|---|---|
| top 5 | first 60 s | 9.0% | 40% | +0.42 | 1.01 c |
| ranks 6-20 | first 60 s | 9.6% | 44% | +0.38 | 1.01 c |
| rest | first 60 s | 18.1% | 118% | -0.36 | 0.81 c |
| ours P60, real model | first 60 s | 15.5% | 78% | -0.03 | 0.95 c |
| ours P60, ideal anchor | first 60 s | 0.3% | 0.5% | +1.61 | 1.64 c |
| top 5 | 0-240 s | 10.2% | 51% | +0.32 | 1.02 c |
| ours P240, real model | 0-240 s | 17.0% | 91% | -0.16 | 0.99 c |

Reading: the top makers take bad fills too, about 1 share in 11, against 1 in 6.5 for us; their good fills earn the same
per share as ours. The gap is the bad share alone. At 1 s, our P60 is -$55 over 8 days: removing 4% of its bad-fill loss
breaks even, removing half (to the top makers' 40% ratio) gives about +$620 (~+0.34 c/sh), if no good fills are lost with
them; a real filter will lose some.

## 48. Can the information we have tell the bad fills? (declared 01:57Z; as of 2026-11-04T02:59Z)

User: "check how do we further reduce bad fills, do we need more information or we can do better with current information".

Quality sweep (part A), cut at its declared calibration check: a synthetic anchor with the real model's R^2 (0.15) gives P60
+826 / +62 (8/8 days, +0.60 c/sh at 1 s) and P240 +1,446 / +448 (7/8, +0.66), against the real model's +380 / -25 (-0.03)
and -77 / -841 (-0.16). Evenly spread noise still sees big moves; the real model does not, so R^2 cannot rank real anchors.
Levels 0.3..0.9 dropped (outputs of 0.3 for 4 days kept, unused).

Information test (part B, `pf_badfill_info.py`): neutral grid, 1,071,152 exposures per side (an order resting at the known
best, every 500 ms, live 0-240 s, 8 days; statuses 0). Swept through within (tau + 22, tau + 522] ms: bid 11.4%, ask 11.6%;
bad (v <= -2 c/sh): 4.2%; mean v -0.23 c per exposure. Scores pull the lowest 5-30% of exposures per side per test day
(leave one day out); shares of the day's bad value avoided / good value lost; net = bad avoided minus good lost, as a share
of the bad value; day-clustered 95% intervals (G = 8):

| pull | base: current guard input | M1: model, current features | M2: + Binance spot, Hyperliquid, Chainlink relay | ideal (realised move) | random |
|---|---|---|---|---|---|
| 5% | 34.5 / 16.4 / +32.2% | 35.8 / 17.7 / +33.3% (vs base +1.1%, CI +0.4..+1.7, 8/8) | 36.3 / 18.1 / +33.8% (vs M1 +0.5%, +0.2..+0.8, 8/8) | 71.4 / 1.3 / +71.2% | 5.0 / 5.1 / +4.3% |
| 10% | 44.9 / 27.0 / +41.1% | 47.1 / 29.2 / +43.0% (+1.9%, +0.9..+2.9, 8/8) | 47.4 / 29.1 / +43.4% (+0.4%, +0.2..+0.6, 7/8) | 87.4 / 4.3 / +86.9% | 10.0 / 10.1 / +8.5% |
| 20% | 58.4 / 41.6 / +52.5% | 61.5 / 45.2 / +55.1% (+2.6%, +1.6..+3.7, 8/8) | 61.9 / 45.4 / +55.5% (+0.4%, +0.1..+0.8, 8/8) | 96.9 / 11.9 / +95.2% | 19.8 / 20.3 / +16.9% |
| 30% | 67.4 / 51.3 / +60.1% | 71.7 / 56.2 / +63.7% (+3.6%, +2.6..+4.7, 8/8) | 71.9 / 56.6 / +63.9% (+0.2%, -0.1..+0.5, 5/8) | 99.0 / 17.2 / +96.5% | 29.7 / 30.3 / +25.4% |

AUC for bad: base 0.784, M1 0.827, M2 0.829, ideal 0.999, random 0.520. The current M025 guard pulls 25.1% of exposures.

Reading: the information we have already ranks bad exposures well (AUC 0.78); a model trained for the bad-fill question on
the same features does a little better on every day (+1..+4% of the bad value), and the outside feeds add +0.2..+0.5%.
The gap to the ideal (AUC 0.999: 87% of the bad value avoided for 4% of the good at a 10% pull) is information none of
these feeds holds. Better use of current information is worth a few percent; most of the gap needs new information.

## 49. Big orders on 10 venues around bad fills (as of 2026-11-04T03:03Z)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; the bad-fill events (they are our replay fills); the venue taker-side check stands came from a pre-fix replay and is void (outputs deleted).


User: "check if there are any big fill happens from other exchanges", "we pull data from more exchanges". `pf_big_prints.py`
(+ `pf_big_prints_split.py`), 8 read days, every event covered on every venue (statuses 0). Venues: Binance perp / spot,
Hyperliquid, Bybit perp / spot, OKX perp / spot, Coinbase, Kraken, Deribit (data in `data/exchange_public/`, README).
Taker-side check (taker buys on an uptick / sells on a downtick, median over days): 81-99% on every venue. (The first run
showed ~50% for sells on the three API venues: the fetcher writes rows sorted by time then price, which reverses falling
sweeps within a millisecond; ordered by the venue's trade sequence they pass; the big-order counts group by millisecond and
side, so they were the same before and after.) Big order = one venue's trades in one millisecond on one taker side, summed,
at or above the day's 99th percentile (median p99: Binance perp 3.0 BTC, spot 0.49, Bybit perp 2.0, OKX perp 2.7, Coinbase
0.29, Hyperliquid 4.4). Share of events with a big order against the swept side:

| events | well before (-1000..-250 ms) | just before (-250..-25.5, in time for our cancel) | same moment (+-25) | just after (+25..+250) |
|---|---|---|---|---|
| random times (23,040) | 5.9% | 1.9% | 0.5% | 2.1% |
| our good fills (13,036) | 6.3% | 2.2% | 0.6% | 2.0% |
| our bad fills (4,684) | 6.1% | 8.9% | 5.2% | 15.9% |
| market sweeps, good (29,170) | 9.2% | 3.7% | 0.4% | 1.6% |
| market sweeps, bad (36,813) | 15.7% | 14.4% | 1.9% | 5.5% |

Just before, our bad fills: Binance only 2.6%, other venues only 4.9%, both 1.3% (our good fills: 0.8 / 1.3 / 0.1%).
Rough reading for P60 over 8 days: a pull on any venue's big adverse order catches 8.9% of bad fills (~$122 of $1,374) and
2.2% of good fills (~$38 of $1,751), net ~+$84, if the caught fills are average-sized (not a replay).

Reading: other exchanges do carry part of what we miss, and more than half of it is off Binance (OKX, Coinbase, Bybit),
which our model does not watch. But it is a slice: about 1 bad fill in 11 has a big order elsewhere in time for our cancel.
For our bad fills, big orders elsewhere come more often at the same moment or just after (5.2% + 15.9%) than before: the
move appears on every venue together, with nothing on these 10 venues ahead of it. Market bad sweeps are led more often
(14-16% before) than ours, because our guard already pulls on Binance moves, so the bad fills left to us are the ones with
no earlier sign.

## 50. Research flow from here: anchor first, then the adverse-fill predictor (user, 2026-11-04)
> **PARTLY VOID (2026-11-05):** market-data numbers stand; stage 3 (replay); stages 1-2 stand came from a pre-fix replay and is void (outputs deleted).


User: "We design the flow properly, if we need a good anchor first, then check how do we get a good predictor to avoid
adverse fills". Target throughout (user: "we chase top makers performance"): in the replay at our real latency, bad fills
(<= -2 c/sh at 1 s) at most ~10% of shares, bad $ at most ~45-50% of good $, net ~+0.3 c/sh at 1 s (top 5: 9.0%, 40%, +0.42).

Stage 1, the anchor (fair price now). Candidates: A0 Predict mid, A1 Predict microprice, A2 p_bn (Binance mid x the 60 s
median Chainlink-relay / Binance ratio, against the window's start price, with the 30 min realised 1 s volatility and the time
left; already a feature through gap = p_bn - Predict mid), A3 a blend of A1 and A2 (logistic, fitted leaving one day out).
Measures on the 8 read days, live 0-240 s, every 500 ms: settlement accuracy (Brier score, by time left) and convergence
(does A - Predict mid predict Predict's mid change at +1 s / +10 s / +60 s: slope and R^2). Gate, declared before any run:
a candidate is a good anchor if its Brier score beats A0 on >= 6/8 days, or if the gap closes by >= 30% within 10 s (slope,
day-clustered CI excluding 0). Use: quote centre and skew, and an input to stage 2. If none passes, Predict's own price is
the anchor and the edge must come from stage 2.
Stage 2, the adverse-fill predictor given the anchor: the neutral-grid bad-sweep label (§48), signal families anchor gap,
Predict book signals, outside venues, big orders; screened leave one day out against the current guard; then replayed as the
guard. (The Predict-signal screen started 03:54Z, pf_badfill_info2, belongs here; it is kept as input, nothing adopted.)
Stage 3, anchor + guard in the replay against the target, then days the models never saw (11-03 onward).

Stage 1 redeclared before any run (supersedes the candidate list and gate above; user: "we can't use predict mid since ...
mid is 50ms late when we get that"): candidates are fresh Binance-based fair values only, A2 p_bn and A4 a Binance-only
recalibration fitted leaving one day out; Predict's mid as we know it (~50 ms late) is the benchmark, never an input (A1 and
A3 dropped). Added measure, the nowcast: does A - A0 show the move already made on Predict but not yet received (venue mid at
tau minus A0)? Gate: the gap closes >= 30% within 10 s (CI excluding 0) AND settlement Brier <= A0's + 0.005 on >= 6/8 days.

Stage 2 first screen (`pf_badfill_info2.py`, declared 03:53Z; same grid and metrics as §48): M3 = M1 + Predict book signals
(best size and depth changes over 250 ms / 1 s, best price changes, gaps behind the best, sweeps inferred from the book,
resting order count changes, tau_rem, p_bn, pf_onesided); M4 = M3 + the outside-venue features. Net gain over M1 (share of
the bad value, day-clustered 95% CI): M3 +0.6% / +0.4% / +0.4% / +0.2% at 5 / 10 / 20 / 30% pulls (7-8 of 8 days); M4 +1.1 /
+0.7 / +0.7 / +0.4%. AUC for bad: base 0.784, M1 0.827, M3 0.830, M4 0.831, ideal 0.999. Largest gains in M3 (fold 10-23):
Binance 250 ms return 23-25%, Binance 100 ms return 8-9%, Binance book imbalance 8%, gap (p_bn - Predict mid) 7%, Predict mid
6-7%, Predict depth level 1-3 and its 1 s change 4-6%. Reading: with everything we record, the bad-sweep ranking rises from
0.78 (current guard input) to 0.83; Predict's own book adds ~0.003 of AUC. The ceiling of current information is close.

Stage 1 result (`pf_anchor_eval.py`, as redeclared; 1,071,081 rows, 71 without p_bn): no good fresh anchor.
- Settlement Brier: Predict mid as known (~50 ms late) 0.1747; p_bn 0.1782 (+0.0035, CI +0.0012..+0.0057, better on 1/8 days);
  Binance recalibration 0.1787 (+0.0040, better 0/8). Within +0.005 on 6/8 days for both (accuracy part of the gate met).
  Predict's price is the more accurate in every time-left bucket.
- Convergence: Predict's mid change regressed on p_bn - Predict mid: slope +0.003 now (the unseen move), +0.023 at 1 s, +0.021
  at 10 s (CI -0.002..+0.044), +0.011 at 60 s; R^2 <= 0.003. The recalibration is the same. Predict does not move toward the
  Binance fair value: the level gap is mostly the fair-value model's own error (volatility, basis, the binary's pricing).
- Gate (computed): leads False for both -> no good anchor. Per the flow, Predict's own price (late) stays the anchor; the fresh
  information that does lead is Binance's last 100-250 ms move (the guard models' top input), not the fair-value level.

Stage 3 result (`run_guard.sh`, `pf_guard_report.py` -> `step4/report_guard_stage3.json`; parity: the reference arm on the new
prediction files equals the anchor test's run day for day). P60, our real-latency clock, 8 read days (LODO guard) and 11-03 (a
guard fitted on the 8 days; no model here had seen it):

| arm | read mark@stop | 11-01/02 | days > 0 | shares | +1 s c/sh | bad share | bad $ / good $ | 11-03 mark@stop | 11-03 +1 s | 11-03 bad | 11-03 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current guard (M025) | +380 | -25 | 6/8 | 184k | -0.030 | 15.5% | 78% | -224 | -0.310 | 21.8% | 105% |
| new guard instead, 10% | +396 | +7 | 7/8 | 185k | -0.022 | 15.3% | 78% | -215 | -0.257 | 21.6% | 100% |
| new guard instead, 20% | +617 | +25 | 7/8 | 174k | +0.040 | 14.5% | 72% | -163 | -0.157 | 20.7% | 92% |
| new guard instead, 30% | +648 | +89 | 7/8 | 159k | +0.136 | 13.5% | 64% | -138 | +0.002 | 19.3% | 80% |
| new guard added, 10% | +443 | +14 | 8/8 | 182k | -0.004 | 15.1% | 76% | -202 | -0.239 | 21.4% | 98% |
| new guard added, 20% | +655 | +32 | 7/8 | 173k | +0.048 | 14.4% | 71% | -160 | -0.152 | 20.6% | 92% |
| new guard added, 30% | +665 | +95 | 7/8 | 158k | +0.139 | 13.5% | 64% | -138 | -0.004 | 19.3% | 80% |

Reading (before the control): gains grow with the pull rate (30% is the edge of the grid); on 11-03, a bad day for every arm,
the new guard moves +1 s from -0.31 to 0.00 c/sh. Against the target (bad <= ~10%, ratio <= ~45-50%, +0.3): 13.5%, 64%, +0.14.
Open question: the new arms pull more than M025 (159k vs 184k shares), so part of the gain may be pulling more, not ranking
better. Control declared (STATUS): the current guard at stricter thresholds.

Fitted anchor (`pf_anchor_fit.py`, declared 04:30Z; user: "use Binance market to fit the [Predict.fun] price ... a better anchor"):
A5 = LightGBM, leave one day out, Predict's current venue mid from Binance-side inputs only (1,071,081 rows; inputs by gain: z 72%,
ln(S (1 + basis) / K) 25%, the rest ~1% each). Mean absolute error to Predict's current mid: A5 3.67 c, p_bn 4.62 c, the late
Predict mid 0.074 c. Settlement Brier: A0 0.1747, A5 0.1774 (+0.0027, better on 1/8 days), p_bn 0.1782. Convergence (share of
A - A0 that closes): A5 +0.005 now, +0.040 at 1 s, +0.052 at 10 s (CI +0.021..+0.084), +0.061 at 60 s; R^2 <= 0.007. Gate:
leads False -> not a good anchor. Reading: Binance alone cannot reproduce Predict's price level (3.7 c off on average, against
0.07 c for Predict's own price 50 ms late); the level gap is mostly the fit's error, so it says little about where Predict goes.
Predict's late price is close to its current price except at the moments it jumps; the fresh information for those moments is
the Binance change, not the Binance level.

Lead-adjusted anchor (`pf_anchor_lead.py`, declared 05:16Z; user: "predict price + binance move makes sense, it should be more
fresh without that 50ms latency"). 1,071,081 rows (statuses 0). The Predict frame known at tau was stamped a median 81 ms earlier
(p90 132 ms); the Binance-implied change since then, dp, exceeds 0.5 c in 1.5% of rows.
- Settlement Brier: A0 0.1747; A6 = A0 + dp and A7 = A0 + beta dp -0.00004 (CI -0.00005..-0.00002), A8 (learned) -0.00009, each
  better on 8/8 days. Fitted pass-through beta 1.09-1.14 on every held-out day.
- Predict's mid change regressed on the adjustment A - A0 (1.0 = Predict makes the whole adjustment):

| anchor | already made, not yet received | +1 s | +10 s | +60 s |
|---|---|---|---|---|
| A6 = A0 + dp | +0.07 (R^2 0.010) | +1.12 (CI 1.03..1.21; R^2 0.058) | +1.14 (1.06..1.22) | +1.13 (1.06..1.20) |
| A7 = A0 + beta dp | +0.07 | +1.01 (0.92..1.10) | +1.02 | +1.02 |
| A8 = A0 + learned | +0.17 (R^2 0.083) | +1.19 (1.06..1.33; R^2 0.109) | +1.24 | +1.24 |

- Gate (computed): every candidate passes (Brier within +0.005 on 8/8 days AND +10 s slope >= 0.3 with CI excluding 0).
Reading: Predict follows Binance fully, about 1.1 times the Binance-implied change, mostly within the second after our frame,
and keeps it at 60 s (no reversal). The adjustment explains 6% (A6) to 11% (A8) of Predict's 1 s move: it captures the
Binance-led part; the rest of Predict's moves are its own. Of the candidates tried, this is the first good anchor.

Stage 3 matched-pull control (`pf_step5_baseguard_preds.py`, `run_guard_matched.sh`, `pf_guard_control_report.py`): the current guard's
input (s * pred_1000) ranks the exposures through the same per-side V0 mechanism at the same pull quantile k. The first control
(stricter M thresholds) could not reach the new guard's share counts (178k at best) and is inconclusive.

| pull k, guard instead of M025 | base-ranked: shares / +1 s / bad / ratio / mark@stop | new model: shares / +1 s / bad / ratio / mark@stop | gain +1 s, ratio |
|---|---|---|---|
| 10%, 8 days | 189k / -0.056 / 15.8% / 81% / +280 | 185k / -0.022 / 15.3% / 78% / +402 | +0.034, -3 |
| 20%, 8 days | 186k / -0.037 / 15.6% / 79% / +334 | 174k / +0.040 / 14.5% / 72% / +642 | +0.077, -7 |
| 30%, 8 days | 182k / -0.019 / 15.4% / 77% / +408 | 159k / +0.136 / 13.5% / 64% / +737 | +0.154, -13 |
| 30%, 11-03 (unseen) | 19.0k / -0.292 / 21.7% / 103% / -210 | 16.1k / +0.002 / 19.3% / 80% / -138 | +0.295, -24 |

(Added to M025 instead of replacing it: the same picture, +0.157 / -14 at 30% on 8 days, +0.288 / -23 on 11-03.)
Reading: at the same share of time pulled, the base signal's pulls barely reduce fills (its pulls fall when little would fill),
while the trained model's pulls fall where sweeps hit: it gives up 23k more shares at 30%, and those shares averaged about -1.1
c/sh at 1 s (the 8-day 1 s dollars rise by ~$250). The gain is selection, not only volume. Against the target (bad <= ~10%,
ratio <= ~45-50%, +0.3 c/sh): 13.5%, 64%, +0.14 on 8 days. 30% is still the edge of the pull grid.

Latency-aware guard screen (`pf_badfill_info3.py`, declared 05:32Z): M5 = M1 + age_f, dbn_bp, dp (joined for every grid row).
Net vs M1: +0.0% / -0.1% / -0.0% / -0.0% at 5 / 10 / 20 / 30% pulls (CIs within +-0.3%, 3-4 of 8 days better); AUC 0.827 both.
dp takes 7.5% of the gain (rank 6) but replaces what the model already reads from bn_r100 / bn_r250 and gap; age_f and dbn_bp
are near zero. Rule (coded): M5 beats M1 at 0 of 4 pulls -> no replay. Reading: the guard already captures the Binance-led lag;
aligning the Binance move to Predict's frame stamp adds no information to it (the frame age is short and steady, median 81 ms).
The lead-adjusted anchor's remaining use is the quote centre, not the guard.

Longer look-back guard screen (`pf_badfill_info4.py`, declared 06:42Z): M6 = M1 + Binance returns over 5 / 10 / 30 / 60 s,
realised 1 s volatility over 10 / 60 s, largest 1 s move in 10 s, aggressor flow over 10 / 30 s, Predict mid changes over 10 / 30 /
60 s, best-price changes over 10 / 60 s, sweeps over 30 / 60 s per side (no NaN on the grid). The new inputs take 9.7% of the gain
(60 s Binance volatility 4.2%; each other <= 0.7%). Net vs M1: +0.2% (CI +0.0..+0.3, 6/8) / +0.0% / +0.1% / +0.2% (-0.0..+0.5) at
5 / 10 / 20 / 30% pulls; AUC 0.827 -> 0.830. Rule (coded): beats M1 at 1 of 4 pulls -> no replay. Reading: longer windows add a
little regime information (recent volatility) but no usable warning of the bad sweeps; the guard's information is short-horizon.

Pull-rate sweep, 10-70% (`pf_guard_control_report.py` -> `step4/report_guard_control.json`; trained guard replacing M025; the
base-ranked control at the same pull; mark-at-stop summed over all 8 read days; target bad <= ~10%, ratio <= ~45-50%, +0.3):

| pull | new: shares / mark@stop / +1 s c/sh / bad / ratio | base-ranked: shares / mark@stop / +1 s / ratio | 11-03 new: mark@stop / +1 s / bad / ratio |
|---|---|---|---|
| current guard | 184k / +355 / -0.030 / 15.5% / 78% | - | -224 / -0.310 / 21.8% / 105% |
| 30% | 159k / +737 / +0.136 / 13.5% / 64% | 182k / +408 / -0.019 / 77% | -138 / +0.002 / 19.3% / 80% |
| 40% | 140k / +824 / +0.220 / 12.6% / 58% | 179k / +428 / +0.002 / 76% | -85 / +0.182 / 18.6% / 67% |
| 50% | 115k / +920 / +0.338 / 11.6% / 51% | 175k / +480 / +0.022 / 74% | -61 / +0.356 / 16.7% / 57% |
| 60% | 90k / +832 / +0.534 / 10.3% / 42% | 153k / +663 / +0.123 / 67% | -18 / +0.637 / 17.0% / 48% |
| 70% | 66k / +759 / +0.793 / 9.1% / 33% | 122k / +712 / +0.267 / 60% | -1 / +1.072 / 14.1% / 36% |

(Added to M025 instead of replacing it: the same within a few dollars at every pull.) Reading: the per-share target is met
from 50% (+0.34 c/sh) and the top-maker fill quality (bad ~9-10%, ratio ~33-42%) at 60-70%; total mark-at-stop peaks at 50%
(+920 over 8 days vs +355 now) and falls after, as fills shrink (66k shares at 70%). The base-ranked control also gains at
high pulls but far less (ratio 60% at 70% vs 33%). On 11-03 (unseen) every step up helps (mark-at-stop -224 -> -1).
The pull rate is chosen on the 8 read days; it needs days no model has seen (11-04 from 11-05 02:30Z).

One-tick step-back (`run_guard_stepback.sh`, token SBV; parity: the withhold arm rerun equals its gd run day for day):

| pull | withhold, instead: mark@stop read / 11-01-02 / shares / +1 s / bad / ratio | step back 1 tick, instead | step back 1 tick, added to M025 | 11-03: withhold / step back instead (+1 s) |
|---|---|---|---|---|
| 50% | +719 / +201 / 115k / +0.338 / 11.6% / 51% | -221 / -192 / 217k / -0.352 / 20.3% / 110% | +507 / +61 / 153k / +0.081 / 15.5% / 71% | +0.355 / -0.723 |
| 60% | +714 / +118 / 90k / +0.534 / 10.3% / 42% | -219 / -267 / 205k / -0.356 / 20.7% / 110% | +521 / -30 / 140k / +0.116 / 15.6% / 70% | +0.637 / -0.698 |
| 70% | +682 / +78 / 66k / +0.793 / 9.1% / 33% | -159 / -306 / 192k / -0.373 / 21.7% / 112% | +591 / -51 / 127k / +0.146 / 16.6% / 70% | +1.072 / -0.742 |

Reading: one tick back fills far more (192-217k shares, more than the current guard's 184k) and badly: the flagged moments are
when sweeps run through the best, and one tick behind is still in their path. Withholding is better at every pull and on 11-03.

Prediction-priced step-back (`run_guard_predstep.sh`, token SBP; parity passed):

| pull | withhold instead | 1 tick back instead | prediction-priced instead | 11-03 +1 s: withhold / 1 tick / predicted |
|---|---|---|---|---|
| 50% | +719 / +201, 115k, +0.338, 11.6%, 51% | -221 / -192, 217k, -0.352, 20.3%, 110% | +74 / -66, 199k, -0.202, 18.3%, 94% | +0.356 / -0.723 / -0.539 |
| 60% | +714 / +118, 90k, +0.534, 10.3%, 42% | -219 / -267, 205k, -0.356, 20.7%, 110% | +82 / -168, 187k, -0.199, 18.7%, 94% | +0.637 / -0.698 / -0.531 |
| 70% | +682 / +78, 66k, +0.793, 9.1%, 33% | -159 / -306, 192k, -0.373, 21.7%, 112% | +131 / -207, 173k, -0.201, 19.7%, 95% | +1.072 / -0.742 / -0.528 |

(mark@stop read / 11-01-02, shares, +1 s c/sh, bad share, bad $ / good $.) With M025 kept ('added'), the prediction-priced arm is
identical to the one-tick arm by construction: whenever M025 is not withholding the side, the 1 s prediction is above -0.25 c,
so floor(mid + p - 0.5) is one tick behind the best; deeper prices happen only where M025 withholds anyway.
Reading: re-pricing to the predicted price helps against one tick back (+1 s -0.35 -> -0.20 c/sh) but stays far below
withholding: the flagged moments are sweeps that run past the predicted price (the prediction is small against the move).
Withholding is the rule to keep.

## 51. With the maker rebate, and against the top makers (as of 2026-11-04T09:23Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


`pf_rebate_compare.py`. Rebate = 25% of a 2% taker fee on min(p, 1 - p) per share (Predict's September maker-rebate promotion,
UNVERIFIED for November; the replay's stored rebate equals this formula on every arm). Same rebate formula applied to the market
makers' fills. First 60 s of each window, 8 read days, c/sh at 1 s from the fill price:

| | shares / day | +1 s | rebate | +1 s + rebate | bad share | bad $ / good $ | $ / day at 1 s (+ rebate) | mark@stop $ / day (+ rebate) |
|---|---|---|---|---|---|---|---|---|
| ours, current guard | 23.0k | -0.030 | 0.195 | +0.165 | 15.5% | 78% | -6.9 (+37.9) | 44.3 (89.1) |
| ours, trained guard 50% | 14.4k | +0.338 | 0.192 | +0.529 | 11.6% | 51% | 48.6 (76.2) | 115.0 (142.6) |
| ours, trained guard 60% | 11.2k | +0.534 | 0.190 | +0.724 | 10.3% | 42% | 60.0 (81.3) | 104.0 (125.4) |
| ours, trained guard 70% | 8.2k | +0.793 | 0.187 | +0.980 | 9.1% | 33% | 65.0 (80.3) | 94.9 (110.2) |
| market top 5 (together) | 850k | +0.422 | 0.192 | +0.614 | 9.0% | 40% | 3,586 (5,218) | - |
| market ranks 6-20 | 781k | +0.376 | 0.190 | +0.567 | 9.6% | 44% | 2,939 (4,425) | - |
| market rest | 825k | -0.363 | 0.197 | -0.167 | 18.1% | 118% | -2,997 (-1,376) | - |

11-03 (unseen), +1 s + rebate: current -0.115, 50% +0.547, 60% +0.825, 70% +1.260 c/sh; mark@stop with rebate -187 / -41 / -3 / +10.
Reading: per share, the trained guard at 60% matches the top 5 (+0.53 vs +0.42 c/sh, bad 10.3 vs 9.0%, ratio 42 vs 40%) and at
70% beats them on all three; the rebate adds ~0.19 c/sh to everyone alike. In size we are a different business: 8-14k shares a
day against ~170k a day for each top-5 maker, so ~$50-65 a day at 1 s ($76-81 with the rebate) against their ~$700 each. Our
numbers are a modelled replay (back-of-queue fills, 5-25 share orders, first 60 s only); theirs are real fills.

## 52. Bigger orders, quoting past 60 s, dynamic size (declared 10:32Z / 10:37Z; as of 2026-11-04T12:06Z, pull 60%)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


`run_size_dyn.sh` (tags gz, gzd), `pf_size_stop_report.py`; engine size kind AV (trained-guard value) added to `pf_step4_live.py`
(falsifiers: refused without v columns; AV:25:25:0.03 = fixed 25 to the cent on 40 windows). Trained guard V0 (withhold), pull 60%,
8 read days; parity with the gd run 8/8. Money = $ / day with the UNVERIFIED rebate; mark@stop = spread captured minus adverse moves
to the stop (no outcome); settle = to the outcome (adds the carry of the inventory held at the stop). +1 s c/sh without the rebate.

| arm (stop, size) | shares / day | +1 s c/sh | bad | bad$/good$ | mark@stop | settle | worst day (settle) | mean order |
|---|---|---|---|---|---|---|---|---|
| S60, A1 5/25 (now) | 11.2k | +0.53 | 10.3% | 42% | 125 | 195 | -84 | 25 |
| S60, A1 10/50 | 19.7k | +0.52 | 9.7% | 41% | 210 | 320 | -304 | 50 |
| S60, A1 20/100 | 33.0k | +0.52 | 8.8% | 38% | 355 | 485 | -636 | 99 |
| S60, fixed 100 | 34.0k | +0.50 | 8.9% | 39% | 360 | 519 | -635 | 100 |
| S60, AV 20/100 v >= 0.03 | 23.9k | +0.68 | 8.8% | 34% | 292 | 288 | -965 | 90 |
| S60, AV 20/100 v >= 0.06 | 16.5k | +0.91 | 9.1% | 29% | 262 | 410 | -426 | 69 |
| S60, A3 depth 5-100 | 20.8k | +0.20 | 11.3% | 55% | 158 | 352 | -95 | 77 |
| S120, A1 20/100 | 58.9k | +0.43 | 9.3% | 43% | 590 | 868 | -54 | 99 |
| S180, A1 20/100 | 78.9k | +0.42 | 9.4% | 44% | 680 | 1138 | +310 | 99 |
| S240, A1 20/100 | 95.0k | +0.43 | 9.4% | 44% | 603 | 1065 | -128 | 99 |
| S240, A1 5/25 | 31.3k | +0.44 | 10.9% | 48% | 188 | 267 | -154 | 25 |
| S240, AV 20/100 v >= 0.06 | 54.0k | +0.69 | 9.3% | 36% | 537 | 893 | -141 | 73 |

Day-clustered deltas (G = 8, 95% CI):
- Size (S60, vs A1 5/25): 10/50 mark@stop +84 (+32..+137) 8/8; 20/100 +229 (+94..+365) 8/8. Per-share quality holds (bad falls
  10.3 -> 8.8%). Settle +124 / +289, CI spans 0 (carry noise). Window sd 14 -> 48 $, worst day -84 -> -636.
- Stop (vs S60, same size): every mark@stop and settle CI spans 0 except 20/100 S180 settle +653 (+93..+1214) 7/8; mark@stop
  4-6/8 days better. 11-03 (seen once): every longer stop is worse than S60 at every size (e.g. 5/25 mark@stop -3 -> -260 at S240).
- Dynamic size: A1 (bigger when the 1 s prediction favours the side) = fixed size at the same shares (S60 +4 / +3 $, CI spans 0):
  99% of filled shares come from hi-size orders, because the guard already pulls most sides the prediction is against. AV (bigger
  when the guard's value >= par) is better per share (+0.68..+0.91 vs +0.50 c/sh) and beats fixed sizes at the same shares at
  v >= 0.06 (mark@stop +86, +17..+154, 7/8; settle +138, CI spans 0); at v >= 0.03 +18 / +43 (CI spans 0). It fills fewer
  shares, so against A1 at the same lo / hi it makes less (-39..-93 mark@stop, CI spans 0). A3 (20% of displayed depth) is worse
  per share (+0.20 c/sh, ratio 55%) and loses to fixed at the same shares: rejected.

By segment of fill time (S240 arms) against the top 5 in the same segment (+1 s c/sh, bad, ratio):

| segment | top 5 | ours A1 5/25 | ours fixed 100 | ours AV 20/100 v >= 0.06 |
|---|---|---|---|---|
| 0-60 s | +0.42, 9.0%, 40% | +0.53, 10.3%, 42% | +0.50, 8.9%, 39% | +0.91, 9.1%, 29% |
| 60-120 s | +0.33, 9.5%, 47% | +0.27, 11.7%, 57% | +0.27, 10.1%, 51% | +0.49, 10.6%, 44% |
| 120-180 s | +0.27, 10.6%, 54% | +0.43, 11.0%, 50% | +0.34, 9.3%, 48% | +0.73, 8.8%, 34% |
| 180-240 s | +0.19, 12.7%, 66% | +0.55, 10.8%, 47% | +0.36, 8.9%, 45% | +0.58, 8.5%, 38% |

Top makers' own quality falls through the window (bad 9.0 -> 12.7%); ours is weakest in 60-120 s.
Displayed size at the best (time-weighted median, live windows): 260-290 shares in 0-60 s, 300-335 later. Our hi sizes 25 / 50 / 100
are 9% / 18% / 37% of the 0-60 s touch: at 100 we would be a third of the best level, which the replay does not show to anyone.
Reading: in the replay, size scales almost linearly at unchanged quality and is the one lever whose gain is reliable (8/8 days);
quoting past 60 s adds money on average but not reliably, and 11-03 goes the other way. Guard-value sizing improves quality per
share but costs volume. The replay never shows our order to anyone (no own impact): a 100-share order is not free in the real book.

## 53. Reliability review before a freeze (declared 13:41Z..14:50Z; as of 2026-11-04T15:39Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


An independent read-only code review (agent) and checks here. Candidates at pull 60%, S60, V0 withhold: C1 A1:5:25:0, C2 A1:20:100:0,
C3 AV:20:100:0.06, C4 AV:5:25:0.06. Money = mark@stop $ / day WITHOUT the rebate (unverified; ~0.19 c/sh on top), 8 read days.

Findings and fixes:
- **Split transactions (look-ahead, main):** one taker transaction's maker legs carried different replay times (WS-matched vs E2
  book-time estimates); in mixed transactions (39-49% of 0-60 s volume) the legs spread p50 300-360 ms, so the engine saw the book move and
  acted between the legs of one trade. Fix: `PF_TXTIME=ws` (every leg at the earliest WS-timed leg, else earliest E2 leg); 36% of legs
  re-timed, p50 307 ms. Sensitivities: `min` (earliest leg of either kind), trades 25 ms before the frame (`PF_TXSHIFT=-25`).
- **Start price** used from t = 0: quotes withheld until the first `predictMarketChanged` patch with the start price (Tokyo clock;
  every window found, 0 mismatches, p50 2.4 s, p90 3.3 s) - `pf_preds_startknown.py`.
- **Pending-flow features** known at t + 53.1 ms instead of per frame: `PF_PEND_KNOWN=frame` (rows_rl_d2_pk); guard grid rebuilt on
  transaction times (`pf_badfill_grid_tx.py`: 1% of labels changed) and refit (preds_h60).
- **Queue at activation** from the frame before: pessimistic bound `PF_QACT=max`. (The reviewer's second queue claim does not hold:
  after activation ahead = min(ahead, displayed) attributes cancels behind us first.)
- **Request limit:** venue throttle XV in every final arm (costs ~0). Latency tails (cancel 34 ms, fills known 287 ms): ~no loss.
- **Walk-forward guard** (past days only) keeps 103-123% of leave-one-day-out on its 4 days.
- **Padding:** feature builds now skip sealed and later-day hours; the 04:30Z 11-03 build read 11-04 00Z (recorded).

| candidate | as first reported | + start withhold | + one time per transaction | rebuilt guard | + pessimistic queue | trades 25 ms earlier | +1 s c/sh (rebuilt) | days > 0 | 11-03 |
|---|---|---|---|---|---|---|---|---|---|
| C1 5/25 | 104 | 84 | 63 | 61 | 47 | 42 | +0.43 | 7/8 | -4 |
| C2 20/100 | 292 | 236 | 161 | 180 | 122 | 150 | +0.41 | 7/8 | -4 |
| C3 AV 20/100 | 232 | 205 | 125 | 138 | 124 | 99 | +0.76 | 8/8 | +2 |
| C4 AV 5/25 | - | - | - | 45 | 40 | 31 | +0.93 | 8/8 | +2 |

Predicates (declared before reading): C1, C3, C4 pass all (positive on >= 6/8 days with every leave-one-day-out mean > 0, queue bound
> 0, 25 ms shift > 0, +1 s c/sh >= top 5's +0.42); C2 fails the top-5 per-share test (+0.409). Top 5 for reference: +0.42 c/sh, bad 9.0%.
Reading: after every correction we can make here the per-share edge survives and C3 / C4 stay positive on every read day, at about
half the first-reported money. Not reliable yet: 11-03, the only day not used for any choice, is ~0 at the mark for every candidate;
every arm was chosen on the read days (forward days 11-04+ decide); own price impact of 100-share orders and the fill model against
our own live fills cannot be tested here. C3 / C4 sit at Predict's 500 requests / min in their busiest quoting minutes (p99 506).

## 54. Second review, real taker-order key, within-frame timing; window x size (declared 03:01Z; as of 2026-11-05T06:21Z)
> **VOID (2026-11-05):** every replay number in this section came from an engine before the 11-05 review fixes (round i); the outputs are deleted. Method text only - do not quote these numbers.


Second independent review (read-only agent). Verified here at the artifact:
- **Transaction key merged taker orders:** mid / exec_s / first_recv_ns / taker_index / taker_quote / price_executed held > 1 real taker
  order (REST transactionHash + taker hash) in ~40% of keys, so §53's "one time per transaction" moved one taker's fills to another's
  time. `pf_txmap.py` re-reads REST and maps each leg to its taker order: 87-97% of shares per day (status 0 / 5; 5 = equal-content
  legs split by maker signer, used only where the signer index agrees on 100% of uniquely mapped legs; 10-31 without it). The rest
  keep their own time. Engine `PF_TXKEY=tk`; labels `pf_badfill_grid_tx PF_TXKEY=tk` (0.7% of v labels change); guard refit preds_t60(k).
- **Trades sit at the frame stamp:** Predict publishes every ~50 ms, the match happens up to one frame earlier, our latencies are
  HTTP-measured. Central case `PF_TXJIT=frame` (U(0,1) x gap to the market's previous frame, cap 100 ms, one draw per transaction):
  mean 26.4-27.9 ms per day. Bound: one full frame early (`PF_TXSHIFT=-50`) + pessimistic queue.
- **Padding:** the replay loaded Binance 11-04 00Z / sealed hours and book frames to 11-04 00:00:05; now dropped (books by time,
  Binance by hour); rehearsal on 11-03 shows the padding never moved a number. Summaries carry code sha256 + PF_* env.
- Rehearsal: flags off reproduces rbtx 10-24 to the cent (4 arms).

Candidates at S60, $ / day mark@stop WITHOUT rebate, 8 read days (all chosen on these days: screens, not tests):

| candidate | old (rbtx) | stamps, real key (kS) | central (kJ) | bound (kX) | E2-only -300 ms (kE) | earliest leg (kM) | +1 s c/sh kJ | days > 0 kJ | 10-23..27 / 10-31..11-03 kJ | 11-03 kJ |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 A1 5/25 | 61 | 79 | 50 | -0 | -160 | 47 | +0.27 | 5/8 | +98 / -42 | -74 |
| C2 A1 20/100 | 180 | 226 | 155 | 28 | -406 | 155 | +0.31 | 6/8 | +285 / -62 | -65 |
| C3 AV 20/100 | 138 | 145 | 106 | 32 | -172 | 99 | +0.58 | 5/8 | +191 / -23 | +12 |
| C4 AV 5/25 | 45 | 57 | 46 | 18 | -52 | 43 | +0.76 | 6/8 | +74 / -3 | -6 |

Predicates (declared 03:01Z): none passes all. C4: P_DAYS, P_BOUND, P_MIN, P_TOP5 pass, P_E2 fails. C3: P_DAYS fails (5/8),
P_E2 fails. C2: P_E2 and P_TOP5 fail. C1: all but P_MIN fail. The key fix alone raises money slightly (kS - rbtx +7..+46, CIs span 0);
within-frame timing lowers it (kJ - kS -11..-71; C3 CI -71..-6).
- **E2 stress context:** E2 - WS on legs with both (10-24, 11-01): median 0 ms, mean -30, |err| <= 50 ms for 49-57%, E2 later than
  WS by > 50 ms for 20-23%. kE moves all E2-only transactions (42% of volume) 300 ms earlier - about the p90 in the wrong direction
  applied to every one - so it is a far-tail bound; it still fails as declared, and shows the replay's money depends on the timing
  of trades only the book times (~±250 ms p75 error).
- **Recent days:** at the central timing the four latest days (10-31, 11-01, 11-02, 11-03) are negative on average for every
  candidate (positive on 1 of 4 days each); the old pipeline showed them positive (C3 / C4 4/4). The money comes from 10-23..10-27.

Window x size (kJ; AV par 0.06): no longer stop and no bigger size adds (every day-clustered 95% CI spans 0). Longer stops: mark
flat or lower, carry variance large (worst day S240: -338 / -1,497 / -2,507 $ for 5/25, 20/100, 40/200), 11-03 -147..-488 at S240;
throttled requests 3.6-6.1k / day. Per-share quality past 60 s stays above the top 5 (5/25: +0.45..+0.79 c/sh vs top 5 +0.19..+0.33)
- fills are not the problem, inventory carried to the stop is. Size at S60: 20/100 vs 5/25 +60 (-27..+148), 40/200 vs 20/100 +13
(-70..+96); no own price impact modelled (200 sh ~70% of the touch).

Reading: at realistic timing the per-share edge survives on average but the money is small, concentrated in the first five read
days, and about zero or negative on the four latest days. Nothing here supports quoting longer or larger. Not settled by any
replay: the timing of book-timed trades (42% of volume) and own price impact; only own live fills can measure those.

## 55. Round i: fixed engines; corrected C1-C4 and the ladder / improve / per-minute arms (as of 2026-11-05T17:35Z)

First replay numbers on the fixed engines (review rounds 1-4: fill priority among our own orders, best-price-first sweeps, post-only
rejection reset, strategy view from acknowledgements only, request reservation by policy-visible state). Engines pf_step4_live.py
sha 742637b9a2d5bd6d, pf_step4_ladder.py c682229e4fbe606b. Every earlier replay number is VOID (banner). Tokyo clock (order 40 /
cancel 22 / pending cancel 95 / fills known 100 / decision 2 ms), XNA + XMIN1 + XV, S60, preds_t60k unless noted; central timing J
(within-frame match time) and bound X (book-timed trades -50 ms, pessimistic queue at activation). Exploratory and IN-SAMPLE: the
guard and the per-minute lines were fit on these 8 read days; 11-03 beside. $ = mark@stop per day, no rebate, no own price impact.

Gates (all PASS): synthetic fix check `reviews/simulation_20261105/check_fixes_v4.py` (R7 / R8 cases pass; pre-fix files fail them);
regression with every old switch reproduced the pre-review outputs (C1-C4, 10-24 / 11-01); view check on 100 windows: physical
view 117 early re-places, known view 0; ladder vs live engine parity + never-trigger controls equal; own-cross falsifier fires (1);
report cross-check ladder 'today (no GS)' = live C4 on all 9 days; per-lane reconciliation 0 mismatches.

Corrected C1-C4 (live engine, 8 read days, $ / day, days > 0, 11-03):

| candidate | central J | bound X |
|---|---|---|
| C1 A1 5/25 | +49.6 (5/8), -71.5 | -0.4 (4/8), -51.8 |
| C2 A1 20/100 | +140.2 (5/8), -79.7 | +20.8 (6/8), -93.9 |
| C3 AV 20/100 v >= 0.06 | +96.3 (5/8), -1.5 | +29.3 (6/8), +2.0 |
| C4 AV 5/25 v >= 0.06 | +45.7 (6/8), -8.1 | +17.9 (6/8), -6.2 |

Ladder engine, 50 ms predictions ($ / day, days > 0, 11-03; J | X):

| arm | J | X |
|---|---|---|
| today (V0 + GS, AV 5/25) | +45.5 (6/8), -8.1 | +17.7 (6/8), -6.2 |
| pm (per-minute lines PM90, AM 5/25) | +45.5 (7/8), +2.8 | +28.2 (7/8), -6.0 |
| pm + ladder (LD2) | +63.6 (7/8), -12.2 | +52.0 (7/8), -33.1 |
| pm + improve (IM95) | +70.8 (8/8), -34.6 | +39.8 (7/8), -20.2 |
| pm + both | +83.9 (8/8), -34.4 | +56.0 (7/8), -21.5 |
| today + ladder | +48.8 (6/8), -3.6 | +31.9 (6/8), -26.1 |
| pm + ladder 20/100 | +156.1 (7/8), -5.3 | +116.4 (7/8), -41.9 |
| pm + ladder hold (LH) | +17.6 (6/8), -158.0 | -5.8 (4/8), -130.9 |

Day-clustered differences (t interval, 7 df; computed by pf_ladder2_report):
- ladder on pm: J +18.1 (+9.4..+26.9) 8/8, X +23.8 (+9.9..+37.8) 7/8 - above 0 at both timings; the rung lane earns +1.95..+2.78
  c/sh at mark@stop (touch +0.99..+1.38), swept 84-86%. 11-03: -15.0 / -27.1.
- improve on pm: J +25.4 (+2.0..+48.7), X +11.6 (-4.1..+27.2) - not at both. Both on pm: J +38.4 (+8.0..+68.8), X +27.8 (+7.8..+47.7).
- ladder on today (no per-minute lines): +3.4 / +14.2, spans 0 - the ladder pays only with the per-minute pull lines.
- 20/100 vs 5/25 (pm + ladder): J +92.5 (+9.5..+175.5), X +64.4 (+10.6..+118.1) - no own price impact in the replay.
- hold (LH) vs pm + ladder: J -46.0, X -57.8, 0/8 days: rejected.
- fresh-refresh predictions vs the 50 ms grid: 2 of 18 comparisons exclude 0, both the today arm at bound timing (no GS +12.0, +0.7..+23.3;
  with GS +12.3, +0.9..+23.6; 7/8 days; near-identical arms). [Corrected 2026-11-06 02:33Z: was "1 of 18", a miscount of the report.]

Reading: on the fixed engines the per-minute-line + ladder family is positive at both timings over the read days, but this is
in-sample (lines and guard fit on the same days) and 11-03, the one day beside, is negative in nearly every arm at both timings.
No forward day has been replayed on the fixed engines. Requests: ~270-310 / window with the ladder, 5-15k throttled / day.

## 56. Inventory reduction x quoting window, fixed live engine round i (declared 17:20Z; as of 2026-11-05T21:52Z)

Question (user): does reducing inventory or quoting longer help - on the fixed engine only. pf_step4_live.py round i (sha 742637b9a2d5bd6d),
tokyoreal clock, preds_t60k; every arm MICRO + G4,G2,G6,G5,G3,G1 + V0 + GS + XNA + XMIN1 + XV. Per size: quote W s, then stop (carry
all) | CAP<hi> (inventory cap) | reduce-only W -> W+60 (RO<W>+S<W+60>) | both; W = 60 / 120 / 180, W = 240 stop / CAP only. Driver
research/run_invwin.sh, report research/pf_invwin_report.py -> step4/report_invwin.json (log logs/iw_report2.log; the rerun only added
the descriptive column carry_sd_days_lower, every other number equal to logs/iw_report.log). IN-SAMPLE: V0 guard fit on the 8 read days;
11-03 beside. Correction to the 17:20Z declaration: the comparison count is 36 per timing (18 per size), not 26.

Gates and checks:
- Never-trigger controls (10-24, 11-01): S60+CAP1000 and RO300+S60 equal S60 (7 summary keys + per-window parquet); falsifier S60+CAP5
  differs on both days (fills 1095 -> 806, 781 -> 536). CONTROLS PASS (logs/iwctrl_cmp.log).
- Every CAP / RO arm engaged (withheld counters > 0 on the 9 days; the report refuses an arm that never engaged). No missing day-runs.
- Cross-engine: the s5 S60 arm here equals the ladder engine's "today (V0 + GS)" arm of section 55 (lJi / lXi) on 9 / 9 days at both
  timings (mtm_stop, gross, fills, shares, carry_stop, mo1, windows).

AV 5/25 v >= 0.06 ($ / day, 8 read days; mark@stop (days > 0), 11-03 | settle (days > 0), worst day; carry sd / window, |inv| at stop sh, +1 s c/sh):

| arm | J mark | J settle | J carry sd, abs inv, +1 s | X mark | X settle | X carry sd, abs inv, +1 s |
|---|---|---|---|---|---|---|
| S60 | +45.5 (6/8), -8.1 | +54.8 (4/8), -187.9 | 8.04, 11.8, +0.756 | +17.7 (6/8), -6.2 | +37.3 (4/8), -163.1 | 7.29, 10.1, +0.483 |
| S60+CAP25 | +42.8 (7/8), -4.0 | +67.9 (5/8), -106.1 | 6.77, 10.4, +0.729 | +18.9 (7/8), -11.6 | +41.4 (5/8), -122.4 | 6.15, 9.0, +0.479 |
| RO60+S120 | +30.0 (5/8), -84.9 | +56.1 (4/8), -79.7 | 5.93, 8.5, +0.646 | +23.7 (5/8), -75.5 | +44.1 (5/8), -126.1 | 5.34, 7.4, +0.397 |
| RO60+S120+CAP25 | +30.0 (6/8), -49.3 | +67.3 (7/8), -17.3 | 5.02, 7.2, +0.613 | +18.5 (5/8), -59.6 | +44.8 (6/8), -88.8 | 4.51, 6.3, +0.389 |
| S120 | +30.7 (5/8), -62.3 | +79.0 (6/8), -241.8 | 9.70, 18.0, +0.607 | +23.5 (5/8), -53.0 | +69.9 (5/8), -80.5 | 8.46, 15.5, +0.334 |
| S120+CAP25 | +36.4 (6/8), -30.7 | +66.8 (6/8), -122.0 | 7.74, 14.7, +0.594 | +16.7 (6/8), -55.3 | +58.4 (5/8), -78.5 | 7.06, 13.2, +0.328 |
| RO120+S180 | +35.3 (5/8), -8.7 | +78.5 (6/8), -250.7 | 7.84, 14.7, +0.616 | +25.9 (4/8), +15.5 | +74.8 (5/8), -80.5 | 6.65, 12.6, +0.353 |
| RO120+S180+CAP25 | +27.8 (6/8), -10.9 | +69.1 (6/8), -150.4 | 6.02, 11.6, +0.607 | +18.7 (5/8), -18.6 | +62.6 (5/8), -62.7 | 5.38, 10.5, +0.355 |
| S180 | +48.5 (5/8), +30.0 | +122.5 (6/8), -249.0 | 9.90, 21.4, +0.656 | +33.2 (4/8), +34.7 | +112.0 (5/8), -119.5 | 8.80, 18.7, +0.376 |
| S180+CAP25 | +39.3 (5/8), -2.9 | +97.7 (6/8), -136.9 | 7.60, 17.1, +0.648 | +21.9 (4/8), -10.7 | +81.9 (4/8), -50.5 | 6.90, 15.5, +0.381 |
| RO180+S240 | +38.2 (6/8), -117.1 | +96.7 (6/8), -264.3 | 7.05, 18.2, +0.679 | +28.7 (5/8), -73.1 | +84.6 (5/8), -142.1 | 6.27, 16.1, +0.379 |
| RO180+S240+CAP25 | +46.6 (7/8), -72.9 | +79.5 (5/8), -139.4 | 5.15, 14.2, +0.670 | +16.8 (5/8), -76.3 | +55.3 (5/8), -83.5 | 4.80, 13.2, +0.379 |
| S240 | +39.7 (7/8), -161.4 | +89.6 (5/8), -288.1 | 8.73, 23.6, +0.675 | +23.8 (4/8), -107.3 | +49.6 (5/8), -177.2 | 7.67, 20.8, +0.394 |
| S240+CAP25 | +42.0 (7/8), -85.3 | +55.7 (4/8), -168.7 | 6.44, 18.3, +0.653 | +11.3 (5/8), -97.9 | +12.2 (4/8), -147.0 | 5.95, 16.9, +0.405 |

Differences, AV 5/25 v >= 0.06 (mark@stop mean (t interval) days > 0, 11-03; carry sd change, days lower; J | X):

| comparison | J mark | J carry sd | X mark | X carry sd |
|---|---|---|---|---|
| inventory: S60+CAP25 - S60 | -2.6 (-15.3..+10.0) 3/8, +4.2 | -1.27, 8/8 | +1.2 (-8.8..+11.2) 3/8, -5.5 | -1.14, 8/8 |
| inventory: RO60+S120 - S60 | -15.5 (-67.1..+36.1) 2/8, -76.8 | -2.12, 8/8 | +6.0 (-49.3..+61.3) 4/8, -69.3 | -1.94, 8/8 |
| inventory: RO60+S120+CAP25 - S60 | -15.5 (-51.0..+20.1) 3/8, -41.2 | -3.03, 8/8 | +0.9 (-36.0..+37.7) 3/8, -53.5 | -2.78, 8/8 |
| inventory: S120+CAP25 - S120 | +5.7 (-26.5..+37.9) 4/8, +31.6 | -1.97, 8/8 | -6.8 (-31.1..+17.4) 2/8, -2.3 | -1.40, 8/8 |
| inventory: RO120+S180 - S120 | +4.6 (-35.8..+45.0) 5/8, +53.6 | -1.87, 8/8 | +2.4 (-33.1..+37.8) 4/8, +68.5 | -1.81, 8/8 |
| inventory: RO120+S180+CAP25 - S120 | -2.9 (-51.7..+45.9) 3/8, +51.4 | -3.68, 8/8 | -4.8 (-43.3..+33.8) 1/8, +34.4 | -3.09, 8/8 |
| inventory: S180+CAP25 - S180 | -9.1 (-56.7..+38.4) 4/8, -32.9 | -2.31, 8/8 | -11.3 (-37.5..+14.9) 4/8, -45.4 | -1.90, 8/8 |
| inventory: RO180+S240 - S180 | -10.2 (-85.3..+64.9) 6/8, -147.1 | -2.85, 8/8 | -4.6 (-66.8..+57.7) 5/8, -107.8 | -2.53, 8/8 |
| inventory: RO180+S240+CAP25 - S180 | -1.8 (-56.7..+53.0) 4/8, -102.8 | -4.76, 8/8 | -16.5 (-57.7..+24.8) 4/8, -111.0 | -3.99, 8/8 |
| inventory: S240+CAP25 - S240 | +2.3 (-68.3..+72.9) 4/8, +76.1 | -2.28, 8/8 | -12.5 (-59.4..+34.4) 2/8, +9.4 | -1.71, 8/8 |
| window: S120 - S60 | -14.7 (-90.5..+61.0) 3/8, -54.2 | +1.66, 0/8 | +5.8 (-61.7..+73.4) 3/8, -46.9 | +1.18, 0/8 |
| window: S180 - S60 | +3.0 (-84.1..+90.1) 3/8, +38.1 | +1.86, 0/8 | +15.6 (-46.6..+77.7) 3/8, +40.8 | +1.51, 0/8 |
| window: S240 - S60 | -5.8 (-136.5..+125.0) 5/8, -153.3 | +0.68, 2/8 | +6.1 (-80.3..+92.5) 5/8, -101.1 | +0.38, 2/8 |
| window: S120+CAP25 - S60+CAP25 | -6.4 (-62.3..+49.4) 3/8, -26.7 | +0.97, 1/8 | -2.2 (-54.1..+49.6) 2/8, -43.7 | +0.92, 1/8 |
| window: S180+CAP25 - S60+CAP25 | -3.5 (-64.4..+57.4) 2/8, +1.0 | +0.83, 2/8 | +3.0 (-43.0..+49.0) 4/8, +0.9 | +0.75, 1/8 |
| window: S240+CAP25 - S60+CAP25 | -0.8 (-96.1..+94.5) 5/8, -81.3 | -0.33, 5/8 | -7.7 (-62.9..+47.5) 4/8, -86.2 | -0.20, 6/8 |
| window: RO120+S180+CAP25 - RO60+S120+CAP25 | -2.2 (-37.7..+33.4) 4/8, +38.4 | +1.01, 0/8 | +0.2 (-29.9..+30.3) 4/8, +41.1 | +0.87, 0/8 |
| window: RO180+S240+CAP25 - RO60+S120+CAP25 | +16.6 (-61.0..+94.3) 6/8, -23.5 | +0.13, 3/8 | -1.8 (-52.3..+48.7) 5/8, -16.7 | +0.30, 3/8 |

A1 20/100 ($ / day, 8 read days; mark@stop (days > 0), 11-03 | settle (days > 0), worst day; carry sd / window, |inv| at stop sh, +1 s c/sh):

| arm | J mark | J settle | J carry sd, abs inv, +1 s | X mark | X settle | X carry sd, abs inv, +1 s |
|---|---|---|---|---|---|---|
| S60 | +139.6 (5/8), -79.7 | +180.6 (5/8), -877.2 | 43.04, 67.8, +0.305 | +19.9 (6/8), -93.9 | +37.3 (3/8), -675.0 | 40.51, 62.0, +0.069 |
| S60+CAP100 | +141.1 (5/8), -58.3 | +142.8 (5/8), -889.0 | 34.36, 56.6, +0.274 | +10.8 (5/8), -124.9 | -7.0 (3/8), -545.3 | 32.42, 52.2, +0.064 |
| RO60+S120 | +106.3 (5/8), -216.6 | +209.7 (5/8), -820.1 | 31.62, 45.6, +0.231 | -23.2 (4/8), -69.1 | +42.6 (4/8), -493.1 | 29.48, 42.9, +0.016 |
| RO60+S120+CAP100 | +141.5 (6/8), -144.7 | +171.1 (5/8), -872.0 | 24.34, 35.9, +0.207 | -8.7 (5/8), -129.6 | +23.5 (5/8), -502.4 | 23.10, 34.4, +0.017 |
| S120 | +158.6 (5/8), -160.2 | +463.9 (6/8), -1359.3 | 48.38, 93.9, +0.203 | -24.8 (4/8), -169.1 | +170.0 (3/8), -792.1 | 45.64, 86.4, -0.040 |
| S120+CAP100 | +148.0 (5/8), -119.3 | +333.3 (5/8), -891.1 | 36.55, 72.7, +0.169 | -10.2 (4/8), -228.3 | +109.9 (4/8), -623.2 | 34.85, 68.2, -0.047 |
| RO120+S180 | +142.4 (4/8), -44.2 | +397.5 (6/8), -1575.1 | 36.31, 71.6, +0.198 | +17.9 (4/8), -42.1 | +231.7 (5/8), -771.3 | 34.59, 66.8, -0.049 |
| RO120+S180+CAP100 | +107.6 (5/8), +21.9 | +288.6 (5/8), -1147.5 | 27.22, 52.5, +0.173 | +12.0 (4/8), -115.4 | +150.7 (5/8), -734.8 | 25.67, 50.5, -0.059 |
| S180 | +200.7 (5/8), +37.2 | +707.7 (6/8), -501.5 | 47.07, 106.9, +0.194 | +3.9 (4/8), +41.2 | +415.7 (4/8), -534.4 | 44.54, 99.9, -0.052 |
| S180+CAP100 | +161.2 (5/8), +113.5 | +544.4 (5/8), -373.2 | 34.65, 79.9, +0.167 | +37.1 (4/8), -4.3 | +352.9 (4/8), -562.5 | 33.05, 75.9, -0.039 |
| RO180+S240 | +166.7 (5/8), +109.1 | +551.0 (6/8), -528.0 | 34.14, 85.2, +0.203 | -46.4 (5/8), -80.4 | +282.6 (4/8), -687.5 | 31.85, 81.5, -0.052 |
| RO180+S240+CAP100 | +192.5 (5/8), +94.7 | +429.7 (6/8), -280.5 | 23.97, 61.1, +0.178 | +38.9 (5/8), -2.6 | +232.7 (4/8), -552.0 | 22.99, 59.8, -0.038 |
| S240 | +142.8 (5/8), +176.8 | +517.0 (6/8), -910.7 | 39.95, 112.0, +0.208 | -87.9 (4/8), -116.3 | +140.3 (4/8), -1031.8 | 37.62, 106.2, -0.028 |
| S240+CAP100 | +196.9 (5/8), +273.1 | +341.3 (6/8), -735.5 | 28.65, 82.0, +0.195 | +19.4 (5/8), +16.2 | +80.9 (4/8), -1022.6 | 27.45, 78.7, -0.010 |

Differences, A1 20/100 (mark@stop mean (t interval) days > 0, 11-03; carry sd change, days lower; J | X):

| comparison | J mark | J carry sd | X mark | X carry sd |
|---|---|---|---|---|
| inventory: S60+CAP100 - S60 | +1.5 (-31.9..+35.0) 4/8, +21.4 | -8.68, 8/8 | -9.1 (-37.6..+19.3) 3/8, -31.0 | -8.09, 8/8 |
| inventory: RO60+S120 - S60 | -33.3 (-270.2..+203.6) 4/8, -136.9 | -11.42, 8/8 | -43.0 (-294.6..+208.5) 4/8, +24.8 | -11.03, 8/8 |
| inventory: RO60+S120+CAP100 - S60 | +1.9 (-196.0..+199.9) 4/8, -65.0 | -18.70, 8/8 | -28.6 (-213.6..+156.4) 4/8, -35.7 | -17.41, 8/8 |
| inventory: S120+CAP100 - S120 | -10.6 (-151.0..+129.8) 5/8, +40.9 | -11.83, 8/8 | +14.6 (-96.3..+125.6) 4/8, -59.1 | -10.79, 8/8 |
| inventory: RO120+S180 - S120 | -16.2 (-310.5..+278.1) 3/8, +116.0 | -12.08, 8/8 | +42.7 (-172.2..+257.6) 3/8, +127.0 | -11.05, 8/8 |
| inventory: RO120+S180+CAP100 - S120 | -51.0 (-176.2..+74.3) 4/8, +182.1 | -21.16, 8/8 | +36.8 (-158.4..+232.1) 4/8, +53.7 | -19.96, 8/8 |
| inventory: S180+CAP100 - S180 | -39.5 (-307.3..+228.4) 5/8, +76.3 | -12.42, 8/8 | +33.1 (-151.5..+217.8) 5/8, -45.6 | -11.49, 8/8 |
| inventory: RO180+S240 - S180 | -34.0 (-306.3..+238.3) 5/8, +72.0 | -12.93, 8/8 | -50.3 (-273.2..+172.6) 4/8, -121.7 | -12.70, 8/8 |
| inventory: RO180+S240+CAP100 - S180 | -8.2 (-347.0..+330.7) 4/8, +57.6 | -23.10, 8/8 | +35.0 (-211.1..+281.1) 5/8, -43.9 | -21.55, 8/8 |
| inventory: S240+CAP100 - S240 | +54.1 (-263.0..+371.1) 6/8, +96.4 | -11.30, 8/8 | +107.4 (-92.8..+307.6) 6/8, +132.4 | -10.17, 8/8 |
| window: S120 - S60 | +19.0 (-381.5..+419.4) 5/8, -80.5 | +5.34, 0/8 | -44.7 (-431.9..+342.5) 4/8, -75.2 | +5.13, 0/8 |
| window: S180 - S60 | +61.1 (-579.4..+701.6) 3/8, +116.9 | +4.03, 1/8 | -16.0 (-512.7..+480.8) 3/8, +135.1 | +4.03, 2/8 |
| window: S240 - S60 | +3.2 (-622.3..+628.7) 4/8, +256.4 | -3.09, 6/8 | -107.8 (-660.2..+444.6) 3/8, -22.4 | -2.89, 6/8 |
| window: S120+CAP100 - S60+CAP100 | +6.8 (-281.4..+295.1) 4/8, -61.0 | +2.19, 1/8 | -20.9 (-310.6..+268.8) 4/8, -103.4 | +2.43, 3/8 |
| window: S180+CAP100 - S60+CAP100 | +20.1 (-358.5..+398.7) 4/8, +171.8 | +0.29, 4/8 | +26.3 (-312.7..+365.3) 4/8, +120.5 | +0.63, 3/8 |
| window: S240+CAP100 - S60+CAP100 | +55.8 (-320.6..+432.2) 4/8, +331.4 | -5.71, 7/8 | +8.7 (-408.4..+425.7) 5/8, +141.0 | -4.97, 8/8 |
| window: RO120+S180+CAP100 - RO60+S120+CAP100 | -33.9 (-254.2..+186.3) 3/8, +166.6 | +2.88, 1/8 | +20.7 (-153.0..+194.5) 4/8, +14.2 | +2.57, 1/8 |
| window: RO180+S240+CAP100 - RO60+S120+CAP100 | +51.0 (-235.6..+337.6) 5/8, +239.4 | -0.37, 4/8 | +47.6 (-261.9..+357.1) 6/8, +127.0 | -0.11, 4/8 |

Computed over 36 comparisons x 2 timings: ADDS_J 0, ADDS_X 0, ADDS_BOTH 0; mark interval excludes 0 0, settle interval excludes 0 0; inventory comparisons with carry sd lower on 8/8 days: 40 of 40.

Reading: neither lever adds money on the fixed engine. No inventory or window comparison has a mark@stop or settle interval that
excludes 0, in either direction, at either timing. CAP and reduce-only reliably cut carried risk (carry sd lower on every read day in
all 40 inventory comparisons) at no detectable mark@stop cost or gain. Longer windows raise settle point estimates (carry to the outcome,
not spread), with wide intervals, and at bound timing 20/100 the +1 s markout turns negative from W >= 120 s. In-sample; window
comparisons value mark@stop at different stop times; multiplicity 36 comparisons per timing.

## 57. MM vs settlement P&L, +1 s against the top makers, why longer windows fail (declared 2026-11-06 02:06Z; as of 2026-11-06T02:10Z)

Question (user): the 1 s gain compared with the top makers; settlement P&L per config; MM performance and settlement P&L separately; why
longer windows do not work. Reads only the round-i invwin fills (section 56; engine sha 742637b9a2d5bd6d) and the step-1 legs / books of the
same days. Script research/pf_mm_decomp.py -> step4/report_mm_decomp.json (log logs/mm_decomp.log). Descriptive, in-sample.

Per fill (side s, shares q, price p; m0 = venue mid before the fill, m1 = mid at +1 s, mS = mid at the arm's stop + 100 ms, y = outcome
0 / 100): spread s(m0-p), informed s(m1-m0), drift s(mS-m1), carry s(y-mS). +1 s = spread + informed; MM = +1 s + drift (= mark@stop);
settlement = carry; MM + carry = settle. Makers: real legs at the engine's timing anchor (tx ws + taker-order key, without the replay's
jitter / -50 ms shift), bands top 5 / 6-20 / rest by live-window volume per day; their part after 1 s = s(y-m1).

Checks:
- Reconciliation with the engine PASS on every arm-day (56 arm files x 9 days, 1,378,386 fills): settle = engine gross, +1 s = engine mo1,
  MM = engine mtm_stop (|diff| <= $0.01). Falsifier: a copy with one fill's shares doubled fails all four checks.
- Positive control: maker +1 s c/sh with the legs_timed2 timing reproduces report_maker_bad_fills.json in all 6 band x part cells (diff 0.0000).
- Maker retimed-leg count = the engine's on every day (the script refuses otherwise). S240's fills before 60 s = S60's: 36 of 36.
- Statuses: fills without a mid before / at +1 s / at the stop 0 / 0 / 0; maker legs 5,235,731, untimed (excluded)
  51,027, no mid before 1.

A. MM vs settlement per config ($ / day, 8 read days; MM d>0 = days with MM > 0; carry sd = sd of the daily carry; 11-03 beside):


J 5/25 AV:

| config | spread | informed | +1 s | drift | MM | carry | settle | +1 s c/sh | MM d>0 | carry sd | 11-03 MM / carry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S60 | +40.6 | -1.5 | +39.1 | +6.4 | +45.5 | +9.4 | +54.8 | +0.756 | 6/8 | 179.8 | -8.1 / -81.0 |
| S60+CAP25 | +36.6 | -2.4 | +34.2 | +8.6 | +42.8 | +25.0 | +67.9 | +0.729 | 7/8 | 138.3 | -4.0 / -63.6 |
| RO60+S120 | +48.0 | -8.2 | +39.8 | -9.8 | +30.0 | +26.1 | +56.1 | +0.646 | 5/8 | 60.7 | -84.9 / +51.0 |
| RO60+S120+CAP25 | +43.7 | -9.0 | +34.7 | -4.7 | +30.0 | +37.3 | +67.3 | +0.613 | 6/8 | 46.8 | -49.3 / +38.3 |
| S120 | +76.1 | -16.9 | +59.2 | -28.4 | +30.7 | +48.2 | +79.0 | +0.607 | 5/8 | 159.8 | -62.3 / -171.2 |
| S120+CAP25 | +67.7 | -16.2 | +51.6 | -15.1 | +36.4 | +30.3 | +66.8 | +0.594 | 6/8 | 103.0 | -30.7 / -116.7 |
| RO120+S180 | +84.6 | -18.4 | +66.2 | -30.9 | +35.3 | +43.2 | +78.5 | +0.617 | 5/8 | 112.4 | -8.7 / -179.5 |
| RO120+S180+CAP25 | +75.8 | -17.4 | +58.4 | -30.5 | +27.8 | +41.2 | +69.1 | +0.607 | 6/8 | 72.5 | -10.9 / -104.5 |
| S180 | +104.8 | -17.7 | +87.1 | -38.7 | +48.5 | +74.1 | +122.5 | +0.656 | 5/8 | 161.0 | +30.0 / -163.3 |
| S180+CAP25 | +93.1 | -16.6 | +76.5 | -37.1 | +39.3 | +58.3 | +97.7 | +0.648 | 5/8 | 123.0 | -2.9 / -17.9 |
| RO180+S240 | +112.9 | -16.2 | +96.7 | -58.5 | +38.2 | +58.5 | +96.7 | +0.679 | 6/8 | 91.6 | -117.1 / -18.7 |
| RO180+S240+CAP25 | +100.5 | -15.7 | +84.8 | -38.2 | +46.6 | +32.9 | +79.5 | +0.670 | 7/8 | 62.5 | -72.9 / +41.0 |
| S240 | +129.8 | -20.2 | +109.6 | -69.9 | +39.7 | +49.9 | +89.6 | +0.675 | 7/8 | 116.9 | -161.4 / -77.5 |
| S240+CAP25 | +114.3 | -20.7 | +93.6 | -51.6 | +42.0 | +13.7 | +55.7 | +0.653 | 7/8 | 80.6 | -85.3 / +15.0 |

X 5/25 AV:

| config | spread | informed | +1 s | drift | MM | carry | settle | +1 s c/sh | MM d>0 | carry sd | 11-03 MM / carry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S60 | +33.1 | -12.2 | +20.9 | -3.2 | +17.7 | +19.6 | +37.3 | +0.483 | 6/8 | 134.8 | -6.2 / -103.7 |
| S60+CAP25 | +30.5 | -11.3 | +19.2 | -0.3 | +18.9 | +22.5 | +41.4 | +0.479 | 7/8 | 107.0 | -11.6 / -86.0 |
| RO60+S120 | +39.2 | -18.8 | +20.4 | +3.2 | +23.7 | +20.4 | +44.1 | +0.397 | 5/8 | 62.7 | -75.5 / +5.6 |
| RO60+S120+CAP25 | +36.4 | -17.8 | +18.7 | -0.1 | +18.5 | +26.3 | +44.8 | +0.389 | 5/8 | 45.4 | -59.7 / +0.5 |
| S120 | +61.6 | -34.3 | +27.3 | -3.8 | +23.5 | +46.4 | +69.9 | +0.334 | 5/8 | 90.3 | -53.0 / -154.4 |
| S120+CAP25 | +56.3 | -32.0 | +24.4 | -7.7 | +16.7 | +41.7 | +58.4 | +0.328 | 6/8 | 61.0 | -55.3 / -124.3 |
| RO120+S180 | +68.1 | -36.3 | +31.9 | -6.0 | +25.9 | +49.0 | +74.8 | +0.353 | 4/8 | 70.9 | +15.5 / -163.8 |
| RO120+S180+CAP25 | +62.5 | -33.2 | +29.3 | -10.5 | +18.7 | +43.8 | +62.6 | +0.355 | 5/8 | 50.4 | -18.6 / -119.9 |
| S180 | +85.3 | -43.0 | +42.3 | -9.1 | +33.2 | +78.7 | +112.0 | +0.376 | 4/8 | 101.2 | +34.7 / -176.0 |
| S180+CAP25 | +77.3 | -38.5 | +38.8 | -16.8 | +21.9 | +59.9 | +81.9 | +0.380 | 4/8 | 80.2 | -10.7 / -101.7 |
| RO180+S240 | +90.9 | -45.3 | +45.6 | -16.9 | +28.7 | +56.0 | +84.6 | +0.379 | 5/8 | 69.0 | -73.1 / -75.7 |
| RO180+S240+CAP25 | +82.6 | -41.4 | +41.2 | -24.4 | +16.8 | +38.5 | +55.3 | +0.379 | 5/8 | 54.3 | -76.3 / -48.3 |
| S240 | +104.8 | -50.4 | +54.4 | -30.6 | +23.8 | +25.8 | +49.6 | +0.394 | 4/8 | 97.1 | -107.3 / -99.5 |
| S240+CAP25 | +94.5 | -44.4 | +50.1 | -38.9 | +11.3 | +0.9 | +12.2 | +0.405 | 5/8 | 82.2 | -97.9 / -60.8 |

J 20/100 A1:

| config | spread | informed | +1 s | drift | MM | carry | settle | +1 s c/sh | MM d>0 | carry sd | 11-03 MM / carry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S60 | +190.9 | -98.6 | +92.3 | +47.3 | +139.6 | +41.0 | +180.6 | +0.305 | 5/8 | 923.8 | -79.7 / +223.7 |
| S60+CAP100 | +169.8 | -96.4 | +73.4 | +67.7 | +141.1 | +1.6 | +142.8 | +0.274 | 5/8 | 693.8 | -58.3 / +69.0 |
| RO60+S120 | +232.0 | -146.8 | +85.2 | +21.1 | +106.3 | +103.4 | +209.7 | +0.231 | 5/8 | 431.5 | -216.6 / +567.8 |
| RO60+S120+CAP100 | +207.6 | -139.6 | +68.1 | +73.5 | +141.5 | +29.6 | +171.1 | +0.207 | 6/8 | 350.5 | -144.7 / +405.8 |
| S120 | +353.0 | -239.5 | +113.5 | +45.1 | +158.6 | +305.3 | +463.9 | +0.203 | 5/8 | 965.0 | -160.2 / +676.7 |
| S120+CAP100 | +308.1 | -225.7 | +82.5 | +65.5 | +148.0 | +185.3 | +333.3 | +0.169 | 5/8 | 663.2 | -119.3 / +471.4 |
| RO120+S180 | +396.2 | -272.7 | +123.5 | +18.9 | +142.4 | +255.1 | +397.5 | +0.198 | 4/8 | 683.9 | -44.2 / +192.5 |
| RO120+S180+CAP100 | +346.9 | -252.3 | +94.6 | +13.0 | +107.6 | +181.0 | +288.6 | +0.173 | 5/8 | 488.1 | +21.9 / -13.7 |
| S180 | +482.2 | -335.9 | +146.3 | +54.4 | +200.7 | +507.0 | +707.7 | +0.194 | 5/8 | 754.1 | +37.2 / +558.4 |
| S180+CAP100 | +420.0 | -310.3 | +109.8 | +51.4 | +161.2 | +383.2 | +544.4 | +0.167 | 5/8 | 628.1 | +113.4 / +292.8 |
| RO180+S240 | +524.6 | -358.0 | +166.6 | +0.1 | +166.7 | +384.3 | +551.0 | +0.203 | 5/8 | 432.7 | +109.1 / +507.0 |
| RO180+S240+CAP100 | +457.3 | -330.3 | +127.0 | +65.5 | +192.5 | +237.2 | +429.7 | +0.178 | 5/8 | 313.2 | +94.7 / +329.0 |
| S240 | +590.5 | -399.1 | +191.4 | -48.6 | +142.8 | +374.2 | +517.0 | +0.208 | 5/8 | 632.6 | +176.8 / +523.2 |
| S240+CAP100 | +514.3 | -358.6 | +155.7 | +41.2 | +196.9 | +144.3 | +341.3 | +0.195 | 5/8 | 512.6 | +273.1 / +365.3 |

X 20/100 A1:

| config | spread | informed | +1 s | drift | MM | carry | settle | +1 s c/sh | MM d>0 | carry sd | 11-03 MM / carry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S60 | +167.3 | -148.9 | +18.4 | +1.5 | +19.9 | +17.5 | +37.3 | +0.069 | 6/8 | 667.6 | -93.9 / +401.9 |
| S60+CAP100 | +150.5 | -135.3 | +15.2 | -4.5 | +10.8 | -17.8 | -7.0 | +0.064 | 5/8 | 497.4 | -124.9 / +198.7 |
| RO60+S120 | +201.4 | -196.4 | +5.0 | -28.2 | -23.2 | +65.7 | +42.6 | +0.016 | 4/8 | 360.6 | -69.1 / +491.5 |
| RO60+S120+CAP100 | +182.4 | -177.6 | +4.8 | -13.6 | -8.7 | +32.2 | +23.4 | +0.017 | 5/8 | 236.1 | -129.6 / +295.5 |
| S120 | +309.5 | -329.4 | -19.9 | -4.9 | -24.8 | +194.8 | +170.0 | -0.040 | 4/8 | 665.4 | -169.1 / +583.4 |
| S120+CAP100 | +272.5 | -293.0 | -20.5 | +10.3 | -10.2 | +120.1 | +109.9 | -0.047 | 4/8 | 478.7 | -228.3 / +440.0 |
| RO120+S180 | +345.6 | -372.8 | -27.2 | +45.1 | +17.9 | +213.8 | +231.7 | -0.049 | 4/8 | 433.6 | -42.1 / +298.9 |
| RO120+S180+CAP100 | +305.5 | -333.9 | -28.4 | +40.5 | +12.0 | +138.7 | +150.7 | -0.059 | 4/8 | 327.9 | -115.4 / +245.9 |
| S180 | +424.9 | -459.7 | -34.9 | +38.8 | +3.9 | +411.8 | +415.7 | -0.052 | 4/8 | 588.3 | +41.2 / +681.3 |
| S180+CAP100 | +370.7 | -393.7 | -23.0 | +60.1 | +37.1 | +315.9 | +352.9 | -0.039 | 4/8 | 454.0 | -4.4 / +436.2 |
| RO180+S240 | +458.3 | -495.7 | -37.4 | -9.0 | -46.4 | +329.0 | +282.6 | -0.052 | 5/8 | 430.8 | -80.4 / +754.1 |
| RO180+S240+CAP100 | +400.0 | -423.8 | -23.8 | +62.7 | +38.9 | +193.8 | +232.7 | -0.038 | 5/8 | 311.9 | -2.6 / +438.0 |
| S240 | +517.8 | -540.4 | -22.6 | -65.3 | -87.9 | +228.2 | +140.3 | -0.028 | 4/8 | 549.5 | -116.3 / +955.7 |
| S240+CAP100 | +451.1 | -458.1 | -7.0 | +26.4 | +19.4 | +61.4 | +80.9 | -0.010 | 5/8 | 463.5 | +16.2 / +652.3 |

B. By fill time in the window (8 read days pooled; c / share; post 1 s = from +1 s to the outcome; |m-50| = mean distance of the mid
from 50 c at the fill). Ours: S240, which quotes the whole window (S60 = its first row: the fills before 60 s are identical).

| who | bucket s | sh / day | spread | informed | +1 s | post 1 s | settle |
|---|---|---|---|---|---|---|---|
| makers top 5 | 0-60 | 859,445 | +1.26 | -0.83 | +0.43 | +0.05 | +0.48 |
| makers top 5 | 60-120 | 702,219 | +1.23 | -0.91 | +0.33 | -0.11 | +0.22 |
| makers top 5 | 120-180 | 622,788 | +1.23 | -0.96 | +0.27 | +0.16 | +0.43 |
| makers top 5 | 180-240 | 549,276 | +1.34 | -1.14 | +0.20 | -0.79 | -0.59 |
| makers 6-20 | 0-60 | 790,909 | +0.96 | -0.58 | +0.38 | +0.38 | +0.76 |
| makers 6-20 | 60-120 | 671,039 | +0.93 | -0.61 | +0.31 | +0.18 | +0.49 |
| makers 6-20 | 120-180 | 639,750 | +0.93 | -0.66 | +0.27 | -0.40 | -0.13 |
| makers 6-20 | 180-240 | 533,069 | +0.90 | -0.72 | +0.18 | -0.54 | -0.36 |
| makers rest | 0-60 | 833,713 | +1.03 | -1.39 | -0.36 | +0.01 | -0.34 |
| makers rest | 60-120 | 689,622 | +0.93 | -1.18 | -0.25 | +0.49 | +0.24 |
| makers rest | 120-180 | 766,305 | +0.89 | -1.07 | -0.19 | +0.50 | +0.31 |
| makers rest | 180-240 | 898,769 | +0.75 | -0.89 | -0.14 | -0.21 | -0.36 |
| makers all makers | 0-60 | 2,484,068 | +1.09 | -0.94 | +0.15 | +0.14 | +0.29 |
| makers all makers | 60-120 | 2,062,880 | +1.03 | -0.90 | +0.13 | +0.19 | +0.32 |
| makers all makers | 120-180 | 2,028,843 | +1.00 | -0.91 | +0.10 | +0.11 | +0.21 |
| makers all makers | 180-240 | 1,981,115 | +0.95 | -0.92 | +0.04 | -0.46 | -0.42 |
| ours J s5 S240 | 0-60 | 5,162 | +0.79 | -0.03 | +0.76 | +0.33 | +1.08 |
| ours J s5 S240 | 60-120 | 4,581 | +0.78 | -0.34 | +0.44 | +0.13 | +0.57 |
| ours J s5 S240 | 120-180 | 3,546 | +0.81 | -0.02 | +0.79 | +0.35 | +1.14 |
| ours J s5 S240 | 180-240 | 2,951 | +0.85 | -0.09 | +0.76 | -1.88 | -1.12 |
| ours X s5 S240 | 0-60 | 4,328 | +0.76 | -0.28 | +0.49 | +0.39 | +0.87 |
| ours X s5 S240 | 60-120 | 3,835 | +0.74 | -0.58 | +0.16 | +0.77 | +0.94 |
| ours X s5 S240 | 120-180 | 3,098 | +0.77 | -0.28 | +0.48 | +0.75 | +1.23 |
| ours X s5 S240 | 180-240 | 2,526 | +0.77 | -0.30 | +0.48 | -2.95 | -2.47 |
| ours J s20 S240 | 0-60 | 30,293 | +0.63 | -0.33 | +0.30 | +0.30 | +0.60 |
| ours J s20 S240 | 60-120 | 25,683 | +0.63 | -0.55 | +0.08 | +1.06 | +1.15 |
| ours J s20 S240 | 120-180 | 19,620 | +0.66 | -0.49 | +0.17 | +1.02 | +1.19 |
| ours J s20 S240 | 180-240 | 16,263 | +0.67 | -0.39 | +0.28 | -1.46 | -1.19 |
| ours X s20 S240 | 0-60 | 26,717 | +0.62 | -0.55 | +0.07 | +0.09 | +0.16 |
| ours X s20 S240 | 60-120 | 22,589 | +0.63 | -0.80 | -0.17 | +0.81 | +0.64 |
| ours X s20 S240 | 120-180 | 17,943 | +0.65 | -0.73 | -0.08 | +1.36 | +1.27 |
| ours X s20 S240 | 180-240 | 14,105 | +0.66 | -0.57 | +0.09 | -2.04 | -1.95 |

C. Ours (S240) minus top 5 makers, +1 s c / share per bucket: day-clustered mean (t interval, 7 df), days ours > top 5:

| bucket s | J 5/25 | X 5/25 | J 20/100 | X 20/100 |
|---|---|---|---|---|
| 0-60 | +0.27 (-0.27..+0.82) 5/8 | +0.02 (-0.51..+0.55) 4/8 | -0.11 (-0.45..+0.22) 3/8 | -0.34 (-0.66..-0.02) 2/8 |
| 60-120 | +0.09 (-0.21..+0.40) 3/8 | -0.17 (-0.53..+0.18) 3/8 | -0.24 (-0.58..+0.10) 2/8 | -0.48 (-0.90..-0.06) 1/8 |
| 120-180 | +0.54 (-0.05..+1.13) 6/8 | +0.25 (-0.42..+0.91) 4/8 | -0.11 (-0.52..+0.31) 4/8 | -0.37 (-0.89..+0.15) 3/8 |
| 180-240 | +0.71 (+0.05..+1.36) 6/8 | +0.39 (-0.29..+1.07) 5/8 | +0.12 (-0.32..+0.57) 4/8 | -0.11 (-0.59..+0.37) 4/8 |

D. Why longer windows: config minus S60, $ / day, day-clustered mean (t interval, 7 df), 8 read days:

| comparison | spread | informed | drift | MM | carry | settle |
|---|---|---|---|---|---|---|
| J/s5/S120 - S60 | +35.5 (+28.2..+42.9) | -15.4 (-25.9..-5.0) | -34.9 (-105.6..+35.9) | -14.7 (-90.5..+61.0) | +38.9 (-105.3..+183.0) | +24.1 (-82.7..+130.9) |
| J/s5/S180 - S60 | +64.2 (+51.8..+76.7) | -16.1 (-40.6..+8.3) | -45.1 (-118.0..+27.9) | +3.0 (-84.1..+90.1) | +64.7 (-115.9..+245.3) | +67.7 (-100.3..+235.7) |
| J/s5/S240 - S60 | +89.2 (+72.7..+105.8) | -18.7 (-57.1..+19.7) | -76.3 (-193.7..+41.0) | -5.8 (-136.5..+125.0) | +40.5 (-114.7..+195.7) | +34.8 (-120.1..+189.6) |
| J/s5/S240+CAP25 - S60+CAP25 | +77.7 (+63.0..+92.3) | -18.3 (-50.6..+13.9) | -60.2 (-146.8..+26.4) | -0.8 (-96.2..+94.5) | -11.4 (-126.9..+104.2) | -12.2 (-124.0..+99.6) |
| X/s5/S120 - S60 | +28.5 (+23.6..+33.3) | -22.1 (-34.4..-9.8) | -0.6 (-67.8..+66.7) | +5.8 (-61.7..+73.4) | +26.8 (-51.5..+105.1) | +32.6 (-23.9..+89.2) |
| X/s5/S180 - S60 | +52.2 (+43.4..+60.9) | -30.8 (-56.4..-5.1) | -5.8 (-76.7..+65.0) | +15.6 (-46.6..+77.7) | +59.1 (-56.6..+174.9) | +74.7 (-41.1..+190.5) |
| X/s5/S240 - S60 | +71.7 (+57.8..+85.5) | -38.2 (-76.2..-0.3) | -27.3 (-117.8..+63.1) | +6.1 (-80.3..+92.5) | +6.2 (-116.9..+129.2) | +12.3 (-106.5..+131.1) |
| X/s5/S240+CAP25 - S60+CAP25 | +64.0 (+50.5..+77.6) | -33.0 (-68.2..+2.1) | -38.6 (-93.3..+16.1) | -7.7 (-62.9..+47.5) | -21.6 (-130.9..+87.8) | -29.2 (-143.0..+84.5) |
| J/s20/S120 - S60 | +162.1 (+133.0..+191.2) | -140.9 (-228.5..-53.4) | -2.2 (-376.4..+371.9) | +19.0 (-381.5..+419.4) | +264.3 (-360.5..+889.2) | +283.3 (-291.8..+858.4) |
| J/s20/S180 - S60 | +291.3 (+240.7..+341.9) | -237.3 (-388.4..-86.2) | +7.1 (-576.0..+590.2) | +61.1 (-579.4..+701.6) | +466.0 (-288.7..+1220.8) | +527.1 (-132.1..+1186.3) |
| J/s20/S240 - S60 | +399.6 (+333.0..+466.2) | -300.5 (-502.8..-98.2) | -95.9 (-665.2..+473.4) | +3.2 (-622.3..+628.7) | +333.2 (-666.8..+1333.2) | +336.5 (-356.7..+1029.6) |
| J/s20/S240+CAP100 - S60+CAP100 | +344.5 (+286.5..+402.5) | -262.2 (-432.2..-92.2) | -26.5 (-377.9..+324.8) | +55.8 (-320.6..+432.2) | +142.7 (-701.4..+986.7) | +198.5 (-536.6..+933.5) |
| X/s20/S120 - S60 | +142.2 (+117.8..+166.5) | -180.5 (-278.2..-82.8) | -6.4 (-378.6..+365.8) | -44.7 (-431.9..+342.5) | +177.3 (-150.3..+505.0) | +132.6 (-246.9..+512.2) |
| X/s20/S180 - S60 | +257.6 (+215.1..+300.1) | -310.9 (-480.2..-141.5) | +37.3 (-410.7..+485.3) | -16.0 (-512.8..+480.8) | +394.4 (-112.4..+901.1) | +378.4 (-104.1..+860.9) |
| X/s20/S240 - S60 | +350.5 (+286.2..+414.7) | -391.5 (-608.7..-174.3) | -66.8 (-558.3..+424.7) | -107.8 (-660.2..+444.6) | +210.8 (-540.1..+961.6) | +103.0 (-464.2..+670.1) |
| X/s20/S240+CAP100 - S60+CAP100 | +300.6 (+244.1..+357.1) | -322.8 (-514.0..-131.6) | +30.9 (-309.6..+371.3) | +8.7 (-408.4..+425.7) | +79.2 (-584.5..+742.9) | +87.9 (-483.0..+658.8) |

Reading (each claim computed from report_mm_decomp.json):
- MM vs settlement. Spread is positive and informed (1 s adverse selection) negative in all 56 configs; drift (inventory from 1 s to the
  stop) is positive in 24 of 56. Only S60 and S60+CAP25 at 5/25 central timing have an MM (mark@stop) interval above 0. Carry (settlement)
  has a daily sd of 45-180 $ at
  5/25 and 236-965 $ at 20/100, far above its mean; 2 of 56 carry intervals exclude 0 (about chance) and no settle interval does. The
  settlement part is a bet on the outcome, not market making.
- +1 s vs the top makers. Top-5 makers earn +0.43 c/sh at 1 s in 0-60 s, falling every minute to +0.20 in 180-240 s (all makers +0.15 ->
  +0.04). Ours at 5/25 central timing (+0.76 / +0.44 / +0.79 / +0.76) is above the top 5 in every minute (interval above 0 only in
  180-240 s); at 5/25 bound timing the difference is -0.17 .. +0.39, no interval excludes 0; at 20/100 bound timing ours is below the top 5
  in 0-60 s (-0.34) and 60-120 s (-0.48), intervals below 0. Our timing is the replay's
  (jitter / -50 ms shift), theirs the measured anchor, so the X comparison is pessimistic for us.
- Why longer windows fail. Quoting past 60 s adds spread in all 16 comparisons (interval above 0), but informed loss grows with it (below 0
  in 12 of 16: all 8 at 20/100, 3 of 4 at 5/25 bound, 1 of 4 at 5/25 central), and drift falls (point estimate negative in all 8 at 5/25,
  5 of 8 at 20/100; wide intervals); MM change is 0 within the intervals in all 16.
  The market shows the same: later in the window the mid is further from 50 c (all makers' fills: 11 -> 32 c). After the first second, every maker band loses in the last minute (post 1 s: top 5 -0.79, all
  -0.46 c/sh), and every arm of ours that quotes through it (16 arms, >= 690 sh / day) loses -1.5 .. -4.2 c/sh after 1 s although its
  +1 s stays >= 0 at 5/25. Interpretation (not tested): the 1 s guard protects the first second, not the slower move to the outcome, so
  late fills buy inventory on the losing side as the outcome becomes clear.

## 58. Per-market feed refresh over the whole window, fixed live engine round i (declared 2026-11-06 02:38Z; as of 2026-11-06T04:15Z)

Question (user): the per-frame guard refresh was built only for the first 65 s (section 55); test it over the whole window, at both sizes,
bound timing and longer windows. Refreshed = preds_t60kF: the same guard models (k 60) evaluated at every Predict frame's known time in
[ws, ws + 240 s), merged with the 50 ms grid. Grid = the invwin runs of section 56 (preds_t60k), same arm names. Arms MICRO + D1 stack +
V0 + GS + XNA + XMIN1 + XV + S60 / S120 / S180 / S240 at AV:5:25:0.06 (s5) and A1:20:100:0 (s20); fixed live engine (sha
742637b9a2d5bd6d), tokyoreal clock; central J and bound X timing; 9 days. Exploratory, IN-SAMPLE (guard fit on the read days). Driver
research/run_fresh240.sh, report research/pf_fresh240_report.py -> step4/report_fresh240.json (logs fw_ctrl_cmp.log, fw_report.log).

Checks:
- Inputs: on all 9 days preds_t60kF restricted to [ws, ws + 65 s) = preds_t60kE row for row (708,721 .. 736,404 rows / day compared; the
  merge refuses otherwise). Guard thresholds of the 240 s build = the grid build's (t_60) on all 9 days (a 1e-12 perturbation is flagged).
  The every-frame feature rows over 0-65 s = the 65 s build (10-24, 354,977 rows).
- Gate PASS (10-24, 11-01, J, both sizes): S60 on preds_t60kF = S60 on preds_t60kE (summary + per-window); S240 on preds_t60kF differs from
  S240 on the grid (falsifier fires on all 4 day x size cells).
- The report refuses unless refreshed and grid runs have the same engine code and settings apart from the prediction file; no statuses
  (no missing run).
- s5/S60 at X reproduces round i's fresh-minus-grid "today" arm from the ladder engine (12.3 (+0.9..+23.6), 7/8; 11-03 +1.3 here vs +1.4
  there, not investigated).
- Fills at or after the stop (cancel in flight, <= 13 ms): 0-12 per arm over 9 days; those minute bins are omitted below.

A. Per arm, refreshed (F) and grid (G), $ / day over the 8 read days; +1 s c/sh by minute up to the stop; 11-03 beside:

| timing / size / stop | src | MM (days > 0) | +1 s $ | settle | sh / day | +1 s c/sh | by minute | 11-03 MM |
|---|---|---|---|---|---|---|---|---|
| J s5 S60 | F | +50.3 (8/8) | +39.7 | +69.7 | 4978 | +0.797 | +0.80 | -27.3 |
| J s5 S60 | G | +45.5 (6/8) | +39.1 | +54.8 | 5167 | +0.756 | +0.76 | -8.1 |
| X s5 S60 | F | +29.9 (6/8) | +22.5 | +42.0 | 4147 | +0.543 | +0.55 | -4.8 |
| X s5 S60 | G | +17.7 (6/8) | +20.9 | +37.3 | 4334 | +0.483 | +0.49 | -6.2 |
| J s5 S120 | F | +43.6 (5/8) | +59.7 | +91.4 | 9379 | +0.637 | +0.80 / +0.46 | -106.9 |
| J s5 S120 | G | +30.7 (5/8) | +59.2 | +79.0 | 9747 | +0.607 | +0.76 / +0.44 | -62.3 |
| X s5 S120 | F | +25.5 (6/8) | +30.2 | +60.5 | 7923 | +0.381 | +0.55 / +0.20 | -61.1 |
| X s5 S120 | G | +23.5 (5/8) | +27.3 | +69.9 | 8172 | +0.334 | +0.49 / +0.16 | -53.0 |
| J s5 S180 | F | +58.4 (5/8) | +87.4 | +137.6 | 12804 | +0.683 | +0.80 / +0.46 / +0.81 | -2.6 |
| J s5 S180 | G | +48.5 (5/8) | +87.1 | +122.5 | 13289 | +0.656 | +0.76 / +0.44 / +0.79 | +30.0 |
| X s5 S180 | F | +40.8 (6/8) | +46.7 | +93.2 | 10862 | +0.430 | +0.55 / +0.20 / +0.56 | +17.5 |
| X s5 S180 | G | +33.2 (4/8) | +42.3 | +112.0 | 11262 | +0.376 | +0.49 / +0.16 / +0.48 | +34.7 |
| J s5 S240 | F | +65.9 (7/8) | +113.3 | +117.4 | 15626 | +0.725 | +0.80 / +0.46 / +0.81 / +0.92 | -80.3 |
| J s5 S240 | G | +39.7 (7/8) | +109.6 | +89.6 | 16240 | +0.675 | +0.76 / +0.44 / +0.79 / +0.76 | -161.4 |
| X s5 S240 | F | +17.1 (5/8) | +59.5 | +32.1 | 13300 | +0.447 | +0.55 / +0.20 / +0.56 / +0.52 | -110.8 |
| X s5 S240 | G | +23.8 (4/8) | +54.4 | +49.6 | 13787 | +0.394 | +0.49 / +0.16 / +0.48 / +0.48 | -107.3 |
| J s20 S60 | F | +153.7 (5/8) | +95.4 | +155.5 | 30198 | +0.316 | +0.32 | -105.3 |
| J s20 S60 | G | +139.6 (5/8) | +92.3 | +180.6 | 30297 | +0.305 | +0.30 | -79.7 |
| X s20 S60 | F | +30.8 (5/8) | +20.1 | +87.0 | 26891 | +0.075 | +0.08 | -132.3 |
| X s20 S60 | G | +19.9 (6/8) | +18.4 | +37.3 | 26731 | +0.069 | +0.07 | -93.9 |
| J s20 S120 | F | +116.1 (5/8) | +107.8 | +534.2 | 55742 | +0.193 | +0.32 / +0.05 | -421.3 |
| J s20 S120 | G | +158.6 (5/8) | +113.5 | +463.9 | 55994 | +0.203 | +0.30 / +0.08 | -160.2 |
| X s20 S120 | F | -16.2 (5/8) | -18.3 | +207.4 | 49511 | -0.037 | +0.08 / -0.17 | -421.3 |
| X s20 S120 | G | -24.8 (4/8) | -19.9 | +170.0 | 49343 | -0.040 | +0.07 / -0.17 | -169.1 |
| J s20 S180 | F | +120.6 (4/8) | +145.4 | +710.9 | 75359 | +0.193 | +0.32 / +0.05 / +0.19 | -144.5 |
| J s20 S180 | G | +200.7 (5/8) | +146.3 | +707.7 | 75610 | +0.194 | +0.30 / +0.08 / +0.17 | +37.2 |
| X s20 S180 | F | -9.5 (4/8) | -31.5 | +276.4 | 67232 | -0.047 | +0.08 / -0.17 / -0.07 | -290.9 |
| X s20 S180 | G | +3.9 (4/8) | -34.9 | +415.7 | 67257 | -0.052 | +0.07 / -0.17 / -0.08 | +41.2 |
| J s20 S240 | F | +154.1 (5/8) | +188.3 | +534.2 | 91429 | +0.206 | +0.32 / +0.05 / +0.19 / +0.27 | -14.3 |
| J s20 S240 | G | +142.8 (5/8) | +191.4 | +517.0 | 91858 | +0.208 | +0.30 / +0.08 / +0.17 / +0.28 | +176.8 |
| X s20 S240 | F | -53.6 (4/8) | -11.9 | +18.1 | 81175 | -0.015 | +0.08 / -0.17 / -0.07 / +0.14 | -372.2 |
| X s20 S240 | G | -87.9 (4/8) | -22.6 | +140.3 | 81354 | -0.028 | +0.07 / -0.17 / -0.08 / +0.09 | -116.3 |

B. Refreshed minus grid, day-clustered mean (t interval, 7 df), days > 0; 11-03 beside; * = interval excludes 0:

| size / stop | timing | MM | +1 s $ | settle | sh / day | 11-03 MM | +1 s c/sh by minute (up to the stop) |
|---|---|---|---|---|---|---|---|
| s5/S60 | J | +4.8 (-8.8..+18.4) 5/8 | +0.6 (-4.4..+5.6) | +14.8 (-63.9..+93.6) | -188 | -19.2 | 0-60 +0.04 (-0.06..+0.14) |
| s5/S60 | X | +12.3 (+0.9..+23.6)* 7/8 | +1.6 (-1.7..+4.9) | +4.7 (-56.2..+65.7) | -188 | +1.3 | 0-60 +0.05 (-0.04..+0.13) |
| s5/S120 | J | +12.8 (-34.5..+60.1) 6/8 | +0.6 (-5.9..+7.1) | +12.5 (-84.2..+109.2) | -368 | -44.6 | 0-60 +0.04 (-0.06..+0.14); 60-120 +0.03 (-0.05..+0.12) |
| s5/S120 | X | +2.0 (-27.3..+31.3) 4/8 | +2.9 (-2.2..+8.0) | -9.4 (-77.3..+58.6) | -249 | -8.1 | 0-60 +0.05 (-0.04..+0.13); 60-120 +0.03 (-0.05..+0.11) |
| s5/S180 | J | +9.9 (-33.1..+52.9) 5/8 | +0.3 (-8.8..+9.4) | +15.1 (-67.7..+97.9) | -485 | -32.5 | 0-60 +0.04 (-0.06..+0.14); 60-120 +0.03 (-0.05..+0.12); 120-180 +0.02 (-0.10..+0.14) |
| s5/S180 | X | +7.6 (-27.0..+42.1) 5/8 | +4.4 (-0.5..+9.3) | -18.7 (-82.5..+45.1) | -399 | -17.2 | 0-60 +0.05 (-0.04..+0.13); 60-120 +0.03 (-0.05..+0.11); 120-180 +0.08 (-0.05..+0.22) |
| s5/S240 | J | +26.1 (-25.2..+77.5) 7/8 | +3.7 (-6.7..+14.1) | +27.9 (-72.4..+128.1) | -614 | +81.1 | 0-60 +0.04 (-0.06..+0.14); 60-120 +0.03 (-0.05..+0.12); 120-180 +0.02 (-0.10..+0.14); 180-240 +0.19 (+0.00..+0.37)* |
| s5/S240 | X | -6.7 (-52.3..+39.0) 4/8 | +5.2 (-2.0..+12.3) | -17.4 (-68.1..+33.2) | -487 | -3.5 | 0-60 +0.05 (-0.04..+0.13); 60-120 +0.03 (-0.05..+0.11); 120-180 +0.08 (-0.05..+0.22); 180-240 +0.06 (-0.08..+0.21) |
| s20/S60 | J | +14.1 (-40.9..+69.1) 4/8 | +3.0 (-10.9..+16.9) | -25.1 (-161.4..+111.2) | -98 | -25.6 | 0-60 +0.01 (-0.03..+0.06) |
| s20/S60 | X | +10.9 (-24.0..+45.8) 6/8 | +1.7 (-13.5..+16.8) | +49.7 (-58.9..+158.3) | +160 | -38.4 | 0-60 +0.00 (-0.05..+0.06) |
| s20/S120 | J | -42.4 (-181.1..+96.2) 4/8 | -5.7 (-18.9..+7.6) | +70.3 (-108.0..+248.5) | -251 | -261.1 | 0-60 +0.01 (-0.03..+0.06); 60-120 -0.04 (-0.08..+0.01) |
| s20/S120 | X | +8.7 (-95.1..+112.4) 5/8 | +1.6 (-16.0..+19.2) | +37.5 (-128.1..+203.0) | +168 | -252.1 | 0-60 +0.00 (-0.05..+0.06); 60-120 -0.01 (-0.04..+0.02) |
| s20/S180 | J | -80.1 (-247.6..+87.4) 2/8 | -0.9 (-20.8..+18.9) | +3.2 (-232.5..+239.0) | -252 | -181.6 | 0-60 +0.01 (-0.03..+0.06); 60-120 -0.04 (-0.08..+0.01); 120-180 +0.02 (-0.04..+0.09) |
| s20/S180 | X | -13.4 (-120.7..+93.8) 4/8 | +3.3 (-8.8..+15.4) | -139.3 (-260.3..-18.3)* | -25 | -332.1 | 0-60 +0.00 (-0.05..+0.06); 60-120 -0.01 (-0.04..+0.02); 120-180 +0.01 (-0.03..+0.05) |
| s20/S240 | J | +11.3 (-244.0..+266.6) 3/8 | -3.1 (-41.0..+34.6) | +17.1 (-268.7..+303.0) | -430 | -191.1 | 0-60 +0.01 (-0.03..+0.06); 60-120 -0.04 (-0.08..+0.01); 120-180 +0.02 (-0.04..+0.09); 180-240 +0.01 (-0.13..+0.15) |
| s20/S240 | X | +34.3 (-133.6..+202.3) 5/8 | +10.7 (-5.4..+26.8) | -122.1 (-375.1..+130.9) | -179 | -255.9 | 0-60 +0.00 (-0.05..+0.06); 60-120 -0.01 (-0.04..+0.02); 120-180 +0.01 (-0.03..+0.05); 180-240 +0.05 (-0.05..+0.14) |

Reading (each claim computed from report_fresh240.json):
- The refresh past 65 s adds no market-making money. ADDS_BOTH 0 of 8, ADDS_J 0 of 8, ADDS_X 1 of 8 (s5/S60, which cannot use the refresh
  after 65 s: it is the first-65 s result already in section 55). MM intervals excluding 0: 1 of 16 (that one); +1 s $: 0 of 16; settle:
  1 of 16, negative (s20/S180 X -139.3).
- Direction only: at 5/25 the refreshed +1 s c/sh is above the grid in all 8 distinct minute cells (4 minutes x 2 timings; +0.02 .. +0.19), and the MM point
  estimate is positive in 7 of 8; at 20/100 5 of 8 cells and 5 of 8 MM point estimates. The one per-minute cell excluding 0 (s5 J 180-240,
  +0.19, lower end +0.00, 6/8 days) is 1 of 16 distinct cells, about chance. (Minute cells repeat across stops: fills before a stop do not
  depend on later stops; 16 distinct of 40 printed, repeats identical.)
- The refresh pulls quotes more often: shares / day lower in 14 of 16 comparisons (-25 .. -614).
- 11-03: refreshed minus grid MM < 0 in 14 of 16.
- Not measured here: the post-1 s loss by minute (section 57's last-minute diagnosis) on the refreshed runs.

## 59. Whole-window refresh: MM vs settlement and +1 s against the top makers (declared 2026-11-06 05:04Z; as of 2026-11-06T05:05Z)

Question (user): measure the +1 s gain against the top makers, together with settlement P&L, for the refreshed runs of section 58. Reads
only existing fills: refreshed F = fwJ / fwX (preds_t60kF), grid G = invwin runs (preds_t60k); arms S60 / S120 / S180 / S240 at 5/25 AV
and 20/100 A1, J and X; same per-fill terms as section 57 (spread / informed / drift / carry; post-1 s = drift + carry = s(y - m1)).
Script research/pf_mm_decomp_fw.py -> step4/report_mm_decomp_fw.json (log logs/mm_decomp_fw.log). Descriptive, in-sample.

Checks (each fails the run): reconciliation with the engine on every arm-day of both sources (845,103 fills; settle = gross, +1 s = mo1,
MM = mtm_stop, |diff| <= $0.01), falsifier fired (one fill's shares doubled fails all four); the grid side reproduces report_mm_decomp.json
in all 112 config x term cells and a perturbed copy is refused; maker band x minute +1 s = report_mm_decomp.json; refreshed and grid runs
share input bounds and the maker retimed-leg count. Statuses: fills without a mid before / at +1 s / at the stop 0 / 0 / 0; maker legs
5,235,731, untimed (excluded) 51,027.

A. Per config, grid (G) and refreshed (F), $ / day over the 8 read days; MM interval (t, 7 df; * = excludes 0); 11-03 beside:

| timing size stop | src | spread | informed | +1 s | drift | MM (interval) | MM d>0 | carry | carry sd | settle | 11-03 MM / carry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J s5 S60 | G | +40.6 | -1.5 | +39.1 | +6.4 | +45.5 (+7.4..+83.5)* | 6/8 | +9.4 | 180 | +54.8 | -8.1 / -81.0 |
| J s5 S60 | F | +39.2 | +0.4 | +39.7 | +10.7 | +50.3 (+12.7..+88.0)* | 8/8 | +19.3 | 178 | +69.7 | -27.3 / -21.3 |
| J s5 S120 | G | +76.1 | -16.9 | +59.2 | -28.4 | +30.7 (-67.8..+129.2) | 5/8 | +48.2 | 160 | +79.0 | -62.3 / -171.2 |
| J s5 S120 | F | +72.9 | -13.1 | +59.7 | -16.2 | +43.6 (-33.2..+120.3) | 5/8 | +47.9 | 145 | +91.4 | -106.9 / -57.9 |
| J s5 S180 | G | +104.8 | -17.7 | +87.1 | -38.7 | +48.5 (-59.8..+156.8) | 5/8 | +74.1 | 161 | +122.5 | +30.0 / -163.3 |
| J s5 S180 | F | +100.1 | -12.7 | +87.4 | -29.0 | +58.4 (-37.0..+153.8) | 5/8 | +79.3 | 164 | +137.6 | -2.6 / -21.9 |
| J s5 S240 | G | +129.8 | -20.2 | +109.6 | -69.9 | +39.7 (-102.4..+181.8) | 7/8 | +49.9 | 117 | +89.6 | -161.4 / -77.5 |
| J s5 S240 | F | +125.6 | -12.3 | +113.3 | -47.5 | +65.8 (-53.0..+184.7) | 7/8 | +51.6 | 133 | +117.4 | -80.3 / +29.4 |
| X s5 S60 | G | +33.1 | -12.2 | +20.9 | -3.2 | +17.7 (-18.8..+54.1) | 6/8 | +19.6 | 135 | +37.3 | -6.2 / -103.7 |
| X s5 S60 | F | +31.6 | -9.1 | +22.5 | +7.4 | +29.9 (+1.7..+58.2)* | 6/8 | +12.1 | 141 | +42.0 | -4.8 / -115.3 |
| X s5 S120 | G | +61.6 | -34.3 | +27.3 | -3.8 | +23.5 (-62.1..+109.2) | 5/8 | +46.4 | 90 | +69.9 | -53.0 / -154.4 |
| X s5 S120 | F | +60.2 | -30.0 | +30.2 | -4.7 | +25.5 (-50.3..+101.3) | 6/8 | +35.0 | 92 | +60.5 | -61.1 / -217.1 |
| X s5 S180 | G | +85.3 | -43.0 | +42.3 | -9.1 | +33.2 (-34.8..+101.3) | 4/8 | +78.7 | 101 | +112.0 | +34.7 / -176.0 |
| X s5 S180 | F | +83.3 | -36.5 | +46.7 | -5.9 | +40.8 (-34.5..+116.1) | 6/8 | +52.4 | 55 | +93.2 | +17.5 / -184.9 |
| X s5 S240 | G | +104.8 | -50.4 | +54.4 | -30.6 | +23.8 (-73.9..+121.4) | 4/8 | +25.8 | 97 | +49.6 | -107.3 / -99.5 |
| X s5 S240 | F | +102.4 | -42.8 | +59.5 | -42.4 | +17.1 (-86.4..+120.6) | 5/8 | +15.0 | 82 | +32.1 | -110.8 / -66.0 |
| J s20 S60 | G | +190.9 | -98.6 | +92.3 | +47.3 | +139.6 (-64.2..+343.4) | 5/8 | +41.0 | 924 | +180.6 | -79.7 / +223.7 |
| J s20 S60 | F | +190.2 | -94.8 | +95.4 | +58.4 | +153.7 (-51.9..+359.4) | 5/8 | +1.8 | 954 | +155.5 | -105.3 / +499.6 |
| J s20 S120 | G | +353.0 | -239.5 | +113.5 | +45.1 | +158.6 (-360.0..+677.1) | 5/8 | +305.3 | 965 | +463.9 | -160.2 / +676.7 |
| J s20 S120 | F | +351.2 | -243.4 | +107.8 | +8.3 | +116.2 (-305.2..+537.5) | 5/8 | +418.0 | 871 | +534.2 | -421.3 / +1351.0 |
| J s20 S180 | G | +482.2 | -335.9 | +146.3 | +54.4 | +200.7 (-523.3..+924.7) | 5/8 | +507.0 | 754 | +707.7 | +37.2 / +558.4 |
| J s20 S180 | F | +479.7 | -334.3 | +145.4 | -24.8 | +120.6 (-607.0..+848.3) | 4/8 | +590.3 | 628 | +710.9 | -144.5 / +1220.9 |
| J s20 S240 | G | +590.5 | -399.1 | +191.4 | -48.6 | +142.8 (-560.0..+845.6) | 5/8 | +374.2 | 633 | +517.0 | +176.8 / +523.2 |
| J s20 S240 | F | +588.5 | -400.2 | +188.3 | -34.1 | +154.1 (-518.6..+826.9) | 5/8 | +380.0 | 668 | +534.2 | -14.3 / +951.4 |
| X s20 S60 | G | +167.3 | -148.9 | +18.4 | +1.5 | +19.9 (-140.0..+179.7) | 6/8 | +17.5 | 668 | +37.3 | -93.9 / +401.9 |
| X s20 S60 | F | +167.4 | -147.3 | +20.1 | +10.7 | +30.8 (-138.4..+199.9) | 5/8 | +56.3 | 755 | +87.0 | -132.3 / +209.9 |
| X s20 S120 | G | +309.5 | -329.4 | -19.9 | -4.9 | -24.8 (-518.8..+469.2) | 4/8 | +194.8 | 665 | +170.0 | -169.1 / +583.4 |
| X s20 S120 | F | +310.1 | -328.4 | -18.3 | +2.2 | -16.2 (-492.9..+460.6) | 5/8 | +223.6 | 652 | +207.4 | -421.3 / +896.0 |
| X s20 S180 | G | +424.9 | -459.7 | -34.9 | +38.8 | +3.9 (-568.7..+576.5) | 4/8 | +411.8 | 588 | +415.7 | +41.2 / +681.3 |
| X s20 S180 | F | +424.1 | -455.6 | -31.5 | +22.0 | -9.5 (-625.0..+606.0) | 4/8 | +285.9 | 655 | +276.4 | -290.9 / +1172.1 |
| X s20 S240 | G | +517.8 | -540.4 | -22.6 | -65.3 | -88.0 (-710.0..+534.1) | 4/8 | +228.2 | 550 | +140.3 | -116.3 / +955.7 |
| X s20 S240 | F | +516.8 | -528.7 | -11.9 | -41.7 | -53.6 (-642.1..+534.9) | 4/8 | +71.8 | 640 | +18.1 | -372.2 / +1375.5 |

B. Refreshed minus grid per term, day-clustered mean (t interval, 7 df), $ / day; * = excludes 0:

| timing size stop | spread | informed | drift | MM | carry | settle |
|---|---|---|---|---|---|---|
| J s5 S60 | -1.4 (-2.2..-0.6)* | +2.0 (-3.4..+7.3) | +4.3 (-5.9..+14.4) | +4.8 (-8.8..+18.4) | +10.0 (-68.9..+88.8) | +14.8 (-63.9..+93.6) |
| J s5 S120 | -3.2 (-4.7..-1.8)* | +3.8 (-3.4..+11.0) | +12.2 (-31.5..+56.0) | +12.8 (-34.5..+60.1) | -0.3 (-83.7..+83.1) | +12.5 (-84.2..+109.2) |
| J s5 S180 | -4.7 (-6.4..-2.9)* | +5.0 (-4.1..+14.0) | +9.6 (-34.8..+54.1) | +9.9 (-33.1..+52.9) | +5.2 (-53.1..+63.5) | +15.1 (-67.7..+97.9) |
| J s5 S240 | -4.2 (-5.8..-2.7)* | +7.9 (-2.5..+18.4) | +22.4 (-27.3..+72.2) | +26.1 (-25.2..+77.5) | +1.7 (-77.1..+80.5) | +27.9 (-72.4..+128.1) |
| X s5 S60 | -1.5 (-2.7..-0.4)* | +3.1 (-0.3..+6.5) | +10.7 (+0.1..+21.3)* | +12.3 (+0.9..+23.6)* | -7.5 (-67.0..+51.9) | +4.7 (-56.2..+65.7) |
| X s5 S120 | -1.4 (-3.6..+0.9) | +4.2 (-1.9..+10.4) | -0.9 (-32.0..+30.2) | +2.0 (-27.3..+31.4) | -11.4 (-62.3..+39.6) | -9.4 (-77.3..+58.6) |
| X s5 S180 | -2.0 (-4.5..+0.5) | +6.4 (+1.5..+11.4)* | +3.1 (-35.1..+41.4) | +7.6 (-27.0..+42.1) | -26.3 (-89.0..+36.4) | -18.7 (-82.5..+45.1) |
| X s5 S240 | -2.5 (-6.0..+1.1) | +7.6 (+0.8..+14.4)* | -11.8 (-63.2..+39.6) | -6.7 (-52.3..+39.0) | -10.8 (-54.7..+33.2) | -17.4 (-68.1..+33.2) |
| J s20 S60 | -0.7 (-3.5..+2.1) | +3.7 (-9.1..+16.6) | +11.1 (-46.8..+69.0) | +14.1 (-40.9..+69.1) | -39.2 (-169.7..+91.3) | -25.1 (-161.4..+111.2) |
| J s20 S120 | -1.8 (-4.8..+1.2) | -3.9 (-17.9..+10.2) | -36.7 (-177.1..+103.6) | -42.4 (-181.1..+96.2) | +112.7 (-93.5..+318.9) | +70.3 (-108.0..+248.5) |
| J s20 S180 | -2.5 (-7.1..+2.2) | +1.6 (-17.9..+21.1) | -79.2 (-249.6..+91.2) | -80.1 (-247.6..+87.4) | +83.3 (-205.8..+372.5) | +3.2 (-232.5..+239.0) |
| J s20 S240 | -2.1 (-7.3..+3.2) | -1.1 (-37.6..+35.4) | +14.5 (-228.9..+257.8) | +11.3 (-244.0..+266.6) | +5.8 (-190.8..+202.5) | +17.1 (-268.7..+303.0) |
| X s20 S60 | +0.1 (-3.7..+3.9) | +1.6 (-12.2..+15.3) | +9.2 (-21.0..+39.4) | +10.9 (-24.0..+45.8) | +38.8 (-71.0..+148.6) | +49.7 (-58.9..+158.3) |
| X s20 S120 | +0.6 (-5.2..+6.5) | +0.9 (-13.2..+15.0) | +7.1 (-88.5..+102.7) | +8.7 (-95.1..+112.4) | +28.8 (-118.0..+175.6) | +37.5 (-128.1..+203.0) |
| X s20 S180 | -0.8 (-6.5..+4.9) | +4.1 (-4.6..+12.8) | -16.8 (-128.2..+94.7) | -13.4 (-120.7..+93.8) | -125.9 (-265.6..+13.8) | -139.3 (-260.3..-18.3)* |
| X s20 S240 | -1.0 (-8.2..+6.2) | +11.7 (-3.1..+26.5) | +23.6 (-144.4..+191.6) | +34.3 (-133.6..+202.3) | -156.5 (-297.4..-15.5)* | -122.1 (-375.1..+130.9) |

C. By minute of the window, 8 read days pooled, c / share: +1 s / post-1 s (fill +1 s -> outcome); ours = the S240 arm; shares / day in brackets:

| who | 0-60 | 60-120 | 120-180 | 180-240 |
|---|---|---|---|---|
| maker top 5 | +0.43 / +0.05 [859445] | +0.33 / -0.11 [702219] | +0.27 / +0.16 [622788] | +0.20 / -0.79 [549276] |
| all makers | +0.15 / +0.14 [2484068] | +0.13 / +0.19 [2062880] | +0.10 / +0.11 [2028843] | +0.04 / -0.46 [1981115] |
| ours J s5 G | +0.76 / +0.33 [5162] | +0.44 / +0.13 [4581] | +0.79 / +0.35 [3546] | +0.76 / -1.88 [2951] |
| ours J s5 F | +0.80 / +0.59 [4977] | +0.46 / +0.13 [4398] | +0.81 / +0.44 [3430] | +0.92 / -1.64 [2822] |
| ours X s5 G | +0.49 / +0.39 [4328] | +0.16 / +0.77 [3835] | +0.48 / +0.75 [3098] | +0.48 / -2.95 [2526] |
| ours X s5 F | +0.55 / +0.48 [4140] | +0.20 / +0.39 [3774] | +0.56 / +0.40 [2948] | +0.52 / -3.03 [2438] |
| ours J s20 G | +0.30 / +0.30 [30293] | +0.08 / +1.06 [25683] | +0.17 / +1.02 [19620] | +0.28 / -1.46 [16263] |
| ours J s20 F | +0.32 / +0.20 [30198] | +0.05 / +1.49 [25524] | +0.19 / +0.64 [19630] | +0.27 / -1.36 [16077] |
| ours X s20 G | +0.07 / +0.09 [26717] | -0.17 / +0.81 [22589] | -0.08 / +1.36 [17943] | +0.09 / -2.04 [14105] |
| ours X s20 F | +0.08 / +0.27 [26877] | -0.17 / +0.76 [22595] | -0.07 / +0.36 [17755] | +0.14 / -1.99 [13949] |

D. Ours (S240 arm) minus top 5 makers, +1 s c / share per minute, day-clustered mean (t interval, 7 df), days ours > top 5; * = excludes 0:

| timing size | src | 0-60 | 60-120 | 120-180 | 180-240 |
|---|---|---|---|---|---|
| J s5 | G | +0.27 (-0.27..+0.82) 5/8 | +0.09 (-0.21..+0.40) 3/8 | +0.54 (-0.05..+1.13) 6/8 | +0.71 (+0.05..+1.36)* 6/8 |
| J s5 | F | +0.31 (-0.25..+0.87) 6/8 | +0.13 (-0.22..+0.47) 4/8 | +0.56 (-0.07..+1.19) 6/8 | +0.89 (+0.14..+1.64)* 6/8 |
| X s5 | G | +0.02 (-0.51..+0.55) 4/8 | -0.17 (-0.53..+0.18) 3/8 | +0.25 (-0.42..+0.91) 4/8 | +0.39 (-0.29..+1.07) 5/8 |
| X s5 | F | +0.06 (-0.48..+0.60) 4/8 | -0.14 (-0.50..+0.23) 2/8 | +0.33 (-0.42..+1.07) 4/8 | +0.45 (-0.29..+1.20) 5/8 |
| J s20 | G | -0.11 (-0.45..+0.22) 3/8 | -0.24 (-0.58..+0.10) 2/8 | -0.11 (-0.52..+0.31) 4/8 | +0.12 (-0.32..+0.57) 4/8 |
| J s20 | F | -0.10 (-0.43..+0.23) 3/8 | -0.27 (-0.61..+0.07) 2/8 | -0.09 (-0.50..+0.33) 4/8 | +0.13 (-0.42..+0.68) 4/8 |
| X s20 | G | -0.34 (-0.66..-0.02)* 2/8 | -0.48 (-0.90..-0.06)* 1/8 | -0.37 (-0.89..+0.15) 3/8 | -0.11 (-0.59..+0.37) 4/8 |
| X s20 | F | -0.33 (-0.66..-0.00)* 2/8 | -0.49 (-0.90..-0.08)* 1/8 | -0.36 (-0.87..+0.15) 3/8 | -0.06 (-0.55..+0.43) 4/8 |

Reading (each claim computed from report_mm_decomp_fw.json):
- What the refresh changes: at 5/25 it gives up a little spread (point estimate below 0 in 8 of 8, interval below 0 in 5: -1.4 .. -4.7
  $ / day) for less adverse selection (informed above 0 in 8 of 8, interval above 0 in 2: X S180 +6.4, X S240 +7.6). At 20/100 neither
  moves beyond noise. Drift, MM, carry and settle: 1 of 16 intervals excludes 0 each (MM and drift: X 5/25 S60 +12.3 / +10.7; carry X
  20/100 S240 -156.5; settle X 20/100 S180 -139.3).
- MM vs settlement: spread > 0 in all 32 configs, informed < 0 in 31 (J 5/25 S60 refreshed +0.4). MM interval above 0 only at S60 5/25:
  J grid, J refreshed, X refreshed; none at 20/100; none below 0. Carry daily sd 55-180 $ at 5/25, 550-965 $ at 20/100; 2 of 32 carry
  intervals exclude 0 (both above 0), no settle interval does.
- Against the top 5: the refresh does not change the ranking (the same cells exclude 0 for G and F). 5/25 central above the top 5 in all
  4 minutes (points), interval above 0 only in 180-240 s (G +0.71, F +0.89); 5/25 bound 1 of 4 minutes below, none excludes 0; 20/100
  central 3 of 4 below, none excludes 0; 20/100 bound 4 of 4 below, interval below 0 in 0-60 and 60-120 s.
- Last minute: post-1 s in 180-240 s is below 0 for the top 5 (-0.79), all makers (-0.46) and all 8 of our runs (-1.36 .. -3.03 c / share,
  both sources); in 0-180 s ours is above 0 in 24 of 24 cells (pooled over days, so mostly carry; descriptive). The refresh does not change
  it: refreshed minus grid post-1 s excludes 0 in 1 of 16 minute cells (X 20/100 120-180 s, -1.05).
- 11-03: MM < 0 in 26 of 32 configs.

## 60. Summary of round i (sections 55-59), fixed engines (as of 2026-11-06T05:09Z)

Scope: every number below is from the fixed engines (round i), the 8 read days (10-23..27, 10-31, 11-01, 11-02) with 11-03 beside;
IN-SAMPLE (guard, per-minute lines and thresholds fit on the same days); $ / day, no rebate, no own price impact; day-clustered t
intervals (7 df). MM = mark@stop (spread + 1 s adverse selection + drift to the stop); settlement = carry from the stop to the outcome.

Levels (5/25, S60, mark@stop; section 55): today (V0 + GS) J +45.5 (+7.4..+83.5) 6/8, X +17.7 (-18.7..+54.1) 6/8; pm + ladder J +63.6
(+27.3..+99.9) 7/8, X +52.0 (+13.5..+90.6) 7/8; pm + ladder + improve J +83.9 (+37.1..+130.6) 8/8, X +56.0 (+18.3..+93.6) 7/8. 11-03:
pm + ladder -12.2 / -33.1.

Levers, effect on MM (day-clustered mean difference, J | X):

| lever | J | X | reading | section |
|---|---|---|---|---|
| ladder on per-minute lines (pm + ladder - pm) | +18.1 (+9.4..+26.9) | +23.8 (+9.9..+37.8) | above 0 at both timings | 55 |
| price improve on pm | +25.4 (+2.0..+48.7) | +11.6 (-4.1..+27.2) | central only | 55 |
| per-minute lines (pm - today) | -0.0 (-13.4..+13.4) | +10.5 (-0.8..+21.9) | spans 0 | 55 |
| ladder without lines (today + ladder - today) | +3.4 (-17.6..+24.3) | +14.2 (-7.9..+36.3) | spans 0 | 55 |
| hold (pm + ladder hold - pm) | -27.9 (-62.8..+7.0) | -34.0 (-67.3..-0.6) | rejected | 55 |
| 20/100 vs 5/25 (pm + ladder) | +92.5 (+9.5..+175.5) | +64.4 (+10.6..+118.1) | above 0 in the replay, which has no own price impact | 55 |
| inventory cap / reduce-only | 0 of 36 | 0 of 36 | no MM or settle change; cuts carry sd on every read day (risk only) | 56 |
| longer window (S120 / S180 / S240 vs S60) | 0 of 16 MM intervals exclude 0 (both timings) | | spread up, lost to adverse selection and the last minute | 56, 57 |
| feed refresh, first 65 s (today arm) | +4.8 (-8.8..+18.4) | +12.3 (+0.9..+23.6) | bound only | 55, 58 |
| feed refresh, whole window | 0 of 8 above 0 at both | | trades spread for less adverse selection; net within noise | 58, 59 |

What the round says (each line from the cited section's report):
1. The market-making money is spread minus the first second's adverse selection, at small size, in the first minute. Spread > 0 in all 88
   decomposed configs (today-stack arms, sections 57, 59), informed < 0 in 87. Among them a mark@stop interval above 0 exists only for
   5/25 S60 (J S60 grid and refreshed, J S60+CAP25, X S60 refreshed); no 20/100 config and no window past 60 s has one. The ladder-engine
   arms with per-minute lines + ladder (section 55, S60) are above 0 at both timings, at 5/25 and at 20/100.
2. Size costs per share. At S60 on the grid, 1 s adverse selection takes 4% of the spread at 5/25 vs 52% at 20/100 (central), and 37% vs
   89% (bound). At 20/100 our +1 s per share is below the top 5 in 3 of 4 minutes (central) and 4 of 4 (bound). The 20/100 dollar gains
   come from volume in a replay without own price impact, so they are an upper bound.
3. Settlement P&L is outcome noise, not a strategy result. Daily carry sd is 45-180 $ at 5/25 and 236-965 $ at 20/100, against MM means of
   tens of dollars; 4 of 88 carry intervals exclude 0 (all above 0, about the chance rate), no settle interval does. MM is the decision
   metric. Carry is risk, and the inventory cap cut it at no measurable MM cost.
4. Against the top makers (+1 s per share; our replay timing, their measured anchor): at 5/25 central we are above the top 5 in every
   minute, but the interval is above 0 only in 180-240 s. At 5/25 bound timing no interval excludes 0; at 20/100 bound timing we are below
   (interval below 0 in 0-60 and 60-120 s). Our volume is
   2.4-5.2k sh / day per minute at 5/25 and 14-30k at 20/100, against 550-860k for the top 5.
5. The last minute loses for everyone after the first second: post-1 s in 180-240 s is -0.79 c / share for the top 5, -0.46 for all
   makers, and -1.4 .. -4.2 for our arms (sections 57, 59). This is why longer windows add nothing. The whole-window refresh does not
   change it (refreshed minus grid post-1 s, last minute: 0 of 4 cells exclude 0).
6. Bound timing weakens everything. At bound timing the today arm's level interval spans 0, and so does pm's (+28.2, -0.7..+57.0). 11-03 is
   negative in most arms (section 55: nearly all; section 59: 26 of 32 configs).
7. Multiplicity: sections 55-59 make well over a hundred comparisons, so a few intervals excluding 0 by chance are expected. Only one lever
   result, the ladder on per-minute lines, is above 0 at both timings.

Standing candidate (not frozen): 5/25, S60, per-minute lines + ladder (+ improve: central only), V0 + GS guard on the 50 ms grid; today S60
as the control. No longer windows; no 20/100 until own impact is modelled; no whole-window refresh. The open check is a forward replay
of a frozen version on days after 11-03, since all of the above is in-sample.

## 61. Fixed lines, the week split, and pm_ladder_v2 (daily 7-day refit) (as of 2026-11-06T09:38Z)

User question: are the pull / size percentiles updated? In v1 they are not. The models, c60 and lines were fitted once on the 8 read days.

Line stability (read days, out-of-fold v; 11-03 frozen model): the share of minute-0 time each side's v is above v1's pull line 0.0492:
- weekdays (10-23, 10-26, 10-27, 11-02, 11-03): 6-8%;
- weekends: 9-12% on 10-24 / 25 and 14-18% on 10-31 / 11-01.

Week split of round i (mark@stop $ / day; point estimates; 5 and 4 days; same weekdays, Sat-Tue plus a Friday in week 1):

| | week 1, 10-23..27 | week 2, 10-31..11-03 |
|---|---|---|
| candidate, J | +92.6 | +8.4 |
| candidate, X | +76.9 | -0.4 |
| control, J | +74.2 | -3.8 |
| control, X | +39.7 | -15.9 |

v2 (frozen 2026-11-06T09:23Z; predictfun/freeze/pm_ladder_v2/RECEIPT.md):
- v1's arm and rules, with the guard models, c60 and lines refit every day on the last 7 admissible days, using v1's own fit and line
  rules.
- 7 = one weekly cycle, set before any result.
- All parity checks pass: fold, lines, start-known mask, per-day grid, engine.

Rehearsal on week 2 (step4/report_v2_rehearsal.json; in-sample, not like-for-like: v1's values there come from leave-one-day-out
models that saw later days):
- J: v2 +131.8, v1 +33.5, control -15.2 over 4 days; v2 > v1 on 4 of 4 days.
- X: +8.4, -1.4, -63.5; v2 - v1 by day +0.0, +6.2, -6.0, +9.5.
- Only 11-02 / 11-03 have full 7-day windows. There: J v2 +30.0 vs v1 -14.5; X -68.0 vs -71.5.

Forward race declared (predictfun/freeze/forward_20261104_08/RECEIPT.md): v1 + v2 + control on 11-04..11-08.
- Rules: v1 PASS (unchanged), v2 PASS, ROLLING ADOPTED iff v2 PASS and sum (v2 - v1) > 0 at J and X.
- That receipt also repairs v1's procedure: the frozen engine refused every day >= 11-04. The repair is a generated forward-enabled
  engine, byte-identical to round i, plus frozen-model apply.

## 62. Order changes of the frozen candidate: what triggers them (as of 2026-11-06T09:38Z)

User question: do we cancel / replace / change orders when the prediction does not change?

Method: pf_ladder_causes.py, hooked read-only into the frozen engine; outputs byte-identical to round i; every request attributed
(report_causes_v1.json).

The engine reads a side's v only through its bucket:
- none (missing / stale);
- <= pull line;
- between the lines;
- >= size line.

Each lane change is classified against the previous decision. Rows are the 8 read days at J; X is within 1%.

| cause | requests / window (first 60 s) | share |
|---|---|---|
| guard bucket changed | 199.8 (143.8..255.8) | 74.1% |
| best bid / ask price moved | 58.6 (47.8..69.4) | 21.7% |
| best-level sizes only (micro price flip at the same prices) | 8.4 | 3.1% |
| Binance guard | 2.1 | 0.8% |
| stop at 60 s, inventory, re-place after fill / refusal, other | < 1 | < 0.4% |
| total | 269.7 (223.0..316.4) | |

- 97% of the ~5,800 decisions per window change nothing: an unchanged price and size is kept, never re-sent.
- So 26% of requests come without a prediction change. They follow the book: the touch is pulled when the micro price falls behind the
  best (G4), and the rung is moved when the best moves more than 2 ticks away. These rarely undo themselves: 0.9 reversals within 1 s
  per window.
- The churn is the prediction itself. 50.6 reversals per window (15.7..85.4; touch 30.1, rung 20.5) are a bucket change undone within
  1 s, against about 195 bucket-driven changes. Most cancelled orders were cancelled within 100 ms of going live, or before.
- A reversal counter that tested a repeat instead of a reversal first read 0 everywhere. It was fixed, with a selftest that rejects the
  old rule, before these numbers.

Implication, not tested: a dead band on the lines (enter above line + d, leave below line - d) or a minimum hold would remove most
flicker and free throttle tokens (about 5,000 throttled places / day). Whether it adds P&L is unknown. It would be a new version.
