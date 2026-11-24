# Maker3: trend, displayed spread, turnover and inventory — 2026-11-24

**Completed:** the requested saved-data measurements and an interactive chart over all19 already complete BOOK markets on consumed Oct24–27. This is a fixed, selected subset of Maker3's trading history, not his whole business or untouched validation. No strategy replay, model fit or new days were run in this packet. Maker3 is the ex-ante rank3 address0x57FbC891E89184DA071AE9D97E745531cd7F718a.

Open [the chart](http://127.0.0.1:8765/maker3-research.html) through the existing SSH tunnel. [3789484's last two minutes](http://127.0.0.1:8765/maker3-research.html?market=3789484&from=180&to=298) are directly linked. The same [self-contained HTML](../maker_dashboard_20261124/maker3-research.html) can be downloaded. Use the market/window buttons and click a fill or completed pair. The chart includes book/spread, every accepted maker/taker fill, uncertain execution windows, recorded net inventory, completed turnover, marked total and BTC movement. Our existing baseline/wider-band overlays are available for3702960,3789484,3877472 only. A native BUY NO is a YES-equivalent sell; both prices are explicit in the detail.

## Our inventory rules

The current baseline and the last experimental wider-band arm are distinct. Both use at most100 shares per new order, two desired levels per side, desired side quantity200, and client-known inventory after notices. They quote the full supported298s. The wider arm has not replaced the baseline or passed validation.

| Mechanism | Baseline | Wider experiment | Meaning |
|---|---:|---:|---|
|Quantity-skew scale / soft boundary S|100|500|bid multiplier clip(1−I/S,0,1); ask clip(1+I/S,0,1)|
|Release boundary R|50|250|After a restriction, recover inside the release boundary before resuming the adding side|
|Hard cap C|200|1,000|I+B<=C; −I+A<=C, with outstanding bid/ask reservations B/A|

The proposed quantity is clipped by the per-order multiplier, remaining desired side quantity and hard-cap room, then floored. For a fresh slot at I=+250 in the wider arm, this gives at most bid50/ask100; at+450, bid10/ask100. Negative inventory is symmetric. Reservations can make those proposals smaller or zero. Existing orders are not continuously resized merely because the multiplier changes.

At+S, the soft mode requests pulls of live adding bids and withholds new bids; reducing asks remain eligible. It releases at+R, with the sign-crossing correction retained. The hard cap prevents worst-case outstanding exposure; cancel-pending and unnotified fills remain reserved. Cancel frees a desired slot before its reservation is released, so asynchronous requests still obey the cap. There is **quantity skew, no inventory-driven price skew**: price placement remains received best/rung. Pressure and structural rules independently pull/withhold.

The separate wider active-take arm would submit a bounded reducing IOC at known |I|>=500, targeting the release region, at most100 shares per take. It did not activate on the three earlier paths (maximum actual |I|496), so there is no wider-taker economic result. Raising C alone had exact parity; widening S changed turnover and risk. See the [earlier four-arm receipt](../maker3_cap_skew_20261124/report.md).

## 1. A trending window, including the uncomfortable case

The frozen price-only selector chose **3838531,245–275s**: public YES midpoint rises71¢, trend efficiency80.7%. It searched fixed30s windows at5s anchors, required31 usable mids, at least10¢ movement and efficiency>=0.6, and never read maker profit in selecting the example.

In recorded upper-time order, Maker3 has76 legs (74 maker,2 taker), buying15.38 and selling1,621.36 YES-equivalent shares. Reconstructed net trade-flow position moves from−64.85 to−1,670.83. Most selling is in265–275s; the two take legs at269.221s buy22.37 NO shares, equivalent to selling YES at62¢. They add to the negative recorded exposure; not every take is a reducing action. Match bounds are267.221–269.221s, not known private decision times.

Only15.38 shares complete FIFO pairs within that window, net−$0.37. This small realized amount does not measure the much larger inventory exposure. The full market still has+$149.82 completed turnover and−$36.48 residual, net+$113.34 at the common END mark. Thus a rising-price example cannot be presented as a universally successful maker hedge or a consistently losing market. The chart lets the user follow the accumulated exposure through END. Private wallet inventory and other-market hedges are absent.

## 2. Wide displayed spreads do not explain most completed gains

There are2,759 accepted actor legs and25,333.99 FIFO matched shares in the common intervals. Completed net turnover is **+$859.50**; residual less its unclosed fees is **−$1,210.32**, leaving **−$350.81** at the common END marks. The completed-turnover edge is3.39¢ per matched share. This is sell-minus-buy accounting across time, not a simultaneous quoted spread.

| FIFO exit time | All19 net completed turnover | Market3789484 |
|---|---:|---:|
|Before180s|−$93.59|−$55.44|
|180–298s|+$953.10|+$564.07|
|240–298s, subset of late|+$749.52|+$577.52|

Late exits do contribute materially in these markets. Entries can precede the window; these are **exit-attributed completed pairs**, not profits created entirely by late quoting. Across all19,41.0% of filled shares occur after180s; in3789484 it is52.8%. The broader previously measured934-market activity population is different and must not inherit this subset's late-concentration reading.

Wide is frozen as displayed spread>4¢. Of gross positive completed net value, the fraction with **both** FIFO legs aligned to wide books is:

| Book clock | Both legs wide | Exactly one leg wide | Neither wide |
|---|---:|---:|---:|
|Before earliest possible match|0.41%|4.53%|95.06%|
|At execution upper bound|5.31%|11.55%|83.14%|
|1s before lower bound|0.00%|5.44%|94.56%|

The declared >50%-on-both-primary-clocks screen is false. Even allowing one wide leg does not make wide spreads the main source of positive completed turnover in this subset. This does not say wide liquidity is worthless or classify the private spread at submission: book changes inside the execution window remain unidentified.

Wide books occupy1.61% of the supported elapsed time across all19,3.74% after180s. For3789484 they occupy13.04 of118 late seconds (11.1%). Its late passive fills aligned to a wide book are308.58 shares at the pre-window clock versus1,551.93 at the upper clock. At the former clock14 such legs are priced behind that earlier same-side touch and4 inside it; at the latter64 are at touch,13 behind and9 inside. This large timing sensitivity forbids inferring a stable deep-quote or touch-quote rule from fill locations. Exact per-fill book times and original-size availability are visible in the chart.

The edge-versus-movement split is an identity, not a causal attribution. On3789484's late closes, pre-window books assign+$440.09 to entry+exit edge and+$124.44 to midpoint movement, less$0.46 fees. Upper books assign+$128.10 and+$436.42 instead. Actual completed net stays+$564.07. A large difference between fill prices is therefore reliable, but how much is instantaneous spread versus subsequent price movement is clock-sensitive. The report does not choose the more flattering split.

## 3. What the risk, sizing and take evidence supports

The new neutral comparison covers all5,700 market-seconds, including inactive seconds. Ten predeclared screens compare five conditions at D0 and lag1s, matched within market, inventory sign/band and minute. The outcome is an opposing take in the following anchored second.500 raw-side schedule shifts preserve takes and recompute opposing direction at each destination. These are venue-state associations, not private client decisions; multiple hypotheses and consumed data preclude a validation claim.

| Fixed condition | Matched take-rate difference, D0 / lag1, percentage points | Count support |
|---|---:|---|
|Net adding passive fills>=100 shares in5s|−1.03 /−0.48|Insufficient|
|No passive reducing fill in5s|−0.65 /−0.40|Adequate counts; no stable positive association|
|BTC against inventory by>=1bp in5s|−0.80 /−0.69|Insufficient|
|Executable reduction net of entry/exit fees>1¢ above FIFO cost|−0.032 /−0.037|Adequate counts; near zero|
|Displayed spread>4¢|−0.59 /+1.00|Insufficient and changes sign|

The no-passive-reduction D0 lower-tail rank is0.044 but lag1 is0.140; this isolated negative reading is not a chosen rule. Cells with count support can still have little discrimination. Zero passive fills is not proof no reducing quote existed, and high adding flow can outpace existing reducing fills. No new take trigger is established.

Concrete3789484 examples remain useful: before the roughly202.7s takes, the200s anchor has reconstructed+553 shares and220 adding/zero reducing passive shares in the previous5s. Three subsequent native NO buys trim109.39 YES-equivalent shares. Before the248.63s take, the246s anchor has+640 shares,8.33 reducing/zero adding shares in5s; the subsequent NO buy trims74.53. One occurs after rapid accumulation, the other while passive reduction already happens. The112s example's sign changes at lag1 and remains timing-sensitive. These facts support **partial active trims as one tool**, not an identified threshold, skew-failure timer or common motive.

The prior signed-size audit remains the valid size evidence:33 pure-maker hashes,16 originally10 shares and17 originally220, only28 usable sampled post windows. Both sizes occur on adding and reducing sides; original10 is not a universal reducing order. The small selected, day-confounded sample did not support10-when-volatile/220-when-calm. Quote-time uncertainty and missing never-filled orders still prevent recovering simultaneous price skew, per-level total size or a private cap. These limitations are shown alongside the new measurements rather than replacing original size with partial filled quantity.

## Integrity and next step

Frozen contract/measurement/input manifest: **2b540b096**. The first build refused because an exact lowercase address filter excluded mixed-case source addresses, before outputs. **41b546811** adds the committed canonical-address adapter with mixed-case controls; frozen economics source is unchanged. Accepted build hashes are in build.json. All19 common-interval cash/inventory/fees and completed turnover reproduce the prior accepted rank3 accounting; eight source-status legs are excluded explicitly, never silently zeroed. No residual is passed off as completed spread profit.

Checks pass:2,759 raw legs,156 cells,10 risk screens,4 planted corruptions. Independent sums cover raw membership, fees, inventory, FIFO allocation, decomposition and matched-risk cells. Cost-basis inventory disagreements are zero at both clocks. Selftests cover wide spread versus directional drift, no future cost basis, causal anchors, and raw-side shifts. Browser tests cover19 choices, trend/late navigation, filters, fills/native details, FIFO links, desktop and390px/DPR2 mobile; no script errors. The first browser check exposed horizontal overflow, fixed before publication; earlier results retained. Canvas height remains330px on both DPRs. The final native-price wording and optional-overlay label change no data. Scientific stages use the codex slot; browser alone needs an isolated core8 process without the wrapper's virtual-address cap, and its CPU is charged. Existing loopback server is retained; owned browser is closed.

**ONE next priority:** execute the already proposed, unchanged WIDE_PASSIVE versus opening-only baseline on the full19 consumed complete markets, with paired completed turnover, residual, position risk, participation and churn. The selected three examples improved completed turnover but one worsened; the maker evidence here likewise distinguishes profitable turnover from costly residuals. Establish whether the wider soft band generalizes within this fixed development population before adding a size/volatility switch, price skew or a new active-take rule. Existing standing short-CPU authorization covers a bounded frozen run. New days, untouched validation or live trading are not part of it.

Resource allowance<=0.15coreh; charge a conservative60CPU seconds including failed build, measurement, checks, UI/browser and publication (measured stages substantially below that), no strategy replay. Cumulative task approximately2.10489/line5.41559coreh versus caps2.35/5.65, with0.05reserve retained. Sources/outputs are local, and nothing is pushed. Hashes, resources and the failed receipts are retained with this report.
