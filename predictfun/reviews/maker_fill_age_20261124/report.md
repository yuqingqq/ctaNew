# How early were the orders behind good fills posted?

As of 2026-11-24T05:34:38Z. Freeze **0b8f1ef41665e6dda581a6d82324c1a25c6e7213**. Consumed Oct24–27 only; no fit, new data, replay, validation or strategy change. USER question measured with explicit timing uncertainty rather than treating a window midpoint as an actual post.

**No required posting lead is established.** The available records permit timing ranges. The first-fill lower bound is zero by construction: the first fill proves the order existed by its execution, but does not independently prove how much earlier. A later execution of the same hash can establish an earlier identity, not uninterrupted queue residence. How early the order *needed* to be posted requires a counterfactual queue/fill comparison; observed age alone cannot supply that.

## Broad order population, not individual good-fill labels

All4,149 top-three fill-linked hashes are retained. Good aggregate order means that the sum of its counted fills' gross10s and gross60s markouts are both positive. These outcomes are **not the first fill's value**. First-age upper endpoints below are conditional maximum leads, not measured rests; their median is not median actual age. G4 assumptions are weaker than G2.

| Grade | Orders with timing | Orders with counted values | Good aggregate orders | Median upper bound among good orders | Upper bound ≤1 / ≤5 / ≤10 / ≤30s |
|---|---:|---:|---:|---:|---:|
| G2 bounded-conditional | 1774 | 1553 | 502 | 9.614s | 1 / 142 / 258 / 395 |
| G4 scenario-only | 1800 | 1316 | 428 | 1.961s | 94 / 325 / 363 / 391 |

Every good-order first-age lower endpoint is zero. Refused timing grades and missing counted-value statuses stay in orders.csv.gz. The complete census therefore does not establish that the profitable aggregate orders had to rest for5s,10s or longer.

## Exact fill-level subset

The saved fresh-level/front-candidate receipt contains1,725 distinct fill legs:1,526 outside the top-three posting census;199 matched top-three legs. Of those,153 have enabled G2/G4 timing and46 refused timing. The153 split into48 G2 (29 orders) and105 G4 (83 orders). Only36 G2 and51 G4 fills have the common valid1s/10s/60s value cohort. This is a small, selected subset of top-maker fills, **not all their good fills**, and the front candidates are not certified front-of-queue observations.

Good fill is declared as gross value positive at **both10s and60s**. It is a stricter descriptive outcome than positive1s spread capture; fees and later hedging costs are absent, so it is not net profit. There are6 such G2 fills and13 G4 fills. None has a5s minimum inferred age. Four G4 fills have upper bounds≤1s; this is conditional short-post compatibility, not evidence that fast posting caused their gains. One G4 good fill has a positive minimum exceeding1s; the rest still allow a near-fill post.

Every qualifying fill is printed below (multiple rows under one hash are related executions). Bounds use the original hard match window, not the narrower book-decrease scenario. Markouts are inherited from the accepted settlement-clock valuation, up to2s after the possible match clock. They are not precisely10/60s after actual matching. Continuous rest, actual queue rank and hidden cancel/re-entry remain unidentified.

| Date/market | Maker/quote | Timing grade | Posting-to-fill age range | Shares | Gross10s /60s c/share | Hash prefix |
|---|---|---|---:|---:|---:|---|
| 20261024/3699278 | #1 bid 39¢ | bounded | 0.000–18.198s | 3.730 | +5.50 / +17.50 | `0x242d395bbf` |
| 20261024/3700070 | #1 ask 59¢ | bounded | 0.000–3.845s | 30.000 | +11.50 / +1.50 | `0x88be6d1b78` |
| 20261024/3702960 | #3 ask 24¢ | scenario | 0.000–0.650s | 4.000 | +2.50 / +2.50 | `0x97e8d44bbc` |
| 20261024/3702960 | #3 ask 24¢ | scenario | 0.000–0.376s | 3.704 | +1.50 / +10.50 | `0xc4a87ecbe0` |
| 20261024/3702960 | #3 ask 19¢ | scenario | 0.000–0.763s | 4.762 | +3.50 / +13.50 | `0x3868788087` |
| 20261025/3763401 | #3 ask 81¢ | scenario | 0.000–2.958s | 15.765 | +5.50 / +9.50 | `0xa51b0e7041` |
| 20261025/3763401 | #3 ask 72¢ | scenario | 0.000–0.083s | 1.351 | +51.50 / +52.00 | `0xc1fce82c37` |
| 20261025/3776314 | #1 ask 20¢ | bounded | 0.000–3.340s | 0.320 | +15.50 / +16.50 | `0x108657c20d` |
| 20261025/3776314 | #1 ask 20¢ | bounded | 0.000–3.382s | 14.493 | +15.50 / +17.00 | `0x108657c20d` |
| 20261025/3776314 | #1 ask 20¢ | bounded | 0.000–3.533s | 1.530 | +15.50 / +17.50 | `0x108657c20d` |
| 20261026/3804696 | #1 bid 35¢ | scenario | 0.000–51.508s | 2.000 | +9.50 / +15.50 | `0xda63245651` |
| 20261026/3804696 | #2 ask 19¢ | scenario | 0.000–49.836s | 50.000 | +5.00 / +10.50 | `0x604bfb1a4a` |
| 20261026/3850428 | #2 ask 46¢ | scenario | 0.000–1.669s | 3.961 | +1.50 / +1.50 | `0x55b9a28560` |
| 20261026/3850428 | #2 bid 38¢ | bounded | 0.000–27.122s | 5.000 | +0.50 / +55.50 | `0x46c1835720` |
| 20261027/3877472 | #1 ask 59¢ | scenario | 0.000–6.213s | 1.695 | +8.50 / +8.50 | `0xddd8ab1d85` |
| 20261027/3877472 | #1 ask 59¢ | scenario | 0.000–7.095s | 1.639 | +8.50 / +10.50 | `0xddd8ab1d85` |
| 20261027/3877472 | #1 ask 59¢ | scenario | 0.000–7.782s | 1.695 | +8.50 / +9.50 | `0xddd8ab1d85` |
| 20261027/3877472 | #1 ask 59¢ | scenario | 1.484–9.409s | 3.390 | +9.50 / +7.50 | `0xddd8ab1d85` |
| 20261027/3877472 | #2 ask 59¢ | scenario | 0.000–21.760s | 14.117 | +3.50 / +9.50 | `0x7efeb9826a` |

## Matched comparison and interpretation

The frozen comparison was min-age≥5s versus max-age≤1s, separately G2 and G4, matching market/day/maker/side/price,first-versus-later record class and creation context. **Both have zero common strata**. The500 declared allocations per grade are saved as not-evaluated zeros, not a null p-value or evidence of no effect. No thresholds or matching were relaxed after seeing this. We cannot state that earlier orders achieve a higher good-fill rate on these data.

The useful concrete result is the interval table: some good fills are compatible with short posting leads; others admit much longer leads, but their early posting is not identified. We cannot conclude that queue age is irrelevant, nor that all good fills require early queue building. The better next estimand is how much earlier the same-price order must join to capture its **entire subsequent fill sequence**, including early losses and later gains; that requires queue competition and cumulative volume, not first-fill markout alone.

## Checks, limits and next priority

Independent checker reconciles every source join,duplicate count,grade,interval,quantity,value,cell and matched statistic. All six planted corruptions are refused. Synthetic interval controls include positive lower bounds,overlapping windows,unknown execution lower bounds and impossible chronology. No source or strategy file changed. Existing dirty work is preserved by owned-state staging. Result-bearing jobs use PF_SLOT=codex/run_light; resource receipt accounts for all attempts within2.88CPU seconds,task2.00/line5.30 plus.05reserve.

Largest limitation: the cached exact-fill subset covers only199 of the top-three fills; first-post upper bounds are first-fill-derived, and a continuous queue path is latent. ONE next priority: a declared **full top-three same-hash fill-sequence age/value census** from the existing accepted valued-fill artifact (all13,227 top-three BOOK rows), preserving whole-sequence economics and hard match windows; then test whether long-age evidence actually increases before a queue counterfactual. The1GB artifact was deliberately not reread in this small packet. This requires a bounded CPU top-up if it exceeds the remaining allowance; no new days or live activity are needed. Do not infer a required lead time from the upper-bound histogram.
