# Detail-review loop — charter

Opened 2026-08-08. Prompted by the observation that I have been declaring things dead on summary statistics
without examining the details underneath them. Several kills in this session rest on aggregates that hide
structure, and at least three measurements I made myself were never followed up.

## Details I glossed over (the actual agenda)

| # | detail | why it matters | why it was never checked |
|---|---|---|---|
| D1 | **The edge is 100% short-side.** Measured leg Sharpes: A long **−0.71** / A short **+1.30**; B long −0.39 / B short **+0.95**. Both long legs LOSE money. | The L/S book is a profitable short book dragged down by an unprofitable long leg. A short-only construction with a beta/index hedge could be materially better. | I measured it in `dp_phase3_diligence.py`, flagged it, offered it as an option — and never ran it. |
| D2 | **The delisting kill may be over-broad.** The sweep's own numbers: pre-2025-08 all events −0.62 Sharpe, but **ex-crisis events −0.21** (flat, not negative); post-2025-08 +3.38. The killer case (ANCUSDT +148.9% then halt) had **announcement→settlement of 0 days**. | Settlement lead time is STATED IN THE ANNOUNCEMENT — an ex-ante, non-fitted filter. Excluding events that settle inside the holding window is mechanism, not curve-fitting. | I accepted the agent's aggregate kill without reading the sub-sample structure. |
| D3 | **The equity strategy was killed on the wrong universe.** `alpha_v7_honest.py` tests the full S&P 100 (−0.20 hard split). The DEPLOYED config was 11 hand-picked Tier A+B names. | Different universe, different result — possibly. I flagged this caveat myself and then treated the strategy as dead anyway. | Never ran the 11-name config. |
| D4 | **Concentration was never revisited at the corrected cost.** The book holds ~26 names because breadth was assumed to help and cost punished concentration. C1 showed cost is 3-5× lower; C2 showed breadth does NOT convert. | Both premises behind wide diversification are now falsified. A concentrated book may dominate. | The two findings arrived separately and I never combined them. |
| D5 | **The sleeves alternate and I never asked why.** A: +2.21 → +0.69; B: +0.45 → +1.04; by half-year A goes +1.05/+1.60/−0.69 while B goes +1.00/−0.26/+2.57. | If the alternation has a driver (vol regime, dispersion, trend), that is timing information worth more than either sleeve. | Treated as "you can't predict which works" without testing whether you can. |

## Discipline

Unchanged: both eras, block-bootstrap CIs, paired deltas with CI on the delta, chronological hard split for
level claims, pre-registered gates and falsifier before running, every result costed at the measured
small-clip cost and stated with a dollar capacity.

**Extra rule for this loop, given its premise:** where a kill is being revisited, state precisely which
statistic the original kill rested on and what in the detail contradicts it. If nothing does, the kill stands
and gets recorded as re-confirmed rather than quietly re-opened.

---

## Iteration 1 — D1: is this actually a short book?

**The measurement that prompts it** (`dp_phase3_diligence.py`, held-out 2025-01→2026-06, top-40, band):

| leg | Sharpe |
|---|---|
| A long | −0.71 |
| A short | **+1.30** |
| B long | −0.39 |
| B short | **+0.95** |

Both long legs are negative in both sleeves. On 7 of the 8 worst days the short leg was the bigger loser,
which says the shorts also carry the squeeze tail — so this is not a free lunch, it is a different book.

**H1.** A short-only construction — short the bottom-quintile names, hedge the resulting market exposure with
a long index/BTC position rather than with the model's own long picks — beats the symmetric L/S book on
held-out net Sharpe.

**Why it might NOT** (stated first, so the test is fair): the long leg may be earning its keep as a *hedge*
rather than as alpha, in which case removing it raises beta exposure and the volatility more than it raises
return. The measured leg Sharpes do not settle this because they are computed on the residual return, which
already nets out beta.

### Constructions tested
- `ls` symmetric long/short (incumbent)
- `short_only` bottom quintile only, dollar-neutralised with a long BTC/index position sized to the short
  basket's beta
- `short_tilt` asymmetric 70/30 short/long weighting
- `long_only` top quintile only, index-hedged — the negative control; should be poor if the legs are as
  measured

### Pre-registered gates
- **G1.** `short_only` beats `ls` on held-out net Sharpe, paired 7d-block CI on the delta excluding 0.
- **G2.** It also survives in the SELECT window (not a one-window artifact).
- **G3.** Drawdown does not worsen materially — a short book with squeeze tails could improve Sharpe while
  making the tail worse; report maxDD and skew alongside.
- **Falsifier.** G1 fails ⇒ the long leg is doing hedging work the leg-Sharpe decomposition hides, and the
  symmetric book stands.

### Result — **G1 FAILS, and the detail CORRECTS a claim I made to the user.**
(`live/dr_iter1_shortonly.py`)

Net Sharpe by construction, hedge ratios always estimated on the *other* window:

| construction | SELECT net | HOLDOUT net | SELECT DD@10%vol | HOLDOUT DD@10%vol |
|---|---|---|---|---|
| ls (symmetric, incumbent) | **+2.37 SIG** | +0.70 | **−4.90%** | −8.24% |
| short_only | +0.73 | +0.92 | −6.81% | −7.32% |
| short_tilt (70/30) | +1.90 SIG | +0.85 | −5.47% | **−6.39%** |
| **long_only** | **+3.15 SIG** | **+0.09** | −4.44% | **−12.77%** |

**G1 FAIL**: short_only beats ls held-out by only +0.52 [−1.80,+3.28], spanning zero.

**The correction.** I told the user "the edge is 100% short-side, and the tail is a short squeeze," based on
leg Sharpes measured on the BTC-*residual* return (A long −0.71 / A short +1.30). **That does not hold as a
book property.** Built as actual books on raw returns with an out-of-window hedge ratio, **long_only is the
BEST construction in SELECT (+3.15, best matched-vol drawdown) and the WORST in HOLDOUT (+0.09, worst
drawdown −12.77%).** The asymmetry flips completely between windows.

This is precisely the counter-hypothesis pre-registered before running: *"the long leg may be earning its
keep as a hedge rather than as alpha; leg Sharpes computed on the residual already net out beta, so they
cannot settle that."* It was right, and the original claim was an artifact of decomposing on a
beta-neutralised quantity.

**Second correction, caught before reporting.** Raw drawdowns suggested short_tilt roughly halves DD in both
windows (−31.1→−17.4 SELECT, −63.2→−27.1 HOLDOUT). That is mostly a **leverage artifact** — short_tilt runs
at 31.8%/42.4% vol against ls at 63.5%/76.7%. At matched 10% vol it is −5.47 vs −4.90 (worse) in SELECT and
−6.39 vs −8.24 (better) in HOLDOUT. Inconsistent, not a win. This is the same error made with the
sleeve-combination claim; the matched-vol check is now standard before any drawdown claim.

**What the detail actually shows.** No construction dominates. And the pattern is now the same one seen
everywhere in this programme: **sleeve A vs B alternates between windows, long vs short alternates between
windows, and construction ranking alternates between windows.** Nothing about this edge is stable except
that combining unstable things is better than picking one.

**Verdict: the symmetric L/S book stands. The "short book" characterisation is retracted.**

---

## Iteration 2 — D2: was the delisting kill over-broad? **Kill RE-CONFIRMED; my hypothesis refuted.**
(`live/dr_iter2_delist.py`)

Rebuilt from source: 421 delisting articles pulled from the Binance CMS API (articles live under
`data.catalogs[0].articles` — the sweep's path was wrong), prices from Binance Vision which does retain
delisted symbols. 22 USDT-M futures events, 41 symbol-events, 2022-11 → 2026-07.

| period | batches | mean bps | batch-clustered t |
|---|---|---|---|
| pre-2025-08 | 9 | **−76** | −0.71 |
| post-2025-08 | 13 | +1,102 | +2.73 |

Concentration: **top-3 batches = 68% of total P&L.**

**My hypothesis was that the ex-ante settlement-lead filter would rescue the pre-period. It is refuted — but
not for the reason I expected, and I could not test it as designed.** Futures delisting titles do not carry
the settlement date in the format spot titles use, so no lead days parsed and all four filter thresholds
returned identical samples. The filter was never actually applied.

It does not matter, because **the pre-period is already flat-to-negative (−76 bps, t −0.71) before any
crisis-event exclusion.** There is nothing for the filter to rescue: removing disasters from a flat period
leaves a flat period. The entire apparent edge lives in post-2025-08, which is 68% concentrated in three
batches and is in-sample to the discovery.

**Limitations, stated:** my parser found 22 events where the sweep found 64, so this is a partial replication
on a differently-filtered sample, not a clean re-test. Extracting settlement dates would require the CMS
article-detail endpoint. I am not pursuing it, because the pre-period result would have to change sign for
the conclusion to change, and a filter that removes losses cannot do that.

**Verdict: the delisting kill stands, now on a better-stated reason — it is an era effect, not a crisis-event
effect.** Capacity was never the binding issue either: ~13 four-hour windows a year.

---

## Iteration 3 — D3: **MY EQUITY KILL WAS WRONG.** (`live/dr_iter3_eq11.py`)

| execution universe | rebalances | net/day | hard-split Sharpe | 95% CI |
|---|---|---|---|---|
| full15 = `XYZ_IN_SP100` *(what I reported as the kill)* | 175 | −1.73 bps | **−0.20** | [−1.61,+1.17] |
| **tier_ab = the DEPLOYED 11 names** | 40 | **+13.88 bps** | **+1.82** | **[−1.00,+5.14]** |
| tierC = the 4 dropped names alone | — | empty | — | — |

`alpha_v7_honest.py` passes `allowed=set(XYZ_IN_SP100)` — 15 names — which is the deployed Tier A+B **plus
AMD, COST, INTC, LLY**, precisely the four Tier-C names `docs/STATUS.md` records as dropped because "they
hurt backtest at realistic cost." I ran the script, read −0.20, and declared the programme dead on a universe
its own operators had already rejected.

**This also reconciles a discrepancy I flagged and could not explain.** STATUS.md documents +1.67 hard-split;
the honest script returned −0.20; I concluded the documented number was unreproducible. It reproduces
(+1.82 vs +1.67). The entire difference was the execution universe.

**Caveats, which are serious.** Only **40 rebalances in six years** (~7/yr — the strategy is dispersion-gated
and rarely fires), and the CI **[−1.00,+5.14]** spans zero by a wide margin. Per-year figures rest on 1-10
observations and are noise. `tierC` alone produced no portfolio (too few names for top-K), so I cannot confirm
the four dropped names were specifically the cause rather than dilution generally.

**Correct status: UNINFORMATIVE, not validated — and emphatically not "dead", which is what I reported.**

**Why it matters for the target:** this is a different asset class, so it is structurally uncorrelated with
the crypto sleeves in a way no further crypto-signal work can produce. The sleeve arithmetic needs N
uncorrelated sleeves; a real equity sleeve at even +1.5 would be the third, and the first from outside crypto.

---

## Running score of this loop

Three kills examined, **three findings that changed the stated conclusion**:

| detail | what I had said | what the detail shows |
|---|---|---|
| D1 short-side | "the edge is 100% short-side, tail is a short squeeze" | **retracted** — long_only best in SELECT, worst in HOLDOUT; asymmetry flips |
| D2 delisting | killed on crisis events | kill stands, but the reason is an **era effect**; pre-period is flat before any exclusion |
| D3 equity | "dead, −0.20 hard split" | **wrong universe** — deployed 11 names give +1.82, reconciling the repo's documented +1.67 |

The common failure was reading an aggregate and not the composition underneath it.

---

## Iteration 4 — D3b: does the +1.82 survive more observations, and is it uncorrelated with crypto?

**Why.** 40 rebalances cannot distinguish +1.82 from 0. Two things are needed before it counts as a sleeve:
more observations, and evidence it is genuinely orthogonal to the crypto book.

**H4.** The 11-name equity strategy retains a positive Sharpe under walk-forward (many more rebalances than a
single hard split) and its return series is near-uncorrelated with the crypto sleeves.

### Gates
- **G1.** Walk-forward Sharpe on tier_ab > 0 with bootstrap CI excluding 0.
- **G2.** |correlation| to each crypto sleeve < 0.2 on overlapping dates.
- **G3.** Not carried by one year — positive in a majority of years with ≥5 rebalances.
- **Falsifier.** G1 fails ⇒ the +1.82 was a 40-observation artifact and the equity programme closes for real.

### Result — **G1 marginal, G2 unmeasurable, and the UNCONDITIONAL number partly un-does D3's correction.**
(`live/dr_iter4_eqwalk.py`, `live/state/eq_uncond.log`)

| universe | rebalances | ACTIVE Sharpe | CI | **UNCOND Sharpe** | **ann return** | days/yr invested |
|---|---|---|---|---|---|---|
| tier_ab (deployed 11) | 41 | **+2.67** | [−0.01,+5.91] | **+0.66** | **2.27%** | **13.7** |
| full15 (what I killed it on) | 287 | +1.48 | [+0.51,+2.46] SIG | **+0.82** | **10.07%** | 83.8 |

**The decisive number is the unconditional Sharpe, and it reverses the ranking.** tier_ab has the better
*active* Sharpe (+2.67 vs +1.48) and survives the hard split where full15 does not — D3's correction stands on
those grounds. But it is invested **13.7 days a year**, so capital sits idle 96% of the time and the strategy
returns **2.27%/yr** at an unconditional Sharpe of +0.66. full15, despite the weaker active Sharpe, deploys
capital 84 days a year and returns 10.07% at +0.82 unconditional.

**So: I was wrong to call the equity programme dead, and also wrong to imply tier_ab was the better book.**
On the portfolio-relevant basis neither clears the crypto sleeves, and the higher-active-Sharpe variant is the
less useful one.

**G2 could not be evaluated at all**: only **1 overlapping day** between the equity series (41 rebalances
spread over 2016-2025) and the crypto sleeves (2025-01→2026-07). The correlation that was the entire reason
to want an equity sleeve is unmeasurable on these samples.

**G3**: 3 of 3 years with ≥5 rebalances are positive, but that is three observations.

**Verdict: real but small.** Not dead — my original call was wrong. Not a solution either: 2.27%/yr, or
10.07%/yr on a variant that fails the hard split. It does not change the aggregate picture.

---

## Iteration 5 — D5: is the alternation predictable? **NULL — and this one upgrades a conclusion.**
(`live/dr_iter5_alternation.py`)

Twice I concluded "you cannot tell which sleeve will work, so combine" **without testing whether you can**.
Now tested: both sleeves rebuilt over the full 1,002-day span, four PIT state variables, A−B spread by state
tercile, both eras.

Full-span facts (cleaner than the window-specific estimates): **A Sharpe +1.62, B +0.73, correlation −0.012.**

| state variable | early high−low | late high−low | significant? |
|---|---|---|---|
| mkt_vol_30d | −8.7 bps | −0.8 bps | ns / ns |
| xs_disp_20d | +19.6 bps | +95.1 bps | ns / ns |
| mkt_trend_30d | +17.4 bps | −1.5 bps | ns / ns |
| avg_corr_30d | −13.2 bps | +72.8 bps | ns / ns |

**G1: no state variable is significant in either era, let alone both.** And the frozen conditional allocation
rule fails too — tilting on any of the four gives Δ vs fixed 50/50 of +0.21, +0.20, −0.80, +0.18, all
spanning zero.

**Why this null is worth more than the others: it converts a default into a finding.** The "just combine"
conclusion was previously reached because I could not tell which sleeve would work. It is now reached because
the alternation is **measurably unpredictable** across four plausible drivers in both eras, on 1,002 days,
with a full-span correlation of −0.012. Fixed equal weights are correct, and now for a tested reason.

This is the first affirmatively-established result in the programme rather than a failure to reject.

---

## Iteration 7 — FOUNDATION AUDIT. **The base is sound.** (`live/dr_iter7_foundation.py`)

Every number in this session rests on cached walk-forward predictions I had used as ground truth and never
verified — while this repo's history is a catalogue of look-ahead found late (same-day volume in a liquidity
gate, full-sample VPIN sizing, a label shifted by 1 instead of the horizon, a venue premium inflated by bar
misalignment). Auditing constructions while trusting the base is the wrong order. Five checks:

**F1 — no look-ahead.** IC of the prediction against various targets (CLAUDE.md's threshold: |IC|>0.10 is
suspicious):

| era | true target `alpha_A` | next bar `fwd1` | previous bar `lag1` | previous return |
|---|---|---|---|---|
| RECENT | +0.0302 | +0.0234 | −0.0584 | −0.0612 |
| OOS | +0.0210 | +0.0140 | −0.0303 | −0.0355 |

All far below 0.10. The negative loading on *past* returns is not leakage — it is the reversal mechanism
showing itself exactly as expected (predict high where the past return was low).

**F2 — purge and embargo verified directly.** Max training `exit_time` lands **28 hours before** each test
window opens, in both eras. Correctly purged.

**F3 — cache integrity is exact.** Recomputing RECENT from scratch and comparing to the cache over 268,535
predictions: **correlation 1.000000, max absolute difference 0.00e+00.** The cached preds are the current
pipeline, not a stale artifact.

**F4 — label sane, with one notable property.** Cross-sectional mean −0.136 bps (≈0 as a residual should be);
`z_res` sd 0.993; clipping touches 0.015% of rows. But **skew is +4.16** — the 4h residual return is heavily
right-tailed.

**F5 — the panel is survivor-only.** **0 of 175 symbols stop trading more than 30 days before the panel end.**
Delisted names are not merely under-represented, they are absent entirely.

### Two facts that explain earlier results

1. **The signal predicts the NEXT bar at 67-78% of its strength on the current bar** (fwd1 +0.0234 vs +0.0302
   RECENT; +0.0140 vs +0.0210 OOS). It is highly persistent, so consecutive bars' bets overlap heavily. This
   is an independent, mechanical explanation for why breadth never converted (C2: 0.88× vs 2.1× theory) and
   why effective breadth measured ~1.7% of nominal — it is not a statistical accident, it is built into the
   signal's decay profile.
2. **Label skew +4.16 is the structural source of the short-side tail risk.** The squeeze exposure I mistook
   for a property of the *book* in D1 is a property of the *label*: a heavily right-tailed residual return
   means whoever is short is exposed to the tail, regardless of construction.

### What the audit does and does not license
It licenses believing the ~+1.3 estimate is measuring what it claims — no leakage, correct purging,
reproducible inputs, sane labels. **It does not fix the survivor-only universe**, which remains a real and
un-repairable limitation of the data: the names that collapsed and were delisted — precisely the ones a short
book would have traded — were never in the panel.

This is the first time in the programme that a result has been *verified* rather than merely not-yet-refuted.
