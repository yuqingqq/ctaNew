# Beyond-cross-section loop — charter

Opened 2026-08-07. Prompted by the observation that all 14 prior iterations varied components *inside a single
frame*: dollar-neutral cross-sectional L/S, 176 Binance USDM perps, 4h decide / 24h hold, predicting the 4h
BTC-beta-residual return. The nulls are real and well-established — but they are nulls about that frame.

## The structural gap

Markets pay for three distinct things. Everything this repo has ever tested chases one of them.

| source of return | economic role | what we did | status |
|---|---|---|---|
| **Prediction premium** | forecaster | all 14 iterations, two loops | exhausted on free data — 8 signals with strong standalone IC, zero incremental |
| **Risk / carry premium** | insurer, leverage provider | funding tested as a FEATURE and as a cross-sectional SIGNAL (−2.45 Sharpe) | **never held as a POSITION** |
| **Liquidity provision** | market maker | maker fills measured only as COST REDUCTION (+0.85 → +1.2 net) | **never measured as REVENUE** |

The benchmark being chased (6.33 Sharpe, ex-crypto-HFT-market-maker PM) plausibly earns the second and third.
We have only ever tried to earn the first, then treated the other two as ways to lose less.

## Dimensions held fixed across all prior work

| dimension | always | alternatives, and whether data exists |
|---|---|---|
| instrument | Binance USDM perps | spot: **61 Binance / 60 Coinbase / 60 OKX at 1h, ~2.8y** on disk; options: none |
| structure | dollar-neutral XS L/S | carry position, time-series/directional, dispersion, event-driven |
| predictand | 4h forward residual return | volatility (far more predictable), funding itself, spread |
| horizon | 4h decide / 24h hold | weekly/monthly — **cost falls ~50× and stops binding**; sub-hour — needs infra |
| economic role | price-taker predicting | liquidity provider, carry holder |
| universe | same 176 perps | spot pairs, new listings, cross-venue |

## Directions

| # | direction | data | prior |
|---|---|---|---|
| B1 | **Liquidity provision as revenue.** From aggTrades, `is_buyer_maker` gives the aggressor side directly — so the passive counterparty's markout is computable without quotes. Measures whether being the maker is profitable *as a business*, not as a discount. | aggTrades on disk, 31 syms × 1246 d | open |
| B2 | **Funding carry as a held position.** Cash-and-carry: long spot, short perp, delta-neutral, collect funding. This is a risk premium, not a forecast. Never tested — the "carry" sleeve was a cross-sectional L/S *bet on* funding, a different object. | funding 217 syms + bn_spot 61 syms | open |
| B3 | **Horizon extension.** Weekly/monthly rebalanced book. Our binding constraint is cost at turnover 0.4/bar; at monthly rebalance it is ~1/50th. Untested *as a design* (iteration 1 slowed a 4h signal; it did not build one for the horizon). | owned | open |
| B4 | **Cross-venue / spot-perp dislocation.** Three venues on disk; `cross_exchange_features*.parquet` suggests prior partial work — check before re-running. | bn/cb/okx spot | open |
| B5 | **Volatility as the predictand.** Vol is orders of magnitude more predictable than returns. Monetisation without options is the open question. | owned | open |

## Discipline (unchanged)

Both eras; day-clustered CI for per-bar statistics, 7d-block for Sharpe; paired deltas with CI on the DELTA;
chronological hard split for level claims; pre-registered gate + falsifier before running. New rule from
iteration 7: for anything imported from a time-series paper, regress it on the **cross-sectionally demeaned
target first** — if the effect is market-wide it dies there in one command.

Additional discipline for this loop: these are *different businesses*, not different signals. Each must be
costed on its own terms — a carry position pays two legs of fees and borrow; a market-making book earns the
spread but pays adverse selection and needs queue priority we cannot verify offline. State the unmodelled
risk explicitly rather than reporting a gross number.

---

## Iteration 1 — B1: is liquidity provision profitable as a business?

**Why first.** It is the sharpest test of the reframing, it uses data already on disk, and it directly
addresses the benchmark question: an ex-HFT market maker's edge may not be a better forecast at all.

**The measurement.** Binance aggTrades carry `is_buyer_maker`, which gives the aggressor side directly — no
Lee-Ready classification needed. For every trade, the *passive counterparty* filled at price P with
`maker_sign = -aggressor_sign`. Their gross P&L per unit notional over horizon h is the **markout**:

    markout_h = maker_sign * (P_{t+h} - P_t) / P_t = -aggressor_sign * forward_return_h

Volume-weighted across trades this is the gross revenue of providing liquidity, before fees. Net of the
Binance USDM VIP-0 maker fee (2.0 bps; 1.8 with BNB; lower at higher tiers, negative under MM agreements):

    net_LP_bps = markout_bps - maker_fee_bps

**H1.** Passive liquidity provision on these perps earns a positive markout net of the maker fee — i.e. the
spread earned exceeds adverse selection — at some horizon.

**Assumptions to validate**
- **A1 sign convention.** Verify `is_buyer_maker=True` ⇒ aggressor is the SELLER, by checking that the
  aggressor-signed short-horizon return is POSITIVE (trades move price in the aggressor's direction). If this
  comes out backwards the whole study is inverted.
- **A2 horizon.** Markout must be reported across horizons (1s → 5m). Adverse selection grows with horizon;
  the spread is earned immediately. A positive number at one horizon only is not a result.
- **A3 what this does NOT model** — stated, not buried: queue priority (we cannot verify we would be filled),
  inventory risk, the fact that we observe only *executed* aggressive trades, and that a real maker chooses
  when to quote. This bounds the opportunity; it does not simulate a business.

**Pre-registered gates**
- **G1.** Volume-weighted markout net of 2.0 bps maker fee is positive at ≥1 horizon, in BOTH eras.
- **G2.** Positive on ≥1/3 of the 31 symbols, not concentrated in one name.
- **G3.** The horizon profile is monotone-decaying (spread earned up front, adverse selection accumulating) —
  the shape microstructure theory predicts. A non-monotone or rising profile suggests a measurement error.
- **Falsifier.** Markout net of fee is negative at every horizon in either era ⇒ passive provision on these
  instruments does not pay at retail fee tiers, and the maker-execution result stands as a cost reduction
  only, never a revenue line.

### Result — **G1/G2/G3 FAIL. Passive provision does not pay — and fees are NOT the reason.**
(`live/bx_iter1_markout.py`, 31 symbols × 754 sampled days)

**Measurement error caught in the pilot and fixed before the full run.** My first version measured drift from
the *trade price*, which credits zero spread and reports pure adverse selection — every number came out
negative at every horizon including 1s, which was the tell. Corrected by reconstructing the mid from signed
trades (last buy-aggressor ≈ ask, last sell-aggressor ≈ bid). The decomposition now closes arithmetically:
half-spread 1.832 + adverse selection (−2.287) = total (−0.455).

**A1 PASS** — aggressor markout +1.45 bps at 1s: trades move price in the aggressor's direction, so the sign
convention is right.

| | half-spread earned | adverse selection @1s | **maker gross** | net of 2.0 bps fee |
|---|---|---|---|---|
| pooled | +1.83 | −2.4 | **−0.57 (OOS) / −0.69 (RECENT)** | **−2.57 / −2.69** |

**0 of 31 symbols positive** (best GMX −2.02, worst ZEC −2.68). G3 also fails: the horizon profile is not
monotone — it worsens to 5-15s then partially recovers by 300s, which is transient impact reverting.

**The decision-relevant point: gross is negative, so no fee tier rescues this.** At VIP-9 (0.0% maker) the
maker still loses ~0.6 bps per unit; even a −0.5 bp rebate leaves it negative. Adverse selection on Binance
perps exceeds the half-spread for an *undiscriminating* quoter. Market making here cannot be a fee-tier play —
it must be a **selectivity** play: quoting only when toxicity is low, which needs L2 depth, queue position and
latency this repo does not have. That is the same infrastructure wall the execution work hit, now measured
from the revenue side rather than the cost side.

**Caveat retained:** this measures the average over all executed fills. A real maker chooses when to quote, so
this is a *lower bound* on the business — but it establishes the naive version is unprofitable by ~2.6 bps and
that the gap to viability is selectivity, not fees.

---

## Iteration 2 — B2: funding carry as a held position

### Result — **NULL, and for a completely different reason than everything before it.**
(`live/bx_iter2_carry.py`, 61 symbols with spot + funding + perp, 2025-01 → 2026-05, 80,764 symbol-intervals)

**The premium has compressed away.** Annualised funding across the 61 tradeable symbols: **median −0.69%**,
p25 −5.34%, p75 +3.12%, **max +5.41%** (UNI 5.3, LINK 4.9, LDO 4.9). The literature's "8-40% APR" does not
hold in this sample — consistent with the BitMEX finding that extreme funding rates have fallen ~90% since
2016. Aggregate net carry is *negative*: −3.1%/yr (OOS), −6.1%/yr (RECENT).

Top-K by trailing funding, both legs costed at 30 bps round trip:

| era | best config | churn | gross %/yr | gross Sharpe | net Sharpe |
|---|---|---|---|---|---|
| OOS | K=5, 504h | 0.01 | +8.2 | +0.20 [−0.54,+1.00] | +0.20 spans0 |
| RECENT | K=10, 168h | 0.02 | +5.7 | +0.15 [−0.59,+1.00] | +0.14 spans0 |

**No config has net CI > 0 in either era, let alone both.**

**What makes this null different from all the others: cost is not the binding constraint.** At weekly/3-weekly
rebalance the churn is 0.01-0.03, so the 30 bps round trip costs almost nothing and gross ≈ net. The trade
simply does not pay — the risk premium for supplying leverage has been arbitraged down to ~5%/yr at the very
top of the cross-section, with a Sharpe near 0.15. Every prior null in this repo was "signal spanned" or "cost
eats it"; this one is **"the premium isn't there."**

**Data limitation, stated:** the spot cache starts 2025-01, so this covers 16 months, not both standard eras.
A longer spot history could revisit the 2021-2022 period when funding was genuinely rich — but that is a
statement about a regime that no longer exists, not a deployable finding.

---

## Iteration 3 — B3: horizon extension

**Why.** The two non-prediction businesses are now measured and both are closed. Of the dimensions still held
fixed, horizon is the one where our *measured* binding constraint disappears: the book pays 2-4 bps/bar at
turnover 0.40 on a 4h grid; at weekly rebalance that is ~1/50th. Iteration 1 of the cost/turnover loop slowed
a *fast* signal down and found it era-locked — that is not the same as building a signal **native** to a slow
horizon, with a matching label and matching features, which has never been done here.

**H3.** A cross-sectional book with a multi-day label and multi-day features earns a net Sharpe that survives
the hard split, because at that turnover cost stops being the constraint and the surviving signal is the
slower, more persistent component rather than the 4h reversal.

### Pre-registered gates
- **G1.** Gross rank-IC against the horizon-matched residual label, same sign in BOTH eras, day-clustered CI
  excluding 0. Overlapping labels ⇒ CIs must use non-overlapping blocks, not day clusters alone.
- **G2.** Net Sharpe at the calibrated cost with CI > 0 in BOTH eras.
- **G3.** Hard split — select the horizon and K on 2023-06→2024-12, evaluate on 2025-01→2026-06.
- **Falsifier.** G2 fails ⇒ the cross-sectional edge does not exist at slow horizons either, and the
  prediction frame is closed at every horizon we can trade, not just 4h.

### Result — **G1 passes at 1d/3d only, G2 fails everywhere — but the REASON is structural and new.**
(`live/bx_iter3_horizon.py`)

**Construction note, stated for accuracy:** BTCUSDT is not a leg in the panel, so the code fell back to the
equal-weight market basket. The label here is a *market*-beta residual (beta mean 0.999, sd 0.367), not the
incumbent's BTC-beta residual. Defensible, but not identical.

**A second measurement bug caught and fixed before reading anything**: the first run reported IC CIs like
`[+4.63,+11.04]` for an IC of +0.0255 — I had passed a rank-IC series to a *Sharpe* CI function. Replaced with
a block bootstrap on the mean.

| horizon | OOS rank-IC | RECENT rank-IC | OOS gross / net Sh | RECENT gross / net Sh | blocks |
|---|---|---|---|---|---|
| 1d | +0.0255 [+0.015,+0.035] | +0.0305 [+0.017,+0.045] | +0.45 / −0.06 | +2.19 / +1.66 | 794 / 242 |
| 3d | +0.0337 [+0.019,+0.048] | +0.0419 [+0.022,+0.061] | +0.94 / +0.68 | +2.72 / +2.51 SIG | 265 / 81 |
| 7d | +0.0359 [+0.015,+0.057] | +0.0221 [−0.006,+0.048] | +1.03 / +0.91 | +2.23 / +2.11 SIG | 114 / 35 |
| 14d | **+0.0606** [+0.033,+0.087] | +0.0315 [−0.002,+0.065] | +1.26 / +1.22 | +1.52 / +1.48 | 57 / **18** |
| 30d | **+0.0691** [+0.028,+0.106] | +0.0142 [−0.041,+0.076] | +1.36 / +1.34 | −0.69 / −0.71 | 27 / **8** |

**The IC rises monotonically with horizon in OOS — +0.026 at 1d to +0.069 at 30d, every CI excluding zero.
That is 3× the 4h book's +0.021.** Churn per rebalance stays ~0.33-0.41, but rebalances are 30× rarer, so cost
genuinely stops binding: at 14d/30d, gross and net differ by 0.02-0.04 Sharpe.

**And it still fails G2 — because of a wall that is not cost.** Net Sharpe CIs are enormous at every horizon
([−3.52,+3.26] at 1d; [+1.23,+23.97] at 3d) and RECENT contradicts at 7d+. The reason is countable: **the
entire 1,975-day dataset contains only 65 independent 30-day observations.** A 30d book cannot be validated on
5.4 years of data, ever, with any method.

**⇒ The generalisable finding of this loop: the cost constraint and the statistical-power constraint pull in
opposite directions along the horizon axis.** Fast enough to measure reliably ⇒ too expensive to trade
(the 4h frame, cost eats 45% of gross). Slow enough to trade cheaply ⇒ too few independent observations to
establish anything (14d+, where the IC is strongest and the CIs are useless). The 1d-3d band is the only
region where both constraints are simultaneously survivable, and that is where the incumbent already lives.

This is not a null in the usual sense — the OOS IC pattern is real and monotone. It is a statement that the
attractive part of the horizon axis is **unfalsifiable with the data that exists**, and would need either a
much longer history or a much wider universe to test. That is a data problem, not a signal problem, and it is
the first time in 17 iterations the binding constraint has been neither cost nor information.

---

## Loop status

| # | direction | verdict |
|---|---|---|
| B1 | liquidity provision as revenue | **CLOSED** — gross −0.6 bps before fees; 0/31 symbols positive; no fee tier rescues it; viability needs selective quoting = L2 + latency |
| B2 | funding carry as a position | **CLOSED** — the premium has compressed to ≤5.4%/yr at the top of the cross-section, Sharpe ~0.15; median funding is *negative*; cost is not the constraint |
| B3 | horizon extension | **INCONCLUSIVE BY CONSTRUCTION** — OOS IC triples with horizon, but only 65 independent 30d blocks exist; the attractive region is unfalsifiable on available data |
| B4 | cross-venue / spot-perp dislocation | **CLOSED by prior work — do not re-run.** `docs/convexity_v1_optimization_loop.md` #185: the venue premium was 4h-MISALIGNED (venue close@period-end vs Binance close@hh:05), inflating its IC; boundary-aligned it **collapses >50%** (okx −0.016, cb −0.019 OOS) and the portfolio retest HURTS. Verdict: "a noisier proxy for resid_rev (corr −0.55)". As an *arbitrage* rather than a feature it needs sub-second cross-venue execution — the ICAIF screen found taker PnL negative net of fees at a 500 ms horizon. |
| B5 | volatility as predictand | **BLOCKED, not open.** Monetising a vol forecast needs an instrument: options (none on disk; Deribit free history is essentially BTC/ETH only = 2 names, not a cross-section) or a variance swap (none). The two option-free routes are already closed nulls here — vol-targeting as position sizing, and vol/dispersion regime gating. |

---

## Iteration 4 — B6: measure the slow signal with the RIGHT instrument

**Why.** B3 left one live thread and I mis-tested it. The OOS rank-IC rises monotonically with horizon
(+0.026 at 1d → **+0.069 at 30d**, every CI excluding zero) — those ICs are computed on thousands of
name-observations and are properly block-bootstrapped. What was underpowered was the **book Sharpe**, measured
on 27 blocks. I then wrote off the whole horizon as "unfalsifiable", which conflated the two.

The right instrument for a slow signal on limited data is not a portfolio Sharpe — it is the IC, measured with
non-overlapping-block CIs, plus an honest translation to implied Sharpe with breadth accounting. That is
testable now.

**H4.** The slow (14-30d) cross-sectional signal is real: its rank-IC survives a chronological hard split, is
not merely the known short-horizon reversal in disguise, and implies a Sharpe consistent with what B3 measured.

### Pre-registered gates
- **G1.** Hard split: select the horizon on 2023-06→2024-12 by IC, evaluate on 2025-01→2026-06. Held-out IC CI
  (block bootstrap, block = H) must exclude 0.
- **G2.** Same sign in both eras at the selected horizon.
- **G3 (attribution).** The IC must not be carried solely by `mom_s` (the short-horizon reversal already in the
  incumbent). Drop-one attribution at the selected horizon; if removing `mom_s` kills it, this is the existing
  edge re-measured at a longer horizon, not a new one.
- **G4 (honest translation).** Report implied Sharpe = IC × √(independent bets/yr) with breadth stated, and the
  sample length that would be needed to confirm it directly. Do not present it as a validated Sharpe.
- **Falsifier.** G1 fails ⇒ the horizon pattern is an in-sample artifact and the horizon axis closes for good.

### Result — **G1 PASS, G2 PASS(sign), G3 identifies it as a KNOWN sleeve, G4 bounds it at ~+2.**
(`live/bx_iter4_slowsignal.py`)

**G1 PASS — and this is the first gate-passing result in 18 iterations.** Hard split selected 14d on the
SELECT window; held out it gives IC **+0.0553 [+0.0284,+0.0830]**. All three horizons hold out significant:

| H | SELECT IC | HELD-OUT IC | n dates |
|---|---|---|---|
| 7d | +0.0356 [+0.010,+0.059] | **+0.0374 [+0.014,+0.060] SIG** | 507 |
| 14d *(selected)* | +0.0534 [+0.016,+0.090] | **+0.0553 [+0.028,+0.083] SIG** | 500 |
| 30d | +0.0297 [−0.015,+0.077] | **+0.0868 [+0.036,+0.132] SIG** | 484 |

**G2 — passes on sign, but the deployable cell does not.** My gate only checked sign consistency; the
significance pattern is weaker than that implies and must be stated:

| era | full universe | **top-40 (the deployable one)** |
|---|---|---|
| OOS | +0.0606 [+0.033,+0.087] SIG | **+0.0310 [−0.011,+0.071] spans0** |
| RECENT | +0.0261 [−0.007,+0.060] spans0 | +0.0628 [+0.018,+0.109] SIG |

Neither universe is significant in *both* eras, and the cost-viable top-40 book is insignificant in OOS. That
is the same both-era instability that has killed every prior lead, now appearing one level down.

**G3 — attribution says this is not new.** Drop-one over the seven features, OOS, H=14d: only **`mom_skip`
carries** (Δ −0.0023 [−0.0045,−0.0001], and barely). Everything else is neutral; dropping `vol_m` *improves*
IC by +0.0072. `mom_skip` is 3H-minus-H/3 trailing return — i.e. **skip-recent intermediate momentum**, which
`docs/CONCLUSION_2026-08-03.md` already documents as "a real, persistent, DIFFERENT-ROOT (underreaction)
orthogonal sleeve; thin, standalone-only, not a booster."

**So iteration 4 re-derives a documented finding rather than discovering one** — arrived at from the opposite
direction (horizon scan rather than factor screen), which is worth something as independent confirmation, but
it is not new alpha and it should not be presented as such.

**G4 — honest bound.** IC +0.0606 with nominal breadth 40 names × 26 rebalances = 1,043 bets/yr gives a naive
fundamental-law Sharpe of **+1.96** — an upper bound, because the law assumes independent bets and consecutive
14d bets overlap heavily. B3's directly-measured 14d book was +1.26 gross / +1.22 net (OOS). Those are
consistent: the realised number sits below the naive bound, as it must. (The "years needed" line printed by the
script is not a valid power calculation and is disregarded.)

**Verdict: CONFIRMS a known thin sleeve at ~+1.2 realised / +2.0 bound. Does not change the picture.** It is
the same sleeve the sleeve-correlation test (`sd_iter3_sleeves.py`) measured standalone at +0.47 net, spans 0.

---

# Loop close-out (2026-08-07)

Four directions outside the cross-sectional prediction frame. **Nothing adopted; two genuinely new mechanisms
for failure identified.**

| # | direction | verdict | why it failed — and each reason is DIFFERENT |
|---|---|---|---|
| B1 | liquidity provision | CLOSED | gross −0.6 bps *before* fees; adverse selection > half-spread; **no fee tier fixes it** — needs selective quoting (L2 + latency) |
| B2 | funding carry | CLOSED | **the premium is gone** — median funding negative, max +5.4%/yr, Sharpe ~0.15; cost is *not* the constraint |
| B3 | horizon extension | SUPERSEDED by B6 | book Sharpe underpowered (27-65 blocks); I wrongly generalised that to the whole axis |
| B4 | cross-venue | CLOSED by prior work | venue premium was a bar-alignment artifact; corrected IC collapses >50% |
| B5 | volatility | BLOCKED | no instrument to monetise a vol forecast — options absent, option-free routes already null |
| B6 | slow signal, right instrument | **CONFIRMS a known sleeve** | 14d skip-recent momentum, held-out IC +0.055 SIG, but top-40 OOS spans 0 and it is already documented |

**What the four different failure modes tell us.** Before this loop, every null in the repo was one of two
things: "the signal is spanned by the existing factor" or "cost eats it". This loop added three more:
*the premium has been arbitraged away* (B2), *the business needs infrastructure, not information* (B1), and
*the attractive region is unfalsifiable on the available history* (B3). Those are different diagnoses with
different implications — B2 says don't look there again; B1 says the gap is capex, not research; B3 says the
question is answerable only with more data.

**On the 6-Sharpe benchmark, from outside the frame.** The two businesses a market-making PM actually runs are
now measured on this data: passive provision loses 0.6 bps gross before fees, and carry pays ~5%/yr at best
with Sharpe 0.15. Neither is a 6-Sharpe business *on free data at retail access*. Their edge, if real, lives
in selective quoting with L2 and latency, and in fee tiers we do not have — which is the same conclusion the
execution work reached from the cost side, now confirmed from the revenue side.

Scripts `live/bx_*.py`; logs `live/state/bx_iter*.log`. All uncommitted.
