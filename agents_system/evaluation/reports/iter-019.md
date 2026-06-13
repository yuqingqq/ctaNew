# Evaluation Report — iter-019

**Candidate:** Transaction-cost-aware NO-TRADE BAND on the held-book net weights — execute a
per-symbol weight change only if `|target − held| ≥ δ`, else carry (no trade). A pure
cost-efficiency layer; the alpha champion (HL70 regime-hybrid held-book, K=5/side, 6 sleeves) is
UNCHANGED. SOTA-grounded (Baldi-Lanfranchi FoFI 2024; arXiv:2412.11575). ALPHA track.

**Script:** `research/convexity_portable_2026-05-20/scripts/X130_notrade_band.py`
**Review:** PASS (G1 clean; base δ=0 reproduces X117 to the digit; nested-OOS/G4/G6/gross-PnL trustworthy).

## VERDICT: **REJECT**

Champion **UNCHANGED**. The robust cost-only band (δ≤0.02) is a real but tiny saving that is
too small and too universe-specific to adopt; the spectacular δ=0.05 "win" is the bet-changing
trap (gross PnL jumps). Three gates fail honestly: **G3 nested-OOS does not transport** (HL70
+0.24 but EXT −0.03), **G4 < p95** (no edge over random turnover-skip), **G6 paired CI crosses 0**.

## Gate table (@4.5bps, production calibration)

| # | Gate | Result | Pass? |
|---|---|---|---|
| G1 | Look-ahead | PASS (Review). Band reads only current target + executed-through-t−1 weights; gross identical across cost levels confirms pre-cost path is clean. | PASS |
| G2 | In-sample objective | δ=0.02 Calmar +1.76 > base +1.68 (Sharpe +1.97, maxDD −5512). Marginal improvement. | PASS (marginal) |
| G3 | **Nested-OOS of δ (DECISIVE)** | HL70 fwd Calmar +0.84→+1.08 (lift **+0.24**, 5/6 folds) but EXT +0.65→+0.61 (lift **−0.03**, **1/7 folds**). Does NOT transport; selector leans into trap (δ=0.05 chosen 2/6 HL70, 3/7 EXT). | **FAIL** |
| G4 | Matched random-skip placebo (200 seeds) | HL70 δ=0.02 **p64**, δ=0.03 **p36**; EXT δ=0.02 **p26**, δ=0.03 **p40**. All < p95. | **FAIL** |
| G5 | Per-fold | δ=0.02 HL70 5/7; EXT 5/8; S44 3/8. Not robustly ≥6/9; and lift evaporates OOS (G3). | FAIL (and moot) |
| G6 | Paired CI (block-boot, 2000) | HL70 δ=0.02 mean +0.080 bps/cyc CI **[−0.002, +0.191]** crosses 0; EXT crosses 0; S44 δ=0.03 clears but **NEGATIVE** (−0.165, [−0.304, −0.035]) = band HURTS S44. | **FAIL** |
| G7 | Universe robustness | HL70 marginal positive; EXT lift ~0/negative OOS; S44 HURTS. Does not hold on the production+robustness set. | FAIL |
| G8 | Cost realism | δ=0.02 Calmar lift holds at 1/3/4.5 bps (1.99→2.08, 1.81→1.89, 1.68→1.76) but stays tiny; the δ=0.05 "win" is present at every cost (it is a bet change, NOT cost-driven). | n/a — improvement too small to be cost-robustly meaningful |

## G2/G8 — δ-sweep (HL70 @4.5bps; base gross +12272)

| δ | Sharpe | maxDD | Calmar | totPnL | turnover | **grossPnL** | foldsPos |
|---|---|---|---|---|---|---|---|
| 0.000 | +1.93 | −5674 | +1.68 | +10472 | 800 | **+12272** | 7/7 |
| 0.010 | +1.95 | −5627 | +1.72 | +10600 | 792 | **+12381** | 4/7 |
| 0.020 | +1.97 | −5512 | +1.76 | +10663 | 790 | **+12441** | 5/7 |
| 0.030 | +1.91 | −5885 | +1.61 | +10403 | 736 | **+12058** | 5/7 |
| **0.050** | **+2.31** | −4585 | **+2.48** | +12501 | 491 | **+13605** | 5/7 |
| 0.080 | +1.02 | −9045 | +0.63 | +6229 | 341 | +6996 | 1/7 |

**Cost-only vs bet-changing (the gross-PnL tell, verified):**
- **δ≤0.02 is genuinely cost-only** — turnover drops only 800→790, gross PnL stays flat
  (+12272→+12441, ≈+1.4%), Calmar +1.68→+1.76. The whole gain is ~saved cost on sub-δ churn. Real
  but tiny (~+191 bps net over 402d).
- **δ=0.05 is the bet-changing TRAP** — gross PnL JUMPS +12272→**+13605** (+10.9%) and turnover
  COLLAPSES 800→491. It is no longer trading-less-of-the-same-bet; it holds a stale rank-boundary
  book that happens to win in-sample. Gross is identical across all cost levels (1/3/4.5 bps),
  confirming the +13605 is a pre-cost path change, not a cost saving. On EXT (Calmar +0.66→+0.78)
  and S44 (+2.10→+2.35) the same δ=0.05 does not reproduce the HL70 magnitude → single-universe-flavoured.

## G3 — nested-OOS of δ (DECISIVE; independently reproduced from per-cycle parquets)

δ chosen on past walk-forward folds (max past Calmar), applied forward. Trap δ=0.05 kept in menu.

| universe | chosen δ per fold | OOS base Calmar | OOS banded Calmar | **lift** | fwd folds_pos | trap chosen |
|---|---|---|---|---|---|---|
| HL70 | [.03,.03,.03,.02,.05,.05] | +0.84 | +1.08 | **+0.24** | 5/6 | 2/6 |
| EXT | [.03,.02,.03,.05,.01,.05,.05] | +0.65 | +0.61 | **−0.03** | **1/7** | 3/7 |

A cost lever should transport (cost structure is universe-agnostic). It does not: HL70 marginal
positive, EXT slightly negative with only 1/7 forward folds positive. Worse, the forward selector
**leans into the δ=0.05 trap region** (2/6 HL70, 3/7 EXT folds) — the honest forward-chosen band is
NOT the small cost-only band, it rides the bet-changing region that does not generalize.

## G4 — matched random-turnover-skip placebo (200 seeds, @4.5bps)

| universe | δ | skips | real Calmar | placebo p95 | **rank** | tot rank |
|---|---|---|---|---|---|---|
| HL70 | 0.020 | 5683 | +1.76 | +2.00 | **p64** | p66 |
| HL70 | 0.030 | 19169 | +1.61 | +2.50 | **p36** | p80 |
| EXT | 0.020 | 29341 | +0.66 | +0.93 | **p26** | p33 |
| EXT | 0.030 | 37779 | +0.70 | +0.96 | **p40** | p44 |

Skipping the RANK-BOUNDARY sub-δ churn does NOT beat skipping the same NUMBER of random trades.
At δ=0.03 the real band ranks BELOW the median random skip (p36/p40). The honest equivalent is
"trade a bit less at random" — the rank-boundary churn the band removes carries ~no signal
(consistent with iter-016), so removing it specifically beats nothing over removing random turnover.

## G6 — paired block-bootstrap CI (per-cycle banded−base, block=fold, 2000 boots)

| universe | δ | mean diff (bps/cyc) | 95% CI | clears 0? |
|---|---|---|---|---|
| HL70 | 0.02 | +0.080 | [−0.002, +0.191] | NO (crosses) |
| HL70 | 0.03 | −0.029 | [−0.768, +0.663] | NO |
| EXT | 0.02 | −0.006 | [−0.057, +0.038] | NO |
| EXT | 0.03 | +0.030 | [−0.013, +0.079] | NO |
| S44 | 0.02 | −0.025 | [−0.081, +0.032] | NO |
| S44 | 0.03 | −0.165 | [−0.304, −0.035] | clears but **NEGATIVE (HURTS)** |

No δ gives a positive CI clearing zero on HL70/EXT. The only CI that clears zero (S44 δ=0.03) is
negative — the band actively hurts S44. The HL70 δ=0.02 lift (+0.080 bps/cyc) is too thin to
distinguish from zero.

## Decision rationale

Base reproduces X117 exactly (HL70 @4.5bps +1.93/−5674/Calmar +1.68/totPnL +10472) — trustworthy.
The candidate fails the three honest gates that matter: G3 nested-OOS does not transport
(EXT −0.03, 1/7 folds), G4 < p95 (no edge over random turnover-skip, p64/p36/p26/p40), and G6 CI
crosses 0 (lift too thin). The δ=0.05 "Sharpe +2.31 / Calmar +2.48" is the bet-changing trap,
confirmed by the gross-PnL jump (+12272→+13605) and its non-reproduction on EXT/S44. A clean,
useful negative.

## Insight for next research cycle

Even an off-the-shelf SOTA cost-aware no-trade band gives **no robust edge** here, for a structural
reason: the rank-boundary churn it suppresses carries ~no signal (already known from iter-016), so
removing it specifically is statistically indistinguishable from removing a random-equal slice of
turnover (G4 p26–p64). The genuine cost-only saving (δ≤0.02) is real but ~+191 bps/402d — below the
noise floor of the paired CI. **Cost-side levers on THIS book are exhausted**: the held-book's
6-sleeve averaging already amortizes turnover, leaving no concentrated cost-for-no-signal pocket a
band can cheaply recover. Any future "cost trick" that posts a large Sharpe gain should be checked
against GROSS PnL first — if gross moves, it is a bet change masquerading as a cost saving and must
clear nested-OOS + multi-universe transport, which this did not. Direction is dead; redirect to
risk-side / different-signal work rather than execution-efficiency.
