# Research Insights — iter-010 (FAST selloff-onset metrics: do they LEAD AND continue DOWN? — NO-CANDIDATE)

**Human idea:** the prior alt-bear detector used `alt_index_30d` — a SLOW 30d trailing return that
only flags the selloff LATE and (iter-008) was a COINCIDENT BOTTOM-detector: flagged cycles BOUNCED
(median fwd alt +0.0036). The genuinely NEW angle: a FAST-reacting onset metric (a) detects the
rollover days EARLIER and — crucially — (b) captures DOWN-MOMENTUM / crash-continuation. The
hypothesis: a sharp drop CONTINUES short-term (liquidation cascades + vol clustering = down-momentum),
whereas the slow 30d grind mean-reverts. If true, a FAST crash-onset flag captures the continuation
the slow alt30 missed and could power a de-risk gate that actually helps.

**Verdict: NO-CANDIDATE.** The faster metrics **DO flag the rollover earlier** (up to ~21 days
earlier in 2025_q4, ~7 days in 2022_ftx) — so the "detect onset sooner" half of the idea is real. But
the **make-or-break forward-continuation test FAILS**: conditional on every fast flag, the forward
24h alt move still **BOUNCES** (median > 0, %neg < 50) — statistically identical to the slow alt30 it
was meant to beat. The down-momentum / crash-continuation premise does **not** hold at the trade
horizon across episodes. The G4 pre-check on the best fast candidate ranks **p8–p18 < p95** with
**negative episode-LOFO dropping every episode** — the same "run-smaller, not skill" failure as
iter-001/002/007/009. **Champion stays = baseline (HL70 Calmar +1.68).**

Script: `research/convexity_portable_2026-05-20/scripts/iter010_fast_selloff_metrics.py` (rebuilds the
PIT eq-weight alt-index from klines verbatim per X122/X123, derives the fast battery, runs all four
decisive tests). Per-cycle output: `results/iter010_fast_metrics_EXT.parquet`. Reuses the X123 EXT
multi-episode held-book panel (`pnl_base`/regime/fold/is_side/alt30/btc30/alt_fwd_hold) verbatim.

---

## STEP 2 — the FAST onset-metric battery (PIT, trailing, `.shift(1)` lagged, on the eq-weight alt complex)

All built on the same eq-weight alt-index (cum log-ret of the 22 EXT alts ex-BTC/ETH, `.shift(1)`):

| metric | def | bearish-flag region |
|---|---|---|
| `alt_1d` / `alt_3d` / `alt_7d` | short-window alt-index return (6/18/42 bars) | < 0 |
| `alt_dd10` / `alt_dd20` | alt-index level vs trailing 10d/20d running high (drawdown-from-high) | < −5% |
| `alt_rvol_spike` | 3d realized vol / 30d baseline realized vol | > 1.25 |
| `breadth_below_ma7` | fraction of alts below their 7d MA | > 0.6 |
| `breadth_neg_7d` | fraction of alts with negative 7d return | > 0.6 |
| `alt_accel_3d` | `alt_3d` − prior `alt_3d` (2nd difference = is the drop accelerating?) | < 0 |

SLOW reference (the iter-007/008 flag): `alt30 < btc30`.

## STEP 3.(1) — LEAD TIME: **YES, the fast metrics flag earlier.**

Days the fast metric crosses its bearish region BEFORE the slow `alt30<btc30` fires, per episode
(+ = earlier):

| episode | slow first fires | alt_1d | alt_3d | alt_7d | alt_dd20 | alt_rvol | breadth | accel |
|---|---|---|---|---|---|---|---|---|
| 2022_luna | 2022-05-01 | +0.0 | +0.0 | +0.0 | +0.0 | −0.8 | +0.0 | +0.0 |
| 2022_ftx | 2022-11-01 | +0.0 | −0.8 | **+7.3*** | −1.7 | −2.3 | −1.5 | +0.0 |
| 2024_summer | 2024-06-01 | +0.0 | +0.0 | +0.0 | −7.0 | +0.0 | +0.0 | +0.0 |
| **2025_q4** | 2025-09-22 | **+20.8** | **+21.0** | **+21.0** | **+21.0** | −0.5 | **+21.0** | **+21.0** |

(*alt_7d in ftx: the trailing-7d return turned negative ~7d before the slow 30d-relative flag.)

**The faster-onset claim is confirmed.** In **2025_q4** — the episode that IS the −57% HL70 drawdown —
nearly every fast metric fires **~21 days earlier** than the slow `alt30<btc30`. In 2022_ftx the fast
metrics also lead by 1–7 days. (luna/2024_summer fire at the episode boundary because the bear was
already underway at the window start.) So if "detect the rollover sooner" were sufficient, this idea
would win. **It is not sufficient — what matters is whether the forward move continues.**

## STEP 3.(2) — FORWARD CONTINUATION (the make-or-break): **NO — every fast flag still BOUNCES.**

Conditional on each flag firing, the **forward 24h alt-index move** (the trade horizon a gate acts on):

| flag | n | fwdAlt mean | **fwdAlt median** | **fwdAlt %neg** | read |
|---|---|---|---|---|---|
| SLOW alt30<btc30 (ref) | 6949 | +0.0008 | **+0.0023** | **46.9** | BOUNCE (iter-008 confirmed) |
| alt_1d | 4948 | −0.0010 | +0.0017 | 47.6 | bounce/flat |
| alt_3d | 4987 | −0.0012 | +0.0020 | 47.2 | bounce/flat |
| alt_7d | 5144 | −0.0015 | +0.0013 | 48.3 | bounce/flat |
| alt_dd10 | 5832 | −0.0012 | +0.0018 | 47.8 | bounce/flat |
| alt_dd20 | 7229 | −0.0013 | +0.0011 | 48.7 | bounce/flat |
| alt_rvol_spike | 1739 | +0.0019 | +0.0039 | 45.1 | bounce/flat (more bounce!) |
| breadth_below_ma7 | 5113 | −0.0013 | +0.0019 | 47.4 | bounce/flat |
| breadth_neg_7d | 4968 | −0.0014 | +0.0014 | 48.2 | bounce/flat |
| alt_accel_3d | 5208 | −0.0016 | +0.0012 | 48.5 | bounce/flat |

**Every fast metric's forward median is POSITIVE and %neg is BELOW 50 (45–49%) — i.e. it BOUNCES,
exactly like the slow alt30.** The fast-mean is mildly negative (−0.001 to −0.0016) but driven by a
fat left tail; the *typical* (median) forward outcome after a fast flag is a small UP move, and fewer
than half of the cycles are negative. The down-momentum / crash-continuation premise is false at the
24h horizon. The 3-day forward view is identical (all medians +0.001 to +0.004, all %neg 47–49%): the
move does not continue down beyond the trade horizon either.

**Per-episode (the multi-episode test) — forward continuation is a coin-flip, not consistent:**
cell = median fwd alt @24h on flagged cycles (%neg):

| episode | alt_1d | alt_3d | alt_dd20 | alt_rvol | breadth_belowMA | alt_accel | SLOW |
|---|---|---|---|---|---|---|---|
| 2022_luna | −0.0030 (52%) | +0.0003 (50%) | −0.0009 (51%) | +0.0005 (50%) | +0.0031 (47%) | **−0.0084 (56%)** | −0.0054 (53%) |
| 2022_ftx | +0.0038 (43%) | +0.0038 (42%) | +0.0014 (47%) | +0.0010 (48%) | +0.0031 (45%) | +0.0038 (44%) | +0.0031 (44%) |
| 2024_summer | −0.0024 (52%) | −0.0027 (52%) | −0.0009 (51%) | +0.0013 (46%) | −0.0026 (52%) | −0.0030 (53%) | −0.0010 (51%) |
| **2025_q4** | +0.0008 (49%) | +0.0037 (46%) | +0.0012 (49%) | +0.0015 (49%) | +0.0028 (46%) | +0.0020 (47%) | +0.0012 (49%) |

A fast flag shows genuine forward down-continuation (median<0 AND %neg>52) only in **2024_summer**
(weak, 51–53%) and **2022_luna for the accelerator** (−0.0084, 56% — the one true crash). In
**2022_ftx it BOUNCES** (42–48% neg) and in **2025_q4 — the −57% episode itself — it BOUNCES**
(46–49% neg). The earliest-firing detector (q4, +21d lead) is precisely where the forward move
afterwards is a bounce: firing earlier just means flagging further from any continuation. **No fast
metric continues down across ≥2 episodes** (the pre-registered precondition).

## STEP 3.(3) — IC future vs IC past @ trade horizon: **NO real lead (coincident/lagging or wrong-sign).**

| feature | IC_fwd_pnl | IC_past_pnl | IC_fwd_alt | IC_past_alt | \|fut\|>\|past\|? |
|---|---|---|---|---|---|
| alt_1d | −0.028 | +0.002 | −0.013 | +1.000 | no |
| alt_3d | −0.023 | −0.013 | −0.008 | +0.523 | no |
| alt_dd10 | −0.058 | −0.033 | −0.008 | +0.445 | no |
| alt_dd20 | −0.061 | −0.040 | −0.002 | +0.326 | no |
| alt_rvol_spike | +0.044 | +0.063 | **+0.047** | −0.032 | YES |
| breadth_below_ma7 | +0.036 | +0.025 | +0.002 | −0.541 | no |
| alt_accel_3d | +0.002 | +0.010 | −0.001 | +0.364 | no |

- All the **drawdown / short-return metrics are coincident/lagging**: their IC vs the *past* alt move
  (|IC_past_alt| 0.33–1.0) dwarfs their IC vs the *future* alt move (|IC_fwd_alt| ≤ 0.013). By
  construction a short-window drawdown measures the move that *just happened*, not the next one.
- The only metric with |IC_fut|>|IC_past| is **alt_rvol_spike — but with the WRONG sign and tiny
  magnitude** (+0.047): high realized-vol predicts a *higher* (bounce) forward alt move, not a
  continued fall. That is the opposite of a de-risk signal (it would say "vol spiking → expect a
  bounce → stay in"), and at |IC| 0.047 it is noise anyway.
- vs forward **book PnL**, the deepest IC is alt_dd20 at −0.061 — still noise, and |IC_past_pnl|
  −0.040 is comparable (coincident).

## STEP 3.(4) — G4 PRE-CHECK on the best fast candidates: **FAIL p8–p18; LOFO negative every episode.**

Ranked by side-cycle forward down-continuation strength (most-negative fwd alt @24h first), then a
FLAT-side gate vs matched-random-timing FLAT of the same count (300 seeds), on the EXT panel:

| candidate | side-flag fwd_alt_med (%neg) | base→gate Calmar | gate maxDD | placebo p95 / max | **rank** |
|---|---|---|---|---|---|
| alt_accel_3d | −0.0011 (51%) | +0.664 → **+0.376** | −6,154 (WORSE) | +0.874 / +1.135 | **p11** ✗ |
| alt_dd20 | −0.0007 (51%) | +0.664 → **+0.373** | −5,080 (worse) | +0.785 / +1.086 | **p18** ✗ |
| alt_1d | −0.0006 (51%) | +0.664 → **+0.350** | −6,012 (WORSE) | +0.919 / +1.360 | **p8** ✗ |

All three best fast candidates **lower Calmar** (0.664 → 0.35–0.38), make **maxDD worse** (FLATting
side cycles whose forward move bounced removes good cycles), and rank **p8–p18 < p95** — a matched
random FLAT of the same magnitude does *better*. Episode-LOFO lift is **negative dropping every single
episode** (alt_accel: −0.31/−0.20/−0.34/−0.16; alt_dd20: −0.30/−0.34/−0.19/−0.09; alt_1d:
−0.32/−0.35/−0.28/−0.19) — it uniformly hurts; no episode rescues it. This is the iter-001/002/007/009
"run-smaller, not skill" failure mode, now confirmed for the fast-metric family.

---

## STEP 4 — pre-registration & honest decision

**Pre-registered preconditions for proposing a fast-metric de-risk gate (ALL required):**
(a) a fast metric flags the rollover EARLIER than slow alt30 — **PASS** (up to +21d in 2025_q4);
(b) conditional on the fast flag, the forward move CONTINUES DOWN (median<0 AND %neg>52) at the trade
horizon across ≥2 episodes — **FAIL** (every fast flag BOUNCES: full-panel median > 0, %neg 45–49%;
per-episode continuation holds only in 2024_summer + luna-accelerator, BOUNCES in ftx and in 2025_q4
the −57% episode itself);
(c) beats the matched-random-timing placebo (G4 ≥ p95) — **FAIL** (p8–p18; random does better;
episode-LOFO negative dropping every episode).

**→ NO-CANDIDATE. No change proposed this iteration.** Champion stays = baseline (HL70 Calmar +1.68).

## Why this is the expected result, and what it teaches

This is the decisive disposal of the "fast crash-continuation" angle and the symmetric confirmation of
iter-008's bottom-detector finding:

1. **Faster detection ≠ forward edge.** The fast metrics genuinely fire earlier (the human's first
   intuition was right — alt30 is laggy), but firing earlier on a *trailing* drawdown just flags the
   move that already happened. The forward move from a fast flag is a coin-flip that, if anything,
   BOUNCES (median +). The crash-continuation premise (liquidation-cascade down-momentum at 24h) is
   not present in the equal-weight alt complex on free price data: by the time a sharp drop is
   measurable in any trailing window, the cascade has already discounted and the typical next move is
   a bounce/chop. This is exactly the iter-008 result (slow flag bounced, median +0.0036) reproduced
   for every fast variant (medians +0.0011 to +0.0039).
2. **The drawdown metrics are mechanically coincident** (|IC_past_alt| 0.33–1.0 ≫ |IC_fwd_alt| ≤
   0.013): a short-window drawdown IS the recent move, definitionally lagging. The one
   forward-loaded metric (rvol_spike) points the wrong way (vol → bounce).
3. **The 2025_q4 paradox seals it:** the fast metrics lead by ~21 days there — the most they lead
   anywhere — yet the forward move after the flag in q4 is a *bounce* (46–49% neg). Earlier detection
   bought zero forward predictability; it just moved the (still-coin-flip) flag further from the
   eventual losses. A gate built on it FLATs good cycles → maxDD worse, Calmar down, p8 placebo.

**Converges with iters 005/007/008/009:** the four classes of free forward observable (implied-vol,
price/alt-direction-slow, net-short direction, positioning/leverage) plus now the **fast-onset price
family** are ALL coincident/lagging at the 24h trade horizon. iter-008 showed the slow flag is a
coincident *bottom*-detector that bounces; iter-010 shows making the same price observable FASTER
detects the rollover sooner but the forward move is still a bounce — speed does not buy a lead. The
correlated alt deleverage is not forward-separable from a bottom on ANY free price-derived metric, fast
or slow. **The free leading-signal axis is exhausted** (this confirms, does not weaken, the iter-009
conclusion: "fast metrics flag earlier but the forward move is still a coin-flip / they just whipsaw
more").

**Recommendation (unchanged, high confidence):** the only remaining path to a *leading* deleverage
signal is PAID data — Coinglass aggregated liquidation-cascade flow (the actual trigger, observable as
it begins rather than via a lagged price drawdown) or Glassnode on-chain (exchange inflows /
stablecoin supply ratio = pre-positioning of selling capital that price cannot see). Both paid, no key
in env → human scope/budget decision. Free alternatives: accept the structural DD + live kill-switch,
or a bull-only beta pivot.

Artifacts:
- script: `research/convexity_portable_2026-05-20/scripts/iter010_fast_selloff_metrics.py`
- per-cycle features+pnl: `results/iter010_fast_metrics_EXT.parquet`
- console log reproduces all tables (lead-time, forward-continuation full + per-episode, IC fut/past,
  G4 pre-check + episode-LOFO).
