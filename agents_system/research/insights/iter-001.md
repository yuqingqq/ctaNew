# iter-001 — Research insights

## Target
Reduce HL70 baseline maxDD (−5,674 bps / −57%, Calmar +1.68) without materially losing Sharpe (+1.93).
The drawdown is the central unsolved problem (low %pos 39.9%, fat tails, one deep 2025 episode).

## What the prior ledger already proved about THIS drawdown (don't relitigate)
- **It is broad / correlated, not single-name.** X97 lesson: sizing levers that down-weight individual
  volatile alts (inverse-vol) can't fix it, and they HURT Sharpe because the alpha lives in the
  volatile alts. (X97: invvol K=3/5/7 = +1.41..+1.49 vs base; no DD help.)
- **It is a multi-month regime episode**, i.e. a run of persistently-correlated *negative cycles* —
  X98: worst DDs are −4,170 bps Jun–Sep 2023 (140d) and −2,955 bps Nov 2025–Apr 2026 (177d), both
  sideways-mean-rev-dominated, when the V0 cross-sectional edge inverts.
- **K=5 (more legs) already gave the one clean DD win** (X97: −25% DD, Sharpe flat) and is in the
  baseline. Diversifying further within a cycle is tapped out.
- **The failed vol-target (X97) levered UP**: `scale = min(2.0, vt/rv)` → 2× exposure when realized
  vol is low (the calm before storms) → DD got WORSE (−5,991). This is the documented anti-pattern.
- **PIT self-throttles keyed to the strategy's own trailing realized PnL/IC fail nested-OOS** (X99→X100):
  W=30/flat rescued 2026 in-sample but the block-OOS picker chose the wrong window going forward
  (the regime inversion is not forecastable by trailing IC; DDI R²=0.005). Any *tuned-window /
  tuned-action* throttle is a known dead end.
- The 2026 "alpha decay" / sign-flip saga was a **44-sym composition artifact**; HL70 is healthy
  (+3.44 in 2026). So the proposal must be validated on HL70 and not assume the decay pathology.

## The gap I am exploiting
Every *adaptive* DD attempt so far either (a) levered UP into calm (X97 vol-target), or (b) tuned a
window/threshold/action and died nested-OOS (X99/X100 throttle, K3-margin, decay sleeves). What has
**never been tested in honest OOS** is the textbook crash-protection control from the momentum-crash
literature, applied in its **de-lever-ONLY, parameter-free** form:

> scale the WHOLE book down (never up, hard cap at 1.0) when the strategy's OWN trailing realized
> return volatility runs hot relative to its own long-run level.

### Why this is mechanistically right for this book (not generic)
1. **The crash literature variable matches our failure mode.** Barroso & Santa-Clara (2015) show
   momentum/factor crash risk is predictable *by the strategy's own realized variance*, with a
   negative variance→return relationship — variance ramps *during* the crash episode. Our worst DDs
   are 140–177-day grinds, i.e. clusters of correlated negative cycles → the book's realized return
   vol is persistently elevated through them. Acadian ("Serial Killer") makes the same point: drawdowns
   are magnified by *positive serial correlation in strategy returns* and correlation spikes at the
   worst time. A trailing-vol-of-PnL de-lever directly attenuates exposure during exactly those
   serially-correlated hot stretches.
2. **It de-levers on the book's realized P&L vol, not on per-name vol** — so it does NOT strip the
   volatile-alt alpha (the reason inverse-vol failed). When the book is *working* in a high-vol regime
   (X73: high-vol is fine for this strategy), the *book's realized return vol* can still be modest
   (winning legs offset) → no de-lever. It only fires when the book's own P&L is churning/bleeding.
3. **De-lever-ONLY (cap 1.0) removes the X97 failure mode by construction** — it can never add risk
   before a blowup. Worst case it does nothing (stays at 1.0); it cannot make DD worse the way the
   lever-up version did.
4. **Parameter-free target ⇒ structural ⇒ survives nested-OOS.** Set the target vol = the strategy's
   own *trailing-expanding median* per-cycle |PnL| dispersion (PIT, expanding, lagged). There is no
   fitted threshold or window-of-convenience: the de-lever fraction is just `min(1, target/realized)`.
   This sidesteps the X99/X100 death (which came from *choosing* W and action by full-sample sweep).

## Concrete mechanism
- **State variable** `rv_t` = trailing realized std of the *book's per-cycle net PnL* over a short
  trailing window (use the engine's natural HOLD-aligned lag so it is strictly PIT; the window is a
  short fixed structural choice, e.g. ~1 sleeve-cycle of cycles, not swept for Sharpe).
- **Reference** `tgt_t` = expanding (PIT) median of `rv` up to t−lag (parameter-free; no in-sample pick).
- **Scale** `s_t = clip(tgt_t / rv_t, lo=floor, hi=1.0)` — **hard cap 1.0 (never lever up)**; an
  optional floor (e.g. 0.3) prevents fully exiting on a single noisy spike.
- **Apply** `s_t` as a multiplier on the held-book `net` weight vector for cycle t (one line in
  `heldbook`, exactly where X97's `scale` multiplied `net`, but capped at 1.0 and keyed to the
  *expanding-median* reference instead of a fixed `vt` and instead of `min(2.0, …)`).
- Turnover/cost falls out naturally because the per-cycle `turn` is computed on the scaled `net`.

This is the de-lever-only, parameter-free instantiation of Barroso–Santa-Clara constant-vol targeting,
specialized to this held-book.

## Pre-registered success criteria (against evaluation_contract.md)
Primary objective = **Calmar**. ADOPT requires ALL applicable gates:
- **G1 look-ahead**: PASS — `rv_t` and `tgt_t` use only realized PnL strictly before the cycle being
  sized (HOLD-aligned lag, expanding median PIT). No future PnL in the scale.
- **G2 in-sample**: Calmar > +1.68. Target: maxDD reduced by **≥20%** (|maxDD| ≤ ~−4,540 bps) with
  Sharpe loss **≤0.2** (Sharpe ≥ +1.73). Report Sharpe / maxDD / Calmar / totPnL.
- **G3 nested-OOS**: the de-lever has **no tuned parameter** (cap=1.0 fixed; target=expanding median;
  window is a fixed structural choice). State this; G3 waived IF no parameter is swept for Sharpe. If
  the floor or window IS varied, it must be chosen on past blocks and applied forward, nested Calmar
  ≥ +1.68. (Pre-register: prefer the no-sweep variant to keep G3 clean.)
- **G4 matched placebo**: vs a control that applies the *same distribution of scale magnitudes*
  shuffled across cycles (≥100 seeds) — i.e. random de-lever timing with identical average exposure
  reduction. Require real ≥ **p95**. (Tests that the DD cut comes from *when* it de-levers, not merely
  from running smaller on average.)
- **G5 per-fold**: report folds_positive and DD-improvement per fold; require DD improvement in **≥6/9**
  folds OR LOFO showing the DD cut isn't carried by one episode.
- **G6 paired CI**: block-bootstrap paired per-cycle PnL diff vs baseline; **must not cross zero** for
  ADOPT. (Expectation: Sharpe diff CI may straddle zero — that is fine for a DD trade IF Calmar gates
  pass; flag if so. The binding CI for a DD play is on the *Calmar/PnL* side, not raw Sharpe.)
- **G7 universe**: must hold on **HL70** (production) and ≥1 other (44-sym X97 base). The mechanism
  should generalize because it keys on each universe's own PnL vol.
- **G8 cost**: report @1bp (HL maker), @3bp (taker), @4.5bp. De-lever should *help* cost (less gross,
  less turnover); confirm the Calmar gain is not cost-fragile.

## Expected failure modes / things to watch
- **Lagging the spike.** Trailing realized vol rises *after* the first bad cycles, so the de-lever
  trims the *tail/continuation* of a drawdown, not its onset. That is the realistic, honest claim:
  shorten/shallow the deep multi-month episode, not avoid it. If the DD episode is a single sharp cycle
  (not a grind), this won't help — but X98 says ours are grinds, which is the favorable case.
- **It will also clip some upside** (de-lever during high-vol-but-winning stretches) → some Sharpe/PnL
  loss is expected; the bet is Calmar nets positive. Watch %pos and totPnL.
- **Placebo (G4) is the real test.** If a magnitude-matched random de-lever cuts DD just as much, the
  *timing* adds nothing and we REJECT — that would mean "just run smaller" is the whole effect (in which
  case the honest recommendation is a flat gross-down, not a vol-targeted one).
- **Don't reintroduce the X97 bug**: cap MUST be 1.0; no `min(2.0,…)`. Verify in review.
- **Don't tune for 2026**: the X99/X100 trap. Keep the reference parameter-free (expanding median),
  not a hand-picked target level.

## Literature
- Barroso & Santa-Clara, *Momentum has its moments* (2015) — crash risk forecastable by the strategy's
  own realized variance; constant-vol scaling ≈ eliminates crashes, ~doubles Sharpe.
  https://alphaarchitect.com/avoiding-momentum-crashes/
- Daniel & Moskowitz, *Momentum Crashes* — dynamic exposure scaling on forecast mean/variance.
  https://alphaarchitect.com/cross-sectional-momentum/
- Moreira & Muir, *Volatility-Managed Portfolios* (NBER w22208) — scaling factor exposure inverse to
  own variance raises Sharpe/alpha. https://www.nber.org/system/files/working_papers/w22208/w22208.pdf
- Acadian, *Serial Killer: Drawdowns and Serial Correlation* — drawdowns magnified by positive serial
  correlation in strategy returns; correlation spikes at the worst time (motivates de-levering on the
  book's own clustered-loss vol). https://www.acadian-asset.com/investment-insights/owenomics/serial-killer-drawdowns-and-serial-correlation
- Boyd et al., *Multi-period portfolio selection with drawdown control* — adjust risk aversion by
  realized drawdown. https://web.stanford.edu/~boyd/papers/pdf/multiperiod_portfolio_drawdown.pdf

## Plug-in point
Multiply the per-cycle `net` weight vector in the `heldbook(...)` loop of
`research/convexity_portable_2026-05-20/scripts/X117_hl70_pnl_cost.py` (HL70) and the analogous
`hb`/`heldbook` in X116 / X97 — the same line X97 used for its `scale`, but capped at 1.0 and keyed to
the expanding-median reference. Preds cache:
`research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet`.
