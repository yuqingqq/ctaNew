# Funding-Carry — Pre-Registered Plan v1 (2026-05-19)

## Why this is genuinely distinct (not re-derivation)
Every closed arc (portable-alpha, bottleneck, convexity, sell-convexity,
OI/flow) is **directional return-forecasting** at the closed IC≈0.02 ceiling,
or the closed vol detector, or the closed −0.33 short leg. **Funding-carry is
a different return source: you get *paid to hold* (perp funding accrual), not
to forecast next-bar price.** It sidesteps the directional ceiling entirely.
It is also **low-turnover by construction** (funding sign is persistent →
rebalance rarely), so the 4.5 bps churn cost that killed the closed
high-turnover strategies is a far weaker constraint — a genuinely different
cost regime. It is the one canonical, literature-named crypto cross-sectional
premium ("carry") that this codebase only ever used as 1-of-21 LGBM features
and as a cost term, never isolated as a standalone portable strategy.

## Correct funding economics (the prior arc got this wrong — fixed here)
Binance USDM: funding>0 ⇒ **longs pay shorts**; funding<0 ⇒ shorts pay longs.
Carry strategy: **LONG** the most-**negative**-funding names (paid to be
long) and **SHORT** the most-**positive**-funding names (collect from longs),
equal-notional, market-hedged by the L/S construction (+ optional BTC β
overlay). Per-position carry P&L over a hold = **−position_sign ·
funding_rate · (#funding settlements held)** (a long loses funding when
funding>0; a short gains it). This carry accrual is NOT in the locked engine
(R1.aggregate_capped is turnover-cost only) → an explicit **carry-aware
aggregation** is added and code-reviewed before any test.

## Hard priors / lessons carried (must not be re-discovered)
1. **PIT vs realized split:** the *ranking signal* uses funding observable
   at/before entry only (panel `funding_rate` is backward-merged PIT). The
   *return* legitimately includes realized carry accrued over the hold (that
   is the strategy's actual P&L, like price P&L — not look-ahead). No
   realized-funding in the *signal*.
2. **Distinctness is a BLOCKING pre-gate** (red-team's recurring kill): the
   funding-carry rank must be LOW-correlated to (a) LGBM `pred` and (b) the
   vol/`atr_pct` axis. If |OOS rank-corr| > 0.30 to either → STOP
   (collinear = re-derivation of a closed signal). Pre-registered.
3. **Portability is the gate** (R3c: group-disjoint, no sym_id, beta-neutral,
   costed) with the CORRECTED within-group aggregation (oi_flow_test_v2 fix:
   strict per-group pairing, NO cartesian `time` join, honest n_eff=cyc/BLK,
   per-group level-CI, LOFO single-group sign-flip).
4. **Tail-first** (carry has negative skew — small premium, occasional large
   loss on crowded-carry unwind / short-side squeeze). maxDD / p1-worst-cycle
   / CVaR(5%) are pre-registered KILL criteria even if Sharpe>0. Hard
   per-name cap = 1/3 book gross.
5. Power-limited (~5 groups / ~0.74y). MDE-in-Sharpe pre-computed (BLOCKING).
   Three-way verdict (real / detectable-null / underpowered-indeterminate);
   "indeterminate" is TERMINAL for this hypothesis (spawns next loop
   iteration), never an unfalsifiable holding state, never "exhausted".
6. No goalpost-moving; a prediction miss rewrites the diagnosis, not the gate.

## Locked parameters
- Panel `outputs/vBTC_features/panel_variants_with_funding.parquet` (R0-clean).
  Signal = cross-sectional rank of PIT `funding_rate` (primary) and
  `funding_rate_z_7d` (robustness variant). Eval = R3c portable protocol,
  corrected aggregation, seed 20260519, 5 disjoint groups, BLOCK=11,
  block-bootstrap n=2000, one-sided LCB.
- Construction: each rebalance, long bottom-K funding, short top-K funding,
  K∈{5,8}; hold ∈ {24h(ref), 48h, 96h} (carry wants LONGER holds — turnover
  ↓, funding accrual ↑; the OPPOSITE regime of the closed churn strategies);
  hard per-name cap 1/3 gross; BTC β-neutral variant via trailing-288 PIT β.
- Costs (all reported): turnover @4.5 bps/leg; realized √ADV; **carry
  accrual** (the return engine, per economics above, PIT-signal /
  realized-carry); tail-stress 3× √ADV on top-vol-decile.

## Tests (pre-registered; absolute, falsifiable)
- **S0 — distinctness (BLOCKING).** OOS rank-corr of the funding-carry signal
  vs `pred` and vs `atr_pct`. Pre-registered: if |corr|>0.30 to either →
  STOP, log "funding-carry collinear with closed signal — re-derivation",
  iteration is a completed honest negative, spawn next hypothesis. Else proceed.
- **S1 — funding-carry portable, net of cost+carry (decisive).** L/S funding
  rank, β-neutral, R3c portable, corrected aggregation, all costs incl. the
  carry engine. Report gross, net-turnover-cost, net-incl-carry, per-group,
  LOFO, level-CI, honest n_eff/MDE, across hold ∈ {24,48,96h}.
  - **PASS:** portable net (incl. realized carry, turnover, √ADV) Sharpe ≥
    **+0.5**, one-sided LCB>0, ≥4/5 groups positive, no LOFO sign-flip, at
    some hold.
  - **detectable-null:** HCB < +0.2. **indeterminate:** neither (terminal).
  - Pre-registered prediction (honest, falsifiable both ways): carry is a
    known/compressed premium and panel funding is small (≈ −0.2 bps/8h mean);
    expect gross small-positive at long holds, net ≤ +0.3 after √ADV — but a
    *low-turnover* construction may clear where churn strategies failed (the
    genuine open question). A gross < 0 would refute the carry-premium thesis
    itself for this universe (the pre-registered surprise condition).
- **S2 — tail & friction stress (kill even if S1 PASS).** maxDD vs
  R1-uncapped (absolute short-vol bound, not just a ratio), p1 worst-cycle as
  multiple of mean gross, CVaR(5%); single cycle/group >60% of net.
  **Kill** if any breached regardless of Sharpe.
- **S3 — synthesis (post S0–S2, no pre-write).** Sized; exact gate
  passed/missed + margin. If FAIL/indeterminate → re-initiate loop with the
  next genuinely-distinct hypothesis OR, if the genuinely-distinct in-scope
  space is honestly exhausted, state that as the terminal conclusion with the
  precise scope-change conditions to continue (do not spin re-derivations).

## Process
Plan → **3-agent plan review** (methodology/profitability/red-team) → revise
to alignment or, if killed for an unfixable core flaw, record honest negative
and re-initiate. Heavy test sequenced when CPU free. After S0–S2 → **3-agent
results review** vs these gates; fudge/leak/goalpost ⇒ re-initiate. Loop
continues until an honest PASS or honest exhaustion of the genuinely-distinct
in-scope space.

## Out of scope
Directional return-forecasting (closed); the vol detector (closed); the −0.33
pred-ranked short leg (closed); long-skew (closed); paid data; deployment;
re-deriving any closed arc.
