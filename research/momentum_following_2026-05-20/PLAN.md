# Momentum-Following (Cost-Amortized) — Pre-Registered Plan v1 (2026-05-20)

## What is genuinely distinct (the only honest claim — pre-stated and falsifiable)
This is **NOT** a feature claim (`return_1d`/`ema_slope_20_1h`/r24-equiv are
WINNER_21 features → closed feature ceiling; re-deriving them is forbidden).
It is a **cost-regime claim**: the lifecycle probe measured a real, OOS-stable
directional tilt (`r24` directional accuracy **0.515** OOS-symbol vs placebo
0.501, per-group 0.51–0.53 stable) within the high-volatility cohort. Magnitude
arithmetic: 1.5pp edge × ±1.4% moves ≈ **~4 bps/cycle gross** vs **~9 bps
round-trip cost** ⇒ structurally sub-cost-floor *at 4h cadence*. The
distinct, untested claim: **if momentum-sign is persistent (auto-correlated
across multiple 4h bars), a longer-hold standalone construction collects the
same edge across many bars on ONE round-trip cost — different cost regime,
same signal**. The lifecycle event-path (winners +1.5% post-entry, plateaus
to +24h, no dump within 24h) is empirical support that the post-entry path
sustains the move long enough for amortization. Whether persistence is
actually high enough to clear cost is the open, falsifiable question.

## Hard priors carried (every lesson from this session)
1. **R2b longer-holds were monotone-worse on the LGBM-pred-ranked stack**
   (48h +1.21, 72h +0.90 vs 24h +2.23). That is a HARD prior against
   longer holds. The genuinely-distinct claim here is that momentum SIGN is
   more persistent than LGBM pred sign — this must be **measured and pass a
   pre-registered persistence gate BEFORE the strategy test runs**. If
   persistence fails its own gate, STOP — the cost-amortization story is
   already refuted by the data.
2. Distinctness gate: momentum signal must NOT be a relabel of closed
   features. OOS-symbol rank-corr ≤ **0.30** with each of `pred` (the closed
   LGBM signal) AND each WINNER_21 momentum feature (`return_1d`,
   `ema_slope_20_1h`). Tested PRE-strategy. If high, STOP — re-derivation.
3. Realistic cost model: 4.5 bps/leg flat + realized √ADV + tail-stress 3×
   √ADV on top-vol-decile. Funding charged at PIT rate (entry-time observed)
   × #settlements held, sign-correct (long pays when funding>0; short
   receives) — funding is **small** here (≈ ±1 bps/4h) so it does not drive
   the result, but must be priced correctly (prior plans got this wrong).
4. R3c portable protocol, **corrected within-group aggregation**
   (oi_flow_test_v2: strict per-group pairing, NO cartesian `time` join,
   honest n_eff = cycles/BLK), per-group level-CI, LOFO single-group
   sign-flip, three-way verdict (real / detectable-null / underpowered-
   indeterminate; "indeterminate" is TERMINAL for this hypothesis, never an
   unfalsifiable holding state).
5. **Tail-first.** Momentum has known crash risk (Daniel/Moskowitz). 0.74y
   sample contains few/no regime crashes; even a passing Sharpe here is
   "tail not realized in sample," not "tail safe." S3 explicitly states a
   PASS is *diagnostic-only, not deployable*; deployment requires a real
   forward-test + kill-switch sizing.
6. No goalpost-moving; a miss rewrites the diagnosis, not the gate.

## Locked parameters
- Panel: `outputs/vBTC_features/panel_variants_with_funding.parquet`.
- Signal: cross-sectional rank of `r24` (24h trailing return, PIT `.shift(1)`)
  — the data-validated 0.515-OOS-symbol momentum feature. Robustness
  variant: `runup_z` (0.512 OOS). Constructed from `close_wide` cache, exact
  same recipe as the lifecycle probe (already audited).
- Universe filter (PIT, optional variant): high-vol cohort = top-decile
  `atr_pct` per cycle (where the data shows the tilt is concentrated) vs
  no-filter full universe.
- Construction: cross-sectional L/S, **long top-K** by `r24`, **short
  bottom-K**, K∈{5, 8}; BTC β-neutral via trailing-288 PIT β; hard per-name
  cap = 1/3 book gross; **REBALANCE only when sign rank crosses K boundary**
  (low-turnover by design, not by name — the genuinely-different cost mode).
- Hold grid: {**4h** (ref; expected sub-cost-floor per the data); **24h**,
  **48h**, **72h**} via the same N-sleeve overlap machinery; only PASS at
  ≥24h is the genuinely-distinct claim.
- Costs (all reported): flat 4.5/9 bps; realized √ADV; tail-stress 3× √ADV
  on top-vol-decile; PIT funding accrual.
- Stats: corrected within-group aggregation; block-bootstrap (block=11),
  one-sided LCB, n_boot=2000; honest n_eff = cycles/BLK; MDE pre-computed
  BLOCKING before any fit.

## Tests (pre-registered; absolute, falsifiable)

### M0 — Distinctness & Persistence (BLOCKING gates)
- **Distinctness (BLOCKING):** OOS-symbol rank-corr of `r24` signal vs (a) the
  closed LGBM `pred` (from `outputs/vBTC_audit_panel/all_predictions.parquet`)
  and (b) each WINNER_21 momentum feature. Pre-registered: if |corr|>0.30 to
  any → STOP (re-derivation; this iteration = honest negative).
- **Persistence (BLOCKING — the cost-amortization story IS this number):**
  measure auto-correlation of `r24` rank ordering at lags {4h, 24h, 48h, 72h}
  using cross-sectional Kendall τ between rank vectors at t and t+lag, OOS.
  Pre-registered: persistence τ at 24h ≥ **0.40** AND at 48h ≥ **0.25** is
  required to make a 24/48h-hold amortization story coherent. If not →
  STOP (the data has already refuted the mechanism's prerequisite).

### M1 — The decisive test
Run the L/S-K momentum strategy at the {4h, 24h, 48h, 72h} hold grid, with
and without the high-vol-cohort filter, under flat-4.5, √ADV, and tail-stress
costs. R3c portable, corrected aggregation, per-group level-CI, LOFO, MDE
pre-reported.
- **PASS (real, deployable-pending-forward-test):** for some hold ≥ 24h,
  portable net-of-realistic-cost Sharpe ≥ **+0.5**, one-sided LCB > 0,
  ≥4/5 groups positive, no LOFO sign-flip, AND turnover at that hold is
  materially below 4h turnover (cost-amortization mechanism shown).
- **detectable-null:** HCB < +0.2 at ALL holds.
- **indeterminate (TERMINAL for this hypothesis):** neither.
- **Pre-registered prediction** (the genuine open question): at 4h the
  strategy is sub-cost-floor (≤+0.2) by the lifecycle-probe arithmetic; at
  24h/48h IF persistence τ is high enough, gross-per-hold × hits-per-hold
  could clear cost. P(PASS) honestly LOW given crypto momentum compression
  in recent samples (Liu/Tsyvinski-class results have decayed) — but the
  cost-regime argument is non-redundant and the data points here.

### M2 — Tail & deployability discount
For any (hold, cohort) that clears M1: drawdown vs R1-uncapped, p1
worst-cycle, CVaR(5%); explicit momentum-crash stress (worst 30-day BTC
drawdown sub-sample). PASS at M1 is **diagnostic-only**, not deployable —
pre-registered.

### M3 — Synthesis (post M0–M2, no pre-write)
If PASS: sized result + the tail/forward caveats. If FAIL: this is the last
genuinely-distinct in-scope hypothesis the session pointed at; the loop's
honest terminal state is then sealed by measurement (the cost-regime escape
hatch fails too), and the user-decision (paid data / HFT scope / terminate)
becomes the only remaining path.

## Process
Plan → **3-agent plan review** (methodology/profitability/red-team) → align
or kill. Run gated on M0 passing both BLOCKING gates. After M1–M2 → **3-agent
results review** vs these pre-registered gates; fudge/leak/goalpost ⇒
re-initiate.

## Out of scope
The closed LGBM-pred-ranked stack and all its components; the −0.33 short
leg; the vol detector as direction signal; long-skew; convexity-mining;
ALL prior closed arcs. Paid data; live deployment; HFT/order-book.
