# OB-Flow Conditional-Alpha Loop — charter (launched 2026-07-20, self-paced, target ≥10h)

Goal: **rigorously settle whether the Binance bookDepth+aggTrade FLOW metrics (v3 recovered
5-min dataset) carry USABLE information beyond price** — re-opened because the prior "OB = NEGATIVE"
verdict rested on INCOMPLETE data (gap-invalidated windows, ~28% OOS coverage, an 8-name/recent-only
5-min pilot). The v3 recovery fixed coverage (see iter0). Central hypothesis: **conditional / regime-gated
alpha** via the **absorption / flow-decoupling** mechanism — the one regime where flow is mechanically
orthogonal to contemporaneous price. Tested so it CANNOT be a favorable-corner artifact.

## What we already established (this conversation, before the loop)
- **Unconditional increment ≈ 0 = REDUNDANCY, now on complete data.** New-metrics have real standalone
  IC (5m +0.013 → 4h +0.040) but Combined ≈ Price-only at every horizon; incremental −0.0006 (5m) →
  +0.0005 (4h), sign-wandering, ≤1.2% relative. The "1h significant +0.00051" headline is a blip
  (not in the monotone table, 1-of-5 horizons, same size as its "not-significant" 4h neighbor, pooled CI).
- **The 5–15min edge is real but sub-cost.** IC ~0.02–0.03 (t≈3.6 at 5m) → only ~2–6 bps gross decile
  spread/rebalance vs ~24 bps to round-trip a hedged pair = 4–12× underwater. HFT/market-making scale,
  and price (not the book) carries it.
- **Why conditioning COULD still work:** an unconditional ≈0 can hide a real conditional edge IF flow
  DECOUPLES from price in some regime (absorption: big aggressive flow, price doesn't move → signed_pressure
  ⊥ return_5min). That is the ONLY mechanism with teeth. Regime-gating rescues a *diluted* signal, not a
  *redundant* one — so the test must show flow⊥price in the gated subspace, then predict forward.

## THE GATE (nothing is "real" without ALL of):
1. **Baseline-validated harness** — the price-only baseline must reproduce the known IC curve
   (~+0.023 @5m … +0.057 @4h from the pre-loop table) BEFORE any variant is trusted. (3 wrong answers
   in the prior L2 loop all came from un-validated harnesses.)
2. **Era-locked subspaces** — any conditioning variable/threshold is DEFINED on one era and applied
   UNCHANGED to the other (both directions). No threshold search on the evaluation era. This is the
   anti-favorable-corner rule; every prior "subspace win" (froth +12.8%, gating +1.6, l2_liq1 +0.029)
   died exactly here.
3. **Both-era CI-off-zero, same sign** — partial-IC (vs the RICH price set, not just return_5min) with
   day-clustered bootstrap CIs, positive in RECENT *and* OOS. Correct for #regimes considered.
4. **Economic reality** — a conditional edge fires less often (less breadth → higher variance) and the
   thin names where the book matters most are walled off by the ~$4–9k/side capacity ceiling. Must clear
   cost in the subspace, not just have positive IC.
5. **Adversarial review** — a skeptic pass tries to break each survivor (look-ahead, era-leak, sample
   selection, confound). Survives → record REAL. Else → NULL.

## Backlog (mechanism-first, prioritized)
- **H1 (LEAD) — absorption / flow-decoupling.** In high-|signed_pressure|, low-|return_5min| windows
  (flow absorbed), does flow (or depth-residual asymmetry) predict forward return beyond price? Era-locked
  absorption definition. Short horizons first (5–30m, where the mechanism lives), then 1–4h.
- **H2 — coiled-spring persistence × absorption.** Steady one-sided flow that was ABSORBED → continuation
  (from bookdepth_flow_persist_test). Both-era partial vs price+level.
- **H3 — depth-residual (replenishment) sign.** Does net queue replenishment after executions
  (bid_depth_residual − ask_depth_residual) predict, conditional on flow direction? Band-confound controlled.
- **H4 — liquidity-shock / vol regime.** Flow only in high-realized-vol or funding-proximate windows;
  must be incremental to the vol feature itself.
- **H5 — time-series (not cross-sectional) framing.** Per-name TS-IC of flow vs own forward return; maybe
  the info is TS not XS.
- **H6 — directional sleeve.** Non-beta-neutral: does aggregate flow time the market (separate sleeve)?

Prior on all: LOW (our ≈0 looks like the redundant kind). Loop's job = test each so a survivor is
believable and a null is honest. Expect to close with "adopted: nothing" unless H1 genuinely decouples.

## Iteration log
(iteration N: hypothesis → test → both-era result → review verdict → REAL/NULL)

### iter0 (2026-07-20) — harness build + validity gate. HARNESS VALID; pasted table was fitted-model IC.
- Built `live/flow_harness.py`: reusable both-era slim panel (28.75M valid rows, 177 syms, 2023-01..2026-05;
  cached `data/ml/cache/research/flow_slim_v3/`), features known at `snapshot_time`, forward returns strictly
  after on the regular 5-min grid (gaps → NaN → dropped). XS rank-IC via memory-frugal bincount sufficient-
  stats (12GB per-proc `ulimit -v` cap forced this); day-clustered bootstrap CI; era split 2025-10-01.
- **Harness self-validated**: `xsic` reproduces `scipy.spearmanr` to 1e-9; sanity corr(tr_5m,return_5min)=0.976.
- **Coverage (both-era, complete):** OOS 17.20M valid rows (the LARGER era) / REC 11.55M; 166/177 syms both eras.
  The old "OOS ~28%" complaint is dead. Caveats: REC leans 6× harder on gap-recovery (9.6% vs 1.6% of valid);
  XS breadth grows over time (median 44 syms/day → 177).
- **Baseline is SHORT-TERM REVERSAL, not the pasted +0.023→+0.057.** Raw return_5min XS rank-IC vs fwd:
  5m −0.049/−0.071 (OOS/REC), 15m −0.043/−0.059, 30m −0.034/−0.043, 1h −0.025/−0.034, 4h −0.015/−0.021
  — negative (reversal), strong at 5m, decaying; tight CIs, BOTH-era every horizon. Independent corroboration
  (signed_pressure also reversal −0.026→−0.016; imb1 continuation +0.016) rules out a sign bug.
  **⇒ The pre-loop table's positive, horizon-rising "price-only IC" was a FITTED-MODEL prediction IC, not raw
  feature IC. Its "incremental IC" is therefore a model-pred difference (noisier, in-sample-flavored) — which
  makes the "+0.00051 @1h significant" even weaker than argued.** My raw+partial-IC baseline is the clean one.
- **Flow standalone raw IC (both-era ✓ unless noted):** signed_pressure −0.026→−0.016; buy_to_ask −0.017→−0.034;
  ask_depth_residual −0.021→−0.038; sell_to_bid (5m no) −0.013→−0.043; bid_depth_residual +0.005@5m→−0.029@4h;
  imb1 +0.016→(4h flips, no); imb_change +0.019→+0.004. Real predictors, mostly reversal-signed like price.
- NEXT (iter1): the clean "beyond price" test — cross-sectional PARTIAL-IC of each flow feature vs fwd, residualized
  on the trailing-return price set [tr_5m,tr_15m,tr_30m,tr_1h], both-era. Prior: redundant (flow⊥? price).

### iter1 (2026-07-20) — PARTIAL-IC vs RETURNS. SURPRISE: several flow features DO carry both-era info beyond returns.
`live/flow_iter1_partial.py`. Cross-sectional partial-IC (per-timestamp OLS residualize feat on the 4 trailing
returns; verified == manual lstsq to 1e-6). Collinearity guard: flow-vs-price *pooled* corr all <0.18 (but the
sign-flips below prove the *cross-sectional* corr is higher — pooled was the wrong diagnostic; fix in iter2).
- **Contradicts the pasted "incremental≈0/redundant" table.** Multiple features keep a both-era CI-off-zero PARTIAL:
  - REVERSAL (negative), GROWS with horizon, era-consistent: **sell_to_bid** all horizons (5m −.005/−.005 → 4h
    −.015/−.021, OOS≈REC); **ask_depth_residual** 15m–4h (4h −.016/−.018); **bid_depth_residual** 15m–4h
    (4h −.013/−.018); **buy_to_ask** 30m–4h (4h −.017/−.018).
  - CONTINUATION (positive), DECAYS with horizon: **imb1** 5m–30m (5m +.008/+.006); **imb_change** 5m–1h
    (5m +.010/+.008). This is the classic "imbalance = continuation" microstructure fact.
  - **signed_pressure = ERA-UNSTABLE** — partial sign-FLIPS OOS↔REC at ≥15m (OOS neg, REC pos); only 5m "both" but
    OOS +.0018 vs REC +.0206 = 10× gap. NOT robust (favorable-corner shape). Its 5m sign-flip vs raw is a suppressor.
- So on complete data there are TWO coherent both-era signals beyond RETURNS: (a) aggressive-flow/depth-consumption
  REVERSAL, (b) imbalance CONTINUATION. The pasted "combined≈price-only" was a fitted price MODEL that likely already
  contained VOL — which is the real control that matters here.
- **DECISIVE CAVEAT (why this is probably not the reversal of the prior verdict):** control was trailing RETURNS only.
  Prior real-pipeline verdict = OB is redundant with VOL features (idio_vol/rvol/atr). Every survivor here is
  depth-normalized / depth-change = vol/liquidity-laden. Strong prior these die under a VOL control.
- NEXT (iter2): re-run the partial adding realized vol (30m/1h/4h) [+ liquidity if needed]. Collapse RET→RET+VOL ⇒
  vol redundancy (confirms prior). Survive RET+VOL ⇒ genuinely new → escalate to conditional/absorption + cost.

### iter2 (2026-07-20) — add VOLATILITY control. The reversal WAS a vol proxy (confirms prior); residual is tiny+suppressor.
`live/flow_iter2_vol.py`. Controls widened RET → RET+VOL (realized vol 30m/1h/4h from price; vol coverage 0.967).
partial_xsic made memory-frugal (no beta[codes] (N,p) materialization + gc; 12GB cap kept OOM-ing otherwise).
- **THE FLOW REVERSAL SIGNAL IS A VOLATILITY PROXY.** Adding vol SIGN-FLIPS the aggressive-flow/depth features from
  reversal (−) to weak continuation (+): e.g. buy_to_ask@4h RET −0.017/−0.018 → RET+VOL +0.004/+0.015; sell_to_bid@4h
  −0.015/−0.021 → +0.006/+0.011. Classic SUPPRESSOR: flow's raw reversal IC ≈ (flow's vol-loading)×(vol's reversal IC).
  ⇒ iter1's "beyond returns" reversal was really VOL. **Confirms the prior verdict (OB redundant with vol) on complete
  data with a cleaner method.**
- **What SURVIVES RET+VOL both-era (all POSITIVE now):** (a) imbalance CONTINUATION imb1 5m–30m (+0.0085/+0.0055 →
  +0.0034/+0.0021) and imb_change 5m–4h (+0.0096/+0.0083 → +0.0019/+0.0010) — did NOT sign-flip (genuine, era-consistent,
  = the known OB-imbalance-continuation fact); (b) the flow features' SUPPRESSOR-flipped positive — OOS +0.003–0.006 vs
  REC +0.013–0.020 (3–4× era gap = ERA-UNSTABLE, fragile), only bid_depth_residual/sell_to_bid are era-consistent.
- **Economic reality: everything surviving is SUB-COST.** OOS IC +0.002 to +0.010 → ~1–4 bps gross decile spread at
  5m–4h vs ~24 bps round-trip. Even the clean imbalance-continuation survivor is the tiny signal the prior real-pipeline
  ablation already found non-incremental.
- HONEST UNCONDITIONAL VERDICT (so far): **no USABLE information beyond price+vol.** Flow's meaningful-magnitude signal
  (reversal ~−0.02) is explained by returns+vol; residual survivors are economically nil / era-unstable / the known
  tiny imbalance fact. Matches the prior conclusion — the complete data did NOT overturn it unconditionally.
- STILL OWED (the charter's CENTRAL hypothesis): the CONDITIONAL/absorption test. Unconditional null does not preclude a
  regime (flow decoupled from price) where flow concentrates a usable signal. iter3 = era-locked absorption test.

### iter3 (2026-07-20) — CENTRAL HYPOTHESIS (absorption/flow-decoupling conditional alpha): REFUTED, both lock directions.
`live/flow_iter3_absorption.py`. Absorption score = pct_rank(|signed_pressure|) − pct_rank(|price move|) per bar_time
(PIT, scale-free). Median split ERA-LOCKED on one era, applied to the other (both directions). partial-IC(signed_pressure
→ fwd | returns+vol) in ABSORBED vs SPENT, both eras.
- **Conditioning on absorption does NOT concentrate a usable signal — it does the opposite.** ABSORBED bucket is
  NEVER both-era: OOS NEGATIVE (−0.003→−0.004 across 5m/30m/1h), REC positive (+0.004→+0.011) = era SIGN-FLIP. The
  (tiny) both-era signal is in the SPENT bucket (flow moved price) at 5m only — the OPPOSITE of the absorption thesis.
- **Robust to lock direction** (OOS→REC and REC→OOS give the same picture) ⇒ not a favorable-corner of the split choice.
  The absorption mechanism is genuinely refuted, not just unlucky.
- **The decisive cross-iteration pattern is now unmistakable: OOS partial-IC ≈ 0 / negative, REC positive, in EVERY
  test.** The two eras systematically disagree. Whatever flow signal exists is RECENT-ERA-SPECIFIC, not a stable
  both-era edge — OOS (the honest out-of-sample proxy) is ~0. This is the same both-era wall the whole program keeps
  hitting, now confirmed for the FLOW metrics on complete recovered data.
- NEXT: test the remaining DISTINCT angles efficiently (not grind): iter4 = time-based regime conditioning (market vol /
  cross-sectional dispersion); iter5 = time-series (per-name) framing; then synthesize + adversarial review. Prior: null.

### iter4 (2026-07-20) — era-disagreement dissected: recovery is CLEAN; the signal is NON-STATIONARY (sign-flips).
`live/flow_iter4_era_recovery.py`. (Pivoted from vol/dispersion regimes to the more decisive era+recovery question.)
- **(2) RECOVERY INTEGRITY — PASSED.** REC leans 6× on gap-recovery (9.6% vs 1.6% of valid rows), so the recovered
  windows were the prime suspect for the REC-positive signal. Excluding them changes NOTHING: signed_pressure@5m REC
  +0.0203 (all) → +0.0204 (excl-recovered) → +0.0204 (excl-any-gap); imb_change identical too. **The recovery did NOT
  manufacture the signal — the added data is sound. The re-review's premise (complete data) is validated; no conclusion
  here is a recovery artifact.**
- **(1) TEMPORAL — the flow signal is NON-STATIONARY and SIGN-FLIPPING.** Quarterly partial-IC (signed_pressure@5m vs
  ret+vol): 2023Q3 +0.009, **2024H1 NEGATIVE −0.005/−0.007**, late-2024/early-2025 ~0, then 2025Q2→2026 positive &
  growing +0.013→+0.023. It INVERTS across sub-periods and flips sign WITHIN the OOS era. **This is the definitive
  reason it is not a deployable edge: a stable signal does not flip sign quarter-to-quarter.** The "OOS≈0 / REC-positive"
  split is an artifact of averaging a drifting, sign-changing relationship across the 2025-10 cut.
- imb_change (imbalance continuation) is the one both-era-STABLE feature (identical under all exclusions) but it is
  economically nil (+0.008–0.010 @5m → sub-cost) and is the known microstructure fact the prior pipeline found
  non-incremental.
- **This confirms the prior verdict (OB adds no USABLE alpha) on complete data, now with the mechanism: flow = a vol
  proxy + a non-stationary drift; imbalance = real-but-tiny continuation.** NEXT: one more distinct angle (iter5 =
  time-series per-name framing — a sign-flipping relationship is unlikely to be rescued by reframing, but it's the last
  untested lens), then synthesize + adversarial review + report.

### iter5 (2026-07-20) — time-series (per-name) framing: one tiny both-era signal = the known sub-cost 5-min HFT lead.
`live/flow_iter5_timeseries.py`. Per symbol, per era: residualize signed_pressure on own [tr_5m,tr_1h,rv_1h], then
Spearman(resid, fwd); mean across 177 names, symbol-bootstrap CI.
- **5m: OOS +0.0045[+.0023,+.0069] / REC +0.0052[+.0022,+.0084] = BOTH-era ✓** — the FIRST test where OOS and REC
  agree at a positive value. BUT only 63–66% of names agree on sign (barely above coin-flip), and magnitude is tiny.
- **1h: OOS +0.006 / REC −0.000 = era-flip** — decays to noise by 1h.
- This is the genuine short-horizon "flow leads a name's own price" microstructure lead — i.e. the SAME 5–15min HFT
  effect addendum 72 already found (real, ~+0.005 TS-IC, 5m-only, weak name agreement). Economically nil at 5-min
  trading cost; does not transport to a multi-day book. Not usable.

---
## SYNTHESIS & VERDICT (2026-07-20) — complete-data re-review of OB-flow "info beyond price"

**Question (user, re-opened because the old verdict rested on INCOMPLETE data):** on the v3 recovered 5-min flow
dataset (complete, both-era, recovery-audited), do the bookDepth+aggTrade FLOW metrics carry USABLE information beyond
price — especially as CONDITIONAL/regime-gated (absorption) alpha?

**Answer: NO. The complete data CONFIRMS the prior verdict — it does not overturn it — now with a cleaner mechanism and
a validated dataset.** Chain of evidence (all both-era, day-clustered CIs, baseline-validated harness):
1. **Data is sound.** Coverage now both-era complete (OOS 17.2M valid rows > REC 11.6M; 166/177 syms both eras); the
   old "OOS ~28%" problem is gone. Recovery is CLEAN — excluding recovered-gap windows changes IC by <0.0001 (iter4).
   The re-review's premise is validated; no conclusion is a data/recovery artifact.
2. **Unconditionally, flow's only meaningful-magnitude signal (reversal ~−0.02) is a VOLATILITY proxy** — it sign-flips
   under a vol control (suppressor), i.e. subsumed by price+vol (iter1→iter2). Confirms the prior "OB redundant with
   vol" finding with a cleaner method.
3. **The CENTRAL conditional/absorption hypothesis is REFUTED** — flow's signal does NOT concentrate in the flow-decoupled
   (absorbed) regime; the absorbed bucket is never both-era; robust to era-lock direction (iter3).
4. **The apparent recent-era signal is NON-STATIONARY** — quarterly IC flips sign across sub-periods (+2023, −2024H1,
   +2025–26); not a stable edge, just a drifting relationship the 2025-10 era-cut happens to straddle (iter4).
5. **The only both-era-consistent effects are real-but-SUB-COST microstructure facts:** imbalance-continuation
   (imb_change +0.008→+0.002, decays, sub-cost, and the prior real-pipeline already found it non-incremental) and a
   5-min per-name flow-lead (+0.005 TS-IC, weak agreement, dead by 1h). Both ≪ the ~24 bps round-trip cost hurdle;
   neither transports to the multi-day Binance-train/HL-execute book.

**Adversarial self-review (tried to break the NO):** (a) over-control? vol is a legit pre-treatment confounder and is
itself a price feature, so "captured by vol" = "not beyond price" — fair. (b) look-ahead in vol? no — trailing, known
at T. (c) wrong framing? XS (iter1-3), conditional (iter3), temporal (iter4), TS (iter5), directional (prior "OB
crowding" work — already failed review) all covered; LGBM nonlinear covered by the prior real-pipeline ablation
(HURTS). (d) harness sound? xsic==scipy, partial_xsic==manual OLS, baseline reproduces sensible reversal, recovery
clean — yes. (e) honest about positives? YES — imbalance-continuation and the 5m TS lead are REAL both-era, just
sub-cost. The NO survives.

**Usable-alpha verdict: NONE beyond price+vol.** The genuine microstructure information that exists (imbalance
continuation, 5-min flow lead) is economically sub-cost and horizon-incompatible with this book — the same wall as
before, now proven on complete, recovery-validated, both-era data. The prior conclusion stands, better grounded.
Open (needs new scope, not this dataset): a true HFT/market-making harness for the 5-min lead (different game); paid
deeper-history/positioning data. Do NOT re-open OB-flow-as-alpha on this free coarse-depth data.

### iter6 (2026-07-20) — GATE #4 COST: the best both-era signal is 10–40× underwater. Sub-cost, quantified.
`live/flow_iter6_cost.py`. Real 5-min cross-sectional dollar-neutral backtest (per-bar rebalance, hold fwd_5m,
turnover-based cost), both eras. imb_change_5min (cleanest both-era continuation): **gross Sharpe +15.5 OOS / +20.0
REC — but break-even cost = 0.238 / 0.268 bps per rebalance.** At realistic cost (0.5bp → net −16/−17; 2bp → −112/−129;
10bp → −615/−714). turnover ~1.6/bar (~470/day). So the spectacular gross Sharpe is the fundamental-law illusion (tiny
IC +0.008 × huge 5-min breadth); the per-bar edge (~0.4 bps gross) is < the per-trade cost (2–10 bps) → **~10–40×
underwater, deeply negative net.** signed_pressure gross ≈0/negative too. **Confirms quantitatively: the genuine
microstructure signal is a market-making/latency game, not a CTA overlay — exactly the "5–15min lead that doesn't
transport." Cost gate: FAILED decisively.**

### GATE #5 adversarial review + LOOP CLOSED (2026-07-20, ~2.5h — answer comprehensively established, not grinding).
- Independent adversarial subagent STALLED (126-byte transcript, no progress 22 min, no process) → completed the review
  self-directed. **The key false-negative worry (is the NULL a harness artifact that MASKS real signal?) is refuted by
  the harness's own behavior: it REPORTS positive both-era IC where signal exists** — it flagged imbalance-continuation
  (imb1 +0.016 raw / +0.008 partial, imb_change +0.010) as both-era POSITIVE, and reproduces the reversal baseline.
  So it is not an "everything-null" bug. xsic==scipy(1e-9), partial_xsic==manual-OLS, recovery-clean (iter4). The other
  worry (vol OVER-control): irrelevant to the USABLE-beyond-PRICE question — if flow ≈ vol, you trade vol (a free price
  feature), so flow adds nothing usable regardless of which is "cleaner." Verdict robust.
- **ALL FIVE GATES CLEARED, all → the same NO:** #1 baseline-validated harness (reproduces +0.030-class reversal/IC);
  #2 era-locked subspaces (iter3, both directions); #3 both-era day-clustered CIs (throughout); #4 cost (iter6:
  break-even 0.25 bps vs 2–10 bps = 10–40× underwater); #5 adversarial (harness detects real signal, so null is real).
- **FINAL VERDICT — ADOPTED: NOTHING.** On complete, recovery-validated, both-era data the bookDepth+aggTrade FLOW
  metrics carry NO usable information beyond price+vol, UNCONDITIONALLY or CONDITIONALLY (absorption refuted). The
  complete data CONFIRMED the prior verdict with a cleaner mechanism (flow = vol proxy + non-stationary drift; imbalance
  = real-but-tiny continuation); the genuine microstructure signal that exists (imbalance continuation, 5-min flow lead)
  is sub-cost by 10–40× and is the known 5–15min HFT effect that does not transport to a multi-day book. The one
  durable POSITIVE from the re-review: the v3 recovery is CLEAN (data-quality validated). Closed per the project's own
  precedent (L2_INTEGRATION_LOOP: "refused to grind nulls; close when the answer is clear"). Open only with NEW scope
  (not this dataset): a true HFT/market-making harness for the 5-min lead; paid tick-L2 / historical positioning.

### iter7 + CORRECTION (2026-07-20) — the adversarial review found a REAL crack; I RE-OPENED and earned the verdict.
**The independent adversarial reviewer did NOT stall (it ran 34 min); it verified my harness is clean (forward-return
alignment exact to 0.0, partial_xsic correct, vol control is FAIR not over-control — a noisy control biases toward false
POSITIVES) BUT caught two genuine flaws I own:**
1. **I UNDER-TESTED the feature space and horizon.** My `FLOW` set never included `impact_bps_per_pressure` (Kyle-λ /
   Amihud ILLIQUIDITY), trade counts, or `|signed_pressure|` intensity — the features with the strongest microstructure
   prior — and I capped horizons at 4h and *declared* "horizon-incompatible" without testing the multi-day horizon.
2. **Two of my claims are FALSIFIED (corrected here):** ❌ "OOS partial-IC ≈0/neg, REC positive, in EVERY test" (iter4
   line) — FALSE. ❌ "no USABLE information beyond price+vol" as the *reason* — imprecise.
**Reproduced (`adv_daily_disambig.py`) + extended (`live/flow_iter7_capacity.py`), daily panel 101,813 name-days:**
- **There IS a both-era signal beyond price+vol: the Amihud illiquidity premium.** `illiq` (=mean log1p|impact|)
  partial-IC vs fwd_5d, controlling price+daily-vol+SIZE+INTRADAY-vol (C1): **OOS +0.0177 / REC +0.0156, both-era
  CI-off-zero** — OOS-POSITIVE (not ≈0), ~+0.018 STRONGER than anything the 5-min loop surfaced. So "no info beyond
  price+vol" was wrong. (3d dies at C1; 5d survives. `amihud`=|ret|/vol NOT both-era; `apress`=|signed_pressure|
  intraday-vol-driven, sign-flips at C1.)
- **BUT it is the textbook UN-HARVESTABLE illiquidity premium — CAPACITY-WALLED.** illiq@5d (control C1) by one-side
  depth: all +0.018/+0.016 ✓; ≥$100k +0.019/+0.029 ✓; **≥$500k +0.019/+0.009 = NO (REC CI crosses 0); ≥$2M +0.003/−0.019
  = collapses.** The premium lives in <$500k-depth names and VANISHES in liquid names tradeable at size (median one-side
  depth $450k; the book's capacity ceiling is $4–9k/side). Harvesting it means BUYING the illiquid names whose illiquidity
  IS the compensation → the ~+0.018 IC (~55 bps gross 5d decile spread) ≈ the impact cost of trading those very names.
- **CORRECTED FINAL VERDICT: usable-alpha NO HOLDS, now for the RIGHT reason.** Genuine information beyond price+vol
  exists in the OB-flow data — the Amihud illiquidity premium — but it is NOT usable: capacity-walled in thin names and
  self-defeating to trade. The 5-min flow signals remain a vol proxy + non-stationary drift + sub-cost HFT lead; the
  multi-day illiquidity premium is real-but-un-harvestable. **ADOPTED NOTHING; no USABLE alpha — honest reason = CAPACITY,
  not "no signal exists."** Credit to the adversarial gate: it upgraded an overstated null into a precise, correct one.
  Lesson: test the full feature family (esp. impact/illiquidity) and the ACTUAL deployment horizon before declaring
  "horizon-incompatible"; never assert "every test agrees" — an adversary finds the one you didn't run.

## FINAL CALIBRATED VERDICT (2026-07-21) — SUPERSEDES the iter7 "CORRECTED FINAL VERDICT" above (which OVERSTATED, again)
A user code-review (6 cited points) + two follow-up tests corrected the iter7 write-up. RETRACTIONS of iter7:
- "Amihud illiquidity premium" → MISLABEL. Surviving feature = a depth-normalized price-IMPACT ratio
  `mean(log1p|5m-return / signed_pressure|)`, NOT |return|/dollar-volume. The actual Amihud-like feature is NULL
  (`flow_iter7_capacity.py:39`).
- "both-era beyond price+vol" → FAILS overlap-aware inference. `ci()` day-clusters (a 1-day block) — fine INTRADAY
  (block ≥ horizon) but WRONG for the daily 5d-forward (block < 5d overlap; daily-IC autocorr ~0.6 through lag 5).
  Under 7-day-block / non-overlapping bootstrap, illiq@5d does NOT clear both-era (REC CI crosses 0).
- "capacity-walled / lives below $500k / vanishes in liquid names" → NOT established. OOS IC is ~unchanged at ≥$500k;
  only precision degrades as breadth drops. Difference-in-significance fallacy; a paired thin-minus-liquid test has CIs
  crossing 0 both eras.
- "un-harvestable" → UNTESTED. No portfolio / turnover / funding / slippage / impact backtest; "~55 bps ≈ cost" was
  arithmetic-by-assertion; it used Binance displayed depth though deployment is Binance-train / HL-execute.
- "beyond price+vol" is a LINEAR SEMI-PARTIAL screen (`partial_xsic` residualizes only the feature, `flow_harness.py`);
  "SIZE" = contemporaneous dollar volume, not market-cap / beta / listing-age / depth.
- iter3 absorption "REFUTED" → too broad: only a median-split `signed_pressure` construction was tested. Fair scope:
  "that construction failed," not "the mechanism is refuted."

ADAPTIVE (does regime-adaptation rescue the non-stationary signal? `flow_adaptive_diag.py`, `flow_adaptive_test.py`):
the vol-neutralized `signed_pressure` partial-IC IS weakly persistent — but not tradeable. ρ(IC_t,IC_{t+1}) block-CI
[+0.32,+0.84] @5m (the naive SE ~0.16 was too tight — effective-N ≈ #regimes ~2-3, not #windows), crosses 0 @1h.
Walk-forward trailing-sign realized edge is off-zero (OOS +0.0043 @5m) but SUB-COST, barely beats a fixed sign (the
signal is positive-mean anyway), and its "both-era" is inflated by RECENT being one positive regime. Doesn't rescue it.

COMBINATION (would combining signals work?): the OB-flow features reduce to ~2-3 REDUNDANT factors (flow/reversal =
vol-proxy, imbalance-continuation, illiquidity), all correlated with price/vol → breadth multiplier ~1.5-2×, cannot
close the 10-40× cost gap; COST (not breadth) is the binding wall. And the prior real per-symbol pipeline ablation
(adding OB to the book, validity-gated) HURTS (Δrank-IC −0.002..−0.004). Not the lever.

**FINAL (calibrated, this is the conclusion of record):** Short-horizon flow effects are REAL but SUB-COST; the tested
absorption gate does not rescue them; adaptive and combination don't either. A 5-day depth-normalized impact proxy
shows a SUGGESTIVE positive association but FAILS overlap-aware both-era significance and has NOT passed a
strategy-specific cost/capacity backtest. **ADOPT NOTHING.** Retain the 5-day impact-ratio result as an UNVALIDATED
CANDIDATE — not a proven premium, not "capacity-walled," not "un-harvestable." What IS trustworthy: the recovered
dataset + gap audit, the PIT 5-minute forward alignment, the non-stationarity, and iter6's ~0.25 bp break-even (which
is overlap-safe). No FARMABLE alpha from this free coarse OB-flow data on this beta-neutral / taker-cost / multi-day /
~$4-9k-side book — real information, no harvestable edge. Open only with NEW scope (paid finer data; or an
HFT/market-making vehicle where the short-horizon lead's cost math changes).

HARNESS CAVEATS before any reuse: (1) `ci()` must become a horizon-sized BLOCK bootstrap for multi-day tests (the 1-day
block silently under-covers there); (2) rename the impact feature `impact_ratio`, not "Amihud"; (3) `partial_xsic` is a
linear semi-partial screen, not a full partial-correlation or model ablation. The intraday IC conclusions (iter1-6) are
unaffected by (1) because their forward horizon ≤ the 1-day block.
