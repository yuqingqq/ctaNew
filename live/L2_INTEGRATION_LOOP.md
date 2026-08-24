# L2 Integration & Optimization Loop — charter (launched 2026-07-14 ~10:30 UTC, target ~10h)

Goal: **maximize realizable performance improvement from L2 (Binance bookDepth) data**, integrated into the convexity
v4 book. Driven by research + adversarial-review subagents. Honest, gated, both-eras.

## Evidence going in (this session, `RESEARCH_LOOP_20260707.md` add. 72)
- **L2-as-cross-sectional-alpha = NULL/negative**, proven on the REAL pipeline (V0_LEAN per-symbol RidgeCV,
  baseline reproduces +0.0301 = strategy honest rank-IC): +imb_ewma Δ −0.0020 [CI<0] rec / −0.0026 oos; LGBM same
  (Δ −0.0030 [CI<0]). Reversion test: L2 imbalance is CONTINUATION not reversion; price-reversion is real (+0.04 IC)
  but that's already the book's `resid_rev`; 4h-rebalanced reversion dies on cost.
- **L2-execution/capacity = REAL edge (quantified):** $100k impact spans BTC 0.00% → RUNE 0.70% (1248×); full
  universe median 0.64%, 59/176 names >1%. This is the +4.22→~+1.2 gap. `bookdepth_impact.py`.

## THE GATE (nothing is "adopted" without ALL of):
1. **Baseline-validated real pipeline** — the ablation's V0_LEAN baseline MUST reproduce ~+0.030 rec / +0.024 oos
   (else the harness is invalid — see the 3 wrong answers this session). Use `bookdepth_real_ablation.py` machinery.
2. **Both-eras** — Δ (rank-IC or realizable-Sharpe) CI-off-zero in RECENT *and* OOS (day-clustered bootstrap).
3. **Adversarial review** — a review subagent tries to break it (look-ahead, overfitting, favorable-corner,
   confounded harness, cost realism). Survives → adopt. Else → log as null.
Every iteration appended to `RESEARCH_LOOP_20260707.md` + the LOG below.

## Backlog (priority order)
**WS1 — Depth-aware execution (PRIMARY, real edge):**
- E1 impact model: refine per-name impact from bookDepth (reconstruct 0.2/1/2/3/5% one-side depth), validate vs the
  live HL slippage logs (`live/state/convexity_twobook/slippage.csv`) if present.
- E2 impact-aware backtest: apply real impact to the book's actual 1L/2S positions (from `hl_tgt_res_*_honest`
  preds) — flat-cost vs real-impact net selection-spread PnL/Sharpe, both eras.
- E3 depth-aware sizing: size each leg ∝ f(depth) (not equal-$); optimize the rule; measure realizable-Sharpe gain
  vs equal-weight, both eras. Keep dollar-neutral. (cf. `depth_resize.py`.)
- E4 liquidity gating + capacity curve: drop names too thin at target AUM; plot Sharpe vs AUM; find the AUM ceiling.
- E5 HL port: redo E2-E4 on the Hyperliquid book (execution venue, thinner) for the real numbers.

**WS2 — L2 alpha search (GATED, belief-driven; expect null):**
- A1 exhaustive L2 feature × horizon × condition sweep via the REAL pipeline (baseline-validated), both eras.
- A2 directional / non-beta-neutral L2 signal (raw return), as a SEPARATE sleeve (not the beta-neutral book).
- A3 interaction/regime-gated L2 (e.g., L2 only in high-dispersion or specific regimes).

**WS3 — Risk (medium):**
- R1 L2 squeeze-veto for the short leg (limitation #4): veto thin/one-sided-book shorts; does it cut the squeeze
  tail without killing the short edge? Both eras, book-level.

## Iteration log
(iteration N: task → agent(s) → result → review verdict → adopt/null)
- init: charter written; launching E2/E3 (execution backtest+sizing) + A1 (gated alpha sweep) as iteration 1.
- iter1 (WS2 alpha sweep, COMPLETE): gate validated both eras (baseline +0.0301 rec / +0.0170 oos, imb_ewma
  reproduced to the digit). **ALL 6 constructions NULL** — C1 dispersion-inter (HURTS rec/noise oos), C1 regime-inter
  ×btc_rvol (HURTS/noise), C3 shape l2_slope (noise/noise), C4 fragility (HURTS/noise), C2 imb_ewma short-pool
  (HURTS/noise), C3 near-touch (noise; recent-only). Short-leg tests: NO short-side edge (selection-spread CI straddles
  0 everywhere), even with the asymmetry favoring the variant. Model-free cross-check: imbalance is continuation,
  collinear with V0_LEAN momentum, marginally harmful — real redundancy, not a preproc artifact. Scripts
  `live/l2_alpha_constructions.py`, `l2_alpha_diag.py`. **→ WS2 (L2-alpha) CLOSED. No adoption.**
- iter2: WS3/R1 squeeze-veto (RISK/tail) launched.
- iter2 (WS1 execution E2/E3/E4 COMPLETE, `l2_exec_backtest.py`) — MAJOR finding, review pending:
  - Harness VALIDATED: realizable Sharpe +1.21 at $2k/side = the "~+1.2 at real size" anchor; paper (spec-exact) +2.35 rec / +0.98 oos.
  - **CAPACITY CEILING (realizable Sharpe→0): RECENT ~$4k/side, OOS ~$9k/side.** The full-universe 1L/2S book is a
    FEW-$k/leg strategy — short legs are systematically micro-cap (median 1% one-side depth ~$97k; 52% of shorts <$100k).
    The paper edge lives almost entirely in sub-$10k/leg thin-name exposure.
  - **Depth-aware SIZING = NULL** (doesn't extend the ceiling; no cap binds below names' depth; 0% paper-edge recovery
    at tradeable sizes; only softens the −30 catastrophe at institutional size). Refutes the sizing hypothesis.
  - **Liquidity GATING = the effective lever** (adopt-candidate): drop legs w/ one-way impact >0.1–0.2% → +1.6 Sharpe
    @$10k (40% bars kept) / +1.2 @$25k (32% kept) RECENT ≈ 5× capacity extension; fails >$100k. OOS "weaker" → needs
    explicit both-era capacity curve. Caveats: OOS depth-coverage only 28% of bars; Binance≈HL order but ~2× harsher on
    thin alts (one real-fill cross-check). → skeptical REVIEW spawned before adoption.
- iter2 (WS1 execution REVIEW, agent afed1b7ff): impact model + PIT verified CLEAN (no 100× bug, RT=2×, strict prior-bar
  join). **(a) CAPACITY CEILING CONFIRMED + robust + CONSERVATIVE** (real HL cross-check is 0.70× = model harsher than
  reality): ~$4.1k/side rec / ~$9k/side oos, depth-aware sizing useless. STANDS. **(b) LIQUIDITY GATING = BROKEN →
  NOT adopted.** OOS gating curve NEGATIVE in every cell (inert at $10-25k — OOS covered legs have no thin tail, p99
  impact 0.11%); RECENT +1.58/+1.19 bootstrap CIs CROSS ZERO (P=0.88/0.79), best-threshold post-hoc, neighbor cells
  swing ±2 Sharpe, and **~75% of the "win" is ONE name (ZEC)** (drop ZEC → +1.58→+0.39). Eras non-comparable (OOS
  covered window 2023-10..2024-07 only, touch=0.14 fallback 100%, survivorship-optimistic yet still negative). Gating
  is a universe/era artifact, not a portable lever. Scripts `live/l2_review_*.py`.
  **→ WS1 verdict: the CAPACITY CEILING (~$4-9k/side) is the real finding; NO L2 lever (sizing OR gating) robustly
  extends it. No adoption. E5 (HL-book port) is DATA-BLOCKED (no historical HL depth, only live).**
- iter2 (WS3/R1 squeeze-veto COMPLETE, `l2_squeeze_veto.py`): **NULL.** No L2 fragility feature has a both-era squeeze
  IC (asym1 +0.045 rec vanishes −0.005 oos; imbstd/slope consistently NEGATIVE = opposite hypothesis). Veto/downsize:
  best cell dSharpe CIs span ~[−2,+2] (indistinguishable from 0); aggressive modes flip sign across eras (favorable-
  corner); only 3/36 both-era cells help, scattered/incoherent, inside noise; p1-tail cut recent but WORSENS oos. Only
  genuine both-era fact: high imbstd/slope = fatter-tailed shorts = generic VOLATILITY (vol-target at best, not an L2
  squeeze veto). Do not adopt.

## LOOP CLOSED 2026-07-14 (~1.5h; backlog exhausted, not 10h — answer is clear, refused to grind nulls)
**ADOPTED: nothing.** L2 adds no cross-sectional alpha (WS2, 6 constructions, exhaustively gated null), no robust
capacity lever (WS1: sizing null, gating = recent-only/ZEC artifact killed by review), no squeeze-tail reduction
(WS3 null). **The ONE durable, verified output is DIAGNOSTIC: the strategy is capacity-capped at ~$4-9k/side** (edge
concentrated in micro-cap shorts; robust + conservative per the live-HL cross-check). L2's value here = it told us the
binding constraint is SIZE, not signal. Every "positive" this loop surfaced (gating +1.6) was a favorable-corner the
review gate caught — vindicating the strict both-era + baseline-validated + adversarial protocol. Open (needs new
DATA/scope, not this loop): (i) deeper-universe rebuild for scalability; (ii) the 5-15min directional signal as a
separate HFT track; (iii) historical HL depth for a true execution port.

## DATA RE-FETCH IN PROGRESS (2026-07-15) — all conclusions below are PROVISIONAL pending complete-data re-run
User flagged the original L2 fetch was INSUFFICIENT (OOS ~28% coverage, 2024-08→2025-09 middle unfetched). Re-fetching
full continuous coverage. Scope (user-chosen): **the 176-panel universe FULLY**, NOT the 615-symbol illiquid tail
(Binance has 791 bookDepth symbols total; the 615 missing are micro-cap tail — skip unless a deeper-universe rebuild).
Completion plan:
1. Resume fetch `baned9lhw` (ETCUSDT→end, 2023-01-01..2026-06-30) — running; watcher `bey1qbezx` pings on done/stall.
   (Earlier the user's fetch DIED at ENSUSDT #54/176; resumed from ETCUSDT.)
2. THEN tail top-up: all 176 symbols, 2026-07-01..2026-07-14 (latest available is 07-14; loader merges) — the done
   A→ENS symbols stop at 06-30 and need this tail too.
3. THEN re-run `l2_validate_coverage.py` — acceptance: all pre-2025 syms both-era + middle-filled, density>90%,
   imb1-NaN~0, 0 unreadable. Note near-touch (±0.2%) features are RECENT-ONLY (Binance added post-2024) — structural.
4. THEN re-run the validated test chain on complete data: validity gate (+0.030) → `l2_influence_quant`/`_cherrypick`
   → `l2_exec_backtest` → `l2_comovement`. Flag honestly if fuller OOS changes any verdict.

### ✅ COMPLETE-DATA RE-RUN DONE (2026-07-15) — VERDICT UNCHANGED, now definitive.
Data: 176 syms, 2023-01→2026-07-14, median 100% density, 159/176 both-era (validated). OOS folds widened to
2023-06→2025-09 (9 quarters). Results: (a) INFLUENCE — validity gate healthier (RECENT base +0.0301, OOS base +0.0209
vs +0.0170); Δrank-IC(+OB) RECENT −0.0024 [−.0046,−.0001] HURTS, OOS −0.0007 noise; sel-spread RECENT +7.58bps CI~0 /
OOS −3.88bps HURTS; **cherry-pick 0/8 subsets both-era positive.** (b) CAPACITY — ceiling ~$4.1-4.3k/side, depth-aware
null, **gating +1.6 was an artifact (GONE on full data → confirms the review)**, HL sanity 0.70× (conservative).
(c) CO-MOVEMENT — imbalance corr(alt,BTC) RECENT +0.50 / OOS +0.05 (era-specific), liquidity PC1 30-48% both eras.
**Net: the re-fetch fixed OOS coverage + killed the last false positive, and CONFIRMED (not overturned) every
conclusion. OB = no alpha, mildly negative as a feature, diagnostic-only (capacity). L2 thread definitively closed on
complete, both-era-validated data.** (Minor fix: `l2_exec_backtest.py` import `live.bookdepth_impact`.)

## FINAL REVIEWED QUANTIFICATION of OB influence (2026-07-15, `l2_influence_quant.py` + `l2_influence_cherrypick.py`)
Validated pipeline (baseline reproduces +0.0301 rec / +0.0170 oos EXACTLY). Independently reviewed (agent) AND
corroborated by a parallel cherry-pick — both agree:
- **Δ rank-IC (adding OB to the model): −0.0031 [−.0057,−.0007] RECENT (HURTS, ≈−10% of the +0.030 alpha, CI<0);
  −0.0027 [−.0070,+.0015] OOS (noise).** Δ selection-spread within noise both eras (the +2.33→+2.44 rec / −1.47→−0.47
  oos Sharpe bumps are vol artifacts — Δmean-spread CIs straddle 0).
- **Cherry-pick: 0 of 8 OB subsets** (each single feat + liq+shape + all-6) is both-era CI-positive on rank-IC OR
  spread. No hidden positive configuration.
- Mechanism: OB's only real info = LIQUIDITY (l2_liq1 raw IC +0.0288 rec ≈ the whole model's IC) — but REDUNDANT with
  V0_LEAN's vol features (idio_vol/rvol/atr), so adding it only injects noise. PIT clean (exit_time=open_time+4h 100%).
- **VERDICT (reviewed, definitive): OB influence = NEGATIVE-TO-NEUTRAL, never significantly positive. The strategy is
  better WITHOUT the order book as a feature.** Its value is diagnostic (capacity ceiling) only.
