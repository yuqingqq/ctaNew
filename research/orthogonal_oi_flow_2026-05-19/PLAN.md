# Orthogonal OI + aggTrade-Flow — Portable VALIDATION — v3 (2026-05-19)

> v2 superseded by an explicit user correction: **"do not directly trust the
> docs — we need to validate."** v1/v2 used STATUS.md/PROGRESS.md claims
> (Phase P, §5-INT, v9) to declare OI/flow "closed/redundant" and scope them
> out. That is doc-trust, not validation. v3: doc "negatives" are treated as
> **unvalidated hypotheses we measure ourselves**; the flow arm is RESTORED;
> only the doc-independent methodological guards are kept.

## Question
Does adding orthogonal **OI/positioning** and **aggTrade order-flow**
features, full 51-sym, lift the **portable** alpha-residual number (R3c
unseen-symbol Sharpe) in the LGBM-V3.1 construction and a linear model —
**validated by direct measurement**, not inferred from prior write-ups?

## Prior claims = hypotheses to VALIDATE (not trusted, not used to scope out)
Recorded only so the results can be compared to them — none of these gate
what we run:
- doc-claim: OI gate-form, 25-sym, "rejected" (Phase P).
- doc-claim: order-flow/OI interactions, 19-sym in-universe linear, "no
  signal" (§5-INT). doc-claim: OI model-input "−3.05" (v9).
- our own validated result (kept as a prior, it is ours not a doc): B★
  WINNER_21-superset portable Δ −0.58; per-feature |rankIC| ≤ 0.036.
We re-test OI **and** flow as raw model features, full-51, portably — the
exact form prior write-ups never validated — and let the measurement decide.

## Fetch (full 51 — the validation requires it; user-authorized)
- **OI:** `data_collectors/metrics_loader.py` → 28 missing of 51 (23 cached).
  Vision `metrics` zips ~KB/day — cheap/fast.
- **aggTrade flow:** 26 missing of 51 (25 cached). Stream per-symbol-day:
  download Vision zip → 5m features via `features_ml/trade_flow.py` →
  discard raw (no 25–50 GB on disk; bandwidth/parse is the cost, accepted
  for a validated answer). Run in background.
- Full-51 coverage is also the **leak fix** (uniform coverage dissolves the
  NaN→symbol-identity proxy that subset coverage created — see guard 2).

## Arms (lockstep paired-Δ; identical R3c protocol/folds/seed=20260519)
A0 WINNER_21 (no sym_id) · A1 +OI · A2 +flow · A3 +OI+flow.
Models: M1 production LGBM, M2 Ridge (standardized, per-fold train stats —
chases the "positioning/flow may help linear" intuition). Byte-identical
rows/seeds; arms differ ONLY in added feature columns.

## Doc-independent guards (kept — these stand on their own logic, no doc)
1. Hard denylist (`target,alpha,realized,basket,_fwd,btc_target,demeaned,
   return_pct,xs_alpha`) + blocking assert
   `max|rankIC(new_feat,target_A)| < 0.10` (evidence file written).
2. **Coverage→identity leak guard:** after full-51 fetch, assert
   AUC( new-feature-NaN-pattern → R3c held-out-group ) ≤ 0.60 (~0.5 ideal);
   any symbol-cycle still NaN is **dropped identically from BOTH arms**
   (never NaN-passed to LGBM nor 0-imputed to Ridge); the A0 anchor is
   **recomputed on the exact surviving row-set** (the −0.33 51-sym number is
   NOT assumed).
3. Per-feature PIT proof (validated, not assumed): OI `.shift(1)` + backward
   `merge_asof`; flow built `label="left"` within-bar then `.shift(1)` onto
   the panel `open_time` grid by **exact-timestamp join (NOT merge_asof)`;
   per-feature shift table (VPIN already strictly PIT — not double-shifted;
   `signed_volume_z`/within-bar features shifted). Independent shift-proof +
   48-bar look-ahead-IC < 0.10 audit (the `build_btc_oi_features.py` pattern)
   run and reported for BOTH OI and flow. R0 prefix-causal recompute at 3
   interior cuts on the augmented panel.
4. **MDE-in-Sharpe blocking precompute:** N_eff from the ~5 disjoint-group
   replicates over one ~0.74y window; MDE in Sharpe units, correctly scaled
   (NOT √CPY-doubled). If MDE_Sharpe > +0.5 → only PASS or
   "underpowered/no-detectable-lever" admissible, never "exhausted".
5. LOFO single-fold sign-flip kill on each paired Δ.

## Execution order (cost-efficiency only — NOT doc-based scoping)
- **Stage 0 (cheap, starts now):** fetch full-51 OI; run A1 (M1+M2) +
  A2/A3 on the already-cached flow symbols, full leak-guard. Early indicative
  read.
- **Stage 1 (heavy, background, proceeds regardless — user wants flow
  validated):** fetch the 26 missing aggTrade symbols, rebuild full-51 flow,
  re-run A2/A3 at full coverage (this is what makes the flow arm leak-valid;
  the cached-only subset is the confounded survivorship universe and is
  reported only as the Stage-0 indicative figure, never the verdict).

## Pre-registered gate (decision = portable; miss ⇒ rewrite diagnosis)
- **PASS:** some (arm,model) full-coverage portable paired Δ(Ai−A0) ≥ **+0.5**,
  block-bootstrap CI on per-cycle diff excludes 0, no LOFO sign-flip, top-K
  realized-alpha spread not degraded. Sized prize = Δ. (If this contradicts
  the doc "closed" claims, the docs were stale — that is the point.)
- **FAIL (validated, not doc-asserted):** best full-coverage portable Δ ≤
  **+0.2** or paired CI includes 0 → free Binance OI+flow do not raise the
  portable ceiling, **measured directly at full-51 model-feature form**.
  This is OUR validation of (not citation of) the prior negatives.
- MDE>+0.5 ⇒ effect-size estimate + "no detectable lever", not "exhausted".
- Report full-51 AND covered-subset (separate signal from coverage).

## Process
v3 is the user-corrected scope (validate, restore flow, doc-independent
guards only). Proceed to execution; mandated checkpoint = **3-agent results
review** vs these pre-registered gates (leaky/fudged ⇒ re-initiate). Honest
synthesis → bottleneck `B3` orthogonal-data row (sized).

## Out of scope
Paid data; deployment. (Bottleneck B★b continues independently.) No claim is
closed by doc citation — only by our own measurement here.
