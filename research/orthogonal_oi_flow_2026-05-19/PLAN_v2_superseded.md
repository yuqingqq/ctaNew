# Orthogonal OI — Portable Model-Feature Test — v2 (2026-05-19)

> v1 superseded after 3-agent review (Methodology NEEDS-REVISION;
> Profitability NEEDS-REFOCUS; Red-team PROCEED-WITH-CHANGES). v1 fatal
> issues: flow arm re-derives 3-agent-CLOSED work (Steps 94b/95/98/§5-INT;
> sibling bottleneck v3 already dropped it); OI/flow coverage→symbol-identity
> leak (false-PASS vector R3c can't catch); 25–50 GB fetch front-loaded /
> unjustified; OI prior negatives (Phase P, v9) not surfaced. v2 fixes all.

## Decisive reconciliation (this is NOT a re-derivation — the repo roadmap
## explicitly flags it as the worthwhile un-done test)
- **Phase P (2026-05-12, REJECTED):** Binance OI/LS as a **gate**, 25-sym
  coverage — K=3 metrics_only +0.28 (placebo p29), metrics+state +1.38 (p69),
  K=4 negative. `STATUS.md:1651-1654`.
- `STATUS.md:1560` (verbatim): *"Re-test only worthwhile after expanding
  metrics ingestion to the **full 51-sym panel**; … If re-opened, prefer
  **model-feature form** (add to WINNER_21 retrain) over gate form."*
  `STATUS.md:1639`: *"Complete metrics-ingestion to all 51 symbols (then
  re-test Phase P as model-feature form, not gate)."*
- v9: OI as model input rejected Δsh −3.05 (different construction, pre-audit).
- §5-INT: order-flow/OI **interactions**, 19-sym in-universe linear → no
  signal. B★: WINNER_21 superset (incl. 4h flow) portable Δ −0.58.
- ⇒ The single genuinely-untested, roadmap-endorsed cell: **full-51 OI as
  raw model features, in the LGBM-V3.1 *portable* construction.** Flow is
  CLOSED and DROPPED. No aggTrade fetch.

## Fetch (cheap only; gated discipline satisfied)
`data_collectors/metrics_loader.py` → OI/LS for the 28 missing of 51
(23 cached). Binance Vision `metrics` daily zips are ~KB (288 rows/day) —
small, fast, no 25–50 GB pull. Build full-51 OI panel (oi_panel schema:
`oi_chg_{1h,4h,1d}`, `oi_z_{1d,7d}`, `oiv_z_1d`, `ls_count_{z_1d,chg_4h}`,
`ls_top_z_1d`, `ls_taker_{z_1d,chg_4h}`). PIT: each feature `.shift(1)`;
metrics→5m by **backward `merge_asof`** (publish lag); reuse the audited
recipe in `scripts/build_btc_oi_features.py` (has shift-proof + look-ahead-IC
audit). **Full-51 coverage is the leak fix** (uniform coverage dissolves the
NaN→identity proxy that 23-sym coverage created).

## Arms (lockstep paired-Δ; identical R3c protocol/folds/seed=20260519)
- **A0** WINNER_21 (no sym_id). **A1** A0 + full-51 OI features.
- Models: **M1** production LGBM, **M2** Ridge (standardized, per-fold train
  stats) — chases the user's "positioning could help linear" intuition.
- Trained byte-identical rows/seeds; arms differ ONLY in OI columns.

## Mandatory leak/validity guards (every reviewer fix folded)
1. Hard denylist (`target,alpha,realized,basket,_fwd,btc_target,demeaned,
   return_pct,xs_alpha`) + blocking assert `max|rankIC(OIfeat,target_A)|<0.10`.
2. **Coverage-identity guard (the fatal-leak fix):** after full-51 fetch,
   assert AUC( OI-NaN-pattern → R3c held-out-group ) ≈ 0.5 (≤0.60); any
   symbol-cycle still NaN in OI is **dropped identically from BOTH arms**
   (never NaN-passed to LGBM nor 0-imputed to Ridge). The A0 anchor is
   **recomputed on the exact surviving row-set** — the −0.33 51-sym number is
   NOT assumed as the anchor.
3. Per-feature PIT: OI `.shift(1)`, backward `merge_asof`; reuse
   `build_btc_oi_features.py` AUDIT (corr vs unshifted ≪ corr vs shifted;
   48-bar look-ahead-IC < 0.10). R0-style prefix-causal recompute at 3 cuts.
4. **MDE-in-Sharpe blocking precompute:** before fits, compute N_eff (≈5
   disjoint-group replicates over one ~0.74y window) and MDE in Sharpe units
   (correctly scaled, NOT √CPY-doubled). If MDE_Sharpe > +0.5, the only
   admissible verdicts are PASS or "underpowered/no-detectable-lever" —
   never "exhausted."
5. LOFO single-fold sign-flip kill on the paired Δ.

## Pre-registered gate (decision = portable; miss ⇒ rewrite diagnosis)
- **PASS (OI lever real):** some (arm,model) portable paired Δ(A1−A0) ≥ **+0.5**
  with block-bootstrap CI excluding 0 AND no LOFO sign-flip AND top-K
  realized-alpha spread not degraded. Sized prize = Δ.
- **FAIL (earned exhausted):** best portable Δ ≤ **+0.2** or paired CI
  includes 0 → free Binance OI does not raise the portable ceiling **even at
  full-51 in the roadmap-endorsed model-feature form** — definitively closes
  the cheapest free orthogonal-data lever. Recommendation then RANKS the
  implicated next source (on-chain flow / options positioning — NOT more
  free price/positioning data), an explicit scope decision (not executed).
- MDE>+0.5 ⇒ effect-size estimate + "no detectable OI lever", not "exhausted".
- Report full-51 AND OI-covered-subset (separate signal from coverage).

## Process
v2 is a strict narrowing of v1 implementing every reviewer-mandatory fix
(flow dropped, leak-guarded, cheap-only) AND the repo's own roadmap step ⇒
proceed to execution; the mandated checkpoint is the **3-agent results
review** vs these pre-registered gates (leaky/fudged ⇒ re-initiate). Honest
synthesis → bottleneck `B3` orthogonal-data row (sized).

## Out of scope
aggTrade-flow (CLOSED: 94b/95/98/§5-INT); OI gate-form (CLOSED: Phase P);
25–50 GB fetch; paid data; deployment. Bottleneck B★b continues independently.
