# Bottleneck Plan — v2 (2026-05-19)

> v1 superseded after 3-agent review (Red-team **DO-NOT-PROCEED**; Methodology
> **NEEDS-REVISION**; Profitability **NEEDS-REFOCUS**). v1 fatal flaws: (a) froze
> WINNER_21 so "feature-bound" was rigged (never tests the user's actual
> question — is feature *engineering* a lever); (b) B2 oracle ladder was
> order-dependent and used a tautological rank-by-realized-target ceiling
> (Sharpe ≈94, **already on disk** at `outputs/vBTC_phase1d_oracle_v2/`); (c)
> single-bottleneck framing is category-wrong (edge is convexity, not rank-IC);
> (d) re-derives closed oracle/DDI/selector results. v2 fixes all four.

## Reframed question

Not "name THE single bottleneck" (ill-posed — a low IC ceiling and a high
convexity Sharpe coexist by construction here). Instead: **produce a sized,
portability-gated ranked menu of realizable levers** — for each of {feature
engineering, model/target, harness reconfiguration, data/scope}, how much
*portable* Sharpe would a month of effort plausibly buy, given what is already
known and the one thing not yet tested.

## Reconcile (CITE, do not recompute — already on disk / in records)

| closed result | source | implication |
|---|---|---|
| alpha-oracle (rank by realized alpha) Sharpe **+94.7**; model-pred alpha-PnL **−5.0** in clean β-residual framing | `outputs/vBTC_phase1d_oracle_v2/nets_*.npy` | huge signal gap; the clean-framing model edge is ~0/neg — the +2.23 is NOT clean-alpha skill |
| per-cycle IC predictability **R²=0.005** ("genuinely unpredictable noise") | memory Phase DDI | per-cycle mean-IC is not a usable ceiling metric; do not gate on it |
| IC-selector S/N **0.32**; ALL-eligible **+2.45** > top-15-IC **+2.06** | memory; `diag_ic_selection_*` | selector is value-negative in-universe (a realizable harness lever, but in-universe only) |
| in-universe +2.23 = ~5-eff-name convexity, Herfindahl **0.19**, rotates VVV→AXS→PENDLE | `R1c_concentration_truth.json` | the edge is convexity amplification, not rank skill — closed |
| proper portability (full stack, no sym_id, unseen syms) = **−0.33** | `R3c_portability_proper.json` | the in-universe edge does NOT port — closed |
| LambdaRank/seg/cal model reframings rejected (IC ∓0.005) | memory Phase RANK/SEG/CAL | model/loss is not the limiter *for the current features* — closed |

These already answer model-line and in-universe-harness questions. The **one
decision-relevant lever never tested**: does a *richer feature set* (not the
frozen WINNER_21) raise the **portable** number?

## The one new test — B★: Feature-superset ceiling, in harness currency,
## portability-gated

On-disk, untested: **66 numeric panel columns outside WINNER_21**
(microstructure `aggr_ratio_4h`/`tfi_4h`/`avg_trade_size_4h`/`signed_volume_4h`/
`buy_count_4h`; idio higher-moments `idio_skew_1d`/`idio_kurt_1d`/
`idio_max_abs_12b`; `name_factor_loading_1d`/`name_idio_share_1d`; `xs_alpha_*`
dispersion) **+ 25 cached aggTrades `flow_*.parquet`**. Reuse the exact
validated harnesses (`build_audit_panel.train_fold_restricted`,
`phase_ah_sleeve`, `R1_baseline_frontier.aggregate_capped`,
`R3c_portability_proper`). target = `target_A` (R0-clean), no `sym_id`.

Arms (same folds/embargo/label-purge/listing-eligibility; `exit_time` retained;
each new target/feature panel re-passed through R0 prefix-causal check before
use):
- **A0** WINNER_21 (baseline, no sym_id)
- **A1** WINNER_21 + 66 panel superset
- **A2** A1 + aggTrades flow features (merged PIT, `.shift(1)`)

Metrics (harness currency, NOT noise-mean-IC):
1. pooled OOS rank-IC + **top-K(=3) realized-alpha spread bps** (the harness's
   actual objective);
2. in-universe full-V3.1 Sharpe (cap-1/3, equal, flat-4.5 + realized-√ADV);
3. **portable Sharpe = R3c protocol** (group-disjoint, no sym_id, unseen
   symbols, beta-neutral, costed) — the decisive gate.

**Pre-registered (absolute, falsifiable; miss ⇒ rewrite diagnosis not gate):**
- If A1 or A2 lifts the **portable R3c Sharpe to ≥ +0.5 with block-bootstrap
  LCB > 0** (vs A0/WINNER_21 ≈ the −0.33 floor) → **feature engineering is a
  real, fundable lever** (sized prize = the portable-Sharpe delta).
- If best superset portable Sharpe stays ≤ 0 (CI includes 0) → **feature
  engineering on free on-disk data is exhausted** (earned, not assumed —
  because we tested 91 extra features incl. microstructure, not just learners
  on 21).
- Secondary: in-universe top-K spread lift quantifies the (non-portable)
  feature value for completeness.

Power note: pre-compute N_eff and the 80%-power MDE for the R3c portable
metric; if the MDE exceeds +0.5 the result is reported as an effect-size
estimate, and the verdict is stated as "no detectable portable feature lever",
not a false "exhausted".

## B3 — Sized ranked-lever synthesis (the deliverable)

A single table: lever → realizable in-universe Sharpe Δ → **does it port?**
(R3c) → effort → recommendation. Pre-filled from reconciled closed results,
with B★ supplying the only open cell:

| lever | in-univ Δ | ports? | source |
|---|---|---|---|
| model/target reframe | ≈0 | n/a | closed (Phase RANK/DDI) |
| harness: drop IC-selector | ≈+0.4 | **no** (R3c convexity path) | closed (selector S/N 0.32) |
| feature engineering (superset+flow) | **B★** | **B★** | THIS test |
| data/scope (orthogonal/on-chain) | unknown | only un-refuted | recommendation |

Verdict = which lever has a *portable* prize > 0. No pre-written conclusion.

## Process
v2 re-reviewed by the same 3 angles to clear the Red-team DO-NOT-PROCEED before
B★ runs. After B★, 3-agent results review vs these pre-registered gates;
leaky/fudged measurement ⇒ re-initiate. Cite-don't-recompute enforced.

## Out of scope
Recomputing existing oracle/DDI/selector results; the closed portability
conclusion; orthogonal-data acquisition (recommendation only); deployment.
