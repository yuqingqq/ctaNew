# R4 — Synthesis & Decision (2026-05-19)

Written only after R0–R3 ran. No pre-written verdict. All gates were
pre-registered in `PLAN.md` v3; misses rewrote the Diagnosis, never the gate.

## Headline

**A deployable, profitable, universe-robust system exists — it is the
existing V3.1 24h 6-sleeve stack, correctly understood.** The project's
prior blockers ("62–86% VVV ⇒ fragile", "universe-overfit ⇒ can't replay on
a different symbol set", "no-sym_id ⇒ −0.39") were **measurement artifacts**,
not properties of the strategy. Rigorous, pre-registered, leak-checked
reconstruction overturns all three.

## What each test established

| test | pre-registered gate | result |
|---|---|---|
| **R0** integrity | target PIT-clean; flagged smells = convention not leak | **PASS** — `target_A` recompute 1.1e-5·std, prefix-causal **exactly 0**; the feared pooled-norm leak does not exist |
| **R1** baseline+frontier | a cap clears criteria 1–6 | **PASS / deployable.** V3.1 reproduced exactly (+2.23, 7/9). Per-cycle risk diversified (Herfindahl 0.094); ex-VVV +1.99; drop-5 mean +2.14 (0/30 neg); cost-stress +1.96–2.13. Diagnosis premise (H≥0.40, fragile) **refuted** by data |
| **R2** profit levers | lift ≥ +0.3 over R1, LOFO clean, else refuted | **both refuted honestly.** rvol/ret-as-features −1.84; longer holds monotone-worse. By-product: R1 survives realized √ADV (+2.13) and tail-stressed 3× cost (+1.96) ⇒ not a cost artifact |
| **R3** robustness (diagnostic) | informs sizing, never vetoes | drop-k frac-positive **1.00** at k≤5; **out-of-universe (no sym_id, unseen symbols, beta-neutral) pooled Sharpe +1.35, 4/5 groups +** |

## Deliverable (deployable now — decoupled delivery)

- **System:** existing V3.1 production stack (WINNER_21 incl. sym_id, LGBM
  5-seed, rolling-IC top-15, conv_gate + filter_refill + PM_M2, 6 equal-weight
  overlapping 24h sleeves), deployed via the **cap-1/3** or **vol-norm**
  variant to bound single-name *dollar* exposure.
- **Honest expected performance:** Sharpe ≈ **+2.0** net @4.5 bps
  (cap-1/3 +2.06, 8/9 folds; vol-norm +2.13, lowest maxDD −2864; raw +2.23).
  Robust: ex-VVV +1.99; drop-5 mean +2.09; realized √ADV +2.13;
  tail-stressed 3× cost +1.96; out-of-universe (unseen symbols) +1.35.
- **Sizing/kill-switch (R3):** deploy at **0.5–0.7× full size** (haircut for
  out-of-universe heterogeneity — group g3 was −0.99 — and ~1y sample);
  hard kill-switch at cumulative DD **−6,265 bps (1.75× in-sample maxDD)**;
  live per-name dollar-exposure monitor (VVV-type operational risk).

## The one genuine residual risk (operational, not statistical)

86% of *cumulative dollar* PnL historically routed through VVVUSDT (a
low-float meme). Sharpe survives its removal (+1.99) and the construction
diversifies *risk* (H 0.094), but a single name driving the dollar P&L is a
real live liquidity/delisting exposure. Mitigated — not eliminated — by the
cap-1/3 / vol-norm variant + the dollar-exposure monitor + kill-switch. This
is the honest caveat to carry into live deployment, exactly the
frontier-decides-tradeoff the user asked for.

## Deployment-hardening (must-fix before live)

`live/vBTC_paper_bot.py` ships **K=4, no sleeve** — NOT this research stack
(K=3 + 6-sleeve + cap/vol-norm + kill-switch). The shipped artifact and the
validated strategy diverge; the paper-forward currently tests a different
system. Reconciling the bot to the R1/R3 config is the top operational item.

## What is now closed vs open

- **Closed (decisively, honest OOS):** rvol/ret-as-model-features;
  longer-than-24h holds; the "universe-overfit / not-portable" thesis (refuted
  — it *is* portable at +1.35 to unseen symbols); the "cost-amortization
  artifact" concern (refuted — survives 3× tail cost).
- **Open / next highest-EV (data-ranked):** (1) reconcile the live bot to the
  validated stack + wire the kill-switch (operational, highest value — there
  is a deployable edge being mis-shipped); (2) reduce VVV dollar-dominance
  structurally (per-name notional cap in execution, already shown ≈Sharpe-
  neutral at cap-1/3); (3) the g3 negative group — investigate which symbol
  cohort fails to generalise (refines the deployable universe).

Full record: `PLAN.md`, `reviews/ROUND{1,2}_*`, `results/R{0,1,2,3}_FINDINGS.md`,
`results/R*_results.json`, `scripts/R*.py`.
