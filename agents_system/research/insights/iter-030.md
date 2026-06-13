# iter-030 — Per-symbol TIME-SERIES + ALT-INDEX hedge — NO-CANDIDATE

PHASE 2 (broadened scope). Tested the last untested structural corner: a **per-symbol
time-series strategy** (trade each symbol on its OWN per-symbol pred — long pred>0 /
short pred<0, sized by sign or |pred|, 24h hold via 6 sleeves) with the NET book
exposure hedged by the **equal-weight ALT-INDEX** instead of BTC.

## The idea / the fix vs iter-004
iter-004 ran per-symbol pred-timing + a **BTC-beta hedge** → LOST −1.97 Sharpe; the
per-sym TS-IC of pred was real (+0.0116 / t≈4.5) but didn't monetize. iter-006 argued the
DD-driving risk is the **ALT-complex factor** (alts −24% while BTC −7%; BTC under-hedges
alts). HYPOTHESIS: neutralize NET ALT-beta with the equal-weight alt index (PIT trailing
beta-to-alt) to isolate the idiosyncratic beta-residual the pred actually targets — which a
BTC hedge leaves exposed.

## Online research (cited)
Crypto literature: **time-series momentum is strong, cross-sectional momentum is weak** in
crypto — but a key caveat: single-asset TS strategies are **NOT zero-net-investment**, so
they carry net market/factor beta (which is precisely why the hedge instrument matters,
and why per-asset TS "profit" is often disguised directional beta). Beta-neutralizing crypto
books is hard (thin-tail liquidity, volatile short funding). Sources below.

## Build (transport-first, PIT)
- Per-symbol TS weight at cycle t = f(pred_t) {sign, or signed |pred| clipped at 3}, 1/n
  normalized, held 24h via 6 overlapping sleeves (same cadence as baseline).
- Net book beta to the hedge index = Σ w_s·β_s (β trailing-180×4h, `.shift(1)`, PIT).
  Hedge leg = −net_beta units of the index, PnL realized t→t+4h. Cost 4.5 bps/leg incl hedge.
- Three hedges × two sizings, on HL70 (70-sym, 2025-26) AND EXT (23-sym, 2021-26).
- Net-beta neutralization verified by regressing the net PnL stream on the alt-index fwd return.

## Results (net of 4.5 bps cost)

| config | HL70 net Sh | HL70 gross Sh | EXT net Sh | EXT gross Sh | β(net→ALT) HL70 / EXT |
|---|---|---|---|---|---|
| sign, NO hedge   | **−0.92** | −0.67 | **+0.15** | +0.24 | +0.098 / −0.446 |
| sign, BTC hedge (≈iter-004) | **−0.68** | −0.14 | **+0.37** | +0.64 | +0.029 / −0.118 |
| **sign, ALT hedge (the fix)** | **−0.72** | +0.23 | **−2.55** | −1.90 | −0.010 / +0.006 |
| clip, NO hedge   | −0.98 | −0.71 | +0.25 | +0.37 | +0.130 / −0.524 |
| clip, BTC hedge  | −0.81 | −0.28 | +0.38 | +0.68 | +0.037 / −0.133 |
| **clip, ALT hedge** | −0.73 | +0.13 | **−1.69** | −1.09 | −0.015 / +0.011 |

Baseline (cross-sectional book) for reference: HL70 Sharpe **+1.93**.

### Transport check — NO config is positive on BOTH universes
- sign/clip **NO hedge** & **BTC hedge**: SIGN-FLIP (HL70 −0.7 to −1.0 / EXT +0.15 to +0.38) = universe-overfit.
- sign/clip **ALT hedge**: "same-sign" only because it is **negative on both** (HL70 −0.72/−0.73, EXT −2.55/−1.69).

## Mechanism — the hypothesis is REFUTED, and decisively the wrong way round
1. **Net-beta neutralization works as designed**: the alt-hedge drives β(net→ALT) to ≈0
   (−0.010 HL70, +0.006 EXT) vs +0.10/−0.45 (no hedge); BTC hedge only partially. So the
   build is correct — this is a real result, not a hedge-sizing bug.
2. **But on EXT the alt-hedge is CATASTROPHIC** (−2.55 / gross −1.90), the *opposite* of the
   hypothesis. Root cause: on EXT the per-sym TS book is structurally **net-SHORT the alt
   complex** — only **22% of preds are >0** per cycle (avg net-beta −0.69) → it shorted the
   2021-22 alt bear. THAT net-short-alt directional bet is the ONLY thing making EXT positive
   (no-hedge +0.15, BTC-hedge +0.37). The alt-index hedge buys the index back to neutralize
   that beta → it **strips out the profitable directional exposure** → gross collapses to negative.
3. **The pred's edge is directional/factor beta, not the isolated residual.** Per-symbol
   TS-IC: pred→beta-RESIDUAL = +0.0047 t1.55 (HL70, insignificant) / +0.0132 t4.83 (EXT,
   reproduces iter-004); pred→RAW-return = +0.0061 / +0.0123 — essentially the SAME on EXT.
   The residual the pred "targets" carries a tiny, cost-fragile IC; what monetizes (on EXT)
   is the net directional alt bet. **Isolating the residual = removing the alpha.** On HL70
   neither the residual nor the directional version monetizes net of cost (all negative).
4. So the iter-004 "real TS-IC that doesn't monetize" finding is confirmed and explained: the
   residual IC is real but (a) too small to beat cost as a per-symbol TS book, and (b) the
   alt-hedge that was supposed to "isolate" it instead removes the directional component that
   was carrying the EXT profit. **The right hedge does NOT fix iter-004 — it makes it worse.**

## Diversification
Moot — every variant is either negative net Sharpe (alt-hedge, both universes; all HL70) or
universe-overfit sign-flip (no/BTC hedge). A losing or non-transporting sleeve cannot improve
the combined Calmar regardless of correlation.

## Verdict: NO-CANDIDATE — closes the per-symbol-TS structural corner
The per-symbol time-series + alt-index hedge does NOT monetize and does NOT transport.
This independently re-confirms the deep Phase-2 mechanism (iter-029): crypto 4h cross-section
is **BTC-beta-dominated + trend-persistent** — a per-asset directional pred book is a disguised
net-beta bet (profitable only when it accidentally shorts an alt bear, i.e. EXT 2021-22), and
removing that beta to "isolate the residual" removes the alpha. The 4h beta-residual is near
the achievable per-asset ceiling (TS-IC ~0 HL70 / +0.013 EXT, cost-fragile). The cross-sectional
rank book remains strictly better (+1.93) because it nets the directional beta away by
construction (long-K minus short-K) while keeping the residual ranking, rather than betting it
directionally and then hedging it back out.

## Champion unchanged
BASELINE HL70 regime-hybrid held-book (Calmar +1.68 / Sharpe +1.93) + iter-012 vol-norm reactive stop (k=2.0).

## Scripts / artifacts
- `agents_system/research/scratch/iter030_persym_ts_althedge.py` (per-sym TS engine, 3 hedges × 2 sizings, PIT)
- `outputs/iter030/preds4h_{HL70,EXT}.parquet`, `iter030_summary.csv`, `iter030_net_streams.parquet`

## Sources
- Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market (Han/Kang/Ryu, SSRN 4675565)
- Cross-sectional momentum in crypto markets (Starkiller Capital)
- Dynamic time series momentum of cryptocurrencies (ScienceDirect S1062940821000590)
- Moskowitz/Ooi/Pedersen, Time Series Momentum (JFE 2012)
