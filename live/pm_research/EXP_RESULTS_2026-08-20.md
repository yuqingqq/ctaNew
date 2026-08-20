# First model results — P-2026-003 (2026-08-20)

Two experiments, on data already collected. Both run at knowledge time
(`recv_ns`), never payload timestamps.

## EXP-M6 — settlement truth: **CONFIRM**

| convention | n | agree | agree \|margin\|>0.5bp |
|---|---|---|---|
| **S60(T) vs S60(t0)** | 1465 | **99.8%** | **99.9%** |
| S30(T) vs S30(t0) | 1465 | 96.1% | 96.9% |
| S60(T) vs S30(t0) | 1465 | 96.5% | 97.4% |
| meanS60[t0,T] vs S60(t0) | 1465 | 86.9% | 88.2% |

Gate required ≥99.0% pooled and ≥99.5% on the >0.5bp subset. **Passed.**
Settles the open ambiguity: the averaging window is **w = 60 s**, not the
full 300 s range — the full-range reading scores 86.9% and is refuted.
Reading the same grid at knowledge time gives 99.3% vs 99.8% at event time;
that 0.5 pp gap is the size of the look-ahead a careless backtest would bank.

## EXP-BLEND — does our model beat the book? **No, uniformly.**

σ fitted by MLE on realised winners: **1.538 bps/√s ≈ 86%/yr** (plausible).
Walk-forward (fit on prior days, score the next), paired per (window, time):

| r (s) | n | Brier model | Brier book | Δ |
|---|---|---|---|---|
| 270 | 1477 | 0.2368 | 0.2062 | +0.0305 |
| 240 | 1477 | 0.2150 | 0.1899 | +0.0250 |
| 180 | 1477 | 0.1759 | 0.1503 | +0.0256 |
| 120 | 1477 | 0.1302 | 0.1057 | +0.0245 |
| 60 | 1477 | 0.0815 | 0.0531 | +0.0283 |
| 30 | 1477 | 0.0490 | 0.0174 | +0.0316 |

The book wins at **every** horizon by a stable 2.5–3.2 Brier points — not a
horizon-specific weakness, a uniform information deficit. In E-X2's terms
**ŵ → 0**: we hold no informational edge, so this is a **spread-capture /
market-making strategy, not an alpha strategy**. Every p̂-conditioned claim in
the plan is downgraded accordingly.

## The finding that survives: the book is miscalibrated

| book bucket | n | mean book | realised | gap |
|---|---|---|---|---|
| 0.0–0.1 | 1609 | 0.033 | 0.029 | −0.004 |
| 0.1–0.2 | 486 | 0.146 | 0.103 | **−0.043** |
| 0.2–0.3 | 547 | 0.251 | 0.239 | −0.011 |
| 0.3–0.4 | 698 | 0.349 | 0.317 | **−0.032** |
| 0.4–0.5 | 785 | 0.449 | 0.456 | +0.007 |
| 0.5–0.6 | 805 | 0.546 | 0.574 | +0.028 |
| 0.6–0.7 | 725 | 0.645 | 0.703 | **+0.059** |
| 0.7–0.8 | 645 | 0.749 | 0.800 | **+0.051** |
| 0.8–0.9 | 646 | 0.849 | 0.890 | +0.041 |
| 0.9–1.0 | 1916 | 0.968 | 0.992 | +0.024 |

A monotone slope, not a level shift: longshots **over**priced, favourites
**under**priced — the classic favourite–longshot bias, i.e. the book is
underconfident. Magnitudes of 3–6 cents against a 2–4 cent spread. This is the
first time the FLB has been measured on our own data rather than inherited.

## Caveats, stated plainly

1. **1.4 days.** One walk-forward test day; the paired ΔBrier has no usable CI.
2. **Cross-window dependence.** 1,477 windows ride a handful of underlying
   paths; effective N is far below 8,862.
3. **Up-drift confound.** Realised up-rate is 0.538 against a book mean of
   ~0.52. Some of the positive gaps in high buckets is sample drift, not bias.
   The pattern is a SLOPE (negative low, positive high), which drift alone does
   not produce — but drift inflates the high side.

## What this changes

- Strategy identity is decided: **pure MM, not alpha.** Quote around the book
  with inventory skew; drop the p̂-as-edge thesis.
- The FLB is the candidate edge and it is a *calibration* trade, not a
  *forecast* trade — it needs no better model, only correct sizing against a
  known bias.
- Next: walk-forward the FLB (fit recalibration on day d−1, test on day d),
  de-drift the sample, then markout the fills a band-quoting maker would
  actually get.
